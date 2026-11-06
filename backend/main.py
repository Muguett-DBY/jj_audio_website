"""FastAPI application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from audio_to_tab.isolate import SUPPORTED_MODELS
from audio_to_tab.phase3 import stem_names_from_artifacts
from backend.account import delete_user_account
from backend.auth import (
    REFRESH_COOKIE,
    create_access_token,
    create_anon_session_token,
    create_anonymous_user,
    create_refresh_token,
    decode_token,
    ensure_bootstrap_admin,
    get_current_user,
    get_user_by_email,
    get_user_by_id,
    verify_artifact_token,
    verify_password,
)
from backend.capabilities import (
    CapabilitiesResponse,
    ProcessingModeError,
    assert_device_allowed,
    get_capabilities,
    probe_host,
    resolve_processing_mode,
)
from backend.config import settings
from backend.contracts import (
    IsolateJobCreateRequest,
    JobCreateRequest,
    JobEvent,
    JobResponse,
    JobStatus,
    LoginRequest,
    TokenResponse,
    UploadResponse,
)
from backend import db as db_module
from backend.db import get_db, init_db
from backend.events import event_bus
from backend.jobs import single_flight
from backend.jobs.manager import JobManager
from backend.jobs.runner import run_isolate_job_async, run_job_async
from backend.limits import is_allowed_audio_content, is_allowed_upload, normalize_filename
from backend.storage import LocalStorage, get_storage, reset_storage

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    ensure_bootstrap_admin()
    reset_storage()
    yield


_prod = settings.env == "production"
app = FastAPI(
    title="Audio Tools API",
    version="0.3.0",
    lifespan=lifespan,
    docs_url=None if _prod else "/docs",
    redoc_url=None if _prod else "/redoc",
    openapi_url=None if _prod else "/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def hide_docs_in_production(request: Request, call_next):
    path = request.url.path
    if settings.env == "production" and (
        path == "/openapi.json" or path == "/redoc" or path.startswith("/docs")
    ):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return await call_next(request)

_origins = settings.cors_origin_list()
_allow_creds = "*" not in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=_allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_dir = Path(settings.data_dir)
job_manager = JobManager(data_dir)

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "midi": "audio/midi",
    "tab": "text/plain",
    "guitar_stem": "audio/wav",
    "zip": "application/zip",
    "vocals": "audio/wav",
    "drums": "audio/wav",
    "bass": "audio/wav",
    "other": "audio/wav",
    "guitar": "audio/wav",
    "lead_guitar": "audio/wav",
    "rhythm_guitar": "audio/wav",
    "guitar1": "audio/wav",
    "guitar2": "audio/wav",
    "piano": "audio/wav",
    "no_vocals": "audio/wav",
    "no_drums": "audio/wav",
    "no_bass": "audio/wav",
    "no_other": "audio/wav",
    "no_guitar": "audio/wav",
    "no_piano": "audio/wav",
    "guitar_split_diagnostics": "application/json",
}


def _job_response(job, user_id: str) -> JobResponse:
    title = (job.title or "").strip() or None
    return JobResponse(
        id=job.id,
        status=job.status,
        kind=job.kind,
        stage=job.stage,
        message=job.message,
        error=job.error,
        artifacts=job_manager.artifact_urls(job, user_id),
        title=title,
    )


def _rate_key(request: Request) -> str:
    user = getattr(request.state, "user_id", None)
    if user:
        return f"user:{user}"
    return get_remote_address(request)


def _resolved_processing(mode: str, *, requested_duration_sec: float | None = None):
    probe = probe_host(settings)
    try:
        return resolve_processing_mode(
            mode, probe, settings, requested_duration_sec=requested_duration_sec
        )
    except ProcessingModeError as exc:
        raise HTTPException(400, str(exc)) from None


def _assert_device_or_400(device: str) -> None:
    try:
        assert_device_allowed(device, probe_host(settings))
    except ProcessingModeError as exc:
        raise HTTPException(400, str(exc)) from None


async def _enqueue_or_run(background_tasks: BackgroundTasks, kind: str, job_id: str) -> None:
    if settings.use_worker:
        from arq import create_pool
        from arq.connections import RedisSettings

        redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        fn = "process_isolate_job" if kind == "isolate" else "process_tab_job"
        await redis.enqueue_job(fn, job_id)
        await redis.close()
    else:
        if kind == "isolate":
            background_tasks.add_task(run_isolate_job_async, job_manager, job_id)
        else:
            background_tasks.add_task(run_job_async, job_manager, job_id)


def _isolate_single_flight_enabled() -> bool:
    """Serialize Demucs: arq max_jobs=1, or in-process single-flight when no worker."""
    return bool(settings.single_flight_jobs or not settings.use_worker)


# Kept for capabilities/diagnostics; isolate create no longer 503s on busy.


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok", "env": settings.env, "auth": settings.require_auth}


@app.get("/v1/system/capabilities", response_model=CapabilitiesResponse)
def system_capabilities() -> CapabilitiesResponse:
    return get_capabilities(settings)


@app.post("/v1/auth/login", response_model=TokenResponse)
@limiter.limit(settings.rate_limit_jobs)
def login(request: Request, body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id)
    response.set_cookie(
        REFRESH_COOKIE,
        refresh,
        httponly=True,
        secure=settings.env == "production",
        samesite="lax",
        max_age=settings.refresh_token_days * 86400,
        path="/v1/auth",
    )
    return TokenResponse(access_token=access, email=user.email)


@app.post("/v1/auth/session", response_model=TokenResponse)
@limiter.limit(settings.rate_limit_jobs)
def create_session(request: Request, db: Session = Depends(get_db)):
    if not settings.demo_mode:
        raise HTTPException(404)
    user = create_anonymous_user(db)
    access = create_anon_session_token(user.id, user.email)
    return TokenResponse(access_token=access, email=user.email)


@app.post("/v1/auth/refresh", response_model=TokenResponse)
def refresh_token(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(401, "Missing refresh token")
    payload = decode_token(token, "refresh")
    from backend.refresh_store import refresh_jti_revoked, revoke_refresh_jti

    presented = str(payload.get("jti") or "")
    if presented and refresh_jti_revoked(presented):
        raise HTTPException(401, "Refresh token revoked")
    user = get_user_by_id(db, payload["sub"])
    if not user:
        raise HTTPException(401, "User not found")
    access = create_access_token(user.id, user.email)
    if presented:
        revoke_refresh_jti(presented)
    new_refresh = create_refresh_token(user.id)
    response.set_cookie(
        REFRESH_COOKIE,
        new_refresh,
        httponly=True,
        secure=settings.env == "production",
        samesite="lax",
        max_age=settings.refresh_token_days * 86400,
        path="/v1/auth",
    )
    return TokenResponse(access_token=access, email=user.email)


@app.post("/v1/auth/logout")
def logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(REFRESH_COOKIE)
    if token:
        try:
            payload = decode_token(token, "refresh")
        except HTTPException:
            payload = {}
        jti = str(payload.get("jti") or "")
        if jti:
            from backend.refresh_store import revoke_refresh_jti

            revoke_refresh_jti(jti)
    response.delete_cookie(REFRESH_COOKIE, path="/v1/auth")
    return {"status": "ok"}


@app.get("/v1/auth/me")
def me(user=Depends(get_current_user)) -> dict:
    return {"id": user.id, "email": user.email}


@app.delete("/v1/auth/me")
def delete_me(
    response: Response,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Delete the current user account and associated uploads/job artifacts (C20)."""
    if user.is_admin and not user.is_anonymous:
        raise HTTPException(
            400,
            "Admin accounts cannot be self-deleted from the API. Contact the operator.",
        )
    counts = delete_user_account(db, user, get_storage())
    response.delete_cookie(REFRESH_COOKIE, path="/v1/auth")
    return {"status": "deleted", **counts}


@app.post("/v1/uploads/audio", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_upload, key_func=_rate_key)
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
) -> UploadResponse:
    request.state.user_id = user.id
    filename = normalize_filename(file.filename or "audio.wav")
    ok, reason = is_allowed_upload(filename, file.content_type)
    if not ok:
        raise HTTPException(400, reason)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    chunk_size = 1024 * 1024
    total = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB limit")
        chunks.append(chunk)

    header = b"".join(chunks[:1]) if chunks else b""
    ok_content, content_reason = is_allowed_audio_content(header)
    if not ok_content:
        raise HTTPException(400, content_reason)

    upload_id, _key = job_manager.save_upload_stream(filename, chunks, user.id)
    return UploadResponse(upload_id=upload_id, filename=filename)


@app.post("/v1/jobs", response_model=JobResponse)
@limiter.limit(settings.rate_limit_jobs, key_func=_rate_key)
async def create_job(
    request: Request,
    body: JobCreateRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user),
) -> JobResponse:
    request.state.user_id = user.id
    if body.model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model. Choose from: {', '.join(SUPPORTED_MODELS)}")
    if not body.upload_id and not body.youtube_url:
        raise HTTPException(400, "Provide upload_id or youtube_url")
    if body.youtube_url and not settings.allow_youtube:
        raise HTTPException(403, "YouTube ingestion is disabled")

    if body.processing_mode:
        resolved = _resolved_processing(body.processing_mode)
        max_dur = resolved.max_duration_sec
        demucs_quality = resolved.quality
        demucs_device = resolved.device
    else:
        max_dur = min(body.max_duration_sec, settings.max_job_duration_sec)
        demucs_quality = "balanced"
        demucs_device = "cpu"

    single_flight.acquire_or_503(enabled=settings.single_flight_jobs)
    try:
        job = job_manager.create_job(
            user_id=user.id,
            upload_id=body.upload_id,
            youtube_url=body.youtube_url,
            title=body.title,
            separate_stems=body.separate_stems,
            max_duration_sec=max_dur,
            mix_aware_filtering=body.mix_aware_filtering,
            tempo_bpm_override=body.tempo_bpm_override,
            onset_threshold=body.onset_threshold,
            frame_threshold=body.frame_threshold,
            demucs_quality=demucs_quality,
            demucs_device=demucs_device,
            model=body.model,
            guitar_refine=body.guitar_refine,
            guitar_checkpoint=body.guitar_checkpoint,
            low_end_restore_db=body.low_end_restore_db,
            sub_bass_debleed=body.sub_bass_debleed,
        )
    except FileNotFoundError as exc:
        if settings.single_flight_jobs:
            single_flight.release()
        raise HTTPException(404, str(exc)) from None
    except Exception:
        if settings.single_flight_jobs:
            single_flight.release()
        raise

    # The runner takes ownership of the slot only after scheduling succeeds.
    try:
        await _enqueue_or_run(background_tasks, "tab", job.id)
    except BaseException:
        if settings.single_flight_jobs:
            single_flight.release()
        raise
    return _job_response(job, user.id)


@app.post("/v1/isolate/jobs", response_model=JobResponse)
@limiter.limit(settings.rate_limit_jobs, key_func=_rate_key)
async def create_isolate_job(
    request: Request,
    body: IsolateJobCreateRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user),
) -> JobResponse:
    request.state.user_id = user.id
    if body.model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model. Choose from: {', '.join(SUPPORTED_MODELS)}")

    start_sec = body.start_sec
    if body.end_sec is not None:
        region_length = body.end_sec - start_sec
    elif body.max_duration_sec is not None:
        region_length = body.max_duration_sec
    else:
        region_length = None

    if body.processing_mode:
        resolved = _resolved_processing(
            body.processing_mode, requested_duration_sec=region_length
        )
        quality = resolved.quality
        device = resolved.device
        max_dur = resolved.max_duration_sec
    else:
        if body.quality not in ("fast", "balanced", "high", "extreme"):
            raise HTTPException(400, "quality must be fast|balanced|high|extreme")
        _assert_device_or_400(body.device)
        quality = body.quality or settings.default_isolate_quality
        device = body.device
        max_dur = region_length
        if max_dur is None:
            max_dur = settings.max_job_duration_sec
        else:
            max_dur = min(max_dur, settings.max_job_duration_sec)

    # Serialize Demucs in the runner (wait for slot), not at create — so clients
    # can enqueue several songs. arq max_jobs=1 covers the worker path.
    try:
        job = job_manager.create_isolate_job(
            user_id=user.id,
            upload_id=body.upload_id,
            model=body.model,
            quality=quality,
            device=device,
            start_sec=start_sec,
            max_duration_sec=max_dur,
            two_stems=body.two_stems,
            lead_rhythm=body.lead_rhythm,
            lead_rhythm_mode=body.lead_rhythm_mode,
            guitar_checkpoint=body.guitar_checkpoint,
            two_pass=body.two_pass,
            guitar_refine=body.guitar_refine,
            emit_stems=body.emit_stems,
            fold_other_into_guitar=body.fold_other_into_guitar,
            fold_other_mode=body.fold_other_mode,
            dual_guitar=body.dual_guitar,
            low_end_restore_db=body.low_end_restore_db,
            sub_bass_debleed=body.sub_bass_debleed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None

    await _enqueue_or_run(background_tasks, "isolate", job.id)
    return _job_response(job, user.id)


@app.get("/v1/jobs", response_model=list[JobResponse])
def list_jobs(
    kind: str | None = None,
    limit: int = 20,
    user=Depends(get_current_user),
) -> list[JobResponse]:
    """Newest-first job list for the serial queue UI."""
    from backend.contracts import JobKind

    parsed_kind: JobKind | None = None
    if kind:
        try:
            parsed_kind = JobKind(kind)
        except ValueError as exc:
            raise HTTPException(400, "kind must be tab or isolate") from exc
    rows = job_manager.list_owned(user.id, kind=parsed_kind, limit=min(50, max(1, limit)))
    return [_job_response(j, user.id) for j in rows]


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, user=Depends(get_current_user)) -> JobResponse:
    job = job_manager.get_owned(job_id, user.id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_response(job, user.id)


@app.get("/v1/jobs/{job_id}/stems")
def job_stem_names(job_id: str, user=Depends(get_current_user)) -> dict[str, list[str]]:
    """Read-only stem names for a job this user owns."""
    job = job_manager.get_owned(job_id, user.id)
    if not job:
        raise HTTPException(404, "Job not found")
    kinds = [str(artifact.kind) for artifact in getattr(job, "artifacts", []) or []]
    return {"stems": stem_names_from_artifacts(kinds)}


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str, user=Depends(get_current_user)) -> JobResponse:
    job = job_manager.get_owned(job_id, user.id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_manager.request_cancel(job_id)
    job = job_manager.get(job_id)
    assert job
    return _job_response(job, user.id)


@app.get("/v1/artifacts/{job_id}/{kind}")
def get_artifact(
    job_id: str,
    kind: str,
    token: str | None = None,
    user=Depends(get_current_user),
):
    job = job_manager.get_owned(job_id, user.id)
    if not job:
        raise HTTPException(404, "Job not found")
    if token and not verify_artifact_token(job_id, kind, user.id, token):
        raise HTTPException(403, "Invalid or expired artifact token")
    if not token:
        # Allow authenticated owner without token (SPA can use bearer)
        pass
    key = job.artifacts.get(kind)
    if not key:
        raise HTTPException(404, f"Artifact '{kind}' not found")

    storage = get_storage()
    filename = Path(key).name
    media = MEDIA_TYPES.get(kind, "application/octet-stream")

    if isinstance(storage, LocalStorage):
        path = storage.local_path(key)
        if not path.exists():
            raise HTTPException(404, "Artifact file missing")
        return FileResponse(path, media_type=media, filename=filename)

    tmp = storage.open_temp(key, suffix=Path(key).suffix)

    def _iter():
        with open(tmp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        tmp.unlink(missing_ok=True)

    return StreamingResponse(_iter(), media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{filename}"'
    })


# Public signed download (no session) — token proves ownership
@app.get("/v1/artifacts/{job_id}/{kind}/signed")
def get_artifact_signed(job_id: str, kind: str, token: str, uid: str):
    if not verify_artifact_token(job_id, kind, uid, token):
        raise HTTPException(403, "Invalid or expired artifact token")
    job = job_manager.get(job_id)
    if not job or job.user_id != uid:
        raise HTTPException(404, "Job not found")
    key = job.artifacts.get(kind)
    if not key:
        raise HTTPException(404, f"Artifact '{kind}' not found")
    storage = get_storage()
    filename = Path(key).name
    media = MEDIA_TYPES.get(kind, "application/octet-stream")
    if isinstance(storage, LocalStorage):
        path = storage.local_path(key)
        if not path.exists():
            raise HTTPException(404, "Artifact file missing")
        return FileResponse(path, media_type=media, filename=filename)
    tmp = storage.open_temp(key, suffix=Path(key).suffix)

    def _iter():
        with open(tmp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        tmp.unlink(missing_ok=True)

    return StreamingResponse(
        _iter(),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/v1/jobs/{job_id}/ws")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    token = websocket.query_params.get("access_token")
    db = db_module.SessionLocal()
    try:
        if not token:
            await websocket.close(code=4401)
            return
        try:
            payload = decode_token(token, "access")
        except HTTPException:
            await websocket.close(code=4401)
            return
        user = get_user_by_id(db, payload["sub"])
        if not user:
            await websocket.close(code=4401)
            return
        job = job_manager.get_owned(job_id, user.id)
    finally:
        db.close()

    if not job:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    q = await job_manager.subscribe(job_id)
    redis_task = None
    if settings.use_worker:
        redis_task = asyncio.create_task(event_bus.listen_redis(job_id, q))

    try:
        await websocket.send_json(
            {
                "job_id": job_id,
                "stage": job.stage,
                "message": job.message,
                "status": job.status.value,
                "kind": job.kind.value,
                "artifacts": job_manager.artifact_urls(job, job.user_id),
            }
        )
        # Heartbeat + event loop
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=20.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "job_id": job_id})
                # Refresh from DB for poll-on-reconnect semantics
                job = job_manager.get(job_id)
                if job and job.status in (
                    JobStatus.succeeded,
                    JobStatus.failed,
                    JobStatus.cancelled,
                ):
                    await websocket.send_json(
                        JobEvent(
                            job_id=job_id,
                            stage=job.stage,
                            message=job.message,
                            status=job.status,
                            artifacts=job_manager.artifact_urls(job, job.user_id),
                            kind=job.kind,
                        ).model_dump()
                    )
                    break
                continue

            payload = event.model_dump() if hasattr(event, "model_dump") else event
            if isinstance(payload, dict) and "artifacts" in payload:
                # Refresh signed URLs
                j = job_manager.get(job_id)
                if j:
                    payload["artifacts"] = job_manager.artifact_urls(j, j.user_id)
            await websocket.send_json(payload)
            status = payload.get("status") if isinstance(payload, dict) else None
            if status in (
                JobStatus.succeeded.value,
                JobStatus.failed.value,
                JobStatus.cancelled.value,
            ):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await event_bus.unsubscribe(job_id, q)
        if redis_task:
            redis_task.cancel()


def main() -> None:
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
