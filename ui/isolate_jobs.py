"""Serial desktop isolate jobs — process-level worker, progress on disk.

Demucs stays off the Streamlit request thread so page switches do not abort a run.
Only one job runs at a time (8–16 GB RAM).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from ui.common import DATA_DIR, delete_run, run_output_dir

_JOBS_SUBDIR = "isolate_jobs"
_QUEUE_LOCK = threading.Lock()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None
_WORKER_STOP = threading.Event()
_ACTIVE_JOB_ID: str | None = None
_ACTIVE_PROCESS: multiprocessing.Process | None = None
_ANALYSIS_LOCK = threading.Lock()
_ANALYSIS_THREAD: threading.Thread | None = None
_ANALYSIS_STOP = threading.Event()
_ANALYSIS_PROCESS: multiprocessing.Process | None = None
_ANALYSIS_RUN: str | None = None
_ANALYSIS_SKIPPED: set[str] = set()
_FOREGROUND_NAME = "background_analysis_foreground.json"
_STATUS_NAME = "background_analysis_status.json"


@dataclass
class IsolateJobSpec:
    """Serializable job request for desktop isolate."""

    id: str
    audio_path: str
    output_dir: str
    title: str
    model: str = "htdemucs_6s"
    quality: str = "fast"
    device: str = "cpu"
    start_sec: float = 0.0
    max_duration_sec: float | None = None
    two_stems: str | None = None
    guitar_checkpoint: str | None = None
    two_pass: bool = False
    guitar_refine: bool = False
    emit_stems: list[str] | None = None
    fold_other_into_guitar: bool = True
    fold_other_mode: str = "best_effort"
    custom_stems: list[str] = field(default_factory=list)
    track_options: list[str] = field(default_factory=list)
    source_fingerprint: str | None = None
    source_kind: str | None = None
    region_label: str | None = None
    owner: str | None = None
    created_at: float = 0.0
    audio_duration_sec: float | None = None
    prior_timing: dict[str, Any] | None = None
    low_end_restore_db: float = 0.0
    sub_bass_debleed: bool = False
    demucs_segment: int | None = 8
    demucs_jobs: int = 1
    demucs_shifts: int | None = None
    demucs_overlap: float | None = None
    adaptive_fold_gain: bool = False
    bleed_gate: bool = False
    bass_bleed_mitigation: bool = False
    guitar_ensemble: bool = False
    reseparate_from: str | None = None
    parent_run_dir: str | None = None
    # Desktop mix-tab id the job was started from (``__new__:…``). On success
    # the UI replaces that draft slot with ``run_dir`` instead of opening a
    # second tab.
    origin_tab: str | None = None
    metronome_accent: bool = True
    metronome_rate: float = 1.0
    metronome_sound: str = "classic"
    lead_rhythm: bool = False
    lead_rhythm_mode: str = "confident"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsolateJobSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        raw_emit = filtered.get("emit_stems")
        if raw_emit is not None:
            filtered["emit_stems"] = list(raw_emit)
        return cls(**filtered)


def isolate_config_kwargs_from_spec(spec: IsolateJobSpec) -> dict[str, Any]:
    """Map a desktop job spec onto IsolateConfig fields (no inference)."""
    return {
        "model": spec.model,
        "quality": spec.quality,
        "device": spec.device,
        "start_sec": spec.start_sec,
        "max_duration_sec": spec.max_duration_sec,
        "two_stems": spec.two_stems,
        "guitar_checkpoint": spec.guitar_checkpoint,
        "two_pass": spec.two_pass,
        "guitar_refine": spec.guitar_refine,
        "emit_stems": tuple(spec.emit_stems) if spec.emit_stems else None,
        "fold_other_into_guitar": spec.fold_other_into_guitar,
        "fold_other_mode": spec.fold_other_mode,
        "lead_rhythm": bool(spec.lead_rhythm),
        "lead_rhythm_mode": str(spec.lead_rhythm_mode or "confident"),
        "low_end_restore_db": spec.low_end_restore_db,
        "sub_bass_debleed": spec.sub_bass_debleed,
        "demucs_segment": spec.demucs_segment,
        "demucs_jobs": spec.demucs_jobs,
        "demucs_shifts": spec.demucs_shifts,
        "demucs_overlap": spec.demucs_overlap,
        "adaptive_fold_gain": spec.adaptive_fold_gain,
        "bleed_gate": spec.bleed_gate,
        "bass_bleed_mitigation": spec.bass_bleed_mitigation,
        "guitar_ensemble": spec.guitar_ensemble,
        "metronome_accent": spec.metronome_accent,
        "metronome_rate": spec.metronome_rate,
        "metronome_sound": spec.metronome_sound,
    }


def jobs_root() -> Path:
    root = DATA_DIR / _JOBS_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_dir(job_id: str) -> Path:
    """Path to a job folder. Does not create it — reads must not resurrect deletes."""
    return jobs_root() / job_id


def _ensure_job_dir(job_id: str) -> Path:
    path = _job_dir(job_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _spec_path(job_id: str) -> Path:
    return _job_dir(job_id) / "spec.json"


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging name per write. With a shared "<name>.tmp", the worker thread
    # and a UI poll writing at the same moment overwrote and then consumed each
    # other's staging file; the loser retried replace() on a path that no longer
    # existed and raised FileNotFoundError after burning all 8 attempts.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    last_exc: OSError | None = None
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.02 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    if last_exc is not None:
        raise last_exc


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_status(job_id: str) -> dict[str, Any]:
    return {
        "id": job_id,
        "status": "queued",
        "stage": "pending",
        "message": "Queued",
        "progress": 0.0,
        "error": None,
        "artifacts": {},
        "updated_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "completed_stages": [],
        "pause_requested_at": None,
        "paused_at": None,
    }


def jobs_status_signature(jobs: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Stable id+status snapshot so the UI can detect queue changes between polls."""
    return tuple((str(job.get("id") or ""), str(job.get("status") or "")) for job in jobs)


# Quoted first so macOS "Application Support" (and similar) is not chopped at a space.
_QUOTED_ABS_PATH_RE = re.compile(r"""['"](?:[A-Za-z]:\\|/[^'"]+)['"]""")
_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\|/(?:Users|home|Applications|opt|usr|Library|private)/)\S+"
)
_MISSING_FILE_RE = re.compile(
    r"(?:Errno\s*2|No such file or directory|ENOENT)",
    re.IGNORECASE,
)
_ERROR_MAX_CHARS = 160


def format_job_error(error: str | None) -> str:
    """One short user-facing sentence. Drop HF hub noise and filesystem paths."""
    text = (error or "").strip()
    if not text:
        return "Separation failed"
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "HF_TOKEN" in line or "unauthenticated requests to the HF Hub" in line:
            continue
        cleaned_lines.append(line)
    cleaned = " ".join(cleaned_lines).strip() or text
    if _MISSING_FILE_RE.search(cleaned):
        return "Missing stem file"
    cleaned = _QUOTED_ABS_PATH_RE.sub("", cleaned)
    cleaned = _ABS_PATH_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;-")
    if len(cleaned) > _ERROR_MAX_CHARS:
        head = cleaned[: _ERROR_MAX_CHARS - 1]
        # Cut on a word boundary; a mid-word or trailing-preposition cut ("failed
        # to load the") reads as a rendering bug rather than a truncated message.
        space = head.rfind(" ")
        if space > _ERROR_MAX_CHARS // 2:
            head = head[:space]
        cleaned = head.rstrip(" ,.;:-") + "…"
    return cleaned or "Separation failed"


def write_status(job_id: str, **updates: Any) -> dict[str, Any] | None:
    """Update ``status.json``. A missing job folder is left missing.

    Partial updates must not recreate a deleted job as ``queued`` via
    ``_default_status``. An explicit ``status`` may create the file only when
    the folder already exists (enqueue creates the folder first).
    """
    job_dir = _job_dir(job_id)
    if not job_dir.is_dir():
        return None
    path = _status_path(job_id)
    current = _read_json(path)
    if current is None:
        if "status" not in updates:
            return None
        current = _default_status(job_id)
    current.update(updates)
    current["id"] = job_id
    current["updated_at"] = time.time()
    _write_json(path, current)
    return current


def read_status(job_id: str) -> dict[str, Any] | None:
    return _read_json(_status_path(job_id))


def read_spec(job_id: str) -> IsolateJobSpec | None:
    data = _read_json(_spec_path(job_id))
    if not data:
        return None
    try:
        return IsolateJobSpec.from_dict(data)
    except Exception:
        return None


def requeue_job(job_id: str) -> str | None:
    """Re-queue a failed job from its stored spec, as a fresh job.

    Returns the new job id, or None when the spec or its source audio is gone —
    the caller needs to distinguish "retrying" from "cannot retry" so the button
    never appears to do nothing.
    """
    spec = read_spec(job_id)
    if spec is None:
        return None
    if not spec.audio_path or not Path(spec.audio_path).is_file():
        return None
    # Fresh output dir: the failed run's dir can hold half-written stems that
    # would otherwise be mistaken for results of the retry.
    retry = replace(spec, id="", created_at=0.0, output_dir=str(run_output_dir()))
    return enqueue_job(retry)


BATCH_FILE_LIMIT = 20
SECTION_BATCH_BLOCK = "Separate one file when a section is selected."
YOUTUBE_BATCH_BLOCK = "Clear the YouTube URL before queueing several files."


@dataclass
class LocalBatchResult:
    job_ids: list[str]
    omitted: int
    blocked: str | None
    failed: list[str]


def enqueue_local_batch(
    paths: list[Path],
    *,
    section_selected: bool,
    youtube_url: str,
    make_spec: Callable[[Path], IsolateJobSpec],
) -> LocalBatchResult:
    """Queue local files on the existing serial worker. One job at a time."""
    if section_selected:
        return LocalBatchResult([], 0, SECTION_BATCH_BLOCK, [])
    if str(youtube_url or "").strip():
        return LocalBatchResult([], 0, YOUTUBE_BATCH_BLOCK, [])
    accepted = list(paths[:BATCH_FILE_LIMIT])
    omitted = max(0, len(paths) - BATCH_FILE_LIMIT)
    failed: list[str] = []
    ready: list[Path] = []
    for path in accepted:
        if path.is_file():
            ready.append(path)
        else:
            failed.append(path.name)
    started = time.time()
    job_ids: list[str] = []
    for index, path in enumerate(ready):
        spec = make_spec(path)
        spec.audio_path = str(path)
        spec.created_at = started + index * 0.001
        spec.start_sec = 0.0
        spec.max_duration_sec = None
        spec.region_label = None
        job_ids.append(enqueue_job(spec))
    return LocalBatchResult(job_ids, omitted, None, failed)


def enqueue_job(spec: IsolateJobSpec) -> str:
    """Persist a queued job and ensure the serial worker is running."""
    if not spec.id:
        spec.id = uuid.uuid4().hex
    if not spec.created_at:
        spec.created_at = time.time()
    _ensure_job_dir(spec.id)
    _write_json(_spec_path(spec.id), spec.to_dict())
    write_status(
        spec.id,
        status="queued",
        stage="pending",
        message="Queued",
        progress=0.0,
        error=None,
        artifacts={},
        title=spec.title,
        created_at=spec.created_at,
        source_kind=spec.source_kind,
        source_fingerprint=spec.source_fingerprint,
        origin_tab=spec.origin_tab,
    )
    ensure_worker_started()
    return spec.id


def _abort_flag_path(job_id: str) -> Path:
    return _job_dir(job_id) / "abort.flag"


def _checkpoint_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "checkpoint"


def _request_abort(job_id: str) -> None:
    path = _abort_flag_path(job_id)
    if not path.parent.is_dir():
        return
    path.write_text("1", encoding="utf-8")


def _clear_abort_flag(job_id: str) -> None:
    _abort_flag_path(job_id).unlink(missing_ok=True)


def _terminate_active_process(job_id: str) -> None:
    with _QUEUE_LOCK:
        proc = _ACTIVE_PROCESS
        active_id = _ACTIVE_JOB_ID
    if proc is None or active_id != job_id:
        return
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2.0)


def _delete_job_folder(job_id: str) -> bool:
    path = jobs_root() / job_id
    if not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def _artifacts_from_completed_run(output_dir: Path) -> dict[str, str] | None:
    """Stem paths from meta.json or wavs on disk when the child exited mid-success."""
    meta_path = output_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if isinstance(meta, dict):
            raw = meta.get("artifacts") or {}
            if isinstance(raw, dict) and raw:
                out = {
                    str(k): str(v)
                    for k, v in raw.items()
                    if str(v).lower().endswith(".wav") and Path(str(v)).is_file()
                }
                if out:
                    return out
    if not output_dir.is_dir():
        return None
    wavs = {
        p.stem: str(p)
        for p in output_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".wav"
        and not p.name.startswith(".")
        and not p.stem.endswith("_diagnostics")
        and p.stem != "current_mix"
    }
    return wavs or None


_WORKER_EXITED_BEFORE_START = "Worker exited before the job started"
_SETUP_FAIL_LEAVE = frozenset({"cancelled", "pausing", "failed", "succeeded", "paused"})
_ORPHAN_IN_FLIGHT = frozenset({"running", "pausing", "cancelled"})


def _fail_unstarted_job(job_id: str) -> None:
    """Mark a job failed when its worker child died before leaving ``queued``."""
    write_status(
        job_id,
        status="failed",
        stage="error",
        message=_WORKER_EXITED_BEFORE_START,
        error=_WORKER_EXITED_BEFORE_START,
        finished_at=time.time(),
    )


def _finalize_job_after_process(job_id: str) -> None:
    """Apply pause/cancel cleanup after the child process exits.

    A child that exits while status is still ``queued`` (or the status file is
    gone but the folder remains) must become ``failed``. Leaving ``queued``
    makes the worker retry the same id forever and the UI stays on Waiting.
    A deleted folder is left deleted — do not recreate it.
    """
    if not _job_dir(job_id).is_dir():
        return
    status = read_status(job_id)
    if not status or status.get("status") == "queued":
        _fail_unstarted_job(job_id)
        return
    st = status.get("status")
    if st == "pausing":
        write_status(
            job_id,
            status="paused",
            message="Paused",
            paused_at=time.time(),
        )
        _clear_abort_flag(job_id)
    elif st == "cancelled":
        _delete_job_folder(job_id)
    elif st == "running":
        # Child may have written stems/meta then exited before status=succeeded
        # (crash, SIGTERM race). Prefer success when the run is clearly complete
        # so the OS toast is not a false "failed".
        spec = read_spec(job_id)
        artifacts = None
        if spec is not None and spec.output_dir:
            artifacts = _artifacts_from_completed_run(Path(spec.output_dir))
        if artifacts:
            write_status(
                job_id,
                status="succeeded",
                stage="done",
                message="Complete",
                progress=1.0,
                artifacts=artifacts,
                finished_at=time.time(),
                title=spec.title if spec else status.get("title"),
                run_dir=spec.output_dir if spec else status.get("run_dir"),
                source_fingerprint=spec.source_fingerprint if spec else None,
                source_kind=spec.source_kind if spec else None,
                source_audio_path=spec.audio_path if spec else None,
                error=None,
                traceback=None,
            )
            return
        write_status(
            job_id,
            status="failed",
            stage="error",
            message="Separation stopped unexpectedly",
            error="Process exited before completion",
            finished_at=time.time(),
        )


def pause_job(job_id: str) -> bool:
    """Pause a running job: abort between stages and keep checkpoints on disk."""
    if not job_id or "/" in job_id or "\\" in job_id or job_id in {".", ".."}:
        return False
    status = read_status(job_id)
    if not status or status.get("status") != "running":
        return False
    _request_abort(job_id)
    write_status(
        job_id,
        status="pausing",
        message="Pausing…",
        pause_requested_at=time.time(),
    )
    _terminate_active_process(job_id)
    ensure_worker_started()
    return True


def resume_job(job_id: str) -> bool:
    """Re-queue a paused job so it continues from the last saved stage."""
    if not job_id or "/" in job_id or "\\" in job_id or job_id in {".", ".."}:
        return False
    status = read_status(job_id)
    if not status or status.get("status") != "paused":
        return False
    if read_spec(job_id) is None:
        return False
    _clear_abort_flag(job_id)
    write_status(
        job_id,
        status="queued",
        stage="pending",
        message="Queued",
        progress=float(status.get("progress") or 0.0),
        error=None,
        pause_requested_at=None,
        paused_at=status.get("paused_at"),
    )
    ensure_worker_started()
    return True


def remove_job(job_id: str) -> bool:
    """Drop a job from the list (queued / paused / failed / done).

    Running jobs are cancelled, their worker process is terminated, and the
    folder is removed once the process exits.
    """
    if not job_id or "/" in job_id or "\\" in job_id or job_id in {".", ".."}:
        return False
    status = read_status(job_id)
    if status and status.get("status") in {"running", "pausing"}:
        _request_abort(job_id)
        write_status(
            job_id,
            status="cancelled",
            stage="error",
            message="Removed from queue",
            error="Removed from queue",
            finished_at=time.time(),
        )
        _terminate_active_process(job_id)
        ensure_worker_started()
        return True
    return _delete_job_folder(job_id)


def delete_finished_job(job: dict[str, Any]) -> dict[str, Any]:
    """Remove a succeeded job record and its run directory (stems on disk)."""
    if job.get("status") != "succeeded":
        return {"ok": False, "run_dir": None}
    job_id = str(job.get("id") or "")
    run_dir = str(job.get("run_dir") or "") or None
    removed = bool(job_id) and remove_job(job_id)
    deleted_run = bool(run_dir) and delete_run(run_dir)
    return {"ok": bool(removed or deleted_run), "run_dir": run_dir}


def delete_all_finished_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    """Delete every succeeded job. Returns run_dirs that were targeted."""
    dirs: list[str] = []
    for job in jobs:
        if job.get("status") != "succeeded":
            continue
        result = delete_finished_job(job)
        run_dir = result.get("run_dir")
        if run_dir:
            dirs.append(str(run_dir))
    return dirs


def job_has_stem_wavs(job: dict[str, Any]) -> bool:
    """True when a job still has at least one stem wav on disk."""
    for path in (job.get("artifacts") or {}).values():
        wav = Path(str(path))
        if wav.suffix.lower() == ".wav" and wav.is_file():
            return True
    run_dir = job.get("run_dir")
    if run_dir:
        folder = Path(str(run_dir))
        if folder.is_dir():
            return any(p.is_file() and p.suffix.lower() == ".wav" for p in folder.iterdir())
    return False


def jobs_visible_in_queue(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop succeeded jobs whose wavs are gone (run deleted outside Queue)."""
    visible: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("status") == "succeeded" and not job_has_stem_wavs(job):
            continue
        visible.append(job)
    return visible


def _same_run_path(a: str, b: str) -> bool:
    """True when two run folder strings are the same path (slash and case)."""
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right:
        return False
    if left == right:
        return True
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(
        os.path.normpath(right)
    )


def succeeded_jobs_for_mix_tab(tab_id: str) -> list[dict[str, Any]]:
    """Succeeded queue jobs for a mix tab id (run folder or its origin draft)."""
    wanted = str(tab_id or "").strip()
    if not wanted:
        return []
    found: list[dict[str, Any]] = []
    for job in list_jobs(limit=200):
        if job.get("status") != "succeeded":
            continue
        if _same_run_path(str(job.get("run_dir") or ""), wanted):
            found.append(job)
            continue
        if str(job.get("origin_tab") or "") == wanted:
            found.append(job)
    return found


def delete_library_run(run_dir: str) -> bool:
    """Remove a mixer run and every succeeded queue job that points at it."""
    wanted = str(run_dir or "")
    if not wanted:
        return False
    ok = False
    for job in succeeded_jobs_for_mix_tab(wanted):
        result = delete_finished_job(job)
        ok = ok or bool(result.get("ok"))
    ok = bool(delete_run(wanted)) or ok
    return ok


def video_offset_sec_for_run(run_dir: str | Path | None) -> float:
    """Region start saved on the job spec for this run. Missing spec means 0."""
    if not run_dir:
        return 0.0
    try:
        wanted = Path(run_dir).resolve()
    except OSError:
        wanted = Path(str(run_dir))
    root = jobs_root()
    if not root.is_dir():
        return 0.0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        spec = _read_json(child / "spec.json") or {}
        out = str(spec.get("output_dir") or "")
        if not out:
            continue
        try:
            same = Path(out).resolve() == wanted
        except OSError:
            same = Path(out) == wanted
        if not same:
            continue
        try:
            return max(0.0, float(spec.get("start_sec") or 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def list_jobs(*, limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first job status rows (queued / running / done / failed)."""
    root = jobs_root()
    rows: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        status = _read_json(child / "status.json")
        if not status:
            continue
        spec = _read_json(child / "spec.json") or {}
        status.setdefault("title", spec.get("title") or status.get("id"))
        status.setdefault("created_at", spec.get("created_at") or 0.0)
        if spec.get("source_kind"):
            status.setdefault("source_kind", spec.get("source_kind"))
        if spec.get("source_fingerprint"):
            status.setdefault("source_fingerprint", spec.get("source_fingerprint"))
        if spec.get("origin_tab"):
            status.setdefault("origin_tab", spec.get("origin_tab"))
        rows.append(status)
    rows.sort(key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
    return rows[:limit]


_IN_FLIGHT_STATUSES = frozenset(
    {"queued", "running", "failed", "cancelled", "paused", "pausing"}
)


def list_in_flight_jobs(*, limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first jobs that still belong in the compact queue (not succeeded)."""
    return [job for job in list_jobs(limit=max(limit * 2, 20)) if job.get("status") in _IN_FLIGHT_STATUSES][:limit]


def queued_job_ids() -> list[str]:
    """Oldest-first queued job ids (public wrapper)."""
    return _queued_job_ids()


def queued_wait_caption(job_id: str, queued_ids: list[str]) -> str:
    """Waiting line with queue position; never a fake percent."""
    from ui.isolate_state import queued_wait_caption as _caption

    return _caption(job_id, queued_ids)


def merge_library_runs(
    recent: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Finished runs for the mixer switcher, newest first, one row per run_dir."""
    by_dir: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if job.get("status") != "succeeded":
            continue
        run_dir = str(job.get("run_dir") or "")
        if not run_dir:
            continue
        by_dir[run_dir] = {**job, "run_dir": run_dir}
    for run in recent:
        run_dir = str(run.get("run_dir") or "")
        if not run_dir:
            continue
        existing = by_dir.get(run_dir, {})
        by_dir[run_dir] = {**existing, **run, "run_dir": run_dir}
    rows = list(by_dir.values())
    rows.sort(key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
    return rows


def _release_dead_process_locked() -> multiprocessing.Process | None:
    """Return the live child, or drop a finished one so it cannot latch the UI.

    Caller must hold ``_QUEUE_LOCK``. A missing process is not cleared: the
    worker may have claimed ``_ACTIVE_JOB_ID`` and not stored the process yet.
    """
    global _ACTIVE_JOB_ID, _ACTIVE_PROCESS
    proc = _ACTIVE_PROCESS
    if proc is not None and proc.is_alive():
        return proc
    if proc is not None:
        _ACTIVE_PROCESS = None
        _ACTIVE_JOB_ID = None
    return None


def active_job_id() -> str | None:
    """Job id the worker is starting, running, or stopping (process still alive)."""
    with _QUEUE_LOCK:
        _release_dead_process_locked()
        return _ACTIVE_JOB_ID


def worker_busy() -> bool:
    """True only while the isolate child process is alive."""
    with _QUEUE_LOCK:
        return _release_dead_process_locked() is not None


def jobs_active() -> bool:
    """True while a separation job is queued, starting, or running."""
    try:
        return bool(worker_busy() or active_job_id() or queued_job_ids())
    except Exception:
        return False


_SEPARATION_OVERLAY_STATUSES = frozenset({"queued", "running"})


def separation_in_progress() -> bool:
    """True when disk status shows a job still separating (queued/running).

    Ignores in-memory ``_ACTIVE_JOB_ID`` so a just-finished job cannot keep the
    full-page overlay stuck after Mixer opens.
    """
    try:
        for job in list_jobs(limit=30):
            if job.get("status") in _SEPARATION_OVERLAY_STATUSES:
                return True
        return False
    except Exception:
        return False


def _use_inline_worker() -> bool:
    return os.environ.get("ISOLATE_JOBS_INLINE", "").strip() == "1"


def _queued_job_ids() -> list[str]:
    """Oldest-first queued jobs."""
    rows = []
    for child in jobs_root().iterdir():
        if not child.is_dir():
            continue
        status = _read_json(child / "status.json")
        if not status or status.get("status") != "queued":
            continue
        spec = _read_json(child / "spec.json") or {}
        rows.append((float(spec.get("created_at") or 0.0), child.name))
    rows.sort(key=lambda t: t[0])
    return [job_id for _, job_id in rows]


def _job_process_entry(job_id: str) -> None:
    _run_one_job(job_id)


def _run_one_job(job_id: str) -> None:
    """Run one job. Setup errors become ``failed`` instead of a dead ``queued`` row."""
    try:
        _run_one_job_inner(job_id)
    except Exception as exc:
        latest = read_status(job_id)
        if latest and latest.get("status") in _SETUP_FAIL_LEAVE:
            return
        detail = str(exc).strip() or exc.__class__.__name__
        write_status(
            job_id,
            status="failed",
            stage="error",
            message="Separation failed",
            error=detail,
            finished_at=time.time(),
        )


def _run_one_job_inner(job_id: str) -> None:
    from audio_to_tab.isolate import IsolateConfig, JobAborted, separate_stems
    from audio_to_tab.hardware import (
        get_desktop_probe,
        recommended_cpu_threads,
        resolve_safe_device,
    )
    from ui.common import write_run_metadata
    from ui.isolate_state import (
        estimated_stage_seconds,
        estimate_job_seconds,
        expects_guitar_stem,
        intra_stage_fraction,
        isolation_stages_for_job,
        resolve_active_stage,
        stage_progress_percent,
    )
    from ui.media import cleanup_mix_artifacts

    spec = read_spec(job_id)
    if spec is None:
        write_status(job_id, status="failed", stage="error", message="Missing job spec", error="Missing job spec")
        return

    current = read_status(job_id)
    if current and current.get("status") in ("cancelled", "pausing"):
        return

    _clear_abort_flag(job_id)
    completed_stages = set(current.get("completed_stages") or []) if current else set()

    guitar_job = expects_guitar_stem(
        model=spec.model,
        two_stems=spec.two_stems,
        custom_stems=spec.custom_stems,
    )
    # A re-separate child shares its parent's run directory; analysing there
    # would overwrite the parent's analysis with a subset mix. Skip both
    # optional post-processing stages for those jobs.
    run_analysis = spec.reseparate_from is None and Path(spec.audio_path).is_file()
    run_tab = run_analysis and guitar_job
    stages = isolation_stages_for_job(
        expects_guitar=guitar_job,
        guitar_refine=spec.guitar_refine,
        analyse=run_analysis,
        tab=run_tab,
    )
    clip_sec = spec.audio_duration_sec
    if clip_sec is None:
        clip_sec = spec.max_duration_sec
    # Never run a job on an accelerator this host cannot use (CUDA stays fail-
    # closed; an MPS request on an ineligible Mac downgrades to CPU).
    probe = get_desktop_probe()
    try:
        spec.device = resolve_safe_device(spec.device, probe)
    except RuntimeError as exc:
        write_status(
            job_id,
            status="failed",
            stage="error",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
        )
        return
    job_estimate, estimate_conf = estimate_job_seconds(
        audio_duration_sec=clip_sec,
        quality=spec.quality,
        device=spec.device,
        stages=stages,
        last_run=spec.prior_timing,
        model=spec.model,
        expects_guitar=guitar_job,
        two_pass=spec.two_pass,
        guitar_refine=spec.guitar_refine,
        cpu_threads=recommended_cpu_threads(),
    )
    started = time.time()
    eta_fields = {
        "audio_duration_sec": clip_sec,
        "job_estimate_sec": job_estimate,
        "estimate_confidence": estimate_conf,
        "stages": list(stages),
        "quality": spec.quality,
        "device": spec.device,
        "model": spec.model,
    }
    write_status(
        job_id,
        status="running",
        stage="ingest",
        message="Preparing audio…",
        progress=0.0,
        started_at=started,
        title=spec.title,
        stage_started_at=started,
        error=None,
        traceback=None,
        **eta_fields,
    )

    config = IsolateConfig(**isolate_config_kwargs_from_spec(spec))
    if spec.reseparate_from:
        config.emit_metronome = False

    progress_state: dict[str, Any] = {"stage": "ingest", "stage_started_wall": started}

    def on_progress(stage: str, message: str) -> None:
        now = time.time()
        if stage != progress_state["stage"]:
            progress_state["stage_started_wall"] = now
        progress_state["stage"] = stage
        active = resolve_active_stage(stage, stages)
        elapsed_in = now - float(progress_state["stage_started_wall"])
        stage_est = estimated_stage_seconds(active, stages, job_estimate)
        intra, _est = intra_stage_fraction(elapsed_in, stage_est)
        pct = stage_progress_percent(active, stages=stages, intra=intra)
        write_status(
            job_id,
            status="running",
            stage=active,
            message=message,
            progress=pct,
            title=spec.title,
            **eta_fields,
            stage_started_at=progress_state["stage_started_wall"],
        )

    def should_abort() -> bool:
        return _abort_flag_path(job_id).is_file()

    def on_stage_complete(stage: str) -> None:
        latest = read_status(job_id) or {}
        stages_done = list(latest.get("completed_stages") or [])
        if stage not in stages_done:
            stages_done.append(stage)
        write_status(job_id, completed_stages=stages_done)

    # Cap each long-running Demucs subprocess so a wedged child cannot block the
    # worker forever; 2.5× the (already generous) job estimate + 300s slack keeps
    # legit slow CPU runs safe while still catching true hangs. The abort path
    # (pause/stop) also terminates the child mid-run via the Popen handle.
    subprocess_timeout_sec = (job_estimate * 2.5 + 300.0) if job_estimate and job_estimate > 0 else None
    analysis_result: Any | None = None
    analysis_warning: str | None = None
    tab_warning: str | None = None

    try:
        artifacts = separate_stems(
            audio_path=Path(spec.audio_path),
            output_dir=Path(spec.output_dir),
            config=config,
            on_progress=on_progress,
            should_abort=should_abort,
            checkpoint_dir=_checkpoint_dir(job_id),
            completed_stages=completed_stages,
            on_stage_complete=on_stage_complete,
            subprocess_timeout_sec=subprocess_timeout_sec,
        )
        cleanup_mix_artifacts(Path(spec.output_dir))

        if run_analysis:
            latest = read_status(job_id)
            if should_abort() or (latest and latest.get("status") in ("cancelled", "pausing")):
                return
            on_progress("analysis", "Analysing song…")
            analysis_result, analysis_path, analysis_diagnostics, analysis_warning = (
                _run_analysis_stage(
                    Path(spec.output_dir),
                    artifacts,
                    source_path=spec.audio_path,
                    source_fingerprint=spec.source_fingerprint,
                    duration_sec=clip_sec,
                    progress=on_progress,
                )
            )
            if analysis_path is not None:
                artifacts["analysis"] = analysis_path
            if analysis_diagnostics is not None:
                artifacts["analysis_diagnostics"] = analysis_diagnostics
            on_stage_complete("analysis")
            write_status(
                job_id,
                analysis_warning=analysis_warning,
                analysis_error=analysis_warning if analysis_result is None else None,
            )

        if run_tab:
            latest = read_status(job_id)
            if should_abort() or (latest and latest.get("status") in ("cancelled", "pausing")):
                return
            on_progress("tab", "Transcribing tab…")
            tab_artifacts, tab_warning = _run_tab_stage(
                Path(spec.output_dir),
                artifacts,
                title=spec.title,
                analysis=analysis_result,
                progress=on_progress,
            )
            if tab_artifacts:
                artifacts.update(tab_artifacts)
            on_stage_complete("tab")
            write_status(job_id, tab_warning=tab_warning)

        artifact_map = {k: str(v) for k, v in artifacts.items()}
        # Re-separate children belong to the parent run on disk — never a new
        # Recent run — so only the normal path writes run metadata.
        if spec.reseparate_from is None:
            metro_meta = None
            diag_path = artifacts.get("metronome_diagnostics")
            if diag_path:
                try:
                    raw = json.loads(Path(diag_path).read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        metro_meta = {
                            "bpm": raw.get("bpm"),
                            "beat_count": raw.get("beat_count"),
                            "confidence": raw.get("confidence"),
                            "source": raw.get("source"),
                        }
                        render = raw.get("render")
                        if isinstance(render, dict):
                            metro_meta["render"] = {
                                "accent": render.get("accent"),
                                "rate": render.get("rate"),
                                "sound": render.get("sound"),
                            }
                except Exception:
                    metro_meta = None
            write_run_metadata(
                Path(spec.output_dir),
                page="isolate",
                title=spec.title,
                artifacts=artifacts,
                owner=spec.owner,
                source_kind=spec.source_kind,
                source_fingerprint=spec.source_fingerprint,
                config={
                    "model": spec.model,
                    "quality": spec.quality,
                    "device": spec.device,
                    "two_pass": spec.two_pass,
                    "guitar_refine": spec.guitar_refine,
                    "low_end_restore_db": spec.low_end_restore_db,
                    "sub_bass_debleed": spec.sub_bass_debleed,
                    "demucs_segment": spec.demucs_segment,
                    "demucs_jobs": spec.demucs_jobs,
                    "demucs_shifts": spec.demucs_shifts,
                    "demucs_overlap": spec.demucs_overlap,
                    "fold_other_mode": spec.fold_other_mode,
                    "adaptive_fold_gain": spec.adaptive_fold_gain,
                    "bleed_gate": spec.bleed_gate,
                    "bass_bleed_mitigation": spec.bass_bleed_mitigation,
                    "guitar_ensemble": spec.guitar_ensemble,
                },
                metronome=metro_meta,
                analysis=_analysis_metadata(analysis_result),
            )
        latest = read_status(job_id)
        if latest and latest.get("status") in ("cancelled", "pausing"):
            return
        write_status(
            job_id,
            status="succeeded",
            stage="done",
            message="Complete",
            progress=1.0,
            artifacts=artifact_map,
            finished_at=time.time(),
            title=spec.title,
            run_dir=spec.output_dir,
            source_fingerprint=spec.source_fingerprint,
            source_kind=spec.source_kind,
            region_label=spec.region_label,
            source_audio_path=spec.audio_path,
            custom_stems=spec.custom_stems,
            clip_length=spec.max_duration_sec,
            reseparate_from=spec.reseparate_from,
            parent_run_dir=spec.parent_run_dir,
            error=None,
            traceback=None,
            analysis_warning=analysis_warning,
            analysis_error=analysis_warning if (run_analysis and analysis_result is None) else None,
            tab_warning=tab_warning,
            **eta_fields,
        )
    except JobAborted:
        return
    except Exception as exc:
        latest = read_status(job_id)
        if latest and latest.get("status") in ("cancelled", "pausing"):
            return
        write_status(
            job_id,
            status="failed",
            stage="error",
            message="Separation failed",
            error=str(exc),
            finished_at=time.time(),
            title=spec.title,
            traceback=traceback.format_exc()[-2000:],
        )


def _execute_job(job_id: str) -> None:
    global _ACTIVE_JOB_ID, _ACTIVE_PROCESS
    if _use_inline_worker():
        try:
            _run_one_job(job_id)
        finally:
            _finalize_job_after_process(job_id)
        return
    proc: multiprocessing.Process | None = None
    try:
        proc = multiprocessing.Process(
            target=_job_process_entry,
            args=(job_id,),
            name=f"isolate-{job_id[:8]}",
        )
        proc.start()
        with _QUEUE_LOCK:
            _ACTIVE_PROCESS = proc
            _ACTIVE_JOB_ID = job_id
        proc.join()
    finally:
        with _QUEUE_LOCK:
            if proc is not None and _ACTIVE_PROCESS is proc:
                _ACTIVE_PROCESS = None
        # Child exit (or start failure) must not leave status=queued.
        _finalize_job_after_process(job_id)


def _reconcile_orphaned_jobs() -> None:
    """Finalize in-flight jobs left on disk when no child is alive.

    Queued jobs are real work and are left for the worker. ``running`` /
    ``pausing`` / ``cancelled`` with no live process cannot make progress:
    finish, pause, or delete them so the strip is not stuck waiting.
    """
    with _QUEUE_LOCK:
        proc = _ACTIVE_PROCESS
        if proc is not None and proc.is_alive():
            return
    root = jobs_root()
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir():
            continue
        status = _read_json(child / "status.json")
        if not status or status.get("status") not in _ORPHAN_IN_FLIGHT:
            continue
        _finalize_job_after_process(str(status.get("id") or child.name))


def _worker_loop() -> None:
    global _ACTIVE_JOB_ID, _ACTIVE_PROCESS
    _reconcile_orphaned_jobs()
    while not _WORKER_STOP.is_set():
        try:
            with _QUEUE_LOCK:
                queued = _queued_job_ids()
                job_id = queued[0] if queued else None
                if job_id is None:
                    _ACTIVE_JOB_ID = None
            if job_id is None:
                _WORKER_STOP.wait(0.4)
                continue
            current = read_status(job_id)
            if current and current.get("status") == "cancelled":
                _finalize_job_after_process(job_id)
                continue
            try:
                with _QUEUE_LOCK:
                    _ACTIVE_JOB_ID = job_id
                _execute_job(job_id)
            finally:
                with _QUEUE_LOCK:
                    if _ACTIVE_JOB_ID == job_id:
                        _ACTIVE_JOB_ID = None
                    if _ACTIVE_PROCESS is not None and not _ACTIVE_PROCESS.is_alive():
                        _ACTIVE_PROCESS = None
        except Exception:
            _WORKER_STOP.wait(1.0)


def _stop_worker_thread() -> None:
    """Join the serial worker so a later test cannot inherit a live thread.

    Tests reset ``_WORKER_THREAD`` with monkeypatch, which restores the old
    value on teardown and orphans the thread the test started. That thread
    keeps polling ``DATA_DIR`` and races the next case (status flips to
    ``failed`` / ``running`` under the test's own writes).
    """
    global _WORKER_STARTED, _WORKER_THREAD, _ACTIVE_JOB_ID, _ACTIVE_PROCESS
    thread = _WORKER_THREAD
    _WORKER_STOP.set()
    if (
        thread is not None
        and thread.is_alive()
        and thread is not threading.current_thread()
    ):
        thread.join(timeout=2.0)
    with _WORKER_LOCK:
        if _WORKER_THREAD is thread:
            _WORKER_THREAD = None
        _WORKER_STARTED = False
        _ACTIVE_JOB_ID = None
        _ACTIVE_PROCESS = None
    _WORKER_STOP.clear()


def ensure_worker_started() -> None:
    """Start the serial worker, or restart it if the thread died."""
    if os.environ.get("ISOLATE_JOBS_WORKER", "1").strip() == "0":
        return
    global _WORKER_STARTED, _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        thread = threading.Thread(target=_worker_loop, name="isolate-job-worker", daemon=True)
        thread.start()
        _WORKER_THREAD = thread
        _WORKER_STARTED = True


_DIAGNOSTIC_KEYS = frozenset(
    {
        "guitar_split_diagnostics",
        "stem_presence_diagnostics",
        "bass_bleed_diagnostics",
        "guitar_prerefine",
        "guitar_refined",
    }
)
_MIXER_RESET_KEYS = (
    "isolate_volumes_db",
    "isolate_master_volume_db",
    "isolate_mixer_state",
    "isolate_mix_ready",
    "isolate_mix_fp",
)


def _existing_wav_artifacts(artifacts: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, path in (artifacts or {}).items():
        if name in _DIAGNOSTIC_KEYS:
            continue
        wav = Path(str(path))
        if wav.suffix.lower() != ".wav" or not wav.exists():
            continue
        out[str(name)] = wav
    return out


def _results_fingerprint(stem_paths: dict[str, Path]) -> str:
    payload = "|".join(f"{k}:{v}" for k, v in sorted((n, str(p)) for n, p in stem_paths.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


_ANALYSIS_FILENAME = "analysis.json"
_ANALYSIS_DIAGNOSTICS_FILENAME = "analysis_diagnostics.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _first_existing_artifact(
    artifacts: dict[str, Any],
    keys: tuple[str, ...],
) -> Path | None:
    for key in keys:
        value = artifacts.get(key)
        if value and Path(str(value)).is_file():
            return Path(str(value))
    return None


def _load_mono(path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float64)
    return mono, int(sr)


def _write_harmony_sum(
    artifacts: dict[str, Any],
    source_path: str | Path,
) -> tuple[Path | None, str]:
    """Sum guitar+piano for the chord stage. Bass and ``other`` stay out.

    ``other`` densifies 12-bin chroma on distorted mixes (docs/chord-recognition/08
    FM5); bass is slash-only via ``bass_path``. Kind is ``guitar_piano_sum`` when
    both stems contribute, else ``guitar_sum``.
    """
    import numpy as np
    import soundfile as sf

    guitar = _first_existing_artifact(artifacts, ("guitar",))
    if guitar is None:
        return None, ""
    names = ("guitar", "piano")
    pieces: list[Any] = []
    used: list[str] = []
    sample_rate = 0
    for name in names:
        path = _first_existing_artifact(artifacts, (name,))
        if path is None:
            continue
        try:
            mono, sr = _load_mono(path)
        except Exception:
            continue
        if sample_rate == 0:
            sample_rate = sr
        elif sr != sample_rate and mono.size:
            import librosa

            mono = librosa.resample(mono, orig_sr=sr, target_sr=sample_rate)
        pieces.append(mono)
        used.append(name)
    if not pieces or sample_rate <= 0:
        return None, ""
    length = max(int(piece.size) for piece in pieces)
    stacked = np.zeros(length, dtype=np.float64)
    for piece in pieces:
        stacked[: piece.size] += piece
    sum_rms = float(np.sqrt(np.mean(stacked**2))) if stacked.size else 0.0
    source = Path(source_path)
    if source.is_file() and sum_rms > 0:
        mix = None
        mix_sr = sample_rate
        try:
            mix, mix_sr = _load_mono(source)
        except Exception:
            mix = None
        if mix is not None and mix_sr != sample_rate and mix.size:
            import librosa

            mix = librosa.resample(mix, orig_sr=mix_sr, target_sr=sample_rate)
        mix_rms = float(np.sqrt(np.mean(np.square(mix)))) if mix is not None and mix.size else 0.0
        if mix_rms > 0 and sum_rms > 2.0 * mix_rms:
            return None, "too_loud"
    peak = float(np.max(np.abs(stacked))) if stacked.size else 0.0
    if peak > 0:
        stacked *= 0.99 / peak
    dest = guitar.parent / "harmony_sum.wav"
    sf.write(str(dest), stacked.astype(np.float32), sample_rate, subtype="PCM_16")
    kind = "guitar_piano_sum" if "piano" in used else "guitar_sum"
    return dest, kind


def _analysis_sources(
    artifacts: dict[str, Any],
    source_path: str | Path,
) -> tuple[Path | None, Path | None, Path | None, str | None]:
    """Pick harmony/vocals/metronome inputs without assuming a stem set.

    Priority is guitar(+piano) sum, then ``no_vocals``, then ``other``, then the
    full mix. Bass is not part of the sum.
    """
    vocals = _first_existing_artifact(artifacts, ("vocals",))
    metronome = artifacts.get("metronome_diagnostics")
    metro_path = Path(str(metronome)) if metronome and Path(str(metronome)).is_file() else None
    harmony: Path | None = None
    kind: str | None = None
    if _first_existing_artifact(artifacts, ("guitar",)) is not None:
        summed, summed_kind = _write_harmony_sum(artifacts, source_path)
        if summed is not None:
            harmony, kind = summed, summed_kind
    if harmony is None:
        no_vocals = _first_existing_artifact(artifacts, ("no_vocals",))
        if no_vocals is not None:
            harmony, kind = no_vocals, "no_vocals"
    if harmony is None:
        other = _first_existing_artifact(artifacts, ("other",))
        if other is not None:
            harmony, kind = other, "other"
    source = Path(source_path)
    if harmony is None and source.is_file():
        harmony, kind = source, "mix_fallback"
    return harmony, vocals, metro_path, kind


def _run_analysis_stage(
    run_dir: Path,
    artifacts: dict[str, Any],
    *,
    source_path: str,
    source_fingerprint: str | None,
    duration_sec: float | None,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[Any | None, Path | None, Path | None, str | None]:
    """Run analysis; failures become warnings so a usable separation still ships."""
    from audio_to_tab.analysis import analyze_audio, save_analysis

    run_dir = Path(run_dir)
    harmony, vocals, metronome, harmony_kind = _analysis_sources(artifacts, source_path)
    source = Path(source_path)
    fingerprint = (source_fingerprint or "").strip()
    if not fingerprint:
        hash_target = harmony if harmony is not None else source
        fingerprint = _sha256_file(hash_target) if hash_target.is_file() else "unknown"
    duration = float(duration_sec or 0.0)
    diagnostics_path = run_dir / _ANALYSIS_DIAGNOSTICS_FILENAME
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "source_path": str(source),
        "harmony_path": str(harmony) if harmony is not None else None,
        "harmony_source_kind": harmony_kind,
        "vocals_path": str(vocals) if vocals is not None else None,
        "metronome_diagnostics": str(metronome) if metronome is not None else None,
        "duration_sec": duration,
        "created_at": time.time(),
    }

    if harmony is None:
        warning = "no audio source for analysis"
        diagnostics["error"] = warning
        _write_json_atomic(diagnostics_path, diagnostics)
        return None, None, diagnostics_path, warning

    try:
        result = analyze_audio(
            harmony,
            harmony_path=harmony,
            harmony_source_kind=harmony_kind,
            bass_path=_first_existing_artifact(artifacts, ("bass",)),
            vocals_path=vocals,
            metronome_diagnostics=metronome,
            source_fingerprint=fingerprint,
            duration_sec=duration,
            progress=progress,
        )
        analysis_path = save_analysis(run_dir / _ANALYSIS_FILENAME, result)
        try:
            from audio_to_tab.chart import build_chart
            from audio_to_tab.export_musicxml import write_musicxml
            from audio_to_tab.export_text import write_chart_text

            document = build_chart(result, title=source.stem)
            write_musicxml(document, run_dir / "chart.musicxml")
            write_chart_text(document, run_dir / "chart.txt")
        except Exception as exc:
            diagnostics["export_error"] = str(exc)
        diagnostics.update(
            {
                "bpm": result.bpm,
                "bpm_confidence": result.bpm_confidence,
                "key": result.key,
                "key_confidence": result.key_confidence,
                "beats_per_measure": result.beats_per_measure,
                "chord_count": len(result.chords),
                "section_count": len(result.sections),
                "lyrics_line_count": len(result.lyrics),
                "lyrics_language": result.lyrics_language,
                "models": dict(result.models),
                "warnings": list(result.warnings),
            }
        )
        _write_json_atomic(diagnostics_path, diagnostics)
        warning = None
        if result.warnings:
            warning = "; ".join(result.warnings[:4])
            if len(result.warnings) > 4:
                warning += f" (+{len(result.warnings) - 4} more)"
        return result, analysis_path, diagnostics_path, warning
    except Exception as exc:
        warning = f"analysis failed: {exc}"
        diagnostics["error"] = str(exc)
        _write_json_atomic(diagnostics_path, diagnostics)
        return None, None, diagnostics_path, warning


def analyse_existing_run(run_dir: Path | str) -> tuple[bool, str]:
    """Analyse a finished separation in place (Home ``Analyse`` action).

    Returns ``(ok, message)``. This is synchronous on purpose: it is a one-shot
    repair for old runs that predate analysis, not a second job queue.
    Background backlog uses ``start_background_analysis`` instead.
    """
    if worker_busy():
        return False, "Wait for the current separation to finish, then analyse."
    return _analyse_run_in_place(run_dir)


def _analyse_run_in_place(run_dir: Path | str) -> tuple[bool, str]:
    """Write analysis.json and the meta summary for one finished run."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "This run has no readable meta.json."
    if not isinstance(meta, dict):
        return False, "This run has no readable meta.json."
    raw_artifacts = meta.get("artifacts")
    artifacts: dict[str, Any] = (
        dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
    )
    harmony, _vocals, _metronome, _harmony_kind = _analysis_sources(artifacts, "")
    if harmony is None:
        return False, "No guitar/instrumental stem to analyse."
    try:
        import soundfile as sf

        duration = float(sf.info(str(harmony)).duration)
    except Exception:
        duration = 0.0
    result, analysis_path, diagnostics_path, warning = _run_analysis_stage(
        run_dir,
        artifacts,
        source_path="",
        source_fingerprint=str(meta.get("source_fingerprint") or ""),
        duration_sec=duration,
    )
    if analysis_path is None:
        return False, warning or "Analysis failed."
    artifacts["analysis"] = str(analysis_path)
    if diagnostics_path is not None:
        artifacts["analysis_diagnostics"] = str(diagnostics_path)
    meta["artifacts"] = artifacts
    meta["analysis"] = _analysis_metadata(result)
    try:
        _write_json_atomic(meta_path, meta)
    except (OSError, TypeError, ValueError) as exc:
        return False, f"Analysis finished but meta.json could not be updated: {exc}"
    if warning:
        return True, f"Analysis ready with warnings: {warning}"
    return True, "Analysis ready. Open Practice."


def _analyse_process_result(run_dir: str) -> int:
    """0 when the summary is written, 1 when this run should not be retried."""
    try:
        ok, _message = _analyse_run_in_place(run_dir)
    except Exception:
        return 1
    return 0 if ok else 1


def _analyse_process_entry(run_dir: str) -> None:
    """Child-process target. Stays off the Streamlit script thread."""
    sys.exit(_analyse_process_result(run_dir))


def foreground_state_path() -> Path:
    return DATA_DIR / _FOREGROUND_NAME


def write_analysis_foreground(active_run: str | None) -> None:
    """Remember the song open in the mixer so background analysis can step aside."""
    payload = {"active_run": str(active_run) if active_run else None}
    try:
        _write_json_atomic(foreground_state_path(), payload)
    except (OSError, TypeError, ValueError):
        return


def read_analysis_foreground() -> str | None:
    data = _read_json(foreground_state_path())
    if not isinstance(data, dict):
        return None
    run_dir = data.get("active_run")
    text = str(run_dir or "").strip()
    return text or None


def background_status_path() -> Path:
    return DATA_DIR / _STATUS_NAME


def write_background_analysis_status(
    *,
    state: str,
    title: str = "",
    remaining: int = 0,
    run_dir: str = "",
    finished_run: str = "",
    skipped: list[str] | None = None,
    path: Path | None = None,
) -> None:
    """Record the backlog line the UI shows. Tests pass ``path`` so user data stays untouched."""
    if path is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return
    payload = {
        "state": state if state in {"analysing", "waiting", "idle"} else "idle",
        "title": title,
        "remaining": max(0, int(remaining)),
        "run_dir": run_dir,
        "finished_run": finished_run,
        "skipped": sorted(str(item) for item in (skipped or [])),
    }
    target = path or background_status_path()
    current = _read_json(target)
    if current == payload:
        return
    try:
        _write_json_atomic(target, payload)
    except (OSError, TypeError, ValueError):
        return


def read_background_analysis_status() -> dict[str, Any]:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return {}
    data = _read_json(background_status_path())
    return data if isinstance(data, dict) else {}


def _publish_analysis_progress(
    rows: list[dict],
    *,
    state: str,
    run_dir: str = "",
    finished_run: str = "",
) -> None:
    from ui.pages.home import analysis_summary, run_label

    title = ""
    remaining = 0
    for row in rows:
        item = str(row.get("run_dir") or "")
        if not item or item in _ANALYSIS_SKIPPED:
            continue
        if analysis_summary(row):
            continue
        remaining += 1
        if item == run_dir:
            title = run_label(row)
    if remaining <= 0:
        state = "idle"
        run_dir = ""
        title = ""
    write_background_analysis_status(
        state=state,
        title=title,
        remaining=remaining,
        run_dir=run_dir if state == "analysing" else "",
        finished_run=finished_run,
        skipped=sorted(_ANALYSIS_SKIPPED),
    )


def advance_background_analysis(
    rows: list[dict],
    *,
    separation_busy: bool,
    skipped: set[str],
    active_run: str | None,
    analyse,
) -> str | None:
    """Analyse one eligible finished run. Returns that run dir, or None when idle.

    ``skipped`` gains a run only after a no-stem or failed analysis. A busy
    separator leaves the set unchanged and analyses nothing.
    """
    from ui.pages.home import auto_analysis_should_skip, next_auto_analysis

    chosen = next_auto_analysis(
        rows,
        busy=separation_busy,
        skipped=skipped,
        active_run=active_run,
    )
    if not chosen:
        return None
    try:
        ok, message = analyse(chosen)
    except Exception as exc:
        ok, message = False, str(exc)
    if auto_analysis_should_skip(ok, message):
        skipped.add(chosen)
    return chosen


def _stop_analysis_child(proc: multiprocessing.Process) -> None:
    """End a background analysis child from the analysis thread, not the UI thread."""
    if proc.is_alive():
        proc.terminate()
    proc.join(timeout=1.0)


def _library_for_background() -> list[dict]:
    from ui.pages.home import library_rows

    return library_rows(None, limit=10000)


def _poll_background_analysis() -> None:
    """Start or reap one analysis child. Returns immediately."""
    global _ANALYSIS_PROCESS, _ANALYSIS_RUN
    from ui.pages.home import next_auto_analysis

    proc = _ANALYSIS_PROCESS
    finished_run = ""
    if proc is not None:
        separation_busy = jobs_active() or worker_busy()
        if separation_busy:
            _stop_analysis_child(proc)
            _ANALYSIS_PROCESS = None
            _ANALYSIS_RUN = None
            _publish_analysis_progress(_library_for_background(), state="waiting")
            return
        if proc.is_alive():
            active = read_analysis_foreground()
            if not (active and active == _ANALYSIS_RUN):
                return
            others = next_auto_analysis(
                _library_for_background(),
                busy=False,
                skipped=set(_ANALYSIS_SKIPPED),
                active_run=active,
            )
            if not others or others == active:
                return
            _stop_analysis_child(proc)
            _ANALYSIS_PROCESS = None
            _ANALYSIS_RUN = None
        else:
            code = proc.exitcode
            run_dir = _ANALYSIS_RUN or ""
            proc.join(timeout=0)
            _ANALYSIS_PROCESS = None
            _ANALYSIS_RUN = None
            if run_dir and code not in (0, None):
                _ANALYSIS_SKIPPED.add(run_dir)
            elif code == 0 and run_dir:
                finished_run = run_dir
    if jobs_active() or worker_busy():
        _publish_analysis_progress(
            _library_for_background(),
            state="waiting",
            finished_run=finished_run,
        )
        return
    rows = _library_for_background()
    active = read_analysis_foreground()
    chosen = next_auto_analysis(
        rows,
        busy=False,
        skipped=set(_ANALYSIS_SKIPPED),
        active_run=active,
    )
    if not chosen or jobs_active() or worker_busy():
        _publish_analysis_progress(rows, state="idle", finished_run=finished_run)
        return
    child = multiprocessing.Process(
        target=_analyse_process_entry,
        args=(chosen,),
        name="att-background-analysis",
        daemon=True,
    )
    child.start()
    _ANALYSIS_PROCESS = child
    _ANALYSIS_RUN = chosen
    _publish_analysis_progress(
        rows,
        state="analysing",
        run_dir=chosen,
        finished_run=finished_run,
    )


def _background_analysis_loop() -> None:
    while not _ANALYSIS_STOP.is_set():
        try:
            _poll_background_analysis()
        except Exception:
            pass
        _ANALYSIS_STOP.wait(0.8 if _ANALYSIS_PROCESS is not None else 2.0)


def start_background_analysis() -> None:
    """Start the one-song analysis thread once. A no-op under pytest."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if os.environ.get("ATT_BACKGROUND_ANALYSIS", "1").strip() == "0":
        return
    # A spawned child re-runs ui/app.py as ``__mp_main__``. It must never start
    # its own analysis thread, or every child would spawn another.
    if multiprocessing.parent_process() is not None:
        return
    global _ANALYSIS_THREAD
    with _ANALYSIS_LOCK:
        if _ANALYSIS_THREAD is not None and _ANALYSIS_THREAD.is_alive():
            return
        _ANALYSIS_STOP.clear()
        thread = threading.Thread(
            target=_background_analysis_loop,
            name="att-background-analysis",
            daemon=True,
        )
        thread.start()
        _ANALYSIS_THREAD = thread


def _run_tab_stage(
    run_dir: Path,
    artifacts: dict[str, Any],
    *,
    title: str,
    analysis: Any | None,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Path], str | None]:
    """Transcribe the guitar stem once and persist both the text and player JSON.

    Song analysis already ran; tab is additive. A no-note guitar part is a
    warning, not a job failure — the separated stems remain useful.
    """
    from audio_to_tab.pipeline import (
        EmptyTranscriptionError,
        PipelineConfig,
        run_pipeline,
    )
    from audio_to_tab.rhythm import quantize_tab_document
    from audio_to_tab.tab_generate import (
        midi_to_tab,
        tab_document_to_dict,
        tab_measure_spans,
    )
    from audio_to_tab.tempo import resolve_tempo

    guitar = _first_existing_artifact(artifacts, ("guitar",))
    if guitar is None:
        return {}, None

    output_dir = Path(run_dir) / "tab"
    bpm_override = analysis.bpm if analysis is not None and analysis.bpm else None
    beats_per_measure = int(analysis.beats_per_measure) if analysis is not None else 4
    config = PipelineConfig(
        separate_stems=False,
        title=title,
        tempo_bpm_override=bpm_override,
        beats_per_measure=beats_per_measure,
    )

    def tab_progress(_stage: str, message: str) -> None:
        if progress is not None:
            progress("tab", message)

    try:
        pipeline_artifacts = run_pipeline(
            audio_path=guitar,
            output_dir=output_dir,
            config=config,
            on_progress=tab_progress,
        )
    except EmptyTranscriptionError:
        return {}, "tab skipped: no usable guitar notes detected"
    except Exception as exc:
        return {}, f"tab transcription failed: {exc}"

    tab_artifacts: dict[str, Path] = {}
    if pipeline_artifacts.get("midi"):
        tab_artifacts["tab_midi"] = Path(pipeline_artifacts["midi"])
    if pipeline_artifacts.get("tab"):
        tab_artifacts["tab"] = Path(pipeline_artifacts["tab"])
    if pipeline_artifacts.get("pdf"):
        tab_artifacts["tab_pdf"] = Path(pipeline_artifacts["pdf"])

    warning = None
    try:
        midi_path = tab_artifacts.get("tab_midi")
        if midi_path is None:
            raise ValueError("tab pipeline did not write a MIDI file")
        tempo = resolve_tempo(
            audio_path=guitar,
            midi_path=midi_path,
            override_bpm=bpm_override,
        )
        document = midi_to_tab(
            str(midi_path),
            tempo_bpm=tempo.bpm,
            tempo_confidence=tempo.confidence,
            beats_per_measure=beats_per_measure,
        )
        document = quantize_tab_document(document, beats_per_measure=beats_per_measure)
        payload = {
            "schema_version": 1,
            "document": tab_document_to_dict(document),
            "measures": tab_measure_spans(document),
        }
        tab_artifacts["tab_json"] = _write_json_atomic(output_dir / "tab.json", payload)
    except Exception as exc:
        warning = f"tab player data unavailable: {exc}"

    return tab_artifacts, warning


def _analysis_metadata(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "bpm": result.bpm,
        "bpm_confidence": result.bpm_confidence,
        "key": result.key,
        "key_confidence": result.key_confidence,
        "beats_per_measure": result.beats_per_measure,
        "chord_count": len(result.chords),
        "section_count": len(result.sections),
        "lyrics_line_count": len(result.lyrics),
        "lyrics_language": result.lyrics_language,
        "warnings": list(result.warnings)[:8],
    }


def _presence_from_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    path = artifacts.get("stem_presence_diagnostics")
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rewrite_parent_meta(parent_run_dir: str | None, session: Any) -> None:
    """Replace only ``artifacts`` in the parent run's meta.json with the session's.

    Preserves every other key (config, title, created_at, source_kind, ...).
    Writes atomically. Silent no-op if the run dir is missing or meta cannot be read.
    """
    if not parent_run_dir:
        return
    meta_path = Path(parent_run_dir) / "meta.json"
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    meta["artifacts"] = dict(session.get("isolate_artifacts") or {})
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(meta_path)


def _move_reseparate_children(children: dict[str, Path], parent_run_dir: Path) -> dict[str, Path]:
    """Move each child wav into the parent run dir; return a parent-pathed map.

    The re-separate job separated into a fresh output dir, so ``children`` are the
    only new stems there. Moving them into the parent dir keeps the parent as the
    single library location the mixer/Recent point at.
    """
    parent_run_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, path in children.items():
        dest = parent_run_dir / path.name
        try:
            Path(path).replace(dest)
        except OSError:
            dest = path
        out[str(key)] = dest
    return out


def _apply_reseparate_job_to_session(
    session: Any,
    status: dict[str, Any],
) -> bool:
    """Merge a re-separate job's child stems into the parent run in session.

    The source stem is replaced in the mixer by its children (composite keys like
    ``other::guitar``), the run pointer re-points to the parent, and the parent's
    meta.json ``artifacts`` is rewritten. The parent's source fingerprint/kind stay
    authoritative. Returns False (leaving the session unchanged) when no audible
    children were produced.
    """
    from ui.isolate_state import children_from_outputs, merge_reseparate

    run_dir = str(status.get("run_dir") or "")
    parent_run_dir = str(status.get("parent_run_dir") or "")
    source_stem = str(status.get("reseparate_from") or "").strip()
    if not run_dir or not parent_run_dir or not source_stem:
        return False

    raw_children = children_from_outputs(Path(run_dir))
    if not raw_children:
        return False

    parent = Path(parent_run_dir)
    moved = _move_reseparate_children(raw_children, parent)
    # Composite keys (parent::child) are built here; session artifacts are
    # string-valued everywhere, so stringify the moved child paths.
    children = {f"{source_stem}::{k}": str(v) for k, v in moved.items()}

    current_artifacts = session.get("isolate_artifacts") or {}
    session["isolate_artifacts"] = merge_reseparate(
        current_artifacts, source_stem, children
    )

    selected = dict(session.get("isolate_selected_stems") or {})
    selected.pop(source_stem, None)
    for key in children:
        selected[key] = True
    session["isolate_selected_stems"] = selected

    # Children belong to the parent run; re-point the mixer's run directory there.
    session["isolate_run_dir"] = parent_run_dir
    session["isolate_viewing_run_dir"] = parent_run_dir
    # Do NOT overwrite the parent's authoritative source fingerprint / kind.

    _rewrite_parent_meta(parent_run_dir, session)

    for key in _MIXER_RESET_KEYS:
        session.pop(key, None)

    session["isolate_consumed_job_id"] = status.get("id")
    return True


def apply_succeeded_job_to_session(
    session: Any,
    status: dict[str, Any],
    *,
    viewing_mode: str | None = None,
) -> bool:
    """Copy a finished job's artifacts into Streamlit session_state keys.

    Returns True when artifacts were applied. Missing wavs return False and
    leave session unchanged.
    """
    if status.get("status") != "succeeded":
        return False
    if status.get("reseparate_from"):
        return _apply_reseparate_job_to_session(session, status)
    artifacts = status.get("artifacts") or {}
    wav_paths = _existing_wav_artifacts(artifacts)
    if not wav_paths:
        return False
    produced = list(wav_paths.keys())
    custom = list(status.get("custom_stems") or [])
    presence = _presence_from_artifacts(artifacts)
    from ui.isolate_state import custom_selected_stems, infer_source_kind

    if custom:
        selected = custom_selected_stems(produced, custom)
    else:
        selected = {
            name: bool(presence.get(name, {}).get("present", True)) for name in produced
        }

    session["isolate_artifacts"] = dict(artifacts)
    session["isolate_base_name"] = status.get("title") or "tracks"
    run_dir = status.get("run_dir")
    if run_dir:
        session["isolate_run_dir"] = run_dir
    if status.get("source_audio_path"):
        session["isolate_source_audio_path"] = status["source_audio_path"]
    if status.get("source_fingerprint"):
        session["isolate_results_source_fp"] = status["source_fingerprint"]
    session["isolate_source_kind"] = infer_source_kind(
        source_kind=status.get("source_kind"),
        source_fingerprint=status.get("source_fingerprint"),
    )
    if status.get("region_label"):
        session["isolate_region_label"] = status["region_label"]
    else:
        session.pop("isolate_region_label", None)
    clip_length = status.get("clip_length")
    if clip_length is None:
        clip_length = status.get("max_duration_sec")
    if clip_length is not None:
        session["isolate_clip_length"] = float(clip_length)
    else:
        session.pop("isolate_clip_length", None)
    session["isolate_results_fp"] = _results_fingerprint(wav_paths)
    session["isolate_selected_stems"] = selected
    session["isolate_consumed_job_id"] = status.get("id")
    mode = viewing_mode if viewing_mode is not None else str(status.get("id") or "latest")
    session["isolate_viewing_job_id"] = mode
    if run_dir:
        session["isolate_viewing_run_dir"] = str(run_dir)
    for key in _MIXER_RESET_KEYS:
        session.pop(key, None)
    return True


def apply_finished_job_poll_outcome(
    session: Any,
    status: dict[str, Any],
    *,
    notify_only: bool,
) -> bool:
    """Handle a newly finished isolate job during UI poll.

    Always promotes the job's origin draft tab into the finished ``run_dir``
    strip slot. Stem load + listen-picker retarget happen only when
    ``notify_only`` is False and apply succeeds (keeps strip focus and mixer
    artifacts in sync).

    Returns True when the Mixer workspace should open (auto-apply loaded).
    On notify-only, sets ``isolate_flash`` and leaves picker/artifacts alone.
    """
    from ui.isolate_state import LISTEN_PICKER_KEY, promote_job_origin_tab

    promote_job_origin_tab(session, status)
    if notify_only:
        title = status.get("title") or "track"
        session["isolate_flash"] = f"**{title}** finished. Its tab is open above."
        return False

    job_id = str(status.get("id") or "").strip() or "latest"
    if not apply_succeeded_job_to_session(session, status, viewing_mode=job_id):
        return False

    run_dir = str(status.get("run_dir") or "").strip()
    if run_dir:
        session["isolate_listen_applied_dir"] = run_dir
        session[LISTEN_PICKER_KEY] = run_dir
    produced = [
        n
        for n, path in (status.get("artifacts") or {}).items()
        if Path(path).suffix.lower() == ".wav" and not str(n).endswith("_diagnostics")
    ]
    session["isolate_flash"] = (
        f"Separated {len(produced)} tracks. The mixer and downloads are open below."
    )
    return True
