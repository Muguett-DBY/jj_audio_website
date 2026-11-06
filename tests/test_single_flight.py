"""Single-flight job gate tests."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from tests.backend_test_utils import configure_backend, fake_wav_bytes, login_headers


@pytest.fixture()
def single_flight_client(tmp_path, monkeypatch):
    main_mod = configure_backend(
        tmp_path,
        monkeypatch,
        ATT_REQUIRE_AUTH="false",
        ATT_DEMO_MODE="true",
        ATT_SINGLE_FLIGHT_JOBS="true",
        ATT_MAX_JOB_DURATION_SEC="90",
    )
    from backend.jobs import single_flight

    single_flight.release()
    with TestClient(main_mod.app) as client:
        yield client, main_mod, single_flight
    single_flight.release()


@pytest.fixture()
def tab_job_upload(single_flight_client):
    client, _, _ = single_flight_client
    headers = login_headers(client)
    upload = client.post(
        "/v1/uploads/audio",
        headers=headers,
        files={"file": ("mix.wav", fake_wav_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    return headers, {"upload_id": upload.json()["upload_id"], "max_duration_sec": 15}


@pytest.mark.parametrize("enqueue_error", [RuntimeError, FileNotFoundError])
def test_tab_enqueue_failure_releases_slot(
    single_flight_client, tab_job_upload, monkeypatch, enqueue_error
):
    client, main_mod, single_flight = single_flight_client
    headers, body = tab_job_upload
    release = Mock(wraps=single_flight.release)
    monkeypatch.setattr(single_flight, "release", release)

    async def failed_enqueue(*_args, **_kwargs):
        raise enqueue_error("enqueue failed")

    monkeypatch.setattr(main_mod, "_enqueue_or_run", failed_enqueue)
    with pytest.raises(enqueue_error, match="enqueue failed"):
        client.post("/v1/jobs", headers=headers, json=body)

    release.assert_called_once_with()
    assert not single_flight.is_busy()

    async def noop_enqueue(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main_mod, "_enqueue_or_run", noop_enqueue)
    retried = client.post("/v1/jobs", headers=headers, json=body)
    assert retried.status_code == 200, retried.text
    assert single_flight.is_busy()
    release.assert_called_once_with()


@pytest.mark.parametrize("slot_held", [False, True])
def test_unsupported_tab_model_has_no_job_or_slot_side_effects(
    single_flight_client, tab_job_upload, monkeypatch, slot_held
):
    client, main_mod, single_flight = single_flight_client
    headers, body = tab_job_upload
    if slot_held:
        assert single_flight.try_acquire()
    acquire = Mock(wraps=single_flight.acquire_or_503)
    release = Mock(wraps=single_flight.release)
    create = Mock(wraps=main_mod.job_manager.create_job)
    enqueue = Mock()
    monkeypatch.setattr(single_flight, "acquire_or_503", acquire)
    monkeypatch.setattr(single_flight, "release", release)
    monkeypatch.setattr(main_mod.job_manager, "create_job", create)

    async def noop_enqueue(*args):
        enqueue(*args)

    monkeypatch.setattr(main_mod, "_enqueue_or_run", noop_enqueue)
    rejected = client.post(
        "/v1/jobs", headers=headers, json={**body, "model": "not-a-model"}
    )
    assert rejected.status_code == 400, rejected.text
    assert rejected.json()["detail"] == (
        f"Unsupported model. Choose from: {', '.join(main_mod.SUPPORTED_MODELS)}"
    )
    create.assert_not_called()
    enqueue.assert_not_called()
    acquire.assert_not_called()
    release.assert_not_called()
    assert single_flight.is_busy() is slot_held


@pytest.mark.parametrize("model", ["htdemucs_6s", "guitar_scnet"])
def test_supported_tab_model_is_stored_and_scheduled(
    single_flight_client, tab_job_upload, monkeypatch, model
):
    client, main_mod, single_flight = single_flight_client
    headers, body = tab_job_upload
    queued = []

    async def noop_enqueue(_background_tasks, kind, job_id):
        queued.append((kind, job_id))

    monkeypatch.setattr(main_mod, "_enqueue_or_run", noop_enqueue)
    created = client.post("/v1/jobs", headers=headers, json={**body, "model": model})
    assert created.status_code == 200, created.text
    job_id = created.json()["id"]
    job = main_mod.job_manager.get(job_id)
    assert job is not None
    assert job.isolate_model == model
    assert queued == [("tab", job_id)]
    assert single_flight.is_busy()


def test_tab_enqueue_cancellation_releases_slot(
    single_flight_client, tab_job_upload, monkeypatch
):
    _, main_mod, single_flight = single_flight_client
    headers, body = tab_job_upload
    token = headers["Authorization"].removeprefix("Bearer ")
    user = SimpleNamespace(id=main_mod.decode_token(token, "access")["sub"])
    release = Mock(wraps=single_flight.release)
    monkeypatch.setattr(single_flight, "release", release)

    async def cancelled_enqueue(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(main_mod, "_enqueue_or_run", cancelled_enqueue)
    # Call the handler directly: HTTP middleware can translate cancellation.
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            inspect.unwrap(main_mod.create_job)(
                request=main_mod.Request({"type": "http"}),
                body=main_mod.JobCreateRequest(**body),
                background_tasks=main_mod.BackgroundTasks(),
                user=user,
            )
        )
    release.assert_called_once_with()
    assert not single_flight.is_busy()


def test_tab_enqueue_success_holds_slot_until_runner_finishes(
    single_flight_client, tab_job_upload, monkeypatch
):
    client, main_mod, single_flight = single_flight_client
    headers, body = tab_job_upload
    queued = []

    async def noop_enqueue(_background_tasks, kind, job_id):
        queued.append((kind, job_id))

    monkeypatch.setattr(main_mod, "_enqueue_or_run", noop_enqueue)
    created = client.post("/v1/jobs", headers=headers, json=body)
    assert created.status_code == 200, created.text
    assert single_flight.is_busy()
    assert len(queued) == 1
    assert queued[0][0] == "tab"

    blocked = client.post("/v1/jobs", headers=headers, json=body)
    assert blocked.status_code == 503, blocked.text
    assert len(queued) == 1
    assert single_flight.is_busy()

    from backend.jobs import runner

    async def finished_job(*_args):
        return None

    monkeypatch.setattr(runner, "_run_job_async", finished_job)
    asyncio.run(runner.run_job_async(main_mod.job_manager, queued[0][1]))
    assert not single_flight.is_busy()


def test_disabled_tab_enqueue_failure_preserves_another_held_slot(
    single_flight_client, tab_job_upload, monkeypatch
):
    client, main_mod, single_flight = single_flight_client
    headers, body = tab_job_upload
    monkeypatch.setattr(main_mod.settings, "single_flight_jobs", False)
    assert single_flight.try_acquire()
    release = Mock(wraps=single_flight.release)
    monkeypatch.setattr(single_flight, "release", release)

    async def failed_enqueue(*_args, **_kwargs):
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(main_mod, "_enqueue_or_run", failed_enqueue)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        client.post("/v1/jobs", headers=headers, json=body)
    release.assert_not_called()
    assert single_flight.is_busy()


def test_isolate_job_create_succeeds_while_runner_slot_held(single_flight_client, monkeypatch):
    """Create enqueues with HTTP 200 even when the Demucs slot is held.

    Isolate create no longer 503s on busy; the runner waits for the slot.
    """
    client, main_mod, single_flight = single_flight_client
    headers = login_headers(client)

    upload = client.post(
        "/v1/uploads/audio",
        headers=headers,
        files={"file": ("mix.wav", fake_wav_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    upload_id = upload.json()["upload_id"]

    assert single_flight.try_acquire()

    async def noop_enqueue(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main_mod, "_enqueue_or_run", noop_enqueue)

    created = client.post(
        "/v1/isolate/jobs",
        headers=headers,
        json={"upload_id": upload_id, "model": "htdemucs", "quality": "fast", "max_duration_sec": 15},
    )
    assert created.status_code == 200, created.text
    single_flight.release()


def test_single_flight_disabled_allows_concurrent_create(tmp_path, monkeypatch):
    main_mod = configure_backend(
        tmp_path,
        monkeypatch,
        ATT_REQUIRE_AUTH="false",
        ATT_SINGLE_FLIGHT_JOBS="false",
    )
    from backend.jobs import single_flight

    single_flight.release()
    # Simulate a held slot; with feature off, creates must still succeed
    assert single_flight.try_acquire()

    async def noop_enqueue(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main_mod, "_enqueue_or_run", noop_enqueue)

    with TestClient(main_mod.app) as client:
        headers = login_headers(client)
        upload = client.post(
            "/v1/uploads/audio",
            headers=headers,
            files={"file": ("mix.wav", fake_wav_bytes(), "audio/wav")},
        )
        assert upload.status_code == 200
        res = client.post(
            "/v1/isolate/jobs",
            headers=headers,
            json={
                "upload_id": upload.json()["upload_id"],
                "model": "htdemucs",
                "quality": "fast",
                "max_duration_sec": 15,
            },
        )
        assert res.status_code == 200, res.text

    single_flight.release()
