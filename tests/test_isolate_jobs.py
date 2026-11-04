"""Tests for desktop serial isolate jobs."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from ui.isolate_jobs import (
    IsolateJobSpec,
    apply_finished_job_poll_outcome,
    apply_succeeded_job_to_session,
    delete_all_finished_jobs,
    delete_finished_job,
    delete_library_run,
    enqueue_job,
    ensure_worker_started,
    format_job_error,
    job_has_stem_wavs,
    jobs_active,
    jobs_status_signature,
    jobs_visible_in_queue,
    list_in_flight_jobs,
    list_jobs,
    merge_library_runs,
    pause_job,
    queued_wait_caption,
    read_spec,
    read_status,
    remove_job,
    resume_job,
    separation_in_progress,
    succeeded_jobs_for_mix_tab,
    video_offset_sec_for_run,
)
from ui.isolate_state import (
    is_stopping_previous_job,
    paused_job_caption,
    queued_wait_caption as state_queued_wait_caption,
    status_strip_waiting_caption,
    stopping_previous_caption,
)


def write_status(job_id: str, **updates):
    """Test seed: create the job folder, then write. Production writes do not."""
    import ui.isolate_jobs as jobs

    jobs._ensure_job_dir(job_id)
    return jobs.write_status(job_id, **updates)


@pytest.fixture()
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_TOOLS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ISOLATE_JOBS_INLINE", "1")
    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "0")
    # Re-bind DATA_DIR used by isolate_jobs
    import ui.common as common
    import ui.isolate_jobs as jobs

    # Stop any worker the previous test orphaned before DATA_DIR moves.
    jobs._stop_worker_thread()
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_WORKER_STARTED", False)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", None)
    monkeypatch.setattr(jobs, "_ACTIVE_PROCESS", None)
    monkeypatch.setattr(jobs, "_WORKER_THREAD", None)
    yield tmp_path
    # Teardown runs before monkeypatch restores _WORKER_THREAD, so the
    # thread this test started is still visible and can be joined.
    jobs._stop_worker_thread()


def test_enqueue_persists_queued_status(jobs_dir: Path):
    spec = IsolateJobSpec(
        id="job1",
        audio_path=str(jobs_dir / "a.wav"),
        output_dir=str(jobs_dir / "out"),
        title="Track A",
        created_at=time.time(),
    )
    (jobs_dir / "a.wav").write_bytes(b"x")
    jid = enqueue_job(spec)
    assert jid == "job1"
    status = read_status(jid)
    assert status is not None
    assert status["status"] == "queued"
    assert status["title"] == "Track A"
    rows = list_jobs(limit=5)
    assert any(r["id"] == "job1" for r in rows)
    assert jobs_active() is True
    assert separation_in_progress() is True


def test_video_offset_sec_for_run_reads_spec_start(jobs_dir: Path):
    run = jobs_dir / "out"
    run.mkdir()
    (jobs_dir / "a.wav").write_bytes(b"x")
    enqueue_job(
        IsolateJobSpec(
            id="job-region",
            audio_path=str(jobs_dir / "a.wav"),
            output_dir=str(run),
            title="Clip",
            start_sec=12.5,
            created_at=time.time(),
        )
    )
    assert video_offset_sec_for_run(run) == 12.5
    assert video_offset_sec_for_run(jobs_dir / "missing") == 0.0
    assert video_offset_sec_for_run(None) == 0.0


def test_jobs_active_false_when_idle(jobs_dir: Path):
    assert jobs_active() is False
    assert separation_in_progress() is False


def test_separation_in_progress_true_when_queued(jobs_dir: Path):
    write_status("q1", status="queued", stage="pending", message="Queued", progress=0.0)
    assert separation_in_progress() is True


def test_separation_in_progress_true_when_running(jobs_dir: Path):
    write_status("r1", status="running", stage="separate", message="Go", progress=0.4)
    assert separation_in_progress() is True


def test_separation_in_progress_false_when_succeeded_even_if_memory_busy(
    jobs_dir: Path, monkeypatch
):
    """Mixer open after success must not keep overlay via sticky _ACTIVE_JOB_ID."""
    import ui.isolate_jobs as jobs

    write_status("done1", status="succeeded", stage="done", message="Done", progress=1.0)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", "done1")
    monkeypatch.setattr(jobs, "_ACTIVE_PROCESS", None)
    assert jobs_active() is True  # memory still looks busy
    assert separation_in_progress() is False  # disk says done — overlay must clear


def test_write_status_updates_progress(jobs_dir: Path):
    write_status("j2", status="running", stage="separate", message="Go", progress=0.4)
    status = read_status("j2")
    assert status["status"] == "running"
    assert status["progress"] == 0.4


def test_worker_runs_separate_stems_serially(jobs_dir: Path, monkeypatch):
    """Worker invokes separate_stems and marks job succeeded (mocked engine)."""
    import ui.isolate_jobs as jobs

    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "1")

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()
    calls: list[str] = []

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        calls.append(str(audio_path))
        on_progress("separate", "mock")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr(
        "audio_to_tab.isolate.separate_stems",
        fake_separate,
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.ensure_cuda_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.get_desktop_probe",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "ui.media.cleanup_mix_artifacts",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "ui.common.write_run_metadata",
        lambda *_a, **_k: None,
    )

    # Reset worker so a fresh thread picks up our DATA_DIR
    monkeypatch.setattr(jobs, "_WORKER_STARTED", False)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", None)

    spec = IsolateJobSpec(
        id="serial1",
        audio_path=str(audio),
        output_dir=str(out),
        title="Serial",
        created_at=time.time(),
    )
    enqueue_job(spec)
    ensure_worker_started()

    deadline = time.time() + 5
    while time.time() < deadline:
        status = read_status("serial1")
        if status and status.get("status") in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    status = read_status("serial1")
    assert status is not None
    assert status["status"] == "succeeded", status
    assert calls == [str(audio)]
    assert "vocals" in (status.get("artifacts") or {})


def test_worker_runs_analysis_and_tab_for_guitar_job(jobs_dir: Path, monkeypatch):
    """Analysis and tab land in the run with player-ready JSON, not just stems."""
    import ui.isolate_jobs as jobs
    from audio_to_tab.analysis import AnalysisResult, ChordEvent, Section
    from audio_to_tab.tab_generate import TabDocument, TabEvent, TabNote

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()
    guitar = out / "guitar.wav"
    vocals = out / "vocals.wav"
    harmony = out / "no_vocals.wav"
    calls: dict[str, object] = {}

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        on_progress("separate", "mock")
        guitar.write_bytes(b"g")
        vocals.write_bytes(b"v")
        harmony.write_bytes(b"h")
        return {"guitar": guitar, "vocals": vocals, "no_vocals": harmony}

    def fake_analyze(audio_path, **kwargs):
        calls["analysis_audio"] = Path(audio_path)
        return AnalysisResult(
            source_fingerprint=kwargs["source_fingerprint"],
            duration_sec=kwargs["duration_sec"],
            bpm=120.0,
            bpm_confidence="high",
            key="C major",
            key_confidence="high",
            beats=(0.0, 0.5, 1.0),
            downbeats=(0.0,),
            chords=(ChordEvent(0.0, 2.0, "C", 0.9),),
            sections=(Section(0.0, 8.0, "Full song", 0.4),),
            models={"chords": "mock"},
        )

    def fake_run_pipeline(*, audio_path, output_dir, config, on_progress=None, **kwargs):
        calls["tab_audio"] = Path(audio_path)
        calls["tab_config"] = config
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        midi = output_dir / "transcription.mid"
        tab = output_dir / "transcription.tab"
        pdf = output_dir / "transcription.pdf"
        midi.write_bytes(b"midi")
        tab.write_text("e|--3--|", encoding="utf-8")
        pdf.write_bytes(b"pdf")
        if on_progress:
            on_progress("transcribe", "mock")
        return {"midi": midi, "tab": tab, "pdf": pdf}

    def fake_midi_to_tab(*args, **kwargs):
        return TabDocument(
            events=[
                TabEvent(
                    start=0.0,
                    notes=[TabNote(start=0.0, end=0.5, string=0, fret=3, pitch=55)],
                )
            ],
            tempo_bpm=120.0,
        )

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("audio_to_tab.analysis.analyze_audio", fake_analyze)
    monkeypatch.setattr("audio_to_tab.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("audio_to_tab.tab_generate.midi_to_tab", fake_midi_to_tab)
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock()
    )

    spec = IsolateJobSpec(
        id="analysis-run",
        audio_path=str(audio),
        output_dir=str(out),
        title="Song",
        created_at=time.time(),
        audio_duration_sec=8.0,
    )
    jobs.enqueue_job(spec)
    jobs._run_one_job(spec.id)

    status = jobs.read_status(spec.id)
    assert status["status"] == "succeeded", status
    artifacts = status["artifacts"]
    assert Path(artifacts["analysis"]).is_file()
    assert Path(artifacts["analysis_diagnostics"]).is_file()
    assert Path(artifacts["tab_json"]).is_file()
    assert Path(artifacts["tab_pdf"]).is_file()
    assert calls["analysis_audio"] == harmony
    assert calls["tab_audio"] == guitar
    assert calls["tab_config"].separate_stems is False
    assert status["analysis_warning"] is None
    assert status["analysis_error"] is None

    from audio_to_tab.analysis import load_analysis

    loaded = load_analysis(Path(artifacts["analysis"]))
    assert loaded is not None and loaded.bpm == 120.0
    payload = json.loads(Path(artifacts["tab_json"]).read_text(encoding="utf-8"))
    assert payload["document"]["events"]
    assert payload["measures"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["analysis"]["bpm"] == 120.0
    assert "analysis" in meta["artifacts"]


def test_worker_keeps_separation_when_analysis_fails(jobs_dir: Path, monkeypatch):
    """A broken analysis stage is a warning; the stems still succeed."""
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()
    vocals = out / "vocals.wav"
    calls: list[str] = []

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        on_progress("separate", "mock")
        vocals.write_bytes(b"v")
        return {"vocals": vocals}

    def fake_analyze(*args, **kwargs):
        calls.append("analysis")
        raise RuntimeError("model exploded")

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("audio_to_tab.analysis.analyze_audio", fake_analyze)
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock()
    )

    spec = IsolateJobSpec(
        id="analysis-fail",
        audio_path=str(audio),
        output_dir=str(out),
        title="Song",
        two_stems="vocals",
        created_at=time.time(),
        audio_duration_sec=8.0,
    )
    jobs.enqueue_job(spec)
    jobs._run_one_job(spec.id)

    status = jobs.read_status(spec.id)
    assert status["status"] == "succeeded", status
    assert calls == ["analysis"]
    assert "analysis" not in (status.get("artifacts") or {})
    assert Path(status["artifacts"]["analysis_diagnostics"]).is_file()
    assert status["analysis_warning"].startswith("analysis failed:")
    assert status["analysis_error"].startswith("analysis failed:")


def test_re_separate_skips_analysis_and_tab(jobs_dir: Path, monkeypatch):
    """A child re-separate must not clobber the parent's analysis artifacts."""
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "other.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()
    guitar = out / "guitar.wav"
    calls: list[str] = []

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        on_progress("separate", "mock")
        guitar.write_bytes(b"g")
        return {"other::guitar": guitar}

    def fail_analyze(*args, **kwargs):
        calls.append("analysis")
        raise AssertionError("analysis must not run for a re-separate child")

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("audio_to_tab.analysis.analyze_audio", fail_analyze)
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock()
    )

    spec = IsolateJobSpec(
        id="resp-no-analysis",
        audio_path=str(audio),
        output_dir=str(out),
        title="Other",
        created_at=time.time(),
        reseparate_from="other",
        parent_run_dir=str(jobs_dir / "parent"),
    )
    jobs.enqueue_job(spec)
    jobs._run_one_job(spec.id)

    status = jobs.read_status(spec.id)
    assert status["status"] == "succeeded", status
    assert calls == []
    assert "analysis" not in (status.get("artifacts") or {})
    assert "analysis" not in status["stages"]
    assert "tab" not in status["stages"]
    assert not (out / "analysis.json").exists()


def test_remove_job_deletes_queued_folder(jobs_dir: Path):
    spec = IsolateJobSpec(
        id="dropme",
        audio_path=str(jobs_dir / "a.wav"),
        output_dir=str(jobs_dir / "out"),
        title="Drop me",
        created_at=time.time(),
    )
    (jobs_dir / "a.wav").write_bytes(b"x")
    enqueue_job(spec)
    assert (jobs_dir / "isolate_jobs" / "dropme").is_dir()
    assert remove_job("dropme") is True
    assert not (jobs_dir / "isolate_jobs" / "dropme").exists()
    assert all(r["id"] != "dropme" for r in list_jobs())


def test_remove_running_job_marks_cancelled(jobs_dir: Path):
    write_status("run1", status="running", stage="separate", message="Go")
    assert remove_job("run1") is True
    status = read_status("run1")
    assert status is not None
    assert status["status"] == "cancelled"
    assert (jobs_dir / "isolate_jobs" / "run1" / "abort.flag").is_file()


def test_stop_on_slow_job_starts_next_queued(jobs_dir: Path, monkeypatch):
    """Stopping a running job frees the worker so the next queued job can start."""
    import ui.isolate_jobs as jobs
    from audio_to_tab.isolate import JobAborted

    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "1")

    audio_a = jobs_dir / "a.wav"
    audio_b = jobs_dir / "b.wav"
    audio_a.write_bytes(b"a")
    audio_b.write_bytes(b"b")
    out_a = jobs_dir / "out_a"
    out_b = jobs_dir / "out_b"
    out_a.mkdir()
    out_b.mkdir()

    started = threading.Event()
    calls: list[str] = []

    def slow_separate(*, audio_path, output_dir, config, on_progress, should_abort=None, **kwargs):
        calls.append(str(audio_path))
        first = len(calls) == 1
        started.set()
        # Only the in-flight job needs to stay cancellable. The follow-up job
        # used to sleep the same ~4s, and on the 3-vCPU macOS runner that
        # second sleep was still in "Preparing audio…" when the 8s deadline hit.
        if first:
            for _ in range(200):
                if should_abort and should_abort():
                    raise JobAborted("stop")
                time.sleep(0.02)
        on_progress("separate", "mock")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", slow_separate)
    monkeypatch.setattr("audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None)
    monkeypatch.setattr("audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock())
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "_WORKER_STARTED", False)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", None)
    monkeypatch.setattr(jobs, "_ACTIVE_PROCESS", None)
    monkeypatch.setattr(jobs, "_WORKER_THREAD", None)

    enqueue_job(
        IsolateJobSpec(
            id="slow",
            audio_path=str(audio_a),
            output_dir=str(out_a),
            title="Slow",
            created_at=time.time(),
        )
    )
    enqueue_job(
        IsolateJobSpec(
            id="next",
            audio_path=str(audio_b),
            output_dir=str(out_b),
            title="Next",
            created_at=time.time() + 1,
        )
    )
    ensure_worker_started()

    assert started.wait(timeout=5), "slow job did not start"
    assert remove_job("slow") is True

    deadline = time.time() + 15
    while time.time() < deadline:
        status = read_status("next")
        if status and status.get("status") == "succeeded":
            break
        time.sleep(0.05)

    status = read_status("next")
    assert status is not None
    assert status["status"] == "succeeded", status
    assert str(audio_b) in calls
    assert read_status("slow") is None


def test_pause_job_persists_completed_stages(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs
    from audio_to_tab.isolate import JobAborted

    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "1")

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()

    def staged_separate(
        *,
        audio_path,
        output_dir,
        config,
        on_progress,
        should_abort=None,
        checkpoint_dir=None,
        completed_stages=None,
        on_stage_complete=None,
        subprocess_timeout_sec=None,
    ):
        on_progress("ingest", "prep")
        if on_stage_complete:
            on_stage_complete("ingest")
        # Long stage so the job cannot finish before the test pauses it (race under load).
        for _ in range(300):
            if should_abort and should_abort():
                raise JobAborted("pause")
            time.sleep(0.02)
        on_progress("separate", "sep")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", staged_separate)
    monkeypatch.setattr("audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None)
    monkeypatch.setattr("audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock())
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", lambda *_a, **_k: None)

    spec = IsolateJobSpec(
        id="pause1",
        audio_path=str(audio),
        output_dir=str(out),
        title="Pause me",
        created_at=time.time(),
    )
    enqueue_job(spec)
    ensure_worker_started()

    deadline = time.time() + 8
    while time.time() < deadline:
        status = read_status("pause1")
        if status and "ingest" in (status.get("completed_stages") or []):
            break
        time.sleep(0.05)

    assert pause_job("pause1") is True

    deadline = time.time() + 5
    while time.time() < deadline:
        status = read_status("pause1")
        if status and status.get("status") == "paused":
            break
        time.sleep(0.05)

    status = read_status("pause1")
    assert status is not None
    assert status["status"] == "paused"
    assert "ingest" in (status.get("completed_stages") or [])


def test_resume_job_skips_completed_stages(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "1")

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()
    seen_completed: list[set[str]] = []

    def staged_separate(
        *,
        audio_path,
        output_dir,
        config,
        on_progress,
        should_abort=None,
        checkpoint_dir=None,
        completed_stages=None,
        on_stage_complete=None,
        subprocess_timeout_sec=None,
    ):
        seen_completed.append(set(completed_stages or ()))
        on_progress("separate", "sep")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", staged_separate)
    monkeypatch.setattr("audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None)
    monkeypatch.setattr("audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock())
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", lambda *_a, **_k: None)

    write_status(
        "resume1",
        status="paused",
        title="Resume me",
        completed_stages=["ingest"],
        created_at=time.time(),
    )
    spec = IsolateJobSpec(
        id="resume1",
        audio_path=str(audio),
        output_dir=str(out),
        title="Resume me",
        created_at=time.time(),
    )
    jobs._write_json(jobs._spec_path("resume1"), spec.to_dict())
    assert resume_job("resume1") is True
    ensure_worker_started()

    deadline = time.time() + 5
    while time.time() < deadline:
        status = read_status("resume1")
        if status and status.get("status") == "succeeded":
            break
        time.sleep(0.05)

    assert seen_completed
    assert "ingest" in seen_completed[0]


def test_stopping_vs_waiting_captions():
    assert "Stopping" in stopping_previous_caption()
    assert state_queued_wait_caption(
        "b",
        ["a", "b"],
        stopping_previous=True,
    ) == "Up next. Waiting for the previous job to stop."
    assert status_strip_waiting_caption(["a"], stopping_previous=True) == stopping_previous_caption()
    assert is_stopping_previous_job("job1", "cancelled") is True
    assert is_stopping_previous_job("job1", "running") is False
    assert "separate" in paused_job_caption(["ingest", "separate"])


def test_delete_finished_job_removes_job_and_run_dir(jobs_dir: Path):
    run_dir = jobs_dir / "run_done"
    run_dir.mkdir()
    (run_dir / "vocals.wav").write_bytes(b"x")
    write_status(
        "s1",
        status="succeeded",
        title="Done",
        run_dir=str(run_dir),
        created_at=1.0,
    )
    result = delete_finished_job(
        {"id": "s1", "status": "succeeded", "run_dir": str(run_dir)}
    )
    assert result["ok"] is True
    assert result["run_dir"] == str(run_dir)
    assert not (jobs_dir / "isolate_jobs" / "s1").exists()
    assert not run_dir.exists()


def test_delete_finished_job_skips_running(jobs_dir: Path):
    write_status("r1", status="running", title="Go")
    result = delete_finished_job({"id": "r1", "status": "running"})
    assert result["ok"] is False
    assert read_status("r1")["status"] == "running"


def test_delete_all_finished_jobs_only_succeeded(jobs_dir: Path):
    run_a = jobs_dir / "run_a"
    run_b = jobs_dir / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    write_status("s1", status="succeeded", title="A", run_dir=str(run_a), created_at=1.0)
    write_status("s2", status="succeeded", title="B", run_dir=str(run_b), created_at=2.0)
    write_status("r1", status="running", title="Go", created_at=3.0)
    dirs = delete_all_finished_jobs(list_jobs(limit=10))
    assert set(dirs) == {str(run_a), str(run_b)}
    assert not run_a.exists()
    assert not run_b.exists()
    assert read_status("r1")["status"] == "running"


def test_delete_library_run_removes_job_and_run_dir(jobs_dir: Path):
    run_dir = jobs_dir / "run_lib"
    run_dir.mkdir()
    wav = run_dir / "vocals.wav"
    wav.write_bytes(b"x")
    write_status(
        "s1",
        status="succeeded",
        title="Done",
        run_dir=str(run_dir),
        artifacts={"vocals": str(wav)},
        created_at=1.0,
    )
    assert delete_library_run(str(run_dir)) is True
    assert not (jobs_dir / "isolate_jobs" / "s1").exists()
    assert not run_dir.exists()
    assert all(r["id"] != "s1" for r in list_jobs())


def test_mix_tab_delete_finds_job_by_slash_style_or_origin(jobs_dir: Path):
    run_dir = jobs_dir / "run_origin"
    run_dir.mkdir()
    wav = run_dir / "vocals.wav"
    wav.write_bytes(b"x")
    write_status(
        "s-origin",
        status="succeeded",
        title="Done",
        run_dir=str(run_dir),
        origin_tab="__new__:abc",
        artifacts={"vocals": str(wav)},
        created_at=1.0,
    )
    by_origin = succeeded_jobs_for_mix_tab("__new__:abc")
    assert [job["id"] for job in by_origin] == ["s-origin"]
    by_slash = succeeded_jobs_for_mix_tab(run_dir.as_posix())
    assert [job["id"] for job in by_slash] == ["s-origin"]
    assert delete_library_run(run_dir.as_posix()) is True
    assert not run_dir.exists()
    assert all(r["id"] != "s-origin" for r in list_jobs())


def test_delete_library_run_still_deletes_folder_without_job(jobs_dir: Path):
    run_dir = jobs_dir / "orphan_run"
    run_dir.mkdir()
    (run_dir / "drums.wav").write_bytes(b"x")
    assert delete_library_run(str(run_dir)) is True
    assert not run_dir.exists()


def test_jobs_visible_in_queue_skips_succeeded_without_wavs(jobs_dir: Path):
    gone = jobs_dir / "gone"
    gone.mkdir()
    keep = jobs_dir / "keep"
    keep.mkdir()
    wav = keep / "vocals.wav"
    wav.write_bytes(b"x")
    jobs = [
        {
            "id": "s-gone",
            "status": "succeeded",
            "run_dir": str(gone),
            "artifacts": {"vocals": str(gone / "vocals.wav")},
        },
        {
            "id": "s-keep",
            "status": "succeeded",
            "run_dir": str(keep),
            "artifacts": {"vocals": str(wav)},
        },
        {"id": "q1", "status": "queued"},
    ]
    visible = jobs_visible_in_queue(jobs)
    assert [j["id"] for j in visible] == ["s-keep", "q1"]
    assert job_has_stem_wavs(jobs[0]) is False
    assert job_has_stem_wavs(jobs[1]) is True


def test_jobs_status_signature_changes_when_queued_job_fails():
    queued = [{"id": "c5e7", "status": "queued", "updated_at": 1.0}]
    failed = [{"id": "c5e7", "status": "failed", "updated_at": 2.0}]
    assert jobs_status_signature(queued) != jobs_status_signature(failed)
    assert jobs_status_signature(failed) == jobs_status_signature(failed)


def test_list_in_flight_jobs_includes_paused(jobs_dir: Path):
    write_status("p1", status="paused", title="Paused", created_at=1.0)
    write_status("q1", status="queued", title="Wait", created_at=2.0)
    write_status("s1", status="succeeded", title="Done", created_at=3.0)
    ids = {row["id"] for row in list_in_flight_jobs()}
    assert ids == {"p1", "q1"}


def test_list_in_flight_jobs_excludes_succeeded(jobs_dir: Path):
    write_status("q1", status="queued", title="Wait", created_at=1.0)
    write_status("r1", status="running", title="Go", created_at=2.0)
    write_status("f1", status="failed", title="Nope", created_at=3.0)
    write_status("p1", status="paused", title="Paused", created_at=3.5)
    write_status("s1", status="succeeded", title="Done", created_at=4.0)
    ids = {row["id"] for row in list_in_flight_jobs()}
    assert ids == {"q1", "r1", "f1", "p1"}


def test_queued_wait_caption_is_position_not_percent():
    queued = ["a", "b", "c"]
    assert queued_wait_caption("b", queued) == "Waiting. Position 2 of 3"
    assert "%" not in queued_wait_caption("a", queued)


def test_merge_library_runs_dedupes_by_run_dir():
    recent = [
        {"run_dir": "/runs/one", "title": "From recent", "created_at": 10.0, "artifacts": {"vocals": "/runs/one/vocals.wav"}},
    ]
    jobs = [
        {
            "id": "job1",
            "status": "succeeded",
            "run_dir": "/runs/one",
            "title": "From job",
            "created_at": 11.0,
            "artifacts": {"vocals": "/runs/one/vocals.wav"},
        },
        {
            "id": "job2",
            "status": "succeeded",
            "run_dir": "/runs/two",
            "title": "Other",
            "created_at": 12.0,
            "artifacts": {"drums": "/runs/two/drums.wav"},
        },
    ]
    merged = merge_library_runs(recent, jobs)
    dirs = [row["run_dir"] for row in merged]
    assert dirs == ["/runs/two", "/runs/one"]
    assert merged[1]["title"] == "From recent"


def test_apply_succeeded_job_to_session_sets_viewing_and_keeps_source(tmp_path: Path):
    wav = tmp_path / "vocals.wav"
    wav.write_bytes(b"RIFF")
    session: dict = {"isolate_source_audio_path": "keep-me.wav"}
    ok = apply_succeeded_job_to_session(
        session,
        {
            "id": "job9",
            "status": "succeeded",
            "title": "Clip",
            "run_dir": str(tmp_path),
            "artifacts": {"vocals": str(wav)},
            "source_audio_path": str(tmp_path / "song.wav"),
            "source_fingerprint": "fp1",
            "region_label": "0:10–0:40",
            "clip_length": 30.0,
            "custom_stems": [],
        },
        viewing_mode="latest",
    )
    assert ok is True
    assert session["isolate_source_audio_path"] == str(tmp_path / "song.wav")
    assert session["isolate_viewing_job_id"] == "latest"
    assert session["isolate_viewing_run_dir"] == str(tmp_path)
    assert session["isolate_region_label"] == "0:10–0:40"
    assert session["isolate_clip_length"] == 30.0
    assert session["isolate_results_fp"]
    assert session["isolate_source_kind"] == "file"
    assert "isolate_volumes_db" not in session


def test_apply_succeeded_job_to_session_fills_empty_session(tmp_path: Path):
    wav = tmp_path / "vocals.wav"
    wav.write_bytes(b"RIFF")
    session: dict = {}
    assert apply_succeeded_job_to_session(
        session,
        {
            "id": "job-empty",
            "status": "succeeded",
            "title": "Latest",
            "run_dir": str(tmp_path),
            "artifacts": {"vocals": str(wav)},
        },
        viewing_mode="latest",
    )
    assert session["isolate_artifacts"]["vocals"] == str(wav)
    assert session["isolate_viewing_run_dir"] == str(tmp_path)
    assert session["isolate_viewing_job_id"] == "latest"


def test_apply_succeeded_job_to_session_infers_youtube_kind(tmp_path: Path):
    wav = tmp_path / "vocals.wav"
    wav.write_bytes(b"RIFF")
    session: dict = {}
    ok = apply_succeeded_job_to_session(
        session,
        {
            "id": "yt1",
            "status": "succeeded",
            "title": "Party",
            "run_dir": str(tmp_path),
            "artifacts": {"vocals": str(wav)},
            "source_fingerprint": "youtube:https://youtu.be/abc",
        },
        viewing_mode="latest",
    )
    assert ok is True
    assert session["isolate_source_kind"] == "youtube"


def test_enqueue_persists_source_kind(jobs_dir: Path):
    spec = IsolateJobSpec(
        id="ytjob",
        audio_path=str(jobs_dir / "a.wav"),
        output_dir=str(jobs_dir / "out"),
        title="Party",
        created_at=time.time(),
        source_kind="youtube",
        source_fingerprint="youtube:https://youtu.be/abc",
    )
    (jobs_dir / "a.wav").write_bytes(b"x")
    enqueue_job(spec)
    status = read_status("ytjob")
    assert status is not None
    assert status["source_kind"] == "youtube"
    assert status["source_fingerprint"] == "youtube:https://youtu.be/abc"
    rows = list_jobs(limit=5)
    match = next(r for r in rows if r["id"] == "ytjob")
    assert match["source_kind"] == "youtube"


def test_enqueue_persists_origin_tab(jobs_dir: Path):
    spec = IsolateJobSpec(
        id="draftjob",
        audio_path=str(jobs_dir / "a.wav"),
        output_dir=str(jobs_dir / "out"),
        title="Party",
        created_at=time.time(),
        origin_tab="__new__:abc123",
    )
    (jobs_dir / "a.wav").write_bytes(b"x")
    enqueue_job(spec)
    status = read_status("draftjob")
    assert status is not None
    assert status["origin_tab"] == "__new__:abc123"
    stored = read_spec("draftjob")
    assert stored is not None
    assert stored.origin_tab == "__new__:abc123"


def test_apply_succeeded_job_to_session_pins_when_requested(tmp_path: Path):
    wav = tmp_path / "drums.wav"
    wav.write_bytes(b"RIFF")
    session: dict = {}
    apply_succeeded_job_to_session(
        session,
        {
            "id": "old",
            "status": "succeeded",
            "title": "Old",
            "run_dir": str(tmp_path),
            "artifacts": {"drums": str(wav)},
        },
        viewing_mode="old",
    )
    assert session["isolate_viewing_job_id"] == "old"


def test_apply_succeeded_job_to_session_missing_wavs_returns_false(tmp_path: Path):
    session: dict = {"isolate_artifacts": {"keep": "x"}}
    ok = apply_succeeded_job_to_session(
        session,
        {
            "id": "gone",
            "status": "succeeded",
            "title": "Gone",
            "run_dir": str(tmp_path),
            "artifacts": {"vocals": str(tmp_path / "missing.wav")},
        },
    )
    assert ok is False
    assert session["isolate_artifacts"] == {"keep": "x"}


def test_apply_finished_job_poll_notify_only_keeps_picker_and_artifacts(tmp_path: Path):
    from ui.isolate_state import LISTEN_PICKER_KEY, OPEN_MIX_TABS_KEY, open_new_draft_tab

    run_a = tmp_path / "song-a"
    run_b = tmp_path / "song-b"
    run_a.mkdir()
    run_b.mkdir()
    wav_a = run_a / "vocals.wav"
    wav_b = run_b / "vocals.wav"
    wav_a.write_bytes(b"RIFF-a")
    wav_b.write_bytes(b"RIFF-b")

    session: dict = {
        "isolate_viewing_job_id": "job-a",
        "isolate_artifacts": {"vocals": str(wav_a)},
        "isolate_run_dir": str(run_a),
        LISTEN_PICKER_KEY: str(run_a),
        "isolate_listen_applied_dir": str(run_a),
        OPEN_MIX_TABS_KEY: [str(run_a)],
    }
    draft = open_new_draft_tab(session, fresh=True)
    status_b = {
        "id": "job-b",
        "status": "succeeded",
        "title": "Song B",
        "run_dir": str(run_b),
        "origin_tab": draft,
        "artifacts": {"vocals": str(wav_b)},
    }

    open_mixer = apply_finished_job_poll_outcome(session, status_b, notify_only=True)
    assert open_mixer is False
    assert session[LISTEN_PICKER_KEY] == str(run_a)
    assert session["isolate_listen_applied_dir"] == str(run_a)
    assert session["isolate_artifacts"]["vocals"] == str(wav_a)
    assert session["isolate_run_dir"] == str(run_a)
    assert str(run_b) in session[OPEN_MIX_TABS_KEY]
    assert draft not in session[OPEN_MIX_TABS_KEY]
    assert "Song B" in str(session.get("isolate_flash") or "")


def test_apply_finished_job_poll_auto_apply_pins_and_retargets_picker(tmp_path: Path):
    from ui.isolate_state import LISTEN_PICKER_KEY, OPEN_MIX_TABS_KEY

    run = tmp_path / "song-new"
    run.mkdir()
    wav = run / "vocals.wav"
    wav.write_bytes(b"RIFF")
    session: dict = {
        "isolate_viewing_job_id": "latest",
        LISTEN_PICKER_KEY: str(tmp_path / "old"),
        "isolate_listen_applied_dir": str(tmp_path / "old"),
        OPEN_MIX_TABS_KEY: [],
    }
    status = {
        "id": "job-new",
        "status": "succeeded",
        "title": "Fresh",
        "run_dir": str(run),
        "artifacts": {"vocals": str(wav)},
    }
    open_mixer = apply_finished_job_poll_outcome(session, status, notify_only=False)
    assert open_mixer is True
    assert session[LISTEN_PICKER_KEY] == str(run)
    assert session["isolate_listen_applied_dir"] == str(run)
    assert session["isolate_artifacts"]["vocals"] == str(wav)
    assert session["isolate_viewing_job_id"] == "job-new"
    assert str(run) in session[OPEN_MIX_TABS_KEY]
    assert "Separated 1 tracks" in str(session.get("isolate_flash") or "")


def test_apply_finished_job_poll_apply_failure_leaves_picker(tmp_path: Path):
    from ui.isolate_state import LISTEN_PICKER_KEY, OPEN_MIX_TABS_KEY

    run = tmp_path / "song-missing"
    run.mkdir()
    old = str(tmp_path / "kept")
    session: dict = {
        LISTEN_PICKER_KEY: old,
        "isolate_listen_applied_dir": old,
        "isolate_artifacts": {"vocals": str(tmp_path / "old.wav")},
        OPEN_MIX_TABS_KEY: [],
    }
    status = {
        "id": "job-miss",
        "status": "succeeded",
        "title": "Missing",
        "run_dir": str(run),
        "artifacts": {"vocals": str(run / "gone.wav")},
    }
    open_mixer = apply_finished_job_poll_outcome(session, status, notify_only=False)
    assert open_mixer is False
    assert session[LISTEN_PICKER_KEY] == old
    assert session["isolate_listen_applied_dir"] == old
    assert session["isolate_artifacts"]["vocals"] == str(tmp_path / "old.wav")
    assert str(run) in session[OPEN_MIX_TABS_KEY]


def test_worker_persists_eta_fields(jobs_dir: Path, monkeypatch):
    """Worker writes audio_duration_sec / stages / job_estimate_sec before finishing."""
    import ui.isolate_jobs as jobs

    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "1")

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        on_progress("separate", "mock")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None)
    monkeypatch.setattr("audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock())
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "_WORKER_STARTED", False)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", None)
    monkeypatch.setattr(jobs, "_WORKER_THREAD", None)

    spec = IsolateJobSpec(
        id="eta1",
        audio_path=str(audio),
        output_dir=str(out),
        title="Eta",
        audio_duration_sec=90.0,
        created_at=time.time(),
    )
    enqueue_job(spec)
    ensure_worker_started()
    deadline = time.time() + 5
    while time.time() < deadline:
        status = read_status("eta1")
        if status and status.get("status") in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    status = read_status("eta1")
    assert status is not None
    assert status["status"] == "succeeded", status.get("error") or status.get("traceback") or status
    assert status.get("audio_duration_sec") == 90.0
    assert status.get("stages")
    assert status.get("job_estimate_sec")


def test_format_job_error_keeps_demucs_segment_message():
    raw = (
        "Demucs failed: Warning: You are sending unauthenticated requests to the HF Hub. "
        "Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n"
        "Cannot use a Transformer model with a longer segment than it was trained for. "
        "Maximum segment is: 7.8\n"
    )
    cleaned = format_job_error(raw)
    assert "HF_TOKEN" not in cleaned
    assert "unauthenticated" not in cleaned
    assert "Maximum segment is: 7.8" in cleaned


def test_format_job_error_is_one_short_sentence_without_app_paths():
    raw = (
        "guitar-ft unavailable (RuntimeError: Could not load libtorchcodec "
        "from /Applications/AudioTools.app/Contents/Frameworks/libavutil.59.dylib "
        "because FFmpeg 7 is missing); using stock htdemucs_6s\n"
        + ("x" * 400)
    )
    cleaned = format_job_error(raw)
    assert "/Applications/AudioTools.app" not in cleaned
    assert "libavutil" not in cleaned
    assert len(cleaned) <= 160
    assert cleaned


def test_format_job_error_missing_stem_hides_application_support_path():
    raw = (
        "[Errno 2] No such file or directory: "
        "'/Users/JJ/Library/Application Support/AudioTools/runs/"
        "38472114-475f-4f8a-9e22-0c6f6773990c/other.wav'"
    )
    cleaned = format_job_error(raw)
    assert cleaned == "Missing stem file"
    assert "Application" not in cleaned
    assert "Support" not in cleaned
    assert "other.wav" not in cleaned


def test_format_job_error_strips_quoted_path_with_spaces():
    raw = (
        "Could not load codec from "
        "'/Users/JJ/Library/Application Support/AudioTools/libavutil.dylib'"
    )
    cleaned = format_job_error(raw)
    assert "Application Support" not in cleaned
    assert "/Users/" not in cleaned
    assert "libavutil" not in cleaned
    assert "Could not load codec from" in cleaned


def test_spec_round_trips_reseparate_fields():
    import ui.isolate_jobs as jobs

    spec = IsolateJobSpec(
        id="r1",
        audio_path="C:\\a.wav",
        output_dir="C:\\out",
        title="T",
        reseparate_from="other",
        parent_run_dir="C:\\parent",
    )
    restored = jobs.IsolateJobSpec.from_dict(spec.to_dict())
    assert restored.reseparate_from == "other"
    assert restored.parent_run_dir == "C:\\parent"


def test_run_one_job_reseparate_skips_metadata_and_stores_fields(jobs_dir: Path, monkeypatch):
    """Re-separate job does not write a new Recent run; fields travel in status."""
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "other.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()
    parent = jobs_dir / "parent"
    parent.mkdir()

    metadata_calls: list = []

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        on_progress("separate", "mock")
        dest = Path(output_dir) / "guitar.wav"
        dest.write_bytes(b"g")
        return {"other::guitar": dest}

    def fake_write_metadata(*args, **kwargs):
        metadata_calls.append((args, kwargs))

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", fake_write_metadata)
    monkeypatch.setattr(
        "audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock()
    )

    spec = IsolateJobSpec(
        id="resp-run",
        audio_path=str(audio),
        output_dir=str(out),
        title="Other",
        created_at=time.time(),
        reseparate_from="other",
        parent_run_dir=str(parent),
    )
    jobs.enqueue_job(spec)
    jobs._run_one_job(spec.id)
    status = jobs.read_status(spec.id)
    assert status["status"] == "succeeded"
    assert status["reseparate_from"] == "other"
    assert status["parent_run_dir"] == str(parent)
    assert metadata_calls == []


def test_run_one_job_success_clears_prior_error(jobs_dir: Path, monkeypatch):
    """Succeeded status must not keep a leftover ENOENT from a racing fail write."""
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "song.wav"
    audio.write_bytes(b"wav")
    out = jobs_dir / "out"
    out.mkdir()

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        on_progress("separate", "mock")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock()
    )

    spec = IsolateJobSpec(
        id="leftover-err",
        audio_path=str(audio),
        output_dir=str(out),
        title="Brianstorm",
        created_at=time.time(),
    )
    jobs.enqueue_job(spec)
    write_status(
        spec.id,
        status="failed",
        stage="error",
        message="Separation failed",
        error=(
            "[Errno 2] No such file or directory: "
            "'/Users/JJ/Library/Application Support/AudioTools/runs/x/other.wav'"
        ),
        traceback="FileNotFoundError: other.wav\n",
    )
    jobs._run_one_job(spec.id)
    status = jobs.read_status(spec.id)
    assert status is not None
    assert status["status"] == "succeeded"
    assert status.get("error") is None
    assert status.get("traceback") is None


def test_apply_succeeded_job_to_session_reseparate_merges_and_repoints(tmp_path: Path):
    import ui.isolate_jobs as jobs

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_meta = {
        "page": "isolate",
        "title": "Song",
        "config": {"model": "htdemucs_6s", "quality": "balanced"},
        "artifacts": {"other": str(parent / "other.wav")},
    }
    (parent / "meta.json").write_text(json.dumps(parent_meta), encoding="utf-8")

    job_dir = tmp_path / "jobdir"
    job_dir.mkdir()
    t = np.linspace(0, 0.2, 8820, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    child = job_dir / "guitar.wav"
    sf.write(str(child), tone, 44100)

    session: dict = {
        "isolate_results_source_fp": "fp-parent",
        "isolate_source_kind": "file",
        "isolate_artifacts": {"other": str(parent / "other.wav")},
        "isolate_selected_stems": {"other": True},
        "isolate_run_dir": str(parent),
        "isolate_viewing_run_dir": str(parent),
    }
    ok = jobs.apply_succeeded_job_to_session(
        session,
        {
            "id": "resp1",
            "status": "succeeded",
            "run_dir": str(job_dir),
            "parent_run_dir": str(parent),
            "reseparate_from": "other",
            "artifacts": {"other::guitar": str(child)},
        },
        viewing_mode="latest",
    )
    assert ok is True
    assert "other" not in session["isolate_artifacts"]
    assert session["isolate_artifacts"]["other::guitar"] == str(parent / "guitar.wav")
    assert session["isolate_selected_stems"]["other::guitar"] is True
    assert "other" not in session["isolate_selected_stems"]
    assert session["isolate_run_dir"] == str(parent)
    assert session["isolate_viewing_run_dir"] == str(parent)
    assert session["isolate_results_source_fp"] == "fp-parent"
    assert (parent / "guitar.wav").exists()
    meta = json.loads((parent / "meta.json").read_text(encoding="utf-8"))
    assert "other" not in meta["artifacts"]
    assert meta["artifacts"]["other::guitar"] == str(parent / "guitar.wav")
    assert meta["config"] == {"model": "htdemucs_6s", "quality": "balanced"}


def test_apply_succeeded_job_to_session_reseparate_empty_children_noop(tmp_path: Path):
    import ui.isolate_jobs as jobs

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_meta = {
        "artifacts": {"other": str(parent / "other.wav")},
        "config": {"model": "htdemucs_6s"},
    }
    (parent / "meta.json").write_text(json.dumps(parent_meta), encoding="utf-8")

    job_dir = tmp_path / "jobdir"
    job_dir.mkdir()
    zero = np.zeros(8820, dtype=np.float32)
    silent = job_dir / "guitar.wav"
    sf.write(str(silent), zero, 44100)

    session: dict = {
        "isolate_results_source_fp": "fp-parent",
        "isolate_artifacts": {"other": str(parent / "other.wav")},
    }
    ok = jobs.apply_succeeded_job_to_session(
        session,
        {
            "id": "respEmpty",
            "status": "succeeded",
            "run_dir": str(job_dir),
            "parent_run_dir": str(parent),
            "reseparate_from": "other",
            "artifacts": {"other::guitar": str(silent)},
        },
    )
    assert ok is False
    assert session["isolate_artifacts"] == {"other": str(parent / "other.wav")}
    meta = json.loads((parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["artifacts"] == {"other": str(parent / "other.wav")}


def test_rewrite_parent_meta_preserves_config_and_noop_on_missing(tmp_path: Path):
    import ui.isolate_jobs as jobs

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "meta.json").write_text(
        json.dumps(
            {
                "page": "isolate",
                "title": "Song",
                "config": {"model": "htdemucs_6s"},
                "artifacts": {"other": str(parent / "other.wav")},
                "source_kind": "file",
            }
        ),
        encoding="utf-8",
    )
    session = {
        "isolate_artifacts": {
            "other::guitar": str(parent / "guitar.wav"),
            "vocals": str(parent / "vocals.wav"),
        }
    }
    jobs._rewrite_parent_meta(str(parent), session)
    meta = json.loads((parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["artifacts"] == session["isolate_artifacts"]
    assert meta["config"] == {"model": "htdemucs_6s"}
    assert meta["page"] == "isolate"
    assert meta["source_kind"] == "file"

    jobs._rewrite_parent_meta(str(tmp_path / "missing"), session)


def test_finalize_promotes_running_to_succeeded_when_stems_exist(jobs_dir: Path):
    import ui.isolate_jobs as jobs

    out = jobs_dir / "out"
    out.mkdir()
    guitar = out / "guitar.wav"
    guitar.write_bytes(b"RIFF")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "page": "isolate",
                "title": "Song",
                "artifacts": {"guitar": str(guitar)},
            }
        ),
        encoding="utf-8",
    )
    audio = jobs_dir / "a.wav"
    audio.write_bytes(b"x")
    spec = IsolateJobSpec(
        id="job-recover",
        audio_path=str(audio),
        output_dir=str(out),
        title="Song",
        created_at=time.time(),
    )
    enqueue_job(spec)
    write_status(
        "job-recover",
        status="running",
        stage="separate",
        message="Working",
        error="leftover",
        traceback="tb",
    )
    jobs._finalize_job_after_process("job-recover")
    status = read_status("job-recover")
    assert status is not None
    assert status["status"] == "succeeded"
    assert status["artifacts"]["guitar"] == str(guitar)
    assert status.get("error") is None
    assert status.get("traceback") is None


def test_finalize_marks_failed_when_running_without_stems(jobs_dir: Path):
    import ui.isolate_jobs as jobs

    out = jobs_dir / "empty-out"
    out.mkdir()
    audio = jobs_dir / "a.wav"
    audio.write_bytes(b"x")
    spec = IsolateJobSpec(
        id="job-dead",
        audio_path=str(audio),
        output_dir=str(out),
        title="Song",
        created_at=time.time(),
    )
    enqueue_job(spec)
    write_status("job-dead", status="running", stage="separate", message="Working")
    jobs._finalize_job_after_process("job-dead")
    status = read_status("job-dead")
    assert status is not None
    assert status["status"] == "failed"
    assert "Process exited before completion" in str(status.get("error") or "")


def test_finalize_fails_queued_job_when_child_exits(jobs_dir: Path):
    """A child that dies before status leaves queued must not stay Waiting."""
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "a.wav"
    audio.write_bytes(b"x")
    out = jobs_dir / "out"
    out.mkdir()
    enqueue_job(
        IsolateJobSpec(
            id="stuck",
            audio_path=str(audio),
            output_dir=str(out),
            title="Stuck",
            created_at=1.0,
        )
    )
    assert read_status("stuck")["status"] == "queued"
    jobs._finalize_job_after_process("stuck")
    status = read_status("stuck")
    assert status is not None
    assert status["status"] == "failed"
    assert status["error"] == jobs._WORKER_EXITED_BEFORE_START
    assert "stuck" not in jobs.queued_job_ids()
    assert (jobs_dir / "isolate_jobs" / "stuck").is_dir()


def test_queued_child_exit_lets_next_job_run(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    audio_a = jobs_dir / "a.wav"
    audio_b = jobs_dir / "b.wav"
    audio_a.write_bytes(b"a")
    audio_b.write_bytes(b"b")
    out_a = jobs_dir / "out_a"
    out_b = jobs_dir / "out_b"
    out_a.mkdir()
    out_b.mkdir()
    calls: list[str] = []

    def fake_separate(*, audio_path, output_dir, config, on_progress, **kwargs):
        calls.append(Path(audio_path).name)
        on_progress("separate", "mock")
        dest = Path(output_dir) / "vocals.wav"
        dest.write_bytes(b"v")
        return {"vocals": dest}

    monkeypatch.setattr("audio_to_tab.isolate.separate_stems", fake_separate)
    monkeypatch.setattr("audio_to_tab.hardware.ensure_cuda_available", lambda *a, **k: None)
    monkeypatch.setattr("audio_to_tab.hardware.get_desktop_probe", lambda: MagicMock())
    monkeypatch.setattr("ui.media.cleanup_mix_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr("ui.common.write_run_metadata", lambda *_a, **_k: None)

    enqueue_job(
        IsolateJobSpec(
            id="phantom",
            audio_path=str(audio_a),
            output_dir=str(out_a),
            title="Phantom",
            created_at=1.0,
        )
    )
    enqueue_job(
        IsolateJobSpec(
            id="next",
            audio_path=str(audio_b),
            output_dir=str(out_b),
            title="Next",
            created_at=2.0,
        )
    )
    jobs._finalize_job_after_process("phantom")
    jobs._execute_job("next")

    assert read_status("phantom")["status"] == "failed"
    nxt = read_status("next")
    assert nxt is not None
    assert nxt["status"] == "succeeded", nxt
    assert calls == ["b.wav"]


def test_setup_error_marks_job_failed(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "a.wav"
    audio.write_bytes(b"x")
    out = jobs_dir / "out"
    out.mkdir()
    enqueue_job(
        IsolateJobSpec(
            id="setup-boom",
            audio_path=str(audio),
            output_dir=str(out),
            title="Boom",
            created_at=time.time(),
        )
    )

    def boom():
        raise OSError("probe failed")

    monkeypatch.setattr("audio_to_tab.hardware.get_desktop_probe", boom)
    jobs._run_one_job("setup-boom")
    status = read_status("setup-boom")
    assert status is not None
    assert status["status"] == "failed"
    assert "probe failed" in str(status.get("error") or "")
    assert "setup-boom" not in jobs.queued_job_ids()


def test_deleted_job_is_not_resurrected(jobs_dir: Path):
    import ui.isolate_jobs as jobs

    audio = jobs_dir / "a.wav"
    audio.write_bytes(b"x")
    enqueue_job(
        IsolateJobSpec(
            id="dropme",
            audio_path=str(audio),
            output_dir=str(jobs_dir / "out"),
            title="Drop",
            created_at=time.time(),
        )
    )
    assert remove_job("dropme") is True
    job_dir = jobs_dir / "isolate_jobs" / "dropme"
    assert jobs.read_status("dropme") is None
    assert not job_dir.exists()
    assert jobs.write_status("dropme", completed_stages=["ingest"]) is None
    assert not job_dir.exists()
    assert not (job_dir / "status.json").exists()
    assert "dropme" not in jobs.queued_job_ids()


def test_reconcile_fails_stale_running_job(jobs_dir: Path):
    import ui.isolate_jobs as jobs

    write_status("stale-run", status="running", stage="separate", message="Working")
    jobs._reconcile_orphaned_jobs()
    status = read_status("stale-run")
    assert status is not None
    assert status["status"] == "failed"
    assert separation_in_progress() is False


def test_reconcile_paused_cancelled_and_keeps_queued(jobs_dir: Path):
    import ui.isolate_jobs as jobs

    write_status("pause-me", status="pausing", message="Pausing…")
    write_status("cancel-me", status="cancelled", message="Removed from queue")
    write_status("wait-me", status="queued", message="Queued")
    jobs._reconcile_orphaned_jobs()
    assert read_status("pause-me")["status"] == "paused"
    assert read_status("cancel-me") is None
    assert not (jobs_dir / "isolate_jobs" / "cancel-me").exists()
    assert read_status("wait-me")["status"] == "queued"
    assert jobs.queued_job_ids() == ["wait-me"]


def test_worker_start_reconciles_stale_running(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setenv("ISOLATE_JOBS_WORKER", "1")
    monkeypatch.setattr(jobs, "_WORKER_STARTED", False)
    monkeypatch.setattr(jobs, "_WORKER_THREAD", None)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", None)
    monkeypatch.setattr(jobs, "_ACTIVE_PROCESS", None)
    write_status("orphan-run", status="running", stage="separate", message="Working")
    ensure_worker_started()
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = read_status("orphan-run")
        if status and status.get("status") == "failed":
            break
        time.sleep(0.05)
    assert status is not None
    assert status["status"] == "failed"
    assert separation_in_progress() is False


def test_worker_busy_false_when_process_dead(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    proc = MagicMock()
    proc.is_alive.return_value = False
    monkeypatch.setattr(jobs, "_ACTIVE_PROCESS", proc)
    monkeypatch.setattr(jobs, "_ACTIVE_JOB_ID", "ghost")
    write_status("ghost", status="cancelled", message="Removed from queue")
    assert jobs.worker_busy() is False
    assert jobs._ACTIVE_JOB_ID is None
    assert jobs._ACTIVE_PROCESS is None
    assert jobs.active_job_id() is None
    assert not (jobs.active_job_id() and jobs.worker_busy())


def _batch_spec(path: Path, root: Path) -> IsolateJobSpec:
    return IsolateJobSpec(
        id="",
        audio_path=str(path),
        output_dir=str(root / "out" / path.parent.name),
        title=path.stem,
        model="htdemucs_6s",
        device="cpu",
        quality="fast",
    )


def _touch_songs(root: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        folder = root / f"song{index:02d}"
        folder.mkdir()
        name = "song.wav" if index < 2 else f"track{index}.wav"
        path = folder / name
        path.write_bytes(b"x")
        paths.append(path)
    return paths


def test_local_batch_enqueues_three_oldest_first(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "ensure_worker_started", lambda: None)
    monkeypatch.setattr(jobs, "_execute_job", lambda job_id: (_ for _ in ()).throw(AssertionError(job_id)))
    paths = _touch_songs(jobs_dir, 3)
    result = jobs.enqueue_local_batch(
        paths,
        section_selected=False,
        youtube_url="",
        make_spec=lambda path: _batch_spec(path, jobs_dir),
    )
    assert len(result.job_ids) == 3
    assert result.omitted == 0
    assert result.blocked is None
    queued = jobs._queued_job_ids()
    assert queued == result.job_ids
    specs = [read_spec(job_id) for job_id in queued]
    assert [spec.audio_path for spec in specs if spec] == [str(path) for path in paths]
    assert [spec.device for spec in specs if spec] == ["cpu", "cpu", "cpu"]
    assert [spec.model for spec in specs if spec] == ["htdemucs_6s"] * 3
    assert specs[0] is not None and specs[1] is not None
    assert specs[0].title == specs[1].title == "song"
    assert read_status(queued[0])["status"] == "queued"


def test_local_batch_leaves_out_the_21st_file(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "ensure_worker_started", lambda: None)
    paths = _touch_songs(jobs_dir, 21)
    result = jobs.enqueue_local_batch(
        paths,
        section_selected=False,
        youtube_url="",
        make_spec=lambda path: _batch_spec(path, jobs_dir),
    )
    assert len(result.job_ids) == 20
    assert result.omitted == 1
    queued_paths = {
        spec.audio_path
        for job_id in jobs._queued_job_ids()
        if (spec := read_spec(job_id)) is not None
    }
    assert str(paths[20]) not in queued_paths
    assert str(paths[0]) in queued_paths
    assert len(jobs._queued_job_ids()) == 20


def test_local_batch_section_enqueues_nothing(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "ensure_worker_started", lambda: None)
    paths = _touch_songs(jobs_dir, 3)
    result = jobs.enqueue_local_batch(
        paths,
        section_selected=True,
        youtube_url="",
        make_spec=lambda path: (_ for _ in ()).throw(AssertionError(path)),
    )
    assert result.job_ids == []
    assert result.blocked == "Separate one file when a section is selected."
    assert jobs._queued_job_ids() == []


def test_local_batch_youtube_enqueues_nothing(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "ensure_worker_started", lambda: None)
    paths = _touch_songs(jobs_dir, 2)
    result = jobs.enqueue_local_batch(
        paths,
        section_selected=False,
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        make_spec=lambda path: (_ for _ in ()).throw(AssertionError(path)),
    )
    assert result.job_ids == []
    assert result.blocked
    assert "YouTube" in result.blocked
    assert jobs._queued_job_ids() == []


def test_local_batch_reports_a_missing_file_and_keeps_the_rest(jobs_dir: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "ensure_worker_started", lambda: None)
    paths = _touch_songs(jobs_dir, 2)
    missing = jobs_dir / "gone" / "missing.wav"
    result = jobs.enqueue_local_batch(
        [paths[0], missing, paths[1]],
        section_selected=False,
        youtube_url="",
        make_spec=lambda path: _batch_spec(path, jobs_dir),
    )
    assert len(result.job_ids) == 2
    assert result.failed == ["missing.wav"]
    assert [spec.audio_path for job_id in result.job_ids if (spec := read_spec(job_id))] == [
        str(paths[0]),
        str(paths[1]),
    ]


def test_isolate_uploader_accepts_multiple_files():
    source = Path(__file__).resolve().parents[1].joinpath("ui", "pages", "isolate.py").read_text(
        encoding="utf-8"
    )
    assert "accept_multiple_files=True" in source
    assert source.count('key="isolate_extra_tracks"') == 1
    assert source.count("st.file_uploader(") == 2


def _stem(path: Path, scale: float) -> None:
    sr = 22050
    y = np.full(sr, scale, dtype=np.float32)
    sf.write(str(path), y, sr)


def test_harmony_source_prefers_guitar_piano_sum(tmp_path: Path):
    import ui.isolate_jobs as jobs

    for name, scale in (("guitar", 0.2), ("piano", 0.1), ("other", 0.05), ("bass", 0.15)):
        _stem(tmp_path / f"{name}.wav", scale)
    _stem(tmp_path / "no_vocals.wav", 0.4)
    mix = tmp_path / "mix.wav"
    _stem(mix, 0.5)
    artifacts = {name: tmp_path / f"{name}.wav" for name in ("guitar", "piano", "other", "bass", "no_vocals")}

    harmony, _vocals, _metro, kind = jobs._analysis_sources(artifacts, mix)

    assert kind == "guitar_piano_sum"
    assert harmony is not None
    assert harmony.name == "harmony_sum.wav"
    data, _sr = sf.read(str(harmony))
    assert float(np.max(np.abs(data))) == pytest.approx(0.99, abs=0.02)
    assert harmony.resolve() != artifacts["no_vocals"].resolve()
    assert harmony.resolve() != artifacts["other"].resolve()


def test_harmony_source_guitar_only_when_no_piano(tmp_path: Path):
    import ui.isolate_jobs as jobs

    _stem(tmp_path / "guitar.wav", 0.2)
    _stem(tmp_path / "other.wav", 0.05)
    mix = tmp_path / "mix.wav"
    _stem(mix, 0.5)
    artifacts = {"guitar": tmp_path / "guitar.wav", "other": tmp_path / "other.wav"}

    harmony, _vocals, _metro, kind = jobs._analysis_sources(artifacts, mix)

    assert kind == "guitar_sum"
    assert harmony is not None
    assert harmony.name == "harmony_sum.wav"


def test_harmony_source_too_loud_sum_uses_guitar(tmp_path: Path):
    import ui.isolate_jobs as jobs

    _stem(tmp_path / "guitar.wav", 5.0)
    _stem(tmp_path / "piano.wav", 5.0)
    _stem(tmp_path / "other.wav", 0.05)
    _stem(tmp_path / "no_vocals.wav", 0.4)
    mix = tmp_path / "mix.wav"
    _stem(mix, 0.2)
    artifacts = {
        "guitar": tmp_path / "guitar.wav",
        "piano": tmp_path / "piano.wav",
        "other": tmp_path / "other.wav",
        "no_vocals": tmp_path / "no_vocals.wav",
    }

    harmony, _vocals, _metro, kind = jobs._analysis_sources(artifacts, mix)

    assert kind == "guitar_sum"
    assert harmony is not None
    assert harmony.resolve() == artifacts["guitar"].resolve()
    assert not (tmp_path / "harmony_sum.wav").exists()


def test_harmony_source_keeps_other_when_no_guitar(tmp_path: Path):
    import ui.isolate_jobs as jobs
    from audio_to_tab.isolate import fold_other_into_guitar

    other = tmp_path / "other.wav"
    _stem(other, 0.2)
    artifacts = {"other": other}
    fold_other_into_guitar(artifacts)

    assert other.is_file()
    assert "other" in artifacts
    harmony, _vocals, _metro, kind = jobs._analysis_sources(artifacts, tmp_path / "missing.wav")
    assert kind == "other"
    assert harmony == other
