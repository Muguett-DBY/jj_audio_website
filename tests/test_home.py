"""Home screen: practice entry points and analysis status."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_to_tab.analysis import AnalysisResult, ChordEvent, Section, save_analysis
from ui.pages.home import (
    analysis_line,
    analysis_status_markup,
    analysis_status_notice,
    analysis_status_should_refresh_page,
    analysis_summary,
    auto_analysis_should_skip,
    next_auto_analysis,
    row_analysis_subtitle,
)


def _analysis_result() -> AnalysisResult:
    return AnalysisResult(
        duration_sec=64.0,
        bpm=92.4,
        key="G major",
        chords=(
            ChordEvent(start=0.0, end=4.0, symbol="G", confidence=0.0),
            ChordEvent(start=4.0, end=8.0, symbol="C", confidence=0.0),
        ),
        sections=(
            Section(start=0.0, end=32.0, label="verse", confidence=0.6),
            Section(start=32.0, end=64.0, label="chorus", confidence=0.6),
        ),
    )


def test_analysis_status_notice_names_the_song_and_the_queue():
    notice = analysis_status_notice(
        {"state": "analysing", "title": "Island in the Sun", "remaining": 4}
    )
    assert notice == "Getting key and chords for Island in the Sun. 3 more still waiting."
    assert analysis_status_notice({"state": "analysing", "title": "Island in the Sun", "remaining": 1}) == (
        "Getting key and chords for Island in the Sun."
    )
    assert analysis_status_notice({"state": "waiting", "remaining": 2}) == (
        "Key and chords will continue after the current separation. 2 still waiting."
    )
    assert analysis_status_notice({"state": "idle", "remaining": 4}) == ""
    assert analysis_status_notice(None) == ""
    markup = analysis_status_markup(notice + " <script>")
    assert 'class="att-analysis-status"' in markup
    assert "Island in the Sun" in markup
    assert "<script>" not in markup
    assert "role=\"status\"" in markup


def test_row_subtitle_shows_which_song_is_being_analysed():
    status = {"state": "analysing", "run_dir": "/songs/open", "remaining": 2}
    assert row_analysis_subtitle(None, "/songs/open", status) == "Getting key and chords…"
    assert row_analysis_subtitle(None, "/songs/other", status) == "Waiting for key and chords"
    assert row_analysis_subtitle(
        {"bpm": 100, "key": "C", "section_count": 1},
        "/songs/open",
        status,
    ) == "BPM 100 · C · 1 section"
    assert row_analysis_subtitle(None, "/songs/other", {"state": "idle"}) == "Not analysed yet"
    waiting = {"state": "waiting", "remaining": 3}
    assert row_analysis_subtitle(None, "/songs/other", waiting) == "Waiting for key and chords"


def test_home_refreshes_when_analysis_progress_changes_and_the_mixer_does_not():
    assert analysis_status_should_refresh_page("home", "analysing|a|2|", None) is True
    assert analysis_status_should_refresh_page("home", "|||", None) is False
    assert analysis_status_should_refresh_page("home", "analysing|b|1|", "analysing|a|2|") is True
    assert analysis_status_should_refresh_page("", "idle|||done", "analysing|a|1|") is True
    assert analysis_status_should_refresh_page("isolate", "analysing|b|1|", "analysing|a|2|") is False


def test_analysis_line_formats_bpm_key_and_sections():
    summary = {"bpm": 92, "key": "G major", "section_count": 4}
    assert analysis_line(summary) == "BPM 92 · G major · 4 sections"
    assert analysis_line({"bpm": 120.5, "key": None, "section_count": 1}) == (
        "BPM 120.5 · Key — · 1 section"
    )
    assert analysis_line(None) == "Not analysed yet"
    assert analysis_line({"section_count": 0}) == "BPM — · Key — · No sections"


def test_analysis_summary_prefers_meta_and_falls_back_to_analysis_json(tmp_path: Path):
    meta_summary = {"bpm": 100, "key": "C major", "section_count": 3}
    assert analysis_summary({"analysis": meta_summary}) == meta_summary

    path = tmp_path / "analysis.json"
    save_analysis(path, _analysis_result())
    row = {"artifacts": {"analysis": str(path)}}
    summary = analysis_summary(row)
    assert summary is not None
    assert summary["bpm"] == 92.4
    assert summary["key"] == "G major"
    assert summary["section_count"] == 2
    assert summary["chord_count"] == 2


def test_analysis_summary_ignores_missing_or_corrupt_files(tmp_path: Path):
    assert analysis_summary({"artifacts": {"analysis": str(tmp_path / "nope.json")}}) is None
    bad = tmp_path / "analysis.json"
    bad.write_text("{bad", encoding="utf-8")
    assert analysis_summary({"artifacts": {"analysis": str(bad)}}) is None
    assert analysis_summary({}) is None


def test_next_auto_analysis_picks_one_unanalysed_row_without_a_click():
    rows = [
        {"run_dir": "/songs/done", "analysis": {"bpm": 100, "key": "C", "section_count": 2}},
        {"run_dir": "/songs/fresh", "title": "Fresh"},
        {"run_dir": "/songs/also", "title": "Also"},
    ]
    assert next_auto_analysis(rows, busy=False, skipped=set()) == "/songs/fresh"


def test_next_auto_analysis_leaves_an_analysed_row():
    rows = [
        {"run_dir": "/songs/done", "analysis": {"bpm": 131.8, "key": "A#m", "section_count": 6}}
    ]
    assert next_auto_analysis(rows, busy=False, skipped=set()) is None


def test_next_auto_analysis_does_not_retry_a_skipped_failure():
    rows = [{"run_dir": "/songs/nostem", "title": "No stem"}]
    assert next_auto_analysis(rows, busy=False, skipped={"/songs/nostem"}) is None


def test_next_auto_analysis_schedules_nothing_while_the_worker_is_busy():
    rows = [{"run_dir": "/songs/fresh", "title": "Fresh"}]
    assert next_auto_analysis(rows, busy=True, skipped=set()) is None


def test_next_auto_analysis_leaves_the_open_mix_until_it_is_last():
    rows = [
        {"run_dir": "/songs/open", "title": "Open"},
        {"run_dir": "/songs/older", "title": "Older"},
    ]
    assert next_auto_analysis(rows, busy=False, skipped=set(), active_run="/songs/open") == (
        "/songs/older"
    )
    assert next_auto_analysis(
        [{"run_dir": "/songs/open"}],
        busy=False,
        skipped=set(),
        active_run="/songs/open",
    ) == "/songs/open"


def test_auto_analysis_skip_remembers_a_failure_and_not_a_busy_worker():
    assert auto_analysis_should_skip(False, "No guitar/instrumental stem to analyse.") is True
    assert auto_analysis_should_skip(False, "Analysis failed.") is True
    assert (
        auto_analysis_should_skip(
            False, "Wait for the current separation to finish, then analyse."
        )
        is False
    )
    assert auto_analysis_should_skip(True, "Analysis ready. Open Practice.") is False


def test_home_v2_does_not_block_on_automatic_analysis():
    root = Path(__file__).resolve().parents[1]
    source = (root / "ui" / "pages" / "home.py").read_text(encoding="utf-8")
    main = source[source.find("def _main_v2") : source.find("def _matches")]
    assert "next_auto_analysis(" not in main
    assert "home_auto_analysis_skipped" not in main
    assert 'st.session_state["home_v2_pending"] = run_dir' in source
    app = (root / "ui" / "app.py").read_text(encoding="utf-8")
    assert "start_background_analysis(" in app
    assert "_mount_analysis_status()" in app
    assert "run_every=2.0" in app
    jobs = (root / "ui" / "isolate_jobs.py").read_text(encoding="utf-8")
    poll = jobs[jobs.find("def _poll_background_analysis") : jobs.find("def start_background_analysis")]
    assert "st.rerun" not in poll
    assert "proc.join()" not in poll
    assert "is_alive(" in poll


def test_home_source_has_learn_practice_and_analyse_actions():
    source = (
        Path(__file__).resolve().parents[1] / "ui" / "pages" / "home.py"
    ).read_text(encoding="utf-8")
    assert "Learn a song" in source
    assert '"Practice"' in source
    assert '"Analyse"' in source
    assert "analysis_line(" in source
    assert "open_practice_run(" in source
    assert "analyse_existing_run(" in source
    # The existing flows stay put.
    assert "youtube_audio.py" in source
    assert "tab_pdf.py" in source


def test_analyse_existing_run_updates_meta(tmp_path: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    wav = tmp_path / "no_vocals.wav"
    sf.write(str(wav), np.zeros(1000, dtype=np.float32), 44100)
    meta = {
        "page": "isolate",
        "title": "Song",
        "source_fingerprint": "fp",
        "artifacts": {"no_vocals": str(wav)},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    result = _analysis_result()

    def fake_stage(run_dir, artifacts, **_kwargs):
        path = save_analysis(Path(run_dir) / "analysis.json", result)
        diagnostics = Path(run_dir) / "analysis_diagnostics.json"
        diagnostics.write_text("{}", encoding="utf-8")
        return result, path, diagnostics, "draft quality"

    monkeypatch.setattr(jobs, "worker_busy", lambda: False)
    monkeypatch.setattr(jobs, "_run_analysis_stage", fake_stage)

    ok, message = jobs.analyse_existing_run(tmp_path)

    assert ok is True
    assert message.startswith("Analysis ready with warnings:")
    assert "draft quality" in message
    updated = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert updated["analysis"]["bpm"] == 92.4
    assert updated["analysis"]["key"] == "G major"
    assert updated["analysis"]["section_count"] == 2
    assert updated["artifacts"]["analysis"].endswith("analysis.json")
    assert updated["artifacts"]["analysis_diagnostics"].endswith(
        "analysis_diagnostics.json"
    )


def test_analyse_existing_run_refuses_busy_worker(tmp_path: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "worker_busy", lambda: True)
    ok, message = jobs.analyse_existing_run(tmp_path)
    assert ok is False
    assert "separation" in message


def test_analyse_existing_run_requires_meta_and_a_stem(tmp_path: Path, monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "worker_busy", lambda: False)
    ok, message = jobs.analyse_existing_run(tmp_path)
    assert ok is False
    assert "meta.json" in message

    (tmp_path / "meta.json").write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    ok, message = jobs.analyse_existing_run(tmp_path)
    assert ok is False
    assert "stem" in message


def test_home_v2_is_the_default(monkeypatch):
    from ui.design_tokens import ui_v2_enabled

    monkeypatch.delenv("AUDIO_TOOLS_UI_V2", raising=False)
    assert ui_v2_enabled() is True
    monkeypatch.setenv("AUDIO_TOOLS_UI_V2", "0")
    assert ui_v2_enabled() is False


def test_home_v2_props_have_no_run_dir():
    import json

    from ui.pages.home import build_home_props

    secret = "/tmp/secret/song"
    props = build_home_props(rows=[{"run_dir": secret, "title": "x"}])
    encoded = json.dumps(props)
    assert "run_dir" not in encoded
    assert secret not in encoded
    assert props["rows"][0]["token"]
    assert props["rows"][0]["action"] == "analyse"
    assert props["cards"][0]["id"] == "learn"


def test_home_muted_token_clears_contrast():
    from ui.design_tokens import HOME_BG, HOME_TEXT_MUTED, contrast_ratio, home_flag_css

    css = home_flag_css()
    assert "width: 240px" in css
    assert "font-weight: 400" in css
    assert "font-weight: 600" not in css
    assert "att-shell-v2" in css
    assert 'stSidebar"]' in css
    assert "overflow: hidden" in css
    assert contrast_ratio(HOME_TEXT_MUTED, HOME_BG) >= 4.5
    assert contrast_ratio("#F4F7FB", HOME_BG) >= 4.5


def test_home_v2_keeps_the_legacy_path():
    source = (
        Path(__file__).resolve().parents[1] / "ui" / "pages" / "home.py"
    ).read_text(encoding="utf-8")
    assert "def _main_legacy" in source
    assert "st.title(product, anchor=False)" in source
    assert "ui_v2_enabled()" in source
    assert 'home_v2_first_paint_done' in source
    frontend = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "home_component"
        / "frontend"
        / "src"
        / "main.ts"
    ).read_text(encoding="utf-8")
    assert "run_dir" not in frontend
    assert "No songs yet." in frontend
    assert "No songs match that search." in frontend
    assert 'class="skel"' in frontend or "className = \"skel\"" in frontend or 'el("div", "skel")' in frontend
    assert "#F6AD49" in frontend
    assert "#75EBF5" not in frontend
    assert "#E9EEF5" not in frontend


def test_advance_background_analysis_covers_the_backlog_one_song_at_a_time():
    from ui.isolate_jobs import advance_background_analysis

    rows = [
        {"run_dir": "/songs/open"},
        {"run_dir": "/songs/older"},
        {"run_dir": "/songs/oldest"},
    ]
    skipped: set[str] = set()
    seen: list[str] = []

    def analyse(run_dir: str) -> tuple[bool, str]:
        seen.append(run_dir)
        for row in rows:
            if row["run_dir"] == run_dir:
                row["analysis"] = {"bpm": 120, "key": "A", "section_count": 2, "chord_count": 4}
        return True, "Analysis ready. Open Practice."

    kwargs = {
        "rows": rows,
        "separation_busy": False,
        "skipped": skipped,
        "active_run": "/songs/open",
        "analyse": analyse,
    }
    assert advance_background_analysis(**kwargs) == "/songs/older"
    assert advance_background_analysis(**kwargs) == "/songs/oldest"
    assert advance_background_analysis(**kwargs) == "/songs/open"
    assert advance_background_analysis(**kwargs) is None
    assert seen == ["/songs/older", "/songs/oldest", "/songs/open"]
    assert skipped == set()
    assert all(row.get("analysis", {}).get("key") == "A" for row in rows)


def test_advance_background_analysis_waits_for_separation_and_skips_a_missing_stem():
    from ui.isolate_jobs import advance_background_analysis

    rows = [{"run_dir": "/songs/fresh"}, {"run_dir": "/songs/nostem"}]
    skipped: set[str] = set()
    calls: list[str] = []

    def analyse(run_dir: str) -> tuple[bool, str]:
        calls.append(run_dir)
        if run_dir.endswith("nostem"):
            return False, "No guitar/instrumental stem to analyse."
        return False, "Wait for the current separation to finish, then analyse."

    assert (
        advance_background_analysis(
            rows,
            separation_busy=True,
            skipped=skipped,
            active_run=None,
            analyse=analyse,
        )
        is None
    )
    assert calls == []
    assert skipped == set()

    assert (
        advance_background_analysis(
            rows,
            separation_busy=False,
            skipped=skipped,
            active_run=None,
            analyse=analyse,
        )
        == "/songs/fresh"
    )
    assert skipped == set()
    assert (
        advance_background_analysis(
            rows,
            separation_busy=False,
            skipped=skipped,
            active_run=None,
            analyse=analyse,
        )
        == "/songs/fresh"
    )

    rows.pop(0)
    assert (
        advance_background_analysis(
            rows,
            separation_busy=False,
            skipped=skipped,
            active_run=None,
            analyse=analyse,
        )
        == "/songs/nostem"
    )
    assert skipped == {"/songs/nostem"}
    assert (
        advance_background_analysis(
            rows,
            separation_busy=False,
            skipped=skipped,
            active_run=None,
            analyse=analyse,
        )
        is None
    )


class _FakeAnalysisProcess:
    def __init__(self, target, args, name, daemon):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False
        self.exitcode = None
        self._alive = False

    def start(self):
        self.started = True
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False
        self.exitcode = -15

    def join(self, timeout=None):
        self._alive = False


def test_background_poll_starts_one_child_and_does_not_skip_a_busy_stop(monkeypatch):
    import ui.isolate_jobs as jobs

    created: list[_FakeAnalysisProcess] = []

    def fake_process(**kwargs):
        proc = _FakeAnalysisProcess(**kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(jobs.multiprocessing, "Process", fake_process)
    monkeypatch.setattr(jobs, "jobs_active", lambda: False)
    monkeypatch.setattr(jobs, "worker_busy", lambda: False)
    monkeypatch.setattr(jobs, "read_analysis_foreground", lambda: None)
    monkeypatch.setattr(jobs, "_ANALYSIS_PROCESS", None)
    monkeypatch.setattr(jobs, "_ANALYSIS_RUN", None)
    monkeypatch.setattr(jobs, "_ANALYSIS_SKIPPED", set())
    monkeypatch.setattr(
        "ui.pages.home.library_rows",
        lambda owner, limit=40: [{"run_dir": "/songs/fresh"}],
    )

    jobs._poll_background_analysis()
    assert len(created) == 1
    assert created[0].started is True
    assert created[0].args == ("/songs/fresh",)
    assert created[0].target is jobs._analyse_process_entry

    monkeypatch.setattr(jobs, "jobs_active", lambda: True)
    jobs._poll_background_analysis()
    assert jobs._ANALYSIS_PROCESS is None
    assert "/songs/fresh" not in jobs._ANALYSIS_SKIPPED

    created[0]._alive = False
    created[0].exitcode = 1
    monkeypatch.setattr(jobs, "jobs_active", lambda: False)
    monkeypatch.setattr(jobs, "_ANALYSIS_PROCESS", created[0])
    monkeypatch.setattr(jobs, "_ANALYSIS_RUN", "/songs/fresh")
    jobs._poll_background_analysis()
    assert "/songs/fresh" in jobs._ANALYSIS_SKIPPED
    jobs._poll_background_analysis()
    assert len(created) == 1


def test_publish_analysis_progress_names_the_song(monkeypatch):
    import ui.isolate_jobs as jobs

    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(jobs, "write_background_analysis_status", capture)
    monkeypatch.setattr(jobs, "_ANALYSIS_SKIPPED", set())
    jobs._publish_analysis_progress(
        [
            {"run_dir": "/songs/open", "title": "Island in the Sun"},
            {"run_dir": "/songs/older", "title": "Older"},
            {"run_dir": "/songs/done", "title": "Done", "analysis": {"bpm": 100, "key": "C", "section_count": 1}},
        ],
        state="analysing",
        run_dir="/songs/open",
        finished_run="",
    )
    assert captured["state"] == "analysing"
    assert captured["remaining"] == 2
    assert "Island in the Sun" in captured["title"]
    assert captured["run_dir"] == "/songs/open"


def test_analyse_process_result_uses_the_existing_writer(monkeypatch):
    import ui.isolate_jobs as jobs

    monkeypatch.setattr(jobs, "_analyse_run_in_place", lambda run_dir: (True, "ok"))
    assert jobs._analyse_process_result("/songs/fresh") == 0
    monkeypatch.setattr(
        jobs,
        "_analyse_run_in_place",
        lambda run_dir: (False, "No guitar/instrumental stem to analyse."),
    )
    assert jobs._analyse_process_result("/songs/fresh") == 1



def test_row_analysis_state_locks_until_analysed():
    from ui.pages.home import row_analysis_state

    running = {"state": "analysing", "run_dir": "a", "skipped": ["c"]}
    assert row_analysis_state({"key": "C"}, "a", running) == "ready"
    assert row_analysis_state(None, "a", running) == "analysing"
    assert row_analysis_state(None, "b", running) == "waiting"
    assert row_analysis_state(None, "c", running) == "failed"
    assert row_analysis_state(None, "b", {"state": "idle"}) == "idle"
    assert row_analysis_state(None, "b", None) == "idle"


def test_job_record_does_not_keep_a_trashed_mix_on_home(tmp_path, monkeypatch):
    """The broken step: isolate_jobs status still names the old wav after the move."""
    import ui.common as common
    import ui.isolate_jobs as jobs
    import ui.pages.home as home
    from ui.library_song import delete_library_song, recover_library_song

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "DATA_DIR", tmp_path)
    monkeypatch.setattr(home, "DATA_DIR", tmp_path)
    run = tmp_path / "song"
    run.mkdir()
    wav = run / "guitar.wav"
    wav.write_bytes(b"x")
    (run / "meta.json").write_text(
        json.dumps(
            {
                "page": "isolate",
                "title": "Song",
                "created_at": 1,
                "artifacts": {"guitar": str(wav)},
            }
        ),
        encoding="utf-8",
    )
    jobs._ensure_job_dir("s1")
    jobs.write_status(
        "s1",
        status="succeeded",
        title="Song",
        run_dir=str(run),
        artifacts={"guitar": str(wav)},
        created_at=1.0,
    )
    assert [row["run_dir"] for row in home.library_rows(None)] == [str(run)]
    trashed = delete_library_song(run, tmp_path)
    assert trashed is not None
    assert home.library_rows(None) == []
    assert (jobs.jobs_root() / "s1" / "status.json").is_file()
    recover_library_song(trashed, tmp_path)
    assert [row["run_dir"] for row in home.library_rows(None)] == [str(run)]


def test_apply_library_deletes_ignores_unknown_tokens_and_blocks_the_active_job(tmp_path):
    from ui.pages.home import apply_library_deletes

    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "meta.json").write_text("{}", encoding="utf-8")
    active = tmp_path / "active"
    active.mkdir()
    (active / "meta.json").write_text("{}", encoding="utf-8")
    one = tmp_path / "one"
    two = tmp_path / "two"
    three = tmp_path / "three"
    for folder in (one, two, three):
        folder.mkdir()
        (folder / "meta.json").write_text("{}", encoding="utf-8")
    tokens = {
        "aaaa": str(one),
        "bbbb": str(two),
        "cccc": str(three),
        "dddd": str(active),
    }
    status = {"state": "analysing", "run_dir": str(active)}
    moved, failed = apply_library_deletes(
        tokens,
        "aaaa,bbbb,cccc,dddd,not-a-token",
        data_dir=tmp_path,
        status=status,
    )
    assert (moved, failed) == (3, 1)
    assert not one.exists() and not two.exists() and not three.exists()
    assert active.is_dir() and keep.is_dir()
    assert (tmp_path / "library_trash" / "one").is_dir()
    moved, failed = apply_library_deletes(
        tokens, "not-a-token", data_dir=tmp_path, status=None
    )
    assert (moved, failed) == (0, 0)
    assert keep.is_dir() and active.is_dir()


def test_missing_mix_folder_fails_analysis_without_raising(tmp_path: Path):
    from ui.isolate_jobs import _analyse_process_result

    assert _analyse_process_result(str(tmp_path / "gone")) == 1


def test_home_delete_ui_selects_rows_except_the_one_being_analysed():
    frontend = (
        Path(__file__).resolve().parents[1] / "ui" / "home_component" / "frontend" / "src" / "main.ts"
    ).read_text(encoding="utf-8")
    wrapper = (
        Path(__file__).resolve().parents[1] / "ui" / "home_component" / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "trashed=list(props.get(\"trashed\") or [])" in wrapper
    assert 'row.state !== "analysing"' in frontend
    assert 'send(nonce, "delete_songs"' in frontend
    assert 'aria-label", "Select all"' in frontend
    assert 'selectMode && libraryFilter !== "deleted"' in frontend
    assert 'dataset.focusid = "select-mode"' in frontend
    assert "att-home-files-scroll" in frontend
