"""Page-logic unit tests for the Streamlit pages (pure helpers, not st.*).

The pages are guarded by ``if __name__ == "__main__"`` so a bare import
(resolved via ``ui.ensure_src_path()``) executes only definitions — Streamlit
runs the same file with ``__name__ == "__main__"`` and the page body still
renders (see streamlit.navigation.page.Page.run). Only import-safe helpers are
exercised here; anything that touches ``st.*`` or the fs-heavy engine is out of
scope for I-098.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def isolate_page():
    from ui.pages import isolate

    return isolate


@pytest.fixture(scope="module")
def tab_pdf_page():
    from ui.pages import tab_pdf

    return tab_pdf


@pytest.fixture(scope="module")
def youtube_audio_page():
    from ui.pages import youtube_audio

    return youtube_audio


@pytest.fixture(scope="module")
def app_module():
    # ui.app calls st.html at import time; Streamlit reads sys.argv[0] on the
    # first element write when not under `streamlit run`.
    if not sys.argv:
        sys.argv = ["pytest"]
    import ui.app as app

    return app


def test_isolate_page_imports_without_running_main(isolate_page) -> None:
    assert callable(isolate_page.main)


def test_isolate_queue_float_helpers_are_wired(isolate_page) -> None:
    assert isolate_page.ISOLATE_QUEUE_PANEL_OPEN_KEY == "isolate_queue_panel_open"
    assert callable(isolate_page._toggle_queue_panel)
    assert callable(isolate_page._queue_header_fragment)
    assert callable(isolate_page._render_queue_float_panel)
    assert callable(isolate_page._queue_tab_fragment)
    assert callable(isolate_page._render_job_queue_panel)
    assert callable(isolate_page._render_queue_job_row)
    assert callable(isolate_page._render_compact_running_hint)
    assert callable(isolate_page._retry_failed_job)


def test_tab_pdf_page_imports_without_running_main(tab_pdf_page) -> None:
    assert callable(tab_pdf_page.main)


def test_youtube_audio_page_imports_without_running_main(youtube_audio_page) -> None:
    assert callable(youtube_audio_page.main)
    assert callable(youtube_audio_page.save_youtube_audio_to_folder)


def test_save_youtube_audio_to_folder_happy_path(youtube_audio_page, tmp_path) -> None:
    wav = tmp_path / "work" / "Song Title.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"wav")
    dest_dir = tmp_path / "out"
    downloaded: list[str] = []

    def fake_download(url, work_dir):
        downloaded.append(url)
        return wav

    def fake_export(src, dest, filename, fmt):
        out = Path(dest) / f"{Path(filename).stem}.{fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(Path(src).read_bytes())
        return out

    saved = youtube_audio_page.save_youtube_audio_to_folder(
        "https://www.youtube.com/watch?v=abc",
        dest_dir,
        "mp3",
        work_dir=wav.parent,
        download_fn=fake_download,
        export_fn=fake_export,
    )
    assert downloaded == ["https://www.youtube.com/watch?v=abc"]
    assert saved == dest_dir / "Song Title.mp3"
    assert saved.read_bytes() == b"wav"


def test_save_youtube_audio_to_folder_rejects_non_youtube(youtube_audio_page, tmp_path) -> None:
    called = []

    def fake_download(url, work_dir):
        called.append(url)
        raise AssertionError("download should not run")

    with pytest.raises(ValueError, match="Only YouTube"):
        youtube_audio_page.save_youtube_audio_to_folder(
            "https://example.com/a.wav",
            tmp_path,
            "mp3",
            work_dir=tmp_path,
            download_fn=fake_download,
        )
    assert called == []


def test_save_youtube_audio_to_folder_propagates_download_error(
    youtube_audio_page, tmp_path
) -> None:
    from audio_to_tab.ingest import YouTubeDownloadError

    def fake_download(url, work_dir):
        raise YouTubeDownloadError("Could not download this YouTube video.")

    with pytest.raises(YouTubeDownloadError, match="Could not download"):
        youtube_audio_page.save_youtube_audio_to_folder(
            "https://youtu.be/abc",
            tmp_path,
            "mp3",
            work_dir=tmp_path,
            download_fn=fake_download,
        )


def test_save_youtube_audio_skips_download_when_wav_exists(youtube_audio_page, tmp_path) -> None:
    wav = tmp_path / "ready.wav"
    wav.write_bytes(b"staged")
    dest_dir = tmp_path / "out"

    def fake_download(url, work_dir):
        raise AssertionError("should reuse staged wav")

    def fake_export(src, dest, filename, fmt):
        assert Path(src) == wav
        out = Path(dest) / f"ready.{fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ok")
        return out

    saved = youtube_audio_page.save_youtube_audio_to_folder(
        "https://www.youtube.com/watch?v=abc",
        dest_dir,
        "flac",
        work_dir=tmp_path,
        wav_path=wav,
        download_fn=fake_download,
        export_fn=fake_export,
    )
    assert saved.read_bytes() == b"ok"


def test_youtube_search_page_helpers() -> None:
    from audio_to_tab.ingest import YouTubeSearchHit
    from ui.youtube_search import (
        PAGE_SIZE,
        has_next_page,
        hits_as_dicts,
        meta_line,
        normalize_page,
        page_offset,
        result_range_caption,
    )

    assert PAGE_SIZE == 10
    assert page_offset(1) == 0
    assert page_offset(2) == 10
    assert normalize_page(0) == 1
    assert normalize_page("3") == 3
    assert has_next_page(10)
    assert not has_next_page(9)
    assert result_range_caption(2, 10) == "Page 2 · results 11–20"
    assert result_range_caption(3, 0) == "Page 3 · no results"
    hit = YouTubeSearchHit(
        video_id="abc",
        title="T",
        channel="C",
        duration_sec=90,
        url="https://www.youtube.com/watch?v=abc",
        thumbnail_url="https://img.example/t.jpg",
    )
    assert hits_as_dicts([hit])[0]["video_id"] == "abc"
    assert meta_line({"channel": "C"}, duration_label="1:30") == "C · 1:30"


def test_save_youtube_audio_convert_failure_keeps_wav(youtube_audio_page, tmp_path) -> None:
    wav = tmp_path / "ready.wav"
    wav.write_bytes(b"staged")

    def fake_export(src, dest, filename, fmt):
        raise RuntimeError("ffmpeg conversion to mp3 failed")

    with pytest.raises(RuntimeError, match="ffmpeg conversion"):
        youtube_audio_page.save_youtube_audio_to_folder(
            "https://www.youtube.com/watch?v=abc",
            tmp_path / "out",
            "mp3",
            work_dir=tmp_path,
            wav_path=wav,
            download_fn=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no download")),
            export_fn=fake_export,
        )
    assert wav.is_file()
    assert wav.read_bytes() == b"staged"


def test_app_page_icon_points_at_existing_file(app_module) -> None:
    icon = app_module._page_icon()
    if icon:
        assert Path(icon).is_file()


def test_artifact_fingerprint_is_order_independent(isolate_page) -> None:
    a = {"guitar": Path("x/g.wav"), "vocals": Path("x/v.wav")}
    b = {"vocals": Path("x/v.wav"), "guitar": Path("x/g.wav")}
    fa = isolate_page._artifact_fingerprint(a)
    assert fa == isolate_page._artifact_fingerprint(b)
    assert len(fa) == 16
    assert fa.isalnum()
    changed = {"guitar": Path("y/g.wav"), "vocals": Path("x/v.wav")}
    assert fa != isolate_page._artifact_fingerprint(changed)


def test_selection_fingerprint_aliases_artifact_fingerprint(isolate_page) -> None:
    stems = {"guitar": Path("a/g.wav"), "bass": Path("a/b.wav")}
    assert isolate_page._selection_fingerprint(stems) == isolate_page._artifact_fingerprint(stems)


def test_stem_paths_from_artifacts_filters_diagnostics_and_non_wav(isolate_page, tmp_path) -> None:
    guitar = tmp_path / "guitar.wav"
    vocals = tmp_path / "vocals.wav"
    midi = tmp_path / "out.mid"
    for p in (guitar, vocals, midi):
        p.write_bytes(b"\x00")
    artifacts = {
        "guitar": str(guitar),
        "vocals": str(vocals),
        "out_midi": str(midi),
        "stem_presence_diagnostics": str(tmp_path / "sp.json"),
        "bass_bleed_diagnostics": str(tmp_path / "bb.json"),
        "guitar_split_diagnostics": str(tmp_path / "gs.json"),
    }
    result = isolate_page._stem_paths_from_artifacts(artifacts)
    assert set(result) == {"guitar", "vocals"}
    assert all(p.suffix == ".wav" for p in result.values())


def test_load_guitar_split_diagnostics_returns_dict_or_empty(isolate_page, tmp_path) -> None:
    assert isolate_page._load_guitar_split_diagnostics({}) == {}
    good = tmp_path / "gs.json"
    good.write_text(json.dumps({"lead_present": True}), encoding="utf-8")
    assert isolate_page._load_guitar_split_diagnostics({"guitar_split_diagnostics": str(good)}) == {
        "lead_present": True
    }


def test_mixer_export_fingerprint_is_deterministic_and_sensitive(isolate_page) -> None:
    args = (["guitar", "bass"], {"guitar": -3.0, "bass": -6.0}, {}, {}, 0.0)
    f1 = isolate_page._mixer_export_fingerprint(*args)
    assert f1 == isolate_page._mixer_export_fingerprint(*args)
    louder = isolate_page._mixer_export_fingerprint(
        ["guitar", "bass"], {"guitar": 0.0, "bass": -6.0}, {}, {}, 0.0
    )
    assert f1 != louder
    panned = isolate_page._mixer_export_fingerprint(
        ["guitar", "bass"],
        {"guitar": -3.0, "bass": -6.0},
        {},
        {},
        0.0,
        pans={"guitar": -40, "bass": 0},
    )
    assert panned != f1
    # Pan on a muted stem cannot change the file, so it must not invalidate it.
    muted_out = isolate_page._mixer_export_fingerprint(
        ["guitar", "bass"],
        {"guitar": -3.0, "bass": -6.0},
        {"bass": True},
        {},
        0.0,
        pans={"guitar": 0, "bass": -100},
    )
    muted_other = isolate_page._mixer_export_fingerprint(
        ["guitar", "bass"],
        {"guitar": -3.0, "bass": -6.0},
        {"bass": True},
        {},
        0.0,
        pans={"guitar": 0, "bass": 100},
    )
    assert muted_out == muted_other


def test_speed_preset_labels_resolve(isolate_page) -> None:
    for preset_id in isolate_page.SPEED_PRESETS:
        label = isolate_page._speed_preset_radio_label(preset_id)
        assert isinstance(label, str) and label
    with pytest.raises(KeyError):
        isolate_page._speed_preset_radio_label("not-a-preset")


def test_stem_hints_cover_key_stems(isolate_page) -> None:
    assert {"piano", "guitar", "metronome"} <= set(isolate_page.STEM_HINTS)
    assert "default_muted_for" in (
        Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    ).read_text(encoding="utf-8")


def test_hosted_isolate_blocks_over_cap_and_section_until_duration() -> None:
    src = (
        Path(__file__).resolve().parents[1] / "web" / "src" / "pages" / "IsolatePage.tsx"
    ).read_text(encoding="utf-8")
    assert "if (regionOverCap)" in src
    assert "disabled={!sectionReady}" in src
    assert "startBlocked" in src
    assert "stack-secondary" not in src
    tab = (
        Path(__file__).resolve().parents[1] / "web" / "src" / "pages" / "TabPage.tsx"
    ).read_text(encoding="utf-8")
    assert "A tab job is already running" in tab
    assert "disabled={tabJobRunning}" in tab
    assert "stack-secondary" not in tab
    mixer = (
        Path(__file__).resolve().parents[1]
        / "web"
        / "src"
        / "components"
        / "StemMixer.tsx"
    ).read_text(encoding="utf-8")
    assert "SLEEP_RESUME_HINT" in mixer
    assert 'className="primary live-play"' in mixer
    assert 'prev === SLEEP_RESUME_HINT' in mixer


def test_home_relative_age_and_search_and_real_destinations() -> None:
    from ui.pages.home import _matches, relative_age

    now = 1_700_000_000.0
    assert relative_age(now - 30, now=now) == "just now"
    assert relative_age(now - 120, now=now) == "2 minutes ago"
    assert relative_age(now - 3 * 3600, now=now) == "3 hours ago"
    assert relative_age(now - 2 * 86400, now=now) == "2 days ago"
    assert relative_age("nope") == ""
    row = {"title": "Party Song", "source_kind": "file", "run_dir": "/tmp/r"}
    assert _matches(row, "party")
    assert not _matches(row, "mastering")
    page = (
        Path(__file__).resolve().parents[1] / "ui" / "pages" / "home.py"
    ).read_text(encoding="utf-8")
    assert "Mastering" not in page
    assert "Voice Studio" not in page
    assert "Lyric" not in page
    assert 'open_new_separation' in page
    # 0.1.6 removed Home's page cards; YouTube and Tab PDF are sidebar pages now,
    # so the real destinations are asserted on the navigation, not on Home.
    app = (Path(__file__).resolve().parents[1] / "ui" / "app.py").read_text(encoding="utf-8")
    assert 'youtube_audio.py' in app
    assert 'tab_pdf.py' in app
    assert "open_saved_run" in page