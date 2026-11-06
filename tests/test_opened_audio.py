"""OS open-with: queue, new Home tab, and the installer document types."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "clip_name", ["clip.MP4", "clip with space.MP4", "节奏.MP4", "take%20#1.MP4"]
)
def test_argv_keeps_audio_and_video_and_declines_midi(tmp_path: Path, clip_name: str):
    from ui.opened_audio import paths_from_argv

    song = tmp_path / "Song.wav"
    song.write_bytes(b"audio")
    clip = tmp_path / clip_name
    clip.write_bytes(b"video")
    midi = tmp_path / "chart.mid"
    midi.write_bytes(b"MThd")
    missing = tmp_path / "gone.wav"

    accepted, declined = paths_from_argv(
        [
            "AudioTools",
            "-psn_0_12345",
            str(song),
            clip.as_uri(),
            str(midi),
            str(missing),
            str(song),
        ]
    )
    assert accepted == [song, clip]
    assert declined == [midi]


def test_queue_claims_in_order_and_a_second_claim_is_empty(tmp_path: Path):
    from ui.opened_audio import claim_opened_paths, enqueue_opened_paths

    folder = tmp_path / "pending_open"
    enqueue_opened_paths(
        folder,
        [{"path": "/tmp/a.wav"}, {"path": "/tmp/b.wav", "declined": True}],
    )
    first = claim_opened_paths(folder)
    assert [item["path"] for item in first] == ["/tmp/a.wav", "/tmp/b.wav"]
    assert first[1]["declined"] is True
    assert claim_opened_paths(folder) == []
    assert claim_opened_paths(tmp_path / "missing") == []


def test_opened_file_lands_on_a_new_tab_and_the_original_stays(tmp_path: Path, monkeypatch):
    from ui.isolate_state import (
        ISOLATE_OUTPUT_NAME_PENDING_KEY,
        MIX_TAB_NOTICE_KEY,
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        SHELL_VIEW_NEXT_KEY,
        is_new_draft_tab,
    )
    from ui.pages import isolate as page

    original = tmp_path / "Song.wav"
    original.write_bytes(b"original-bytes")
    video = tmp_path / "Clip.mov"
    video.write_bytes(b"video-bytes")
    midi = tmp_path / "Chart.mid"
    midi.write_bytes(b"MThd")
    run = tmp_path / "finished-mix"
    run.mkdir()

    staged_root = tmp_path / "staged"
    staged_root.mkdir()

    def fresh_dir() -> Path:
        dest = staged_root / str(len(list(staged_root.iterdir())) + 1)
        dest.mkdir()
        return dest

    def fake_extract(src: Path, wav: Path) -> Path:
        wav.write_bytes(b"extracted")
        return wav

    monkeypatch.setattr(page, "run_output_dir", fresh_dir)
    monkeypatch.setattr(page, "normalize_audio", fake_extract)
    monkeypatch.setenv(page.PENDING_OPEN_ENV, str(tmp_path / "pending_open"))

    from ui.opened_audio import enqueue_opened_paths

    enqueue_opened_paths(
        tmp_path / "pending_open",
        [
            {"path": str(original)},
            {"path": str(video)},
            {"path": str(midi), "declined": True},
        ],
    )
    session = {
        OPEN_MIX_TABS_KEY: [str(run)],
        SHELL_TAB_KEY: str(run),
        "isolate_youtube_url": "https://youtu.be/keep-me",
    }
    assert page._consume_opened_audio(session) is True

    assert original.read_bytes() == b"original-bytes"
    assert video.read_bytes() == b"video-bytes"
    tabs = session[OPEN_MIX_TABS_KEY]
    assert str(run) in tabs
    drafts = [tab for tab in tabs if is_new_draft_tab(tab)]
    assert len(drafts) == 2
    assert session[SHELL_TAB_KEY] == drafts[-1]
    assert session[SHELL_VIEW_NEXT_KEY] == "home"
    # The focused tab is the video. Its audio is the extracted wav, not the .mov.
    focused = session["isolate_pending_audio_path"]
    assert focused.endswith("Clip.extracted.wav")
    assert Path(focused).read_bytes() == b"extracted"
    assert session["isolate_pending_video_path"].endswith("Clip.mov")
    assert session[ISOLATE_OUTPUT_NAME_PENDING_KEY] == "Clip"
    assert "isolate_youtube_url" not in session
    # Song stays staged on its own tab, and the finished mix is still in the strip.
    song_snap = session["isolate_draft_sources"][drafts[0]]
    assert song_snap["isolate_pending_audio_path"].endswith("Song.wav")
    assert Path(song_snap["isolate_pending_audio_path"]).read_bytes() == b"original-bytes"
    notice = session[MIX_TAB_NOTICE_KEY]
    assert "Opened 2 files on new tabs." in notice
    assert "Chart.mid" in notice
    assert page._consume_opened_audio(session) is False

    enqueue_opened_paths(
        tmp_path / "pending_open",
        [{"path": str(midi), "declined": True}],
    )
    assert page._consume_opened_audio(session) is True
    assert "Chart.mid" in session[MIX_TAB_NOTICE_KEY]
    assert len([tab for tab in session[OPEN_MIX_TABS_KEY] if is_new_draft_tab(tab)]) == 2


def test_installers_register_the_same_extensions_and_not_midi():
    from ui.opened_audio import OPENED_AUDIO_EXTENSIONS

    plist = (ROOT / "packaging" / "make_app.sh").read_text(encoding="utf-8")
    iss = (ROOT / "packaging" / "AudioTools.iss").read_text(encoding="utf-8")
    spec = (ROOT / "packaging" / "audio_tools.spec").read_text(encoding="utf-8")
    assert "CFBundleDocumentTypes" in plist
    assert "LSHandlerRank" in plist and "Alternate" in plist
    assert "ChangesAssociations=yes" in iss
    assert 'Subkey: "Software\\Classes\\AudioTools.audio\\shell\\open\\command"' in iss
    assert '""%1"""' in iss
    for ext in OPENED_AUDIO_EXTENSIONS:
        assert f"<string>{ext}</string>" in plist
        assert f"Software\\Classes\\.{ext}\\OpenWithProgids" in iss
    for blocked in ("mid", "midi"):
        assert f"<string>{blocked}</string>" not in plist
        assert f"\\.{blocked}\\" not in iss
    assert "argv_emulation=True" in spec


def test_isolate_page_claims_opens_on_first_paint_and_on_the_tab_strip():
    page = (ROOT / "ui" / "pages" / "isolate.py").read_text(encoding="utf-8")
    main = page[page.find("def main(") :]
    assert main.find("_consume_opened_audio()") < main.find('key="isolate_sticky_chrome"')
    fragment = page[page.find("def _mix_tab_strip_fragment") : page.find("def _render_busy_tab_close_prompt")]
    assert "if _consume_opened_audio():" in fragment
    assert "_rerun_scroll_top()" in fragment
