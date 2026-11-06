"""Phase 2 rows that are pure functions. No model, no worker, no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_to_tab.ingest import download_direct_media, is_direct_media_url
from ui.desktop_export import write_wav_48_24
from ui.isolate_jobs import IsolateJobSpec, isolate_config_kwargs_from_spec
from ui.isolate_state import resolve_pro_engine_job_fields
from ui.library_song import (
    chord_chart_text,
    delete_library_song,
    filter_library_rows,
    list_trashed_songs,
    locale_text,
    recover_library_song,
    stage_extra_tracks,
)
from ui.opened_audio import paths_from_argv


def test_direct_media_url_accepts_a_file_link_and_rejects_the_rest():
    assert is_direct_media_url("https://cdn.example.com/tracks/song.mp3")
    assert is_direct_media_url("https://files.example.com/clip.MKV?token=1")
    assert not is_direct_media_url("https://www.youtube.com/watch?v=abc")
    assert not is_direct_media_url("https://example.com/watch")
    assert not is_direct_media_url("file:///tmp/song.mp3")
    assert not is_direct_media_url("https://localhost/song.wav")
    assert not is_direct_media_url("https://127.0.0.1/song.wav")


def test_download_direct_media_saves_the_bytes(tmp_path: Path):
    class FakeResponse:
        def read(self, n=-1):
            return b"RIFF"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        path = download_direct_media("https://cdn.example.com/song.wav", tmp_path)
    assert path.name == "song.wav"
    assert path.read_bytes() == b"RIFF"


def test_download_direct_media_rejects_a_page(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError, match="direct audio or video"):
        download_direct_media("https://example.com/playlist", tmp_path)


def test_lead_rhythm_stays_off_until_the_guitar_job_opts_in():
    default = IsolateJobSpec(id="j", audio_path="a.wav", output_dir="out", title="t")
    assert isolate_config_kwargs_from_spec(default)["lead_rhythm"] is False
    opted = IsolateJobSpec(
        id="j",
        audio_path="a.wav",
        output_dir="out",
        title="t",
        lead_rhythm=True,
        lead_rhythm_mode="confident",
    )
    kwargs = isolate_config_kwargs_from_spec(opted)
    assert kwargs["lead_rhythm"] is True
    assert kwargs["lead_rhythm_mode"] == "confident"
    fields = resolve_pro_engine_job_fields(
        {"isolate_lead_rhythm": True},
        is_pro=True,
        model="htdemucs_6s",
        quality="balanced",
        two_stems=None,
        two_pass=False,
        fold_other_into_guitar=True,
        fold_other_mode="best_effort",
    )
    assert fields["lead_rhythm"] is True
    other = resolve_pro_engine_job_fields(
        {"isolate_lead_rhythm": True},
        is_pro=True,
        model="htdemucs",
        quality="balanced",
        two_stems=None,
        two_pass=False,
        fold_other_into_guitar=True,
        fold_other_mode="best_effort",
    )
    assert other["lead_rhythm"] is False


def test_hifi_label_uses_the_installed_fine_tuned_model():
    source = Path("ui/pages/isolate.py").read_text(encoding="utf-8")
    assert "Hi-Fi (fine-tuned Demucs)" in source
    fields = resolve_pro_engine_job_fields(
        {"isolate_htdemucs_ft": True},
        is_pro=True,
        model="htdemucs",
        quality="balanced",
        two_stems=None,
        two_pass=False,
        fold_other_into_guitar=True,
        fold_other_mode="best_effort",
    )
    assert fields["model"] == "htdemucs_ft"


def test_stage_extra_tracks_keeps_audio_and_skips_other_files(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    saved = stage_extra_tracks(
        run,
        [("solo.wav", b"wav"), ("notes.txt", b"no"), ("../escape.mp3", b"mp3")],
    )
    assert [Path(path).name for path in saved] == ["solo.wav", "escape.mp3"]
    assert (run / "extra_tracks" / "solo.wav").read_bytes() == b"wav"
    assert not (run / "escape.mp3").exists()


def test_write_wav_48_24(tmp_path: Path):
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros(441), 44100, subtype="PCM_16")
    dest = tmp_path / "out.wav"
    write_wav_48_24(src, dest)
    info = sf.info(dest)
    assert info.samplerate == 48000
    assert info.subtype == "PCM_24"


def test_chord_chart_text_lists_symbols_and_skips_silence():
    text = chord_chart_text(
        [
            {"start": 0, "symbol": "Am"},
            {"start": 65, "symbol": "N"},
            {"start": 70, "symbol": "G"},
        ]
    )
    assert "0:00  Am" in text
    assert "1:10  G" in text
    assert "N" not in text.splitlines()[-1]


def test_library_filter_is_separate_from_search():
    rows = [
        {"title": "Waltz", "action": "practice"},
        {"title": "Etude", "action": "analyse"},
    ]
    assert filter_library_rows(rows, "all") == rows
    assert filter_library_rows(rows, "practice") == [rows[0]]
    assert filter_library_rows(rows, "analyse") == [rows[1]]


def test_delete_and_recover_a_library_song(tmp_path: Path):
    data = tmp_path / "data"
    run = data / "song"
    run.mkdir(parents=True)
    (run / "meta.json").write_text("{}", encoding="utf-8")
    trashed = delete_library_song(run, data)
    assert trashed is not None
    assert not run.exists()
    assert trashed.parent.name == "library_trash"
    assert list_trashed_songs(data) == [trashed]
    restored = recover_library_song(trashed, data)
    assert restored == run
    assert (run / "meta.json").is_file()
    assert list_trashed_songs(data) == []


def test_delete_library_song_returns_none_when_the_trash_name_is_taken(tmp_path: Path):
    data = tmp_path / "data"
    run = data / "song"
    run.mkdir(parents=True)
    (run / "meta.json").write_text("{}", encoding="utf-8")
    occupied = data / "library_trash" / "song"
    occupied.mkdir(parents=True)
    assert delete_library_song(run, data) is None
    assert (run / "meta.json").is_file()


def test_delete_follows_a_symlink_into_the_data_dir(tmp_path: Path):
    data = tmp_path / "data"
    run = data / "song"
    run.mkdir(parents=True)
    (run / "meta.json").write_text("{}", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(data, target_is_directory=True)
    trashed = delete_library_song(link / "song", data)
    assert trashed is not None
    assert not run.exists()
    assert trashed.parent.name == "library_trash"


def test_locale_defaults_to_english_and_korean_is_the_first_extra():
    assert locale_text("en", "search") == "Search"
    assert locale_text("ko", "search") == "검색"
    assert locale_text("nope", "search") == "Search"


def test_file_url_video_is_accepted_and_midi_is_declined(tmp_path: Path):
    song = tmp_path / "Song.wav"
    song.write_bytes(b"audio")
    clip = tmp_path / "clip.MP4"
    clip.write_bytes(b"video")
    midi = tmp_path / "chart.mid"
    midi.write_bytes(b"MThd")
    accepted, declined = paths_from_argv(["AudioTools", clip.as_uri(), str(song), str(midi)])
    assert accepted == [clip, song]
    assert declined == [midi]


def test_logout_revokes_the_refresh_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ATT_REFRESH_REVOKE_PATH", str(tmp_path / "revoked.json"))
    from backend.auth import create_refresh_token, decode_token
    from backend.refresh_store import refresh_jti_revoked, revoke_refresh_jti

    token = create_refresh_token("user-1")
    jti = decode_token(token, "refresh")["jti"]
    assert refresh_jti_revoked(jti) is False
    revoke_refresh_jti(jti)
    assert refresh_jti_revoked(jti) is True


def test_player_rows_already_in_the_mixer_source():
    source = Path("ui/stem_mixer_component/frontend/src/main.ts").read_text(encoding="utf-8")
    assert "function waveformSeekHtml" in source
    assert 'id="metro-accent"' in source
    assert "if (isTypingTarget(e.target)) return;" in source
    assert 'e.code !== "Space"' in source
