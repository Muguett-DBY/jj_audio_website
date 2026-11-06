"""Tests for multi-stem isolation (Demucs mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from audio_to_tab.isolate import (
    BASS_BLEED_ENERGY_SHARE_FLOOR,
    BASS_BLEED_HPF_CUTOFF_HZ,
    BASS_BLEED_LOW_BAND_HZ,
    GUITAR_HIGH_END_HZ,
    IsolateConfig,
    MIN_REGION_SEC,
    RegionError,
    SUPPORTED_MODELS,
    STEM_PRESENCE_ENERGY_SHARE_MIN,
    STEM_PRESENCE_FLOOR_DB,
    _trim_audio,
    analyze_bass_bleed,
    analyze_guitar_stem_quality,
    apply_bass_bleed_mitigation,
    apply_emit_stems,
    apply_fold_other_into_guitar,
    detect_present_stems,
    effective_demucs_segment,
    effective_isolation_quality,
    fold_other_into_guitar,
    format_region_label,
    merge_two_pass_stems,
    probe_duration_sec,
    resolve_region,
    separate_stems,
    si_sdr,
)
from audio_to_tab.subprocess_util import subprocess_run_kwargs

# Pre-warm librosa's lazily-imported scipy submodules at collection time (not
# inside a test). Some environments trigger a one-off scipy.ndimage import
# chain that shells out via platform.win32_ver(); doing it here keeps that
# away from tests below that mock audio_to_tab.separate.run_process.
import librosa  # noqa: E402

librosa.stft(np.zeros(2048, dtype=np.float32))


def _write_silent_wav(path: Path, n: int = 256, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros((n, 2), dtype=np.float32), sr)


def test_supported_models():
    assert "htdemucs_6s" in SUPPORTED_MODELS
    assert "htdemucs" in SUPPORTED_MODELS
    assert "htdemucs_ft" in SUPPORTED_MODELS


def test_fold_other_into_guitar_mixes_and_drops_other(tmp_path: Path):
    guitar = tmp_path / "guitar.wav"
    other = tmp_path / "other.wav"
    sf.write(str(guitar), np.ones((256, 2), dtype=np.float32) * 0.1, 44100)
    sf.write(str(other), np.ones((256, 2), dtype=np.float32) * 0.2, 44100)
    artifacts = {"guitar": guitar, "other": other, "vocals": tmp_path / "vocals.wav"}
    fold_other_into_guitar(artifacts)
    assert "other" not in artifacts
    assert not other.exists()
    data, _sr = sf.read(str(guitar))
    # guitar 0.1 + other 0.2 * FOLD_OTHER_MIX_GAIN(0.5) → 0.2 peak, which the
    # int16 WAV write quantizes to ~0.19995; assert it clearly exceeds the
    # guitar-only level (0.1) but stays under the 0.2 mix level.
    peak = float(np.max(np.abs(data)))
    assert 0.15 < peak <= 0.21, peak


def test_fold_other_into_guitar_keeps_other_without_guitar(tmp_path: Path):
    other = tmp_path / "other.wav"
    _write_silent_wav(other)
    artifacts = {"other": other, "vocals": tmp_path / "vocals.wav"}
    fold_other_into_guitar(artifacts)
    assert artifacts["other"] == other
    assert other.exists()


def test_fold_other_missing_file_is_soft_skip(tmp_path: Path):
    guitar = tmp_path / "guitar.wav"
    _write_silent_wav(guitar)
    artifacts = {"guitar": guitar, "other": tmp_path / "other.wav"}
    artifacts, diag = apply_fold_other_into_guitar(artifacts, mode="band_limited")
    assert diag.folded is False
    assert "not a file" in diag.reason
    assert artifacts["guitar"] == guitar


def test_fold_other_missing_mid_flight_does_not_raise(tmp_path: Path, monkeypatch):
    """other.wav can vanish after is_file() (peer fold); must not FileNotFoundError."""
    guitar = tmp_path / "guitar.wav"
    other = tmp_path / "other.wav"
    sf.write(str(guitar), np.ones((256, 2), dtype=np.float32) * 0.1, 44100)
    sf.write(str(other), np.ones((256, 2), dtype=np.float32) * 0.2, 44100)
    artifacts = {"guitar": guitar, "other": other}
    real_is_file = Path.is_file

    def is_file_then_drop_other(self: Path) -> bool:
        present = real_is_file(self)
        if present and self.name == "other.wav":
            self.unlink(missing_ok=True)
        return present

    monkeypatch.setattr(Path, "is_file", is_file_then_drop_other)
    artifacts, diag = apply_fold_other_into_guitar(artifacts, mode="band_limited")
    assert diag.folded is False
    assert "not a file" in diag.reason


def test_apply_emit_stems_drops_unlisted_wavs_keeps_diagnostics(tmp_path: Path):
    guitar = tmp_path / "guitar.wav"
    piano = tmp_path / "piano.wav"
    diag = tmp_path / "bass_bleed_diagnostics.json"
    _write_silent_wav(guitar)
    _write_silent_wav(piano)
    diag.write_text("{}", encoding="utf-8")
    artifacts = {
        "guitar": guitar,
        "piano": piano,
        "bass_bleed_diagnostics": diag,
    }
    apply_emit_stems(artifacts, ("vocals", "drums", "bass", "guitar"))
    assert "piano" not in artifacts
    assert not piano.exists()
    assert artifacts["guitar"] == guitar
    assert artifacts["bass_bleed_diagnostics"] == diag


def test_separate_stems_emit_stems_drops_piano(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        demucs_out = Path(cmd[cmd.index("-o") + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(
                model="htdemucs_6s",
                quality="fast",
                max_duration_sec=15,
                emit_stems=("vocals", "drums", "bass", "guitar"),
            ),
        )

    assert {"vocals", "drums", "bass", "guitar"} <= set(artifacts)
    assert "piano" not in artifacts
    assert "other" not in artifacts
    assert not (out_dir / "piano.wav").exists()


def test_separate_stems_recovery_blames_emit_policy_for_missing_bass(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        demucs_out = Path(cmd[cmd.index("-o") + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(
                model="htdemucs_6s",
                quality="fast",
                max_duration_sec=15,
                sub_bass_debleed=True,
                emit_stems=("guitar",),
            ),
        )

    diag_path = artifacts["low_end_recovery_diagnostics"]
    data = json.loads(Path(diag_path).read_text(encoding="utf-8"))
    assert data["attempted"] is False
    assert "dropped by emit policy" in data["reason"]
    assert "bass stem missing" not in data["reason"]


def test_separate_stems_keeps_other_when_fold_disabled(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        demucs_out = Path(cmd[cmd.index("-o") + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(
                model="htdemucs_6s",
                quality="fast",
                max_duration_sec=15,
                fold_other_into_guitar=False,
            ),
        )

    assert "other" in artifacts
    assert artifacts["other"].exists()
    assert "guitar" in artifacts
    assert "piano" in artifacts


def test_separate_stems_raises_region_error_when_start_past_unknown_duration(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate.probe_duration_sec", return_value=None),
    ):
        with pytest.raises(RegionError, match="can't be honored"):
            separate_stems(
                audio,
                out_dir,
                IsolateConfig(model="htdemucs_6s", quality="fast", start_sec=30.0),
            )


def test_separate_stems_raises_region_error_when_start_past_file_end(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate.probe_duration_sec", return_value=12.0),
    ):
        with pytest.raises(RegionError, match="can't be honored"):
            separate_stems(
                audio,
                out_dir,
                IsolateConfig(model="htdemucs_6s", quality="fast", start_sec=30.0),
            )


def test_is_demucs_available_does_not_import_torch(monkeypatch):
    from audio_to_tab.separate import is_demucs_available
    import inspect

    src = inspect.getsource(is_demucs_available)
    assert "find_spec" in src
    assert "import demucs" not in src
    assert "import torch" not in src

    monkeypatch.setattr(
        "audio_to_tab.separate.importlib.util.find_spec",
        lambda name: object() if name == "demucs" else None,
    )
    assert is_demucs_available() is True
    monkeypatch.setattr(
        "audio_to_tab.separate.importlib.util.find_spec",
        lambda name: None,
    )
    assert is_demucs_available() is False


def test_isolate_config_defaults_to_full_song():
    assert IsolateConfig().max_duration_sec is None


def test_isolate_config_default_quality_is_fast():
    assert IsolateConfig().quality == "fast"
    assert IsolateConfig().emit_metronome is True
    assert IsolateConfig().metronome_accent is True
    assert IsolateConfig().metronome_rate == 1.0
    assert IsolateConfig().metronome_sound == "classic"


def test_effective_demucs_segment_clamps_htdemucs_6s_below_transformer_max():
    """htdemucs_6s rejects --segment > 7.8; CLI only accepts ints, so cap at 7."""
    assert effective_demucs_segment("htdemucs_6s", 8) == 7
    assert effective_demucs_segment("htdemucs_6s", 7) == 7
    assert effective_demucs_segment("htdemucs_6s", 20) == 7


def test_effective_demucs_segment_keeps_ram_target_for_four_stem_models():
    assert effective_demucs_segment("htdemucs", 8) == 8
    assert effective_demucs_segment("htdemucs_ft", 8) == 8
    assert effective_demucs_segment("htdemucs", 12) == 10


def test_effective_demucs_segment_omits_flag_when_disabled():
    assert effective_demucs_segment("htdemucs_6s", None) is None
    assert effective_demucs_segment("htdemucs_6s", 0) is None
    assert effective_demucs_segment("htdemucs_6s", -1) is None


def test_separate_stems_requires_demucs(tmp_path: Path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    with patch("audio_to_tab.isolate.is_demucs_available", return_value=False):
        with pytest.raises(RuntimeError, match="Demucs"):
            separate_stems(audio, tmp_path / "out")


def test_separate_stems_rejects_unknown_model(tmp_path: Path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    with patch("audio_to_tab.isolate.is_demucs_available", return_value=True):
        with pytest.raises(ValueError, match="Unsupported model"):
            separate_stems(audio, tmp_path / "out", IsolateConfig(model="not_a_model"))


def test_separate_stems_rejects_cuda_when_unavailable(tmp_path: Path):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.hardware.probe_torch", return_value=(False, False)),
    ):
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            separate_stems(
                audio,
                tmp_path / "out",
                IsolateConfig(model="htdemucs", device="cuda"),
            )


def test_separate_stems_collects_wavs(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    seen_cmd: list[list[str]] = []

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        # Demucs output layout: -o <dir> → <dir>/<model>/<track>/*.wav
        seen_cmd.append(list(cmd))
        assert cmd[1:3] == ["-m", "audio_to_tab.demucs_bootstrap"]
        assert "-n" in cmd and "htdemucs_6s" in cmd
        out_flag = cmd.index("-o")
        demucs_out = Path(cmd[out_flag + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        # Guitar WAV must be loadable for bass-bleed / presence stages.
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", quality="fast", max_duration_sec=15),
        )

    assert {"vocals", "drums", "bass", "guitar", "piano"} <= set(artifacts)
    assert "other" not in artifacts
    assert "guitar_split_diagnostics" not in artifacts
    assert "lead_guitar" not in artifacts
    assert "stem_presence_diagnostics" in artifacts
    for path in artifacts.values():
        assert path.exists()
        assert path.parent == out_dir
    # Default path passes --jobs 1 and --segment for RAM-safe CPU runs.
    # Demucs argparse requires an int; htdemucs_6s max training segment is 7.8
    # so we must pass 7, not 8.
    assert "--jobs" in seen_cmd[0] and "1" in seen_cmd[0]
    assert "--segment" in seen_cmd[0]
    seg_idx = seen_cmd[0].index("--segment")
    assert seen_cmd[0][seg_idx + 1] == "7"


def _write_pulse_wav(path: Path, *, bpm: float = 120.0, duration: float = 8.0, sr: int = 22050) -> None:
    beat = 60.0 / bpm
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    click = int(0.02 * sr)
    i = 0
    while True:
        start = int(round(i * beat * sr))
        if start >= n:
            break
        y[start : min(n, start + click)] = 0.9
        i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.column_stack([y, y]), sr, subtype="PCM_16")


def test_separate_stems_emits_metronome_from_drums(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        demucs_out = Path(cmd[cmd.index("-o") + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        _write_pulse_wav(track_dir / "drums.wav")
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", quality="fast", max_duration_sec=15),
        )

    assert "metronome" in artifacts
    assert Path(artifacts["metronome"]).is_file()
    assert artifacts["metronome_diagnostics"].is_file()
    clicks, _sr = sf.read(str(artifacts["metronome"]), always_2d=True)
    assert float(np.max(np.abs(clicks))) > 0.05


def test_separate_stems_can_skip_metronome(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        demucs_out = Path(cmd[cmd.index("-o") + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(
                model="htdemucs_6s",
                quality="fast",
                max_duration_sec=15,
                emit_metronome=False,
            ),
        )

    assert "metronome" not in artifacts
    assert "metronome_diagnostics" not in artifacts


def test_separate_stems_threads_abort_and_timeout_into_backend(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake")
    captured: dict[str, object] = {}

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_backend(
        audio_path,
        demucs_out,
        cfg,
        *,
        model,
        allow_guitar_ft,
        progress,
        should_abort=None,
        timeout_sec=None,
    ):
        demucs_out.mkdir(parents=True, exist_ok=True)
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(track_dir / "guitar.wav"), np.zeros((256, 2), dtype=np.float32), 44100)
        captured["should_abort"] = should_abort
        captured["timeout_sec"] = timeout_sec

    def abort_flag() -> bool:
        return False

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.isolate._run_separation_backend", side_effect=fake_backend),
    ):
        separate_stems(
            audio,
            tmp_path / "out",
            IsolateConfig(model="htdemucs_6s", quality="fast"),
            should_abort=abort_flag,
            subprocess_timeout_sec=123.0,
        )

    assert captured["should_abort"] is abort_flag
    assert captured["timeout_sec"] == 123.0


def test_separate_stems_passes_two_stems_flag(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    seen: list[list[str]] = []

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        seen.append(cmd)
        out_flag = cmd.index("-o")
        demucs_out = Path(cmd[out_flag + 1])
        track_dir = demucs_out / "htdemucs" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "vocals.wav").write_bytes(b"v")
        (track_dir / "no_vocals.wav").write_bytes(b"nv")
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            tmp_path / "out",
            IsolateConfig(model="htdemucs", two_stems="vocals", max_duration_sec=None),
        )

    assert "--two-stems" in seen[0]
    assert "vocals" in seen[0]
    assert set(artifacts) == {"vocals", "no_vocals", "stem_presence_diagnostics"}


def test_separate_stems_keeps_other_without_guitar(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        out_flag = cmd.index("-o")
        demucs_out = Path(cmd[out_flag + 1])
        track_dir = demucs_out / "htdemucs" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs", quality="fast", max_duration_sec=15),
        )

    assert "other" in artifacts
    assert artifacts["other"].exists()
    assert "guitar" not in artifacts


def _fake_normalize_factory(tmp_path: Path):
    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    return fake_normalize


def _fake_demucs_with_guitar(stereo_left, stereo_right, sr: int = 44100):
    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        out_flag = cmd.index("-o")
        demucs_out = Path(cmd[out_flag + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        stereo = np.column_stack([stereo_left, stereo_right])
        sf.write(str(track_dir / "guitar.wav"), stereo.astype(np.float32), sr)
        return MagicMock(returncode=0, stderr="", stdout="")

    return fake_run


def test_separate_stems_lead_rhythm_when_confident(tmp_path: Path):
    """Lead/Rhythm runs only when lead_rhythm=True (CLI/eval opt-in)."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, endpoint=False)
    left = np.sin(2 * np.pi * 900 * t)
    right = 0.7 * np.sin(2 * np.pi * 180 * t)

    from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar(left, right)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", max_duration_sec=15, lead_rhythm=True),
        )

    assert "guitar" in artifacts
    assert "lead_guitar" in artifacts
    assert "rhythm_guitar" in artifacts
    assert "guitar_split_diagnostics" in artifacts
    assert artifacts["lead_guitar"].read_bytes() != artifacts["rhythm_guitar"].read_bytes()

    assert "stem_presence_diagnostics" in artifacts
    presence = json.loads(artifacts["stem_presence_diagnostics"].read_text(encoding="utf-8"))
    # The combined guitar stem is superseded by its own lead/rhythm split.
    assert presence["guitar"]["present"] is False
    assert "superseded" in presence["guitar"]["reason"]
    assert presence["lead_guitar"]["present"] is True
    assert presence["rhythm_guitar"]["present"] is True


def test_separate_stems_default_skips_lead_rhythm(tmp_path: Path):
    """Default isolate path never calls Lead/Rhythm post-process."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, endpoint=False)
    left = np.sin(2 * np.pi * 900 * t)
    right = 0.7 * np.sin(2 * np.pi * 180 * t)
    called = {"n": 0}

    def fake_split(*_a, **_kw):
        called["n"] += 1
        return {}, None

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar(left, right)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", max_duration_sec=15),
        )

    assert called["n"] == 0
    assert "guitar" in artifacts
    assert "lead_guitar" not in artifacts
    assert "guitar_split_diagnostics" not in artifacts


def test_separate_stems_lead_rhythm_skips_mono_by_default(tmp_path: Path):
    """Mono / low-confidence guitar does not emit Lead + Rhythm by default."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, endpoint=False)
    mono = np.sin(2 * np.pi * 440 * t)

    from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar(mono, mono)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", max_duration_sec=15, lead_rhythm=True),
        )

    assert "guitar" in artifacts
    assert "lead_guitar" not in artifacts
    assert "rhythm_guitar" not in artifacts
    assert "guitar_split_diagnostics" in artifacts
    diag = json.loads(artifacts["guitar_split_diagnostics"].read_text(encoding="utf-8"))
    assert diag["outcome"] == "ambiguous"
    presence = json.loads(artifacts["stem_presence_diagnostics"].read_text(encoding="utf-8"))
    assert presence["guitar"]["present"] is True
    assert "lead_guitar" not in presence or not presence.get("lead_guitar", {}).get("present", False)


def test_separate_stems_lead_rhythm_best_effort_emits_on_mono(tmp_path: Path):
    """best_effort mode still emits Lead + Rhythm on mono guitar."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, endpoint=False)
    mono = np.sin(2 * np.pi * 440 * t)

    from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar(mono, mono)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(
                model="htdemucs_6s",
                max_duration_sec=15,
                lead_rhythm=True,
                lead_rhythm_mode="best_effort",
            ),
        )

    assert "guitar" in artifacts
    assert "lead_guitar" in artifacts
    assert "rhythm_guitar" in artifacts
    diag = json.loads(artifacts["guitar_split_diagnostics"].read_text(encoding="utf-8"))
    assert diag["outcome"] == "lead_rhythm"
    presence = json.loads(artifacts["stem_presence_diagnostics"].read_text(encoding="utf-8"))
    assert presence["guitar"]["present"] is False
    assert presence["lead_guitar"]["present"] is True
    assert presence["rhythm_guitar"]["present"] is True


def test_isolate_config_dual_guitar_aliases_lead_rhythm():
    cfg = IsolateConfig(dual_guitar=True)
    assert cfg.lead_rhythm is True
    assert cfg.lead_rhythm_mode == "best_effort"


def test_detect_present_stems_flags_loud_and_silent(tmp_path: Path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    loud = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    near_silent = np.zeros_like(loud)

    loud_path = tmp_path / "loud.wav"
    silent_path = tmp_path / "silent.wav"
    sf.write(str(loud_path), loud, sr, subtype="PCM_16")
    sf.write(str(silent_path), near_silent, sr, subtype="PCM_16")

    result = detect_present_stems({"loud": loud_path, "silent": silent_path})

    assert result["loud"].present is True
    assert result["silent"].present is False
    assert 0.0 <= result["loud"].confidence <= 1.0
    assert 0.0 <= result["silent"].confidence <= 1.0
    assert result["loud"].mean_dbfs >= STEM_PRESENCE_FLOOR_DB
    assert result["silent"].reason and result["loud"].reason


def test_detect_present_stems_energy_share_can_flag_quiet_but_real_stem(tmp_path: Path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Loud enough in isolation to pass the floor, and its share of the total
    # RMS across the call also clears STEM_PRESENCE_ENERGY_SHARE_MIN.
    a = (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    b = (0.6 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)

    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    sf.write(str(a_path), a, sr, subtype="PCM_16")
    sf.write(str(b_path), b, sr, subtype="PCM_16")

    result = detect_present_stems({"a": a_path, "b": b_path})

    assert result["a"].present is True
    assert result["b"].present is True
    assert result["a"].energy_share >= STEM_PRESENCE_ENERGY_SHARE_MIN
    assert result["b"].energy_share >= STEM_PRESENCE_ENERGY_SHARE_MIN


def test_detect_present_stems_handles_unreadable_file_gracefully(tmp_path: Path):
    bogus = tmp_path / "bogus.wav"
    bogus.write_bytes(b"not-a-real-wav-file")

    result = detect_present_stems({"bogus": bogus})

    assert result["bogus"].present is False
    assert result["bogus"].mean_dbfs < STEM_PRESENCE_FLOOR_DB


def _clean_guitar_chord(sr: int = 44100, dur: float = 2.0) -> np.ndarray:
    """Guitar-like chord: mostly mid content, one modest low-E fundamental."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (
        0.25 * np.sin(2 * np.pi * 82.4 * t)  # open low E — legit low content
        + 0.5 * np.sin(2 * np.pi * 196.0 * t)  # G3
        + 0.6 * np.sin(2 * np.pi * 293.7 * t)  # D4
        + 0.5 * np.sin(2 * np.pi * 370.0 * t)  # F#4
    ).astype(np.float32)


def _distorted_bass_bleed(sr: int = 44100, dur: float = 2.0) -> np.ndarray:
    """Heavy sub-guitar-range content standing in for bled-in distorted bass."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (
        0.9 * np.sin(2 * np.pi * 41.2 * t)  # bass low E1
        + 0.7 * np.sin(2 * np.pi * 55.0 * t)  # A1
        + 0.4 * np.sin(2 * np.pi * 73.4 * t)  # distortion overtone
    ).astype(np.float32)


def test_analyze_bass_bleed_flags_bass_heavy_guitar(tmp_path: Path):
    sr = 44100
    bled = _clean_guitar_chord(sr) + _distorted_bass_bleed(sr)
    path = tmp_path / "guitar.wav"
    sf.write(str(path), np.column_stack([bled, bled]), sr, subtype="PCM_16")

    diag = analyze_bass_bleed(path)

    assert diag.attempted is True
    assert diag.flagged is True
    assert diag.low_band_energy_share is not None
    assert diag.low_band_energy_share >= BASS_BLEED_ENERGY_SHARE_FLOOR
    assert diag.low_band_cutoff_hz == BASS_BLEED_LOW_BAND_HZ
    assert "bleed" in diag.reason.lower()
    assert diag.mitigation_applied is False


def test_analyze_bass_bleed_clean_guitar_not_flagged(tmp_path: Path):
    sr = 44100
    clean = _clean_guitar_chord(sr)
    path = tmp_path / "guitar.wav"
    sf.write(str(path), np.column_stack([clean, clean]), sr, subtype="PCM_16")

    diag = analyze_bass_bleed(path)

    assert diag.attempted is True
    assert diag.flagged is False
    assert diag.low_band_energy_share is not None
    assert diag.low_band_energy_share < BASS_BLEED_ENERGY_SHARE_FLOOR


def test_analyze_bass_bleed_missing_stem(tmp_path: Path):
    diag = analyze_bass_bleed(tmp_path / "missing.wav")
    assert diag.attempted is False
    assert diag.flagged is False
    assert "missing" in diag.reason.lower()


def test_apply_bass_bleed_mitigation_reduces_bleed_without_damaging_clean_signal(tmp_path: Path):
    """
    Synthetic-fixture check backing step 3's decision to implement the HPF:
    the cutoff sits below a guitar's lowest standard-tuned fundamental (~82 Hz),
    so a clean guitar signal should pass through nearly unchanged while a
    bled-heavy signal's low-band share drops measurably (partial, not full).
    """
    sr = 44100
    clean = _clean_guitar_chord(sr)
    bled = clean + _distorted_bass_bleed(sr)

    clean_path = tmp_path / "clean.wav"
    bled_path = tmp_path / "bled.wav"
    sf.write(str(clean_path), np.column_stack([clean, clean]), sr, subtype="PCM_16")
    sf.write(str(bled_path), np.column_stack([bled, bled]), sr, subtype="PCM_16")

    clean_share_before = analyze_bass_bleed(clean_path).low_band_energy_share
    bled_share_before = analyze_bass_bleed(bled_path).low_band_energy_share

    clean_out = apply_bass_bleed_mitigation(clean_path, tmp_path / "clean_mitigated.wav")
    bled_out = apply_bass_bleed_mitigation(bled_path, tmp_path / "bled_mitigated.wav")

    clean_share_after = analyze_bass_bleed(clean_out).low_band_energy_share
    bled_share_after = analyze_bass_bleed(bled_out).low_band_energy_share

    # Clean guitar content is preserved (RMS nearly unchanged, low-band share
    # barely moves) — the cutoff is below its fundamentals.
    clean_data_before, sr_a = sf.read(str(clean_path))
    clean_data_after, sr_b = sf.read(str(clean_out))
    rms_before = float(np.sqrt(np.mean(np.square(clean_data_before))))
    rms_after = float(np.sqrt(np.mean(np.square(clean_data_after))))
    assert rms_after / rms_before > 0.95
    assert abs(clean_share_after - clean_share_before) < 0.05

    # Bled signal's low-band share measurably drops — a partial reduction,
    # not full removal (bleed harmonics above the cutoff remain).
    assert bled_share_after < bled_share_before
    assert bled_share_after > 0.0  # still some low content — not a full fix


def test_apply_bass_bleed_mitigation_attenuates_drop_c_fundamental(tmp_path: Path):
    """I-504: 60 Hz HPF cuts Drop C (~65 Hz) energy; Drop D is above the cutoff."""
    from audio_to_tab.isolate import _lowpass_rms, _to_mono

    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    drop_c = (0.9 * np.sin(2 * np.pi * 65.4 * t)).astype(np.float32)
    path = tmp_path / "drop_c.wav"
    sf.write(str(path), np.column_stack([drop_c, drop_c]), sr, subtype="PCM_16")
    out = apply_bass_bleed_mitigation(path, tmp_path / "drop_c_hpf.wav")

    before, sr_a = sf.read(str(path))
    after, sr_b = sf.read(str(out))
    assert sr_a == sr_b == sr
    low_before = _lowpass_rms(_to_mono(before), sr, 80.0)
    low_after = _lowpass_rms(_to_mono(after), sr, 80.0)
    assert low_after < low_before


def _fake_demucs_with_guitar_mono(mono: np.ndarray, sr: int = 44100):
    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        out_flag = cmd.index("-o")
        demucs_out = Path(cmd[out_flag + 1])
        track_dir = demucs_out / "htdemucs_6s" / "normalized"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocals", "drums", "bass", "piano"):
            (track_dir / f"{name}.wav").write_bytes(b"stem-" + name.encode())
        _write_silent_wav(track_dir / "other.wav")
        stereo = np.column_stack([mono, mono])
        sf.write(str(track_dir / "guitar.wav"), stereo.astype(np.float32), sr)
        return MagicMock(returncode=0, stderr="", stdout="")

    return fake_run


def test_separate_stems_writes_bass_bleed_diagnostics(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    bled = _clean_guitar_chord(sr) + _distorted_bass_bleed(sr)

    from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar_mono(bled, sr)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", max_duration_sec=15),
        )

    assert "bass_bleed_diagnostics" in artifacts
    diag_data = json.loads(artifacts["bass_bleed_diagnostics"].read_text(encoding="utf-8"))
    assert diag_data["attempted"] is True
    assert diag_data["flagged"] is True
    assert diag_data["mitigation_applied"] is False  # default off
    assert diag_data["reason"]


def test_separate_stems_bass_bleed_mitigation_is_opt_in(tmp_path: Path):
    """Default config: flagged but guitar audio is left untouched."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    bled = _clean_guitar_chord(sr) + _distorted_bass_bleed(sr)

    from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar_mono(bled, sr)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", max_duration_sec=15, bass_bleed_mitigation=False),
        )

    diag_data = json.loads(artifacts["bass_bleed_diagnostics"].read_text(encoding="utf-8"))
    assert diag_data["flagged"] is True
    assert diag_data["mitigation_applied"] is False

    # Guitar audio on disk is untouched — re-measuring it matches the recorded share.
    unmitigated_share = analyze_bass_bleed(artifacts["guitar"]).low_band_energy_share
    assert unmitigated_share == pytest.approx(diag_data["low_band_energy_share"], abs=1e-6)


def test_separate_stems_bass_bleed_mitigation_when_enabled(tmp_path: Path):
    """Opt-in mitigation: guitar stem is high-passed and diagnostics record it."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    clean = _clean_guitar_chord(sr)
    bled = clean + _distorted_bass_bleed(sr)

    from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar_mono(bled, sr)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", max_duration_sec=15, bass_bleed_mitigation=True),
        )

    diag_data = json.loads(artifacts["bass_bleed_diagnostics"].read_text(encoding="utf-8"))
    assert diag_data["flagged"] is True
    assert diag_data["mitigation_applied"] is True
    assert "mitigation applied" in diag_data["reason"].lower()

    mitigated_share = analyze_bass_bleed(artifacts["guitar"]).low_band_energy_share
    assert mitigated_share < diag_data["low_band_energy_share"]


def test_isolate_config_default_start_sec():
    assert IsolateConfig().start_sec == 0.0


def test_probe_duration_passes_no_window_kwargs(tmp_path: Path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"x")

    seen: list[dict] = []

    def fake_run(cmd, **kwargs):
        seen.append(kwargs)
        return MagicMock(returncode=0, stdout="120.5\n", stderr="")

    monkeypatch.setattr("audio_to_tab.isolate.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("audio_to_tab.isolate.subprocess.run", fake_run)

    assert probe_duration_sec(wav) == 120.5
    assert seen
    expected = subprocess_run_kwargs()
    if expected:
        assert seen[0].get("creationflags") == expected["creationflags"]
    else:
        assert "creationflags" not in seen[0]


def test_trim_audio_full_file_unchanged(tmp_path: Path):
    src = tmp_path / "norm.wav"
    src.write_bytes(b"wav")
    assert _trim_audio(src, None, start_sec=0.0) == src


def test_trim_audio_zero_or_negative_returns_full_file(tmp_path: Path):
    src = tmp_path / "norm.wav"
    src.write_bytes(b"wav")
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return MagicMock(returncode=0)

    with patch("audio_to_tab.isolate.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "audio_to_tab.isolate.subprocess.run", side_effect=fake_run
    ):
        assert _trim_audio(src, 0.0) == src
        assert _trim_audio(src, -5.0) == src
        assert _trim_audio(src, 0.0, start_sec=10.0) == src
        assert _trim_audio(src, -5.0, start_sec=10.0) == src
        assert _trim_audio(src, -5.0, start_sec=0.0) == src
    assert seen == []


def test_trim_audio_raises_on_ffmpeg_failure(tmp_path: Path):
    src = tmp_path / "norm.wav"
    src.write_bytes(b"wav")

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial")
        return MagicMock(returncode=2, stderr="boom", stdout="")

    with patch("audio_to_tab.isolate.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "audio_to_tab.isolate.subprocess.run", side_effect=fake_run
    ):
        with pytest.raises(RuntimeError, match="ffmpeg trim failed"):
            _trim_audio(src, 30.0)


def test_trim_audio_builds_ss_before_input(tmp_path: Path):
    src = tmp_path / "norm.wav"
    src.write_bytes(b"wav")
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        out = Path(cmd[-1])
        out.write_bytes(b"trimmed")
        return MagicMock(returncode=0)

    with patch("audio_to_tab.isolate.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "audio_to_tab.isolate.subprocess.run", side_effect=fake_run
    ):
        out = _trim_audio(src, 30.0, start_sec=10.0)

    assert out.exists()
    cmd = seen[0]
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    t_idx = cmd.index("-t")
    assert cmd[ss_idx + 1] == "10.0"
    assert ss_idx < i_idx < t_idx
    assert cmd[t_idx + 1] == "30.0"


def test_trim_audio_start_zero_with_length(tmp_path: Path):
    src = tmp_path / "norm.wav"
    src.write_bytes(b"wav")

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"trimmed")
        return MagicMock(returncode=0)

    with patch("audio_to_tab.isolate.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "audio_to_tab.isolate.subprocess.run", side_effect=fake_run
    ):
        out = _trim_audio(src, 90.0, start_sec=0.0)

    assert out.name.endswith("_trim.wav")


def test_resolve_region_rejects_invalid_ranges():
    with pytest.raises(RegionError, match="less than"):
        resolve_region(120.0, 60.0, 30.0)
    with pytest.raises(RegionError, match="exceeds"):
        resolve_region(60.0, 0.0, 90.0)
    with pytest.raises(RegionError, match="at least"):
        resolve_region(120.0, 0.0, 3.0)


def test_resolve_region_clamps_to_cap_without_moving_start():
    start, length, note = resolve_region(120.0, 30.0, 90.0, cap_sec=45.0)
    assert start == 30.0
    assert length == 45.0
    assert note is not None


def test_format_region_label():
    assert format_region_label(32.0, 43.0) == "0:32–1:15"


def test_guitar_ft_fallback_to_stock_demucs_on_load_failure(tmp_path: Path):
    """Missing guitar-ft weights fall back to stock htdemucs_6s without failing the job."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    sr = 44100
    n = sr * 2
    t = np.linspace(0, 2, n, endpoint=False)
    left = np.sin(2 * np.pi * 900 * t)
    right = 0.7 * np.sin(2 * np.pi * 180 * t)

    def fail_guitar_ft(*_a, **_k):
        raise RuntimeError("mock missing guitar-ft weights")

    from audio_to_tab.lead_rhythm import LeadRhythmDiagnostics, split_lead_rhythm_guitar as real_split

    def fake_split(guitar, out, **_kw):
        return real_split(guitar, out, use_basic_pitch=False, **_kw)

    seen: list[str] = []

    def capture(_stage: str, message: str) -> None:
        seen.append(message)

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=_fake_normalize_factory(tmp_path)),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.isolate.run_demucs_guitar_ft_inprocess", side_effect=fail_guitar_ft),
        patch("audio_to_tab.separate.run_process", side_effect=_fake_demucs_with_guitar(left, right)),
        patch("audio_to_tab.lead_rhythm.split_lead_rhythm_guitar", side_effect=fake_split),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(
                model="htdemucs_6s",
                max_duration_sec=15,
                guitar_checkpoint="htdemucs_6s_guitar_ft",
            ),
            on_progress=capture,
        )

    assert "guitar" in artifacts
    assert any(msg.startswith("Trimming to ") for msg in seen)
    assert not any("mock missing" in msg for msg in seen)
    assert not any("guitar-ft unavailable" in msg for msg in seen)
    isolate = Path(__file__).resolve().parents[1] / "src" / "audio_to_tab" / "isolate.py"
    src = isolate.read_text(encoding="utf-8")
    assert "guitar-ft unavailable" not in src


def test_probe_duration_sec_ffmpeg_stderr_fallback(tmp_path: Path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")

    def fake_which(name: str):
        if name == "ffprobe":
            return None
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return None

    with patch("audio_to_tab.isolate.shutil.which", side_effect=fake_which), patch(
        "audio_to_tab.isolate.subprocess.run",
        return_value=MagicMock(
            returncode=1,
            stderr="Duration: 00:02:05.50, start: 0.000000, bitrate: 128 kb/s\n",
        ),
    ):
        dur = probe_duration_sec(audio)
    assert dur == pytest.approx(125.5, abs=0.01)


def _tone_wav(path: Path, freq: float, sr: int = 44100, seconds: float = 0.5, amp: float = 0.2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), np.column_stack([y, y]), sr)
    return path


def test_effective_isolation_quality_floors_fast_for_six_stem_except_faster():
    assert effective_isolation_quality("fast", model="htdemucs_6s") == "balanced"
    assert effective_isolation_quality("fast", model="htdemucs_6s", speed_id="balanced") == "balanced"
    assert effective_isolation_quality("fast", model="htdemucs_6s", speed_id="faster") == "fast"
    assert effective_isolation_quality("high", model="htdemucs_6s") == "high"
    assert effective_isolation_quality("fast", model="htdemucs") == "fast"
    assert IsolateConfig().quality == "fast"


def test_si_sdr_identical_signals_is_high():
    t = np.linspace(0, 1, 2048, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    assert si_sdr(y, y) > 40.0


def test_analyze_guitar_stem_quality_high_end_and_low_band(tmp_path: Path):
    bright = _tone_wav(tmp_path / "bright.wav", 6000.0)
    dark = _tone_wav(tmp_path / "dark.wav", 120.0)
    bright_diag = analyze_guitar_stem_quality(bright)
    dark_diag = analyze_guitar_stem_quality(dark)
    assert bright_diag.attempted is True
    assert bright_diag.high_end_cutoff_hz == GUITAR_HIGH_END_HZ
    assert bright_diag.high_end_energy_share is not None
    assert dark_diag.high_end_energy_share is not None
    assert bright_diag.high_end_energy_share > dark_diag.high_end_energy_share
    assert dark_diag.low_band_energy_share is not None
    assert dark_diag.low_band_energy_share > bright_diag.low_band_energy_share


def test_analyze_guitar_stem_quality_preservation_and_overlap(tmp_path: Path):
    ref = _tone_wav(tmp_path / "ref.wav", 6000.0, amp=0.3)
    dull = _tone_wav(tmp_path / "dull.wav", 200.0, amp=0.3)
    piano = _tone_wav(tmp_path / "piano.wav", 6000.0, amp=0.3)
    bass = _tone_wav(tmp_path / "bass.wav", 80.0, amp=0.3)
    vs_self = analyze_guitar_stem_quality(ref, reference_path=ref)
    vs_dull = analyze_guitar_stem_quality(dull, reference_path=ref)
    assert vs_self.high_end_preservation is not None
    assert vs_dull.high_end_preservation is not None
    assert vs_self.high_end_preservation > vs_dull.high_end_preservation
    assert vs_self.si_sdr is not None and vs_self.si_sdr > 20.0
    overlap = analyze_guitar_stem_quality(
        ref,
        competing_stems={"piano": piano, "bass": bass},
    )
    assert overlap.competitor_overlap is not None
    assert overlap.competitor_overlap["piano"] > overlap.competitor_overlap["bass"]


def test_merge_two_pass_keeps_pass1_rhythm_and_pass2_guitar(tmp_path: Path):
    pass1 = tmp_path / "p1"
    pass2 = tmp_path / "p2"
    dest = tmp_path / "merged"
    _tone_wav(pass1 / "vocals.wav", 800.0, amp=0.15)
    _tone_wav(pass1 / "drums.wav", 200.0, amp=0.15)
    _tone_wav(pass1 / "bass.wav", 60.0, amp=0.2)
    _tone_wav(pass1 / "other.wav", 400.0, amp=0.1)
    _tone_wav(pass2 / "guitar.wav", 900.0, amp=0.25)
    _tone_wav(pass2 / "piano.wav", 1200.0, amp=0.1)
    _tone_wav(pass2 / "other.wav", 300.0, amp=0.05)
    leftover = _tone_wav(pass2 / "vocals.wav", 1000.0, amp=0.05)
    merged = merge_two_pass_stems(
        {"vocals": pass1 / "vocals.wav", "drums": pass1 / "drums.wav", "bass": pass1 / "bass.wav", "other": pass1 / "other.wav"},
        {
            "guitar": pass2 / "guitar.wav",
            "piano": pass2 / "piano.wav",
            "other": pass2 / "other.wav",
            "vocals": leftover,
        },
        dest,
    )
    assert set(merged) >= {"vocals", "drums", "bass", "guitar", "piano", "other"}
    v1, _ = sf.read(str(pass1 / "vocals.wav"))
    v_out, _ = sf.read(str(merged["vocals"]))
    assert np.allclose(v1, v_out)
    g2, _ = sf.read(str(pass2 / "guitar.wav"))
    g_out, _ = sf.read(str(merged["guitar"]))
    assert np.allclose(g2, g_out)
    assert merged["vocals"] != leftover


def test_two_pass_ignored_with_karaoke_split():
    cfg = IsolateConfig(model="htdemucs", two_stems="vocals", two_pass=True)
    assert cfg.two_pass is False


def test_two_pass_disabled_on_four_stem_model():
    cfg = IsolateConfig(model="htdemucs", two_pass=True)
    assert cfg.two_pass is False


def test_separate_stems_two_pass_runs_htdemucs_then_six_stem(tmp_path: Path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-wav")
    out_dir = tmp_path / "stems"
    seen: list[list[str]] = []

    def fake_normalize(src, dest=None):
        dest = Path(dest) if dest else tmp_path / "norm.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"norm")
        return dest

    def fake_run(cmd, capture_output=True, text=True, **_kwargs):
        seen.append(list(cmd))
        model = cmd[cmd.index("-n") + 1]
        src = Path(cmd[-1])
        demucs_out = Path(cmd[cmd.index("-o") + 1])
        track_dir = demucs_out / model / src.stem
        track_dir.mkdir(parents=True, exist_ok=True)
        if model == "htdemucs":
            for name, freq in (("vocals", 700.0), ("drums", 180.0), ("bass", 70.0), ("other", 400.0)):
                _tone_wav(track_dir / f"{name}.wav", freq)
        else:
            for name, freq in (
                ("vocals", 710.0),
                ("drums", 190.0),
                ("bass", 75.0),
                ("guitar", 900.0),
                ("piano", 1100.0),
                ("other", 300.0),
            ):
                _tone_wav(track_dir / f"{name}.wav", freq)
        return MagicMock(returncode=0, stderr="", stdout="")

    with (
        patch("audio_to_tab.isolate.is_demucs_available", return_value=True),
        patch("audio_to_tab.isolate.normalize_audio", side_effect=fake_normalize),
        patch("audio_to_tab.isolate._trim_audio", side_effect=lambda p, *a, **k: p),
        patch("audio_to_tab.separate.run_process", side_effect=fake_run),
    ):
        artifacts = separate_stems(
            audio,
            out_dir,
            IsolateConfig(model="htdemucs_6s", quality="fast", two_pass=True, max_duration_sec=15),
        )

    models = [cmd[cmd.index("-n") + 1] for cmd in seen]
    assert models == ["htdemucs", "htdemucs_6s"]
    assert {"vocals", "drums", "bass", "guitar", "piano"} <= set(artifacts)
    assert "guitar_stem_quality_diagnostics" in artifacts
    quality = json.loads(Path(artifacts["guitar_stem_quality_diagnostics"]).read_text(encoding="utf-8"))
    assert quality["attempted"] is True
    assert "high_end_energy_share" in quality
