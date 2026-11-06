"""Regression tests for the Demucs WAV-writer workaround.

torchaudio >= 2.9 saves through torchcodec, whose wheels support FFmpeg 4-8.
With FFmpeg 9, Demucs finished separating and then crashed while writing the
stems. These tests pin the soundfile-backed writer and the subprocess bootstrap.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile as sf
import torch

from audio_to_tab.demucs_bootstrap import install_soundfile_writer, save_audio
from audio_to_tab.separate import run_demucs


def test_save_audio_writes_stereo_wav(tmp_path):
    wav = torch.zeros(2, 1000)
    wav[0] = 0.5
    wav[1] = -0.5
    out = tmp_path / "stem.wav"

    save_audio(wav, out, samplerate=44100)

    data, sr = sf.read(out, always_2d=True)
    assert sr == 44100
    assert data.shape == (1000, 2)
    assert np.max(np.abs(data)) <= 1.0


def test_save_audio_rescales_hot_signal(tmp_path):
    wav = torch.full((1, 100), 4.0)
    out = tmp_path / "hot.wav"

    save_audio(wav, out, samplerate=8000)

    data, _ = sf.read(out)
    assert np.max(np.abs(data)) <= 1.0
    assert np.max(np.abs(data)) > 0.9


def test_install_patches_demucs_module_references(monkeypatch):
    import demucs.audio
    import demucs.separate

    monkeypatch.setattr(demucs.audio, "save_audio", demucs.audio.save_audio)
    monkeypatch.setattr(demucs.separate, "save_audio", demucs.separate.save_audio)

    install_soundfile_writer()

    assert demucs.audio.save_audio is save_audio
    assert demucs.separate.save_audio is save_audio


def test_run_demucs_subprocess_uses_soundfile_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_process(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("audio_to_tab.separate.run_process", fake_run_process)

    def should_abort():
        return False

    run_demucs(
        ["-n", "htdemucs_6s", "song.wav"],
        should_abort=should_abort,
        timeout_sec=12.5,
    )

    cmd = captured["cmd"]
    assert cmd[0].endswith("python") or "python" in str(cmd[0])
    assert cmd[1:4] == ["-m", "audio_to_tab.demucs_bootstrap", "-n"]
    assert cmd[4:] == ["htdemucs_6s", "song.wav"]
    assert captured["kwargs"]["should_abort"] is should_abort
    assert captured["kwargs"]["timeout_sec"] == 12.5


def test_run_demucs_frozen_installs_writer_before_main(tmp_path, monkeypatch):
    import demucs.audio
    import demucs.separate

    unpatched_writer = Mock(side_effect=AssertionError("original writer was called"))
    monkeypatch.setattr(demucs.audio, "save_audio", unpatched_writer)
    monkeypatch.setattr(demucs.separate, "save_audio", unpatched_writer)
    out = tmp_path / "stem.wav"
    calls = []

    def fake_main(opts=None):
        calls.append(opts)
        assert demucs.audio.save_audio is save_audio
        assert demucs.separate.save_audio is save_audio
        demucs.separate.save_audio(torch.zeros(2, 80), out, samplerate=8000)

    monkeypatch.setattr(demucs.separate, "main", fake_main)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    sub_run = Mock()
    monkeypatch.setattr("audio_to_tab.separate.run_process", sub_run)
    args = ["-n", "htdemucs_6s", "song.wav"]

    run_demucs(args)

    assert calls == [args]
    sub_run.assert_not_called()
    unpatched_writer.assert_not_called()
    data, sr = sf.read(out, always_2d=True)
    assert sr == 8000
    assert data.shape == (80, 2)


def test_bootstrap_module_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "audio_to_tab.demucs_bootstrap", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
