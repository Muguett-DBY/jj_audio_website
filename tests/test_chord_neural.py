"""Opt-in BTC chord scorer — no network in the default suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_to_tab.chord_neural import (
    BTC_WEIGHT_NAME,
    BTC_WEIGHT_SHA256,
    _mir_eval_to_symbol,
    chord_model_available,
    chord_model_requested,
    detect_chords_btc,
)


def test_mir_eval_label_maps_to_product_symbols():
    assert _mir_eval_to_symbol("C") == "C"
    assert _mir_eval_to_symbol("C:min") == "Cm"
    assert _mir_eval_to_symbol("G#:min") == "G#m"
    assert _mir_eval_to_symbol("N") == "N"


def test_chord_model_unavailable_without_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("ATT_CHORD_MODEL", raising=False)
    monkeypatch.delenv("ATT_CHORD_MODEL_DIR", raising=False)
    assert chord_model_requested() is False
    assert chord_model_available(tmp_path) is False
    assert detect_chords_btc(tmp_path / "missing.wav", model_dir=tmp_path) is None


def test_chord_model_requested_env(monkeypatch):
    monkeypatch.setenv("ATT_CHORD_MODEL", "btc")
    assert chord_model_requested() is True
    monkeypatch.setenv("ATT_CHORD_MODEL", "0")
    assert chord_model_requested() is False


@pytest.mark.slow
def test_btc_majmin_on_synthetic_powerchord(tmp_path, monkeypatch):
    """Requires torch + a pinned weight download (opt-in / slow)."""
    pytest.importorskip("torch")
    from audio_to_tab.chord_neural import ensure_chord_model

    model_dir = tmp_path / "btc"
    ensure_chord_model(model_dir)
    assert chord_model_available(model_dir)
    assert (model_dir / BTC_WEIGHT_NAME).stat().st_size > 0

    sr = 22050
    t = np.arange(int(sr * 4.0), dtype=np.float64) / sr
    root = 103.83  # G#2
    y = 0.3 * np.sin(2 * np.pi * root * t) + 0.22 * np.sin(2 * np.pi * root * 1.5 * t)
    wav = tmp_path / "gs5.wav"
    sf.write(wav, y.astype(np.float32), sr)

    events = detect_chords_btc(wav, model_dir=model_dir)
    assert events is not None
    assert events
    roots = {event.symbol.split("m")[0].rstrip("#").rstrip("b") or event.symbol for event in events}
    # Soft check: at least one event mentions G# (maj or min).
    assert any("G#" in event.symbol for event in events), events
    _ = roots, BTC_WEIGHT_SHA256
