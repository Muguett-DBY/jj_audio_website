"""Chord detection: symbol normalisation, boundary handling, and the P0 baseline."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audio_to_tab.analysis import ChordEvent
from audio_to_tab.chords import (
    _QUALITY_PRIOR,
    _TEMPLATE_NAMES,
    _absorb_short_runs,
    _merge_events,
    _normalize_symbol,
    _quality_prior_for_window,
    _snap_to_beat,
    detect_chords,
    estimate_tuning_semitones,
)

SR = 22050
CHORDS = {
    "C": (261.63, 329.63, 392.00),
    "F": (349.23, 440.00, 523.25),
    "G": (392.00, 493.88, 587.33),
}


def _write_progression(path, symbols=("C", "F", "G", "C"), *, chord_sec: float = 2.0) -> None:
    t = np.arange(int(SR * chord_sec * len(symbols)), dtype=np.float64) / SR
    y = np.zeros_like(t)
    for index, symbol in enumerate(symbols):
        start = int(index * chord_sec * SR)
        end = int((index + 1) * chord_sec * SR)
        local = t[start:end] - t[start]
        for frequency in CHORDS[symbol]:
            y[start:end] += 0.12 * np.sin(2 * np.pi * frequency * local)
            y[start:end] += 0.04 * np.sin(2 * np.pi * frequency * 2 * local)
    sf.write(path, y, SR)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C", "C"),
        ("Cm", "Cm"),
        ("Cmin", "Cm"),
        ("C7", "C7"),
        ("Cmaj7", "Cmaj7"),
        ("CM7", "Cmaj7"),
        ("Cm7", "Cm7"),
        ("Cmin7", "Cm7"),
        ("Csus4", "Csus4"),
        ("Csus", "Csus4"),
        ("Cdim", "Cdim"),
        ("C°", "Cdim"),
        ("Caug", "Caug"),
        ("C+", "Caug"),
        ("Bb", "A#"),
        ("C/G", "C/G"),
        ("C/E", "C/E"),
        ("H", None),
        ("Cadd9", None),
        ("C6", "C6"),
        ("N", None),
        ("", None),
        (None, None),
    ],
)
def test_symbol_normalisation(raw, expected):
    assert _normalize_symbol(raw) == expected


def test_snap_to_beat_within_tolerance():
    assert _snap_to_beat(1.04, [0.0, 1.0, 2.0], tolerance=0.06) == 1.0
    assert _snap_to_beat(1.10, [0.0, 1.0, 2.0], tolerance=0.06) == 1.10
    assert _snap_to_beat(0.4, [], tolerance=0.06) == 0.4


def test_merge_events_merges_and_drops_short():
    events = [
        ChordEvent(0.0, 1.0, "C", 0.0),
        ChordEvent(1.0, 2.0, "C", 0.0),
        ChordEvent(2.0, 2.05, "G", 0.0),
        ChordEvent(2.5, 3.5, "F", 0.0),
    ]

    merged = _merge_events(events, min_duration=0.10)

    assert [(e.start, e.end, e.symbol) for e in merged] == [(0.0, 2.0, "C"), (2.5, 3.5, "F")]


def test_detect_chords_on_synthetic_progression(tmp_path):
    path = tmp_path / "progression.wav"
    _write_progression(path)
    beats = tuple(i * 0.5 for i in range(17))

    events = detect_chords(path, beats=beats, bpm=120.0)

    assert events
    symbols = [event.symbol for event in events]
    assert symbols[0] == "C"
    assert "F" in symbols
    assert "G" in symbols
    assert symbols[-1] == "C"
    assert all(0.0 <= event.confidence <= 1.0 for event in events)
    assert any(event.confidence > 0.0 for event in events)
    assert [e.start for e in events] == sorted(e.start for e in events)
    assert all(0.0 <= e.start < e.end <= 8.05 for e in events)
    assert all(e.end - e.start >= 0.10 - 1e-9 for e in events)


def test_detect_chords_without_beats_uses_fixed_grid(tmp_path):
    path = tmp_path / "progression.wav"
    _write_progression(path)

    events = detect_chords(path, beats=(), bpm=None)

    assert events
    assert all(0.0 <= e.confidence <= 1.0 for e in events)
    assert any(e.confidence > 0.0 for e in events)
    assert [e.start for e in events] == sorted(e.start for e in events)


def test_silence_returns_empty(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SR * 2), SR)

    assert detect_chords(path, beats=(0.0, 0.5, 1.0), bpm=120.0) == []


def test_missing_audio_returns_empty(tmp_path):
    assert detect_chords(tmp_path / "nope.wav", beats=(0.0, 0.5), bpm=120.0) == []


def test_missing_model_dir_falls_back_to_templates(tmp_path, caplog):
    path = tmp_path / "progression.wav"
    _write_progression(path)

    with caplog.at_level("WARNING"):
        events = detect_chords(
            path,
            beats=(0.0, 0.5, 1.0, 1.5, 2.0),
            bpm=120.0,
            model_dir=tmp_path / "missing-model",
        )

    assert events  # template baseline still runs
    assert "chord model" in caplog.text.lower()


def test_progress_callback_is_called(tmp_path):
    path = tmp_path / "progression.wav"
    _write_progression(path)
    messages: list[str] = []

    detect_chords(path, beats=(0.0, 0.5, 1.0), bpm=120.0, progress=messages.append)

    assert messages
    assert any("chord" in message.lower() for message in messages)


def _tone(duration: float, *partials: tuple[float, float], sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    y = np.zeros_like(t)
    for frequency, amplitude in partials:
        y += amplitude * np.sin(2 * np.pi * frequency * t)
    return y


def test_tuning_estimated_on_detuned_render():
    cents = 30.0
    ratio = 2 ** (cents / 1200.0)
    y = _tone(
        2.0,
        (261.63 * ratio, 0.2),
        (329.63 * ratio, 0.16),
        (392.00 * ratio, 0.12),
        (523.25 * ratio, 0.05),
    )
    estimate = estimate_tuning_semitones(y, SR)
    assert abs(estimate - 0.30) <= 0.15


def test_maj7_prior_beats_overtone_template(tmp_path):
    # C major triad. B is only the 3rd harmonic of E, quieter than the chord tones.
    y = _tone(
        2.0,
        (261.63, 0.20),
        (329.63, 0.16),
        (392.00, 0.14),
        (329.63 * 3, 0.04),
    )
    path = tmp_path / "overtone.wav"
    sf.write(path, y, SR)
    events = detect_chords(path, beats=tuple(i * 0.5 for i in range(5)), bpm=120.0)
    assert events
    assert all(event.symbol == "C" for event in events)


def test_sounded_major_seventh_stays(tmp_path):
    y = _tone(
        2.0,
        (261.63, 0.18),
        (329.63, 0.16),
        (392.00, 0.14),
        (493.88, 0.15),
        (587.33, 0.03),
    )
    path = tmp_path / "maj7.wav"
    sf.write(path, y, SR)
    events = detect_chords(path, beats=tuple(i * 0.5 for i in range(5)), bpm=120.0, provisional_key="C")
    assert events
    assert events[0].symbol.startswith("Cmaj7")


def test_voiced_dominant_seventh_stays(tmp_path):
    y = _tone(
        2.0,
        (261.63, 0.18),
        (329.63, 0.16),
        (392.00, 0.14),
        (466.16, 0.15),
    )
    path = tmp_path / "c7.wav"
    sf.write(path, y, SR)
    events = detect_chords(path, beats=tuple(i * 0.5 for i in range(5)), bpm=120.0, provisional_key="C")
    assert events
    assert events[0].symbol.startswith("C7")


def test_voiced_minor_seventh_stays(tmp_path):
    y = _tone(
        2.0,
        (261.63, 0.18),
        (311.13, 0.16),
        (392.00, 0.14),
        (466.16, 0.15),
    )
    path = tmp_path / "cm7.wav"
    sf.write(path, y, SR)
    events = detect_chords(path, beats=tuple(i * 0.5 for i in range(5)), bpm=120.0, provisional_key="C")
    assert events
    assert events[0].symbol.startswith("Cm7")


def test_smeared_window_keeps_seventh_prior():
    chroma = np.ones(12, dtype=np.float64)
    chroma[0] = chroma[4] = chroma[7] = chroma[11] = 1.2
    prior = _quality_prior_for_window(chroma)
    maj7_index = _TEMPLATE_NAMES.index("Cmaj7")
    seven_index = _TEMPLATE_NAMES.index("C7")
    assert float(prior[maj7_index]) == pytest.approx(_QUALITY_PRIOR["maj7"])
    assert float(prior[seven_index]) == pytest.approx(_QUALITY_PRIOR["7"])
    assert float(prior[seven_index]) > 0.0
    assert float(prior[seven_index]) != pytest.approx(0.05)


def test_powerchord_without_a_third_stays_a_fifth(tmp_path):
    """Root + fifth, no third, must stay C5. A 0.70 prior was naming it C."""
    y = _tone(2.0, (130.81, 0.25), (196.00, 0.18))
    path = tmp_path / "c5.wav"
    sf.write(path, y, SR)
    events = detect_chords(
        path,
        beats=tuple(i * 0.5 for i in range(5)),
        bpm=120.0,
        provisional_key="C",
    )
    assert events
    assert all(event.symbol.split("/")[0] == "C5" for event in events)


def test_bass_non_chord_tone_does_not_rename_the_root(tmp_path):
    guitar = _tone(2.0, (261.63, 0.20), (329.63, 0.16), (392.00, 0.14))
    bass = _tone(2.0, (87.31, 0.45))
    guitar_path = tmp_path / "guitar.wav"
    bass_path = tmp_path / "bass.wav"
    sf.write(guitar_path, guitar, SR)
    sf.write(bass_path, bass, SR)
    events = detect_chords(
        guitar_path,
        beats=tuple(i * 0.5 for i in range(5)),
        bpm=120.0,
        provisional_key="C",
        bass_path=bass_path,
    )
    assert events
    assert all(event.symbol.split("/")[0].startswith("C") for event in events)


def test_bass_third_may_slash_but_does_not_become_the_root(tmp_path):
    guitar = _tone(2.0, (261.63, 0.20), (329.63, 0.16), (392.00, 0.14))
    bass = _tone(2.0, (82.41, 0.5))
    guitar_path = tmp_path / "guitar.wav"
    bass_path = tmp_path / "bass.wav"
    sf.write(guitar_path, guitar, SR)
    sf.write(bass_path, bass, SR)
    events = detect_chords(
        guitar_path,
        beats=tuple(i * 0.5 for i in range(5)),
        bpm=120.0,
        provisional_key="C",
        bass_path=bass_path,
    )
    assert events
    assert any(event.symbol == "C/E" for event in events)
    assert all(not event.symbol.startswith("E") for event in events)


def test_short_flicker_is_merged():
    events = [
        ChordEvent(0.0, 2.0, "C", 0.8, simple="C", medium="C"),
        ChordEvent(2.0, 2.1, "G", 0.4, simple="G", medium="G"),
        ChordEvent(2.1, 4.0, "C", 0.8, simple="C", medium="C"),
    ]
    absorbed = _absorb_short_runs(events, [0.5, 0.2, 0.5], beat_sec=0.5)
    assert [(event.start, event.end, event.symbol) for event in absorbed] == [(0.0, 4.0, "C")]


def test_same_root_quality_flicker_under_one_beat_stays():
    # One-beat same-root absorb scored clean root 0.896. Within-bar chops stay blocked.
    events = [
        ChordEvent(0.0, 2.0, "C", 0.8, simple="C", medium="C"),
        ChordEvent(2.0, 2.6, "Cmaj7", 0.4, simple="C", medium="C"),
        ChordEvent(2.6, 4.0, "C", 0.8, simple="C", medium="C"),
    ]
    absorbed = _absorb_short_runs(events, [0.5, 0.2, 0.5], beat_sec=1.0)
    assert [event.symbol for event in absorbed] == ["C", "Cmaj7", "C"]


def test_different_root_inside_the_beat_is_kept():
    events = [
        ChordEvent(0.0, 2.0, "C", 0.8, simple="C", medium="C"),
        ChordEvent(2.0, 2.6, "F", 0.4, simple="F", medium="F"),
        ChordEvent(2.6, 4.0, "C", 0.8, simple="C", medium="C"),
    ]
    absorbed = _absorb_short_runs(events, [0.5, 0.2, 0.5], beat_sec=1.0)
    assert [event.symbol for event in absorbed] == ["C", "F", "C"]


def test_confidence_is_not_constant(tmp_path):
    clear = _tone(2.0, (261.63, 0.2), (329.63, 0.16), (392.00, 0.14))
    messy = _tone(
        2.0,
        (261.63, 0.12),
        (277.18, 0.12),
        (311.13, 0.12),
        (369.99, 0.12),
        (415.30, 0.12),
    )
    y = np.concatenate([clear, messy, clear])
    path = tmp_path / "mixed.wav"
    sf.write(path, y, SR)
    events = detect_chords(path, beats=tuple(i * 0.5 for i in range(13)), bpm=120.0)
    confidences = [event.confidence for event in events]
    assert confidences
    assert all(0.0 <= value <= 1.0 for value in confidences)
    assert len(set(round(value, 3) for value in confidences)) > 1


def test_no_chords_in_silence(tmp_path):
    tone = _tone(2.0, (261.63, 0.2), (329.63, 0.16), (392.00, 0.14))
    y = np.concatenate([np.zeros(SR * 2), tone])
    path = tmp_path / "gap.wav"
    sf.write(path, y, SR)
    events = detect_chords(path, beats=tuple(i * 0.5 for i in range(9)), bpm=120.0)
    pitched = [
        event
        for event in events
        if event.symbol.upper() != "N" and event.start < 1.5 and event.end > 0.0
    ]
    assert pitched == []
    assert any(event.symbol.upper() == "N" and event.start < 1.0 for event in events)
