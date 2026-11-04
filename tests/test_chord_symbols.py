"""One grammar: parse, render, levels, capo, and chord tones."""

from __future__ import annotations

from audio_to_tab.chord_symbols import (
    DETECTED_QUALITIES,
    PITCHES,
    Chord,
    capo_shape,
    chord_tones,
    level_symbol,
    parse,
    parse_render,
    render,
    transpose,
)
from audio_to_tab.chords import _TEMPLATE_NAMES


def test_round_trip_every_template():
    assert len(_TEMPLATE_NAMES) == 12 * len(DETECTED_QUALITIES)
    for name in _TEMPLATE_NAMES:
        chord = parse(name)
        assert chord is not None
        assert chord.quality in DETECTED_QUALITIES
        assert render(chord) == name


def test_flats_in_sharps_out_and_slash_bass():
    assert render(parse("Bb")) == "A#"
    assert render(parse("Bbm7")) == "A#m7"
    assert render(parse("Eb")) == "D#"
    chord = parse("C/E")
    assert chord is not None
    assert chord.bass == "E"
    assert render(chord) == "C/E"


def test_garbage_is_none():
    assert parse("H") is None
    assert parse("N") is None
    assert parse("") is None
    assert parse(None) is None
    assert parse("not-a-chord") is None
    assert parse("C/E/G") is None
    assert parse_render("Cadd9") is None
    assert parse("Cadd9") is not None


def test_every_level_rule():
    cmaj7 = parse("Cmaj7")
    assert level_symbol(cmaj7, "simple") == "C"
    assert level_symbol(cmaj7, "medium") == "Cmaj7"
    assert level_symbol(cmaj7, "advanced") == "Cmaj7"
    assert level_symbol(parse("CmMaj7"), "medium") == "CmMaj7"
    assert level_symbol(parse("CmMaj7"), "advanced") == "CmMaj7"
    dm7 = parse("Dm7")
    assert level_symbol(dm7, "simple") == "Dm"
    assert level_symbol(dm7, "medium") == "Dm7"
    g7 = parse("G7/F")
    assert level_symbol(g7, "simple") == "G/F"
    assert level_symbol(g7, "medium") == "G7/F"
    assert level_symbol(parse("Asus4"), "simple") == "Asus4"
    assert level_symbol(parse("Csus2"), "medium") == "Csus2"
    assert level_symbol(parse("C5"), "advanced") == "C5"
    assert level_symbol(parse("C5"), "simple") == "C5"
    maj9 = Chord("C", "maj9")
    assert level_symbol(maj9, "simple") == "C"
    assert level_symbol(maj9, "medium") == "C"
    assert level_symbol(maj9, "advanced") == "Cmaj9"
    six = parse("C6")
    assert level_symbol(six, "simple") == "C"
    assert level_symbol(six, "medium") == "C"
    assert level_symbol(six, "advanced") == "C6"
    altered = Chord("G", "7b9", "F")
    assert level_symbol(altered, "simple") == "G/F"
    assert level_symbol(altered, "medium") == "G7/F"
    assert level_symbol(altered, "advanced") == "G7b9/F"
    half = parse("F#m7b5")
    assert level_symbol(half, "simple") == "F#dim"
    assert level_symbol(half, "medium") == "F#m7b5"
    dim7 = parse("Cdim7")
    assert level_symbol(dim7, "simple") == "Cdim"
    assert level_symbol(dim7, "medium") == "Cdim7"


def test_capo_matches_frontend_shapes():
    assert render(capo_shape(parse("G"), 2)) == "F"
    assert render(capo_shape(parse("Am"), 2)) == "Gm"
    assert render(capo_shape(parse("G/B"), 2)) == "F/A"


def test_chord_tones_and_transpose():
    assert chord_tones(parse("Cmaj7")) == ("C", "E", "G", "B")
    assert chord_tones(parse("Dm7")) == ("D", "F", "A", "C")
    assert chord_tones(parse("C/E")) == ("C", "E", "G")
    assert render(transpose(parse("C"), 2)) == "D"
    assert render(transpose(parse("C/E"), 1)) == "C#/F"
    assert set(PITCHES) == {render(transpose(parse("C"), step)) for step in range(12)}


def test_auto_spelling_follows_the_key():
    assert render(parse("A#"), spelling="auto", key="F") == "Bb"
    assert render(parse("A#"), spelling="auto", key="Dm") == "Bb"
    assert render(parse("A#"), spelling="auto", key="D") == "A#"
    assert render(parse("A#"), spelling="sharp") == "A#"
