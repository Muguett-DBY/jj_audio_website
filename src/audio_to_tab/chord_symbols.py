"""One chord grammar for detection, key estimation, display levels, and export.

Roots are stored as sharps (``Bb`` parses as ``A#``). ``render`` is the only
spelling transform. Display levels never change the root or the bass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
_SHARP_TO_FLAT = {sharp: flat for flat, sharp in _FLAT_TO_SHARP.items()}
_ROOT_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")

# Qualities the template detector may emit. Extensions below are parseable for
# display (``Cmaj9``) but are not detector states, so ``parse_render`` rejects them.
DETECTED_QUALITIES = (
    "maj",
    "min",
    "dim",
    "aug",
    "sus2",
    "sus4",
    "5",
    "6",
    "m6",
    "7",
    "maj7",
    "m7",
    "m7b5",
    "mMaj7",
    "dim7",
)

_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "5": "5",
    "6": "6",
    "m6": "m6",
    "7": "7",
    "maj7": "maj7",
    "m7": "m7",
    "m7b5": "m7b5",
    "mMaj7": "mMaj7",
    "dim7": "dim7",
    "maj9": "maj9",
    "7b9": "7b9",
    "add9": "add9",
    "9": "9",
    "m9": "m9",
}

# Longest alias first. Matching is exact, not prefix, so ``m7`` does not swallow ``m7b5``.
_QUALITY_ALIASES: tuple[tuple[str, str], ...] = (
    ("m7b5", "m7b5"),
    ("min7b5", "m7b5"),
    ("mmaj7", "mMaj7"),
    ("minmaj7", "mMaj7"),
    ("dim7", "dim7"),
    ("maj9", "maj9"),
    ("major9", "maj9"),
    ("maj7", "maj7"),
    ("major7", "maj7"),
    ("min7", "m7"),
    ("minor7", "m7"),
    ("m7", "m7"),
    ("sus4", "sus4"),
    ("sus2", "sus2"),
    ("7b9", "7b9"),
    ("add9", "add9"),
    ("min6", "m6"),
    ("m6", "m6"),
    ("sus", "sus4"),
    ("major", "maj"),
    ("minor", "min"),
    ("maj", "maj"),
    ("min", "min"),
    ("dim", "dim"),
    ("aug", "aug"),
    ("6", "6"),
    ("9", "9"),
    ("7", "7"),
    ("5", "5"),
    ("m", "min"),
)

_TONE_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "mMaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "maj9": (0, 4, 7, 11, 14),
    "7b9": (0, 4, 7, 10, 13),
    "add9": (0, 4, 7, 14),
    "9": (0, 4, 7, 10, 14),
    "m9": (0, 3, 7, 10, 14),
}

# Simple drops extensions to the triad. Suspensions and power chords stay: reducing
# them would assert a third that is not sounding.
_SIMPLE = {
    "maj": "maj",
    "min": "min",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "5": "5",
    "6": "maj",
    "m6": "min",
    "7": "maj",
    "maj7": "maj",
    "m7": "min",
    "m7b5": "dim",
    "mMaj7": "min",
    "dim7": "dim",
    "maj9": "maj",
    "7b9": "maj",
    "add9": "maj",
    "9": "maj",
    "m9": "min",
}
# Medium keeps structural sevenths (7, m7, maj7, mMaj7). Simple still collapses
# them to triads. add9/maj9 are never detector states — they only appear via parse.
_MEDIUM = {
    **_SIMPLE,
    "7": "7",
    "maj7": "maj7",
    "m7": "m7",
    "m7b5": "m7b5",
    "dim7": "dim7",
    "mMaj7": "mMaj7",
    "maj9": "maj",
    "7b9": "7",
    "9": "7",
    "m9": "m7",
}

# Major keys that take flats, plus the same pitch classes when the stored name is sharp.
_FLAT_MAJOR_PCS = {5, 10, 3, 8, 1, 6}  # F, Bb, Eb, Ab, Db, Gb


@dataclass(frozen=True)
class Chord:
    root: str
    quality: str
    bass: str | None = None


def _pitch(letter: str, accidental: str) -> str | None:
    name = letter.upper() + accidental
    name = _FLAT_TO_SHARP.get(name, name)
    if name not in PITCHES:
        return None
    return name


def _split_root(token: str) -> tuple[str, str] | None:
    match = _ROOT_RE.match(token.strip())
    if not match:
        return None
    letter, accidental, rest = match.groups()
    root = _pitch(letter, accidental)
    if root is None:
        return None
    return root, rest


def _quality_from_rest(rest: str) -> str | None:
    text = rest.strip()
    if text == "":
        return "maj"
    if text == "M":
        return "maj"
    if text == "M7":
        return "maj7"
    if text in {"°", "o", "O"}:
        return "dim"
    if text == "+":
        return "aug"
    lowered = text.lower()
    for alias, quality in _QUALITY_ALIASES:
        if lowered == alias:
            return quality
    return None


def parse(symbol: object) -> Chord | None:
    """Parse a symbol into a chord, or ``None`` for ``N``, ``H``, and garbage."""
    if not isinstance(symbol, str):
        return None
    text = symbol.strip()
    if not text or text.upper() in {"N", "H"}:
        return None
    parts = text.split("/")
    if len(parts) > 2 or not parts[0].strip():
        return None
    head = _split_root(parts[0])
    if head is None:
        return None
    root, rest = head
    quality = _quality_from_rest(rest)
    if quality is None:
        return None
    bass: str | None = None
    if len(parts) == 2:
        bass_token = parts[1].strip()
        bass_head = _split_root(bass_token)
        if bass_head is None or bass_head[1].strip() != "":
            return None
        bass = bass_head[0]
    return Chord(root, quality, bass)


def _spell(name: str, spelling: str) -> str:
    if spelling == "flat":
        return _SHARP_TO_FLAT.get(name, name)
    return name


def key_uses_flats(key: str | None) -> bool:
    """Flat spelling for F, Bb, Eb, Ab, Db, Gb and their relative minors."""
    if not isinstance(key, str) or not key.strip():
        return False
    text = key.strip()
    minor = text.endswith("m") and not text.lower().endswith("maj")
    root_token = text[:-1] if minor else text
    parsed = parse(root_token)
    if parsed is None:
        return False
    pc = PITCHES.index(parsed.root)
    if minor:
        pc = (pc + 3) % 12
    return pc in _FLAT_MAJOR_PCS


def render(chord: Chord, *, spelling: str = "sharp", key: str | None = None) -> str:
    """Render a chord. ``spelling='auto'`` follows ``key`` (sharps unless the key is flat)."""
    chosen = spelling
    if spelling == "auto":
        chosen = "flat" if key_uses_flats(key) else "sharp"
    if chosen not in {"sharp", "flat"}:
        chosen = "sharp"
    suffix = _SUFFIX.get(chord.quality, chord.quality if chord.quality != "maj" else "")
    text = f"{_spell(chord.root, chosen)}{suffix}"
    if chord.bass:
        text = f"{text}/{_spell(chord.bass, chosen)}"
    return text


def parse_render(symbol: object) -> str | None:
    """Canonical detected-vocabulary symbol, or ``None``.

    This is the historical ``_normalize_symbol`` contract: extensions such as
    ``Cadd9`` stay ``None`` because the detector cannot emit them.
    """
    chord = parse(symbol)
    if chord is None or chord.quality not in DETECTED_QUALITIES:
        return None
    return render(chord)


def transpose(chord: Chord, semitones: int) -> Chord:
    steps = int(semitones) % 12
    root = PITCHES[(PITCHES.index(chord.root) + steps) % 12]
    bass = None
    if chord.bass:
        bass = PITCHES[(PITCHES.index(chord.bass) + steps) % 12]
    return Chord(root, chord.quality, bass)


def capo_shape(chord: Chord, fret: int) -> Chord:
    """Shape fingered above a capo: the sounding chord transposed down by ``fret``."""
    return transpose(chord, -int(fret))


def level_symbol(chord: Chord, level: str) -> str:
    """Display symbol. Root and bass are unchanged at every level."""
    if level in {"advanced", "chart"}:
        return render(chord)
    table = _MEDIUM if level == "medium" else _SIMPLE
    quality = table.get(chord.quality, "maj")
    return render(Chord(chord.root, quality, chord.bass))


def chord_tones(chord: Chord) -> tuple[str, ...]:
    """Every note in the chord, sharps, bass appended when it is not already a tone."""
    intervals = _TONE_INTERVALS.get(chord.quality, (0, 4, 7))
    root_index = PITCHES.index(chord.root)
    tones: list[str] = []
    for interval in intervals:
        name = PITCHES[(root_index + interval) % 12]
        if name not in tones:
            tones.append(name)
    if chord.bass and chord.bass not in tones:
        tones.append(chord.bass)
    return tuple(tones)


__all__ = [
    "DETECTED_QUALITIES",
    "PITCHES",
    "Chord",
    "capo_shape",
    "chord_tones",
    "key_uses_flats",
    "level_symbol",
    "parse",
    "parse_render",
    "render",
    "transpose",
]
