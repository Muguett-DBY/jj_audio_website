"""Chord detection for the practice player.

Beat-synchronous CQT chroma, weighted templates, and Viterbi smoothing. Scores are
z-normalised per window so quality and key priors are in standard deviations.
``model_dir`` opts into the BTC maj/min scorer when weights are present
(``audio_to_tab.chord_neural``); otherwise the template baseline runs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from statistics import median

import numpy as np

from audio_to_tab.analysis import ChordEvent
from audio_to_tab.chord_symbols import (
    DETECTED_QUALITIES,
    PITCHES,
    level_symbol,
    parse,
    parse_render,
)

logger = logging.getLogger(__name__)

_normalize_symbol = parse_render

_QUALITY_TONES: dict[str, tuple[tuple[str, int], ...]] = {
    "maj": (("root", 0), ("third", 4), ("fifth", 7)),
    "min": (("root", 0), ("third", 3), ("fifth", 7)),
    "dim": (("root", 0), ("third", 3), ("fifth", 6)),
    "aug": (("root", 0), ("third", 4), ("fifth", 8)),
    # The suspended degree replaces the third, so it uses the third's weight.
    "sus2": (("root", 0), ("third", 2), ("fifth", 7)),
    "sus4": (("root", 0), ("third", 5), ("fifth", 7)),
    "5": (("root", 0), ("fifth", 7)),
    "6": (("root", 0), ("third", 4), ("fifth", 7), ("sixth", 9)),
    "m6": (("root", 0), ("third", 3), ("fifth", 7), ("sixth", 9)),
    "7": (("root", 0), ("third", 4), ("fifth", 7), ("seventh", 10)),
    "maj7": (("root", 0), ("third", 4), ("fifth", 7), ("seventh", 11)),
    "m7": (("root", 0), ("third", 3), ("fifth", 7), ("seventh", 10)),
    "m7b5": (("root", 0), ("third", 3), ("fifth", 6), ("seventh", 10)),
    "mMaj7": (("root", 0), ("third", 3), ("fifth", 7), ("seventh", 11)),
    "dim7": (("root", 0), ("third", 3), ("fifth", 6), ("seventh", 9)),
}
_TONE_WEIGHTS = {
    "root": 1.00,
    "third": 0.90,
    "fifth": 0.70,
    "seventh": 0.55,
    "sixth": 0.55,
    "ninth": 0.45,
    "eleventh": 0.35,
    "thirteenth": 0.30,
}
_QUALITY_SUFFIX = {
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
}
# Priors are subtracted from z-scored template scores (positive = less likely).
_QUALITY_PRIOR = {
    "maj": 0.0,
    "min": 0.0,
    "7": 0.0,
    "maj7": 0.55,
    "m7": 0.15,
    "6": 0.45,
    "m6": 0.45,
    "sus4": 0.50,
    "sus2": 0.50,
    "dim": 0.35,
    "aug": 0.35,
    "m7b5": 0.45,
    "dim7": 0.35,
    "mMaj7": 0.50,
    "5": 0.70,
}
_MAJOR_DIATONIC = {
    0: frozenset({"maj", "maj7"}),
    2: frozenset({"min", "m7"}),
    4: frozenset({"min", "m7"}),
    5: frozenset({"maj", "maj7"}),
    7: frozenset({"maj", "7"}),
    9: frozenset({"min", "m7"}),
    11: frozenset({"dim", "m7b5"}),
}
_MINOR_DIATONIC = {
    0: frozenset({"min", "m7"}),
    2: frozenset({"dim", "m7b5"}),
    3: frozenset({"maj", "maj7"}),
    5: frozenset({"min", "m7"}),
    7: frozenset({"min", "m7", "maj", "7"}),
    8: frozenset({"maj", "maj7"}),
    10: frozenset({"maj", "7"}),
}
_HOP = 512
_BINS_PER_OCTAVE = 36
_FALLBACK_WINDOW_SEC = 0.5
_MIN_EVENT_SEC = 0.10
_CHANGE_PENALTY = 0.8
_OFF_DOWNBEAT_PENALTY = 0.15
# A new chord on the slash-fixture downbeat leads by only ~0.11σ. The planned
# +0.10σ bonus (penalty 0.70) holds the old chord for a beat and drops clean
# root WCSR to 0.938. Penalty 0.10 on the downbeat lets that change through;
# off the downbeat the penalty stays 0.95.
_ON_DOWNBEAT_BONUS = 0.70
_SILENCE_NORM = 1e-4
_BASS_ROOT_BONUS = 0.0


def _build_templates() -> tuple[tuple[str, ...], np.ndarray, np.ndarray, tuple[int, ...]]:
    names: list[str] = []
    rows: list[np.ndarray] = []
    priors: list[float] = []
    roots: list[int] = []
    for quality in DETECTED_QUALITIES:
        tones = _QUALITY_TONES[quality]
        for index, pitch in enumerate(PITCHES):
            vector = np.zeros(12, dtype=np.float64)
            for role, interval in tones:
                vector[(index + interval) % 12] += _TONE_WEIGHTS[role]
            rows.append(vector / np.linalg.norm(vector))
            names.append(f"{pitch}{_QUALITY_SUFFIX[quality]}")
            priors.append(_QUALITY_PRIOR[quality])
            roots.append(index)
    return tuple(names), np.stack(rows), np.asarray(priors, dtype=np.float64), tuple(roots)


_TEMPLATE_NAMES, _TEMPLATE_TABLE, _PRIOR_VECTOR, _TEMPLATE_ROOTS = _build_templates()
_TEMPLATE_CHORDS = tuple(parse(name) for name in _TEMPLATE_NAMES)


def estimate_tuning_semitones(y: np.ndarray, sr: int) -> float:
    """Tuning offset in semitones (librosa ``estimate_tuning`` at 12 bins/octave)."""
    import librosa

    return float(librosa.estimate_tuning(y=np.asarray(y), sr=int(sr)))


def _snap_to_beat(time: float, beats: Sequence[float], tolerance: float) -> float:
    if not beats:
        return float(time)
    nearest = min(float(b) for b in beats)
    for beat in beats:
        if abs(float(beat) - time) < abs(nearest - time):
            nearest = float(beat)
    return nearest if abs(nearest - time) <= tolerance else float(time)


def _merge_events(events: list[ChordEvent], *, min_duration: float = _MIN_EVENT_SEC) -> list[ChordEvent]:
    merged: list[ChordEvent] = []
    for event in sorted(events, key=lambda item: (item.start, item.end)):
        if event.end - event.start < min_duration:
            continue
        if merged and merged[-1].symbol == event.symbol and abs(merged[-1].end - event.start) <= 1e-6:
            previous = merged[-1]
            merged[-1] = ChordEvent(
                start=previous.start,
                end=event.end,
                symbol=previous.symbol,
                confidence=min(previous.confidence, event.confidence),
                simple=previous.simple,
                medium=previous.medium,
            )
        else:
            merged.append(event)
    return merged


def _absorb_short_runs(
    events: list[ChordEvent],
    energies: Sequence[float],
    *,
    beat_sec: float,
) -> list[ChordEvent]:
    """Fold runs shorter than half a beat into the louder neighbour. No holes.

    A one-beat window, including same-root quality flicker, scored clean root
    0.896 on ``synth_bench.py`` and stays out.
    """
    if not events:
        return []
    limit = 0.5 * beat_sec if beat_sec > 0 else _MIN_EVENT_SEC
    runs: list[dict[str, float | str]] = []
    for index, event in enumerate(events):
        energy = float(energies[index]) if index < len(energies) else float(event.confidence)
        runs.append(
            {
                "start": float(event.start),
                "end": float(event.end),
                "symbol": event.symbol,
                "confidence": float(event.confidence),
                "simple": event.simple,
                "medium": event.medium,
                "energy": energy,
            }
        )
    guard = 0
    while guard < len(runs) + 8:
        guard += 1
        short_index = next(
            (index for index, run in enumerate(runs) if float(run["end"]) - float(run["start"]) < limit),
            None,
        )
        if short_index is None:
            break
        previous = runs[short_index - 1] if short_index > 0 else None
        following = runs[short_index + 1] if short_index + 1 < len(runs) else None
        if previous is None and following is None:
            break
        if previous is None:
            chosen = following
        elif following is None:
            chosen = previous
        else:
            chosen = previous if float(previous["energy"]) >= float(following["energy"]) else following
        assert chosen is not None
        if chosen is previous:
            previous["end"] = runs[short_index]["end"]
            del runs[short_index]
        else:
            following["start"] = runs[short_index]["start"]
            del runs[short_index]
    collapsed: list[dict[str, float | str]] = []
    for run in runs:
        if (
            collapsed
            and collapsed[-1]["symbol"] == run["symbol"]
            and abs(float(collapsed[-1]["end"]) - float(run["start"])) <= 1e-6
        ):
            collapsed[-1]["end"] = run["end"]
            collapsed[-1]["energy"] = max(float(collapsed[-1]["energy"]), float(run["energy"]))
            collapsed[-1]["confidence"] = min(float(collapsed[-1]["confidence"]), float(run["confidence"]))
        else:
            collapsed.append(run)
    absorbed: list[ChordEvent] = []
    for run in collapsed:
        absorbed.append(
            ChordEvent(
                start=float(run["start"]),
                end=float(run["end"]),
                symbol=str(run["symbol"]),
                confidence=float(run["confidence"]),
                simple=str(run["simple"]),
                medium=str(run["medium"]),
            )
        )
    return absorbed


def _viterbi(scores: np.ndarray, frame_penalty: np.ndarray | float = _CHANGE_PENALTY) -> list[int]:
    frames, states = scores.shape
    if frames == 0:
        return []
    if np.isscalar(frame_penalty):
        penalties = np.full(frames, float(frame_penalty), dtype=np.float64)
    else:
        penalties = np.asarray(frame_penalty, dtype=np.float64)
    dp = scores[0].copy()
    back = np.zeros((frames, states), dtype=np.int16)
    for index in range(1, frames):
        penalty = float(penalties[index])
        order = np.argsort(dp)[::-1]
        best = int(order[0])
        second = int(order[1]) if states > 1 else best
        best_value = float(dp[best])
        second_value = float(dp[second]) - penalty
        for state in range(states):
            stay = float(dp[state])
            other = second_value if state == best else best_value - penalty
            if stay >= other:
                back[index, state] = state
                dp[state] = scores[index, state] + stay
            else:
                back[index, state] = best if state != best else second
                dp[state] = scores[index, state] + other
    path = np.zeros(frames, dtype=np.int16)
    path[-1] = int(np.argmax(dp))
    for index in range(frames - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return [int(value) for value in path]


def _window_confidence(row: np.ndarray) -> float:
    """Top-1 minus top-2 softmax over the three best states, clamped to [0, 1]."""
    if row.size < 2:
        return 0.0
    count = min(3, int(row.size))
    chosen = row[np.argpartition(row, -count)[-count:]]
    shifted = chosen - float(np.max(chosen))
    weights = np.exp(np.clip(shifted, -40.0, 40.0))
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    ordered = np.sort(weights / total)
    return float(np.clip(ordered[-1] - ordered[-2], 0.0, 1.0))


def _notify(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        logger.debug("Chord progress callback failed", exc_info=True)


def _boundaries(beats: Sequence[float], duration: float) -> tuple[list[float], float]:
    valid = sorted(
        {
            float(beat)
            for beat in beats
            if isinstance(beat, (int, float)) and math.isfinite(float(beat)) and 0.0 < float(beat) < duration
        }
    )
    if len(valid) >= 2:
        period = median([b - a for a, b in pairwise(valid) if b > a])
        tolerance = min(0.12, 0.25 * period)
    else:
        tolerance = 0.08
    if not valid:
        valid = [
            round(index * _FALLBACK_WINDOW_SEC, 6)
            for index in range(1, int(duration / _FALLBACK_WINDOW_SEC) + 1)
            if index * _FALLBACK_WINDOW_SEC < duration
        ]
    boundaries = [0.0, *valid, duration]
    deduped: list[float] = []
    for boundary in sorted(set(boundaries)):
        if not deduped or boundary - deduped[-1] > 1e-6:
            deduped.append(boundary)
    return deduped, tolerance


def _key_prior(key: str | None) -> np.ndarray:
    """Subtracted from z-scores: diatonic bonus is negative, outside-key roots are positive."""
    prior = np.zeros(len(_TEMPLATE_NAMES), dtype=np.float64)
    if not key:
        return prior
    text = key.strip()
    minor = text.endswith("m") and not text.lower().endswith("maj")
    root_token = text[:-1] if minor else text
    parsed = parse(root_token)
    if parsed is None:
        return prior
    tonic = PITCHES.index(parsed.root)
    table = _MINOR_DIATONIC if minor else _MAJOR_DIATONIC
    scale = set(table)
    for index, chord in enumerate(_TEMPLATE_CHORDS):
        if chord is None:
            continue
        degree = (PITCHES.index(chord.root) - tonic) % 12
        if degree not in scale:
            prior[index] = 0.10
        elif chord.quality in table[degree]:
            prior[index] = -0.20
    return prior


def _quality_prior_for_window(chroma: np.ndarray) -> np.ndarray:
    """Constant quality prior, relaxed when a major seventh is actually sounding.

    A balanced Cmaj7 voicing leads the C triad by only ~0.37σ. A flat 0.55σ
    prior erases it. The same prior stays on when the seventh bin is below
    0.35× the third, which is the overtone case (measured ~0.18×).
    """
    prior = _PRIOR_VECTOR.copy()
    if chroma.size != 12 or float(np.max(chroma)) <= 0:
        return prior
    for index, chord in enumerate(_TEMPLATE_CHORDS):
        if chord is None:
            continue
        root = _TEMPLATE_ROOTS[index]
        if chord.quality in {"maj7", "mMaj7", "m7", "7"}:
            third_interval = 3 if chord.quality in {"mMaj7", "m7"} else 4
            seventh_interval = 10 if chord.quality in {"m7", "7"} else 11
            third = float(chroma[(root + third_interval) % 12])
            seventh = float(chroma[(root + seventh_interval) % 12])
            root_energy = float(chroma[root])
            # Two ways a seventh is real, measured on the synthetic set:
            # the color tone is ≥0.32 of the root (clean Cmaj7 is 0.38; a
            # realistic IV triad's phantom major seventh is 0.23), or the
            # triad is fully voiced and the seventh is ≥0.35 of the third
            # (a balanced maj7) rather than an overtone (measured 0.18).
            # Requiring both dropped realistic sevenths to 0.889
            # (jazz 0.500, quality_zoom 0.833) on synth_bench.py GT beats.
            strong_color = root_energy > 1e-8 and seventh >= 0.32 * root_energy
            voiced = (
                root_energy > 1e-8
                and third >= 0.45 * root_energy
                and third > 1e-8
                and seventh >= 0.35 * third
            )
            if strong_color or voiced:
                prior[index] = min(float(prior[index]), 0.05)
        elif chord.quality == "5":
            # A bare fifth leads the triad by ~0.79σ (measured C2+G2). The 0.70
            # prior plus the tonic triad's diatonic bonus still names it C.
            # Relax only when neither third is present, so a real triad stays.
            fifth = float(chroma[(root + 7) % 12])
            major_third = float(chroma[(root + 4) % 12])
            minor_third = float(chroma[(root + 3) % 12])
            root_energy = float(chroma[root])
            bare = (
                root_energy > 1e-8
                and fifth >= 0.35 * root_energy
                and major_third < 0.25 * root_energy
                and minor_third < 0.25 * root_energy
            )
            if bare:
                prior[index] = min(float(prior[index]), 0.05)
    return prior


def _bass_bonus(bass_vector: np.ndarray) -> np.ndarray:
    bonus = np.zeros(len(_TEMPLATE_NAMES), dtype=np.float64)
    if bass_vector.size != 12 or float(np.max(bass_vector)) <= 0:
        return bonus
    order = np.argsort(bass_vector)
    peak = int(order[-1])
    second = float(bass_vector[int(order[-2])]) if bass_vector.size > 1 else 0.0
    if float(bass_vector[peak]) < 1.25 * (second + 1e-9):
        return bonus
    for index, root in enumerate(_TEMPLATE_ROOTS):
        if root == peak:
            bonus[index] = _BASS_ROOT_BONUS
    return bonus


def _slash_bass(symbol: str, bass_vector: np.ndarray) -> str:
    chord = parse(symbol)
    if chord is None or bass_vector.size != 12:
        return symbol
    order = np.argsort(bass_vector)
    peak = int(order[-1])
    root_index = PITCHES.index(chord.root)
    if peak == root_index:
        return symbol
    if float(bass_vector[peak]) < 1.6 * (float(bass_vector[root_index]) + 1e-9):
        return symbol
    bass_name = PITCHES[peak]
    quality_tones = {
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
    }
    allowed = {(root_index + interval) % 12 for interval in quality_tones.get(chord.quality, ())}
    if peak not in allowed:
        return symbol
    return f"{symbol}/{bass_name}"


def _is_downbeat(
    start: float,
    *,
    downbeats: Sequence[float],
    beats: Sequence[float],
    meter: int,
    tolerance: float,
) -> bool:
    if downbeats:
        return any(abs(start - float(beat)) <= max(tolerance, 1e-3) for beat in downbeats)
    meter = max(1, int(meter))
    grid = [float(beat) for beat in beats if math.isfinite(float(beat)) and float(beat) >= 0.0]
    if not grid:
        return abs(start) <= tolerance
    nearest = min(range(len(grid)), key=lambda index: abs(grid[index] - start))
    if abs(grid[nearest] - start) > tolerance and abs(start) > tolerance:
        return False
    if abs(start) <= tolerance and grid[0] > tolerance:
        return True
    return nearest % meter == 0


def _beat_seconds(beats: Sequence[float], bpm: float | None) -> float:
    if bpm and bpm > 0:
        return 60.0 / float(bpm)
    spacing = [float(b) - float(a) for a, b in pairwise(beats) if float(b) > float(a)]
    if spacing:
        return float(median(spacing))
    return 0.5


def _decorate(symbol: str) -> tuple[str, str]:
    if symbol.upper() == "N":
        return "N", "N"
    chord = parse(symbol)
    if chord is None:
        return symbol, symbol
    return level_symbol(chord, "simple"), level_symbol(chord, "medium")


def detect_chords(
    audio_path: Path,
    *,
    beats: Sequence[float],
    bpm: float | None,
    model_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
    provisional_key: str | None = None,
    beats_per_measure: int = 4,
    downbeats: Sequence[float] | None = None,
    bass_path: Path | None = None,
) -> list[ChordEvent]:
    """Detect chords on ``audio_path``.

    A fully silent or unreadable file returns ``[]``. Silent regions inside a
    recording are explicit ``N`` events. ``provisional_key`` biases diatonic
    qualities; when omitted it is estimated from the chroma. ``bass_path``
    supplies the low chroma used only for slash bass. It is not added to the
    template chroma. Omit it and the slash is read from this file's own low end.
    """
    path = Path(audio_path)
    # Opt-in neural path (BTC maj/min). Template remains the default.
    try:
        from audio_to_tab.chord_neural import (
            chord_model_available,
            chord_model_requested,
            default_chord_model_dir,
            detect_chords_btc,
            ensure_chord_model,
        )

        neural_dir: Path | None = Path(model_dir) if model_dir is not None else None
        if neural_dir is None and chord_model_requested():
            neural_dir = default_chord_model_dir()
        if neural_dir is not None:
            if not chord_model_available(neural_dir) and chord_model_requested():
                try:
                    ensure_chord_model(neural_dir, progress=progress)
                except Exception:
                    logger.warning("Chord model download failed; using templates", exc_info=True)
            if chord_model_available(neural_dir):
                neural = detect_chords_btc(path, model_dir=neural_dir, progress=progress)
                if neural is not None:
                    return neural
            elif model_dir is not None and not Path(model_dir).is_dir():
                logger.warning("Chord model directory not found: %s", model_dir)
            else:
                logger.debug("Chord model unavailable at %s; using templates", neural_dir)
    except Exception:
        logger.debug("Chord neural import/setup failed; using templates", exc_info=True)

    if not path.is_file():
        logger.warning("Chord audio not found: %s", path)
        return []

    _notify(progress, "chords: loading audio")
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=22050, mono=True)
    except Exception as exc:
        logger.warning("Chord detection could not read %s: %s", path, exc)
        return []
    if y.size < int(sr * 0.2) or float(np.sqrt(np.mean(y**2))) < 1e-5:
        return []

    duration = float(y.size) / float(sr)
    try:
        tuning = estimate_tuning_semitones(y, int(sr))
        tuning_bins = tuning * (_BINS_PER_OCTAVE / 12.0)
        harmonic, _percussive = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=sr,
            hop_length=_HOP,
            tuning=tuning_bins,
            bins_per_octave=_BINS_PER_OCTAVE,
            norm=None,
        )
        bass_y = y
        if bass_path is not None and Path(bass_path).is_file():
            try:
                loaded, _bass_sr = librosa.load(str(bass_path), sr=sr, mono=True)
                if loaded.size >= int(sr * 0.2):
                    bass_y = loaded
            except Exception:
                logger.debug("Bass stem unreadable; slash uses the harmony low end", exc_info=True)
        bass_chroma = librosa.feature.chroma_cqt(
            y=bass_y,
            sr=sr,
            hop_length=_HOP,
            fmin=librosa.note_to_hz("C2"),
            n_octaves=3,
            tuning=tuning_bins,
            bins_per_octave=_BINS_PER_OCTAVE,
            norm=None,
        )
    except Exception as exc:
        logger.warning("Chord chroma failed for %s: %s", path, exc)
        return []
    if chroma.shape[1] < 2:
        return []
    if bass_chroma.shape[1] < chroma.shape[1]:
        bass_chroma = np.pad(
            bass_chroma,
            ((0, 0), (0, chroma.shape[1] - bass_chroma.shape[1])),
        )
    bass_chroma = bass_chroma[:, : chroma.shape[1]]

    boundaries, tolerance = _boundaries(beats, duration)
    frame_count = chroma.shape[1]
    frames = librosa.time_to_frames(np.asarray(boundaries), sr=sr, hop_length=_HOP)
    frames = np.clip(frames, 0, frame_count - 1)

    windows: list[tuple[float, float, np.ndarray, np.ndarray, float]] = []
    for index in range(len(boundaries) - 1):
        start_frame = int(frames[index])
        end_frame = int(frames[index + 1])
        if end_frame <= start_frame:
            continue
        vector = chroma[:, start_frame:end_frame].mean(axis=1)
        bass_vector = bass_chroma[:, start_frame:end_frame].mean(axis=1)
        start_sample = max(0, int(boundaries[index] * sr))
        end_sample = min(y.size, int(boundaries[index + 1] * sr))
        segment = y[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0
        windows.append((boundaries[index], boundaries[index + 1], vector, bass_vector, rms))
    if not windows:
        return []

    if provisional_key is None:
        try:
            from audio_to_tab.key import provisional_key as estimate_provisional_key

            chroma_mean = np.mean([vector for _, _, vector, _, _ in windows], axis=0)
            bass_mean = np.mean([vector for _, _, _, vector, _ in windows], axis=0)
            provisional_key, _score = estimate_provisional_key(chroma_mean, bass_mean)
        except Exception:
            logger.debug("Provisional key failed", exc_info=True)
            provisional_key = None

    _notify(progress, f"chords: analysing {len(windows)} windows")
    matrix = np.stack([vector for _, _, vector, _, _ in windows])
    bass_matrix = np.stack([vector for _, _, _, vector, _ in windows])
    norms = np.linalg.norm(matrix, axis=1)
    rmses = np.asarray([rms for _, _, _, _, rms in windows], dtype=np.float64)
    silent = (norms < _SILENCE_NORM) | (rmses < 1e-4)
    normalised = np.divide(matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] > 1e-9)
    scores = normalised @ _TEMPLATE_TABLE.T
    scores = (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)
    for index in range(len(windows)):
        if not silent[index]:
            scores[index] += _bass_bonus(bass_matrix[index])
    key_prior = _key_prior(provisional_key)
    for index in range(len(windows)):
        if silent[index]:
            continue
        scores[index] -= _quality_prior_for_window(normalised[index])
        scores[index] -= key_prior
    scores[silent] = 0.0

    meter = max(1, int(beats_per_measure or 4))
    downbeat_times = tuple(downbeats or ())
    penalties = np.full(len(windows), _CHANGE_PENALTY + _OFF_DOWNBEAT_PENALTY, dtype=np.float64)
    for index, (start, _end, _vector, _bass, _rms) in enumerate(windows):
        if _is_downbeat(
            start,
            downbeats=downbeat_times,
            beats=beats,
            meter=meter,
            tolerance=tolerance,
        ):
            penalties[index] = _CHANGE_PENALTY - _ON_DOWNBEAT_BONUS
    path_indices = _viterbi(scores, penalties)
    beat_sec = _beat_seconds(boundaries, bpm)

    events: list[ChordEvent] = []
    energies: list[float] = []
    for index, state in enumerate(path_indices):
        start, end, _vector, bass_vector, rms = windows[index]
        start = _snap_to_beat(start, boundaries[1:-1], tolerance)
        end = _snap_to_beat(end, boundaries[1:-1], tolerance)
        start = max(0.0, min(duration, start))
        end = max(start, min(duration, end))
        if silent[index]:
            symbol = "N"
            confidence = 1.0
        else:
            symbol = _TEMPLATE_NAMES[state]
            symbol = _slash_bass(symbol, bass_vector)
            confidence = _window_confidence(scores[index])
        simple, medium = _decorate(symbol)
        events.append(
            ChordEvent(
                start=start,
                end=end,
                symbol=symbol,
                confidence=confidence,
                simple=simple,
                medium=medium,
            )
        )
        energies.append(rms)

    absorbed = _absorb_short_runs(events, energies, beat_sec=beat_sec)
    merged = _merge_events(absorbed)
    _notify(progress, f"chords: {len(merged)} segments")
    return merged


__all__ = ["detect_chords", "estimate_tuning_semitones"]
