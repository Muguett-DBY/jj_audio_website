"""Diagnosis driver for the chord stage. READ-ONLY w.r.t. src/ and data/ — writes only here and /tmp.

Subcommands:
  parity  — prove the instrumented mirror produces identical events to detect_chords
  synth   — synthetic G#5 experiments (the user's example case)
  real    — mirror over real runs, aggregate failure modes, dump failing windows

    .venv311/bin/python bench/scratch-chord-diagnosis/diagnose.py parity|synth|real
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT_DIR = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="chord-diag-"))

import librosa  # noqa: E402
from audio_to_tab.analysis import ChordEvent  # noqa: E402
from audio_to_tab import chords as C  # noqa: E402
from audio_to_tab.chords import (  # noqa: E402
    _bass_bonus,
    _beat_seconds,
    _boundaries,
    _decorate,
    _is_downbeat,
    _key_prior,
    _merge_events,
    _quality_prior_for_window,
    _SILENCE_NORM,
    _slash_bass,
    _snap_to_beat,
    _TEMPLATE_CHORDS,
    _TEMPLATE_NAMES,
    _TEMPLATE_ROOTS,
    _TEMPLATE_TABLE,
    _viterbi,
    _window_confidence,
    detect_chords,
)
from audio_to_tab.chord_symbols import PITCHES, parse  # noqa: E402

MAJOR = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
MINOR = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}
EXTENDED = {"7", "maj7", "m7", "m7b5", "mMaj7", "dim7", "6", "m6"}


def quality_of(symbol: str) -> str:
    chord = parse(symbol.split("/", 1)[0])
    return chord.quality if chord else "N"


def root_of(symbol: str) -> str:
    chord = parse(symbol.split("/", 1)[0])
    return chord.root if chord else "N"


def net_of(symbol: str) -> str:
    chord = parse(symbol.split("/", 1)[0])
    if chord is None:
        return symbol
    suffix = C._QUALITY_SUFFIX.get(chord.quality, chord.quality)
    return f"{chord.root}{suffix}"


def load_like_product(path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Exactly detect_chords lines 556-606: 22 kHz mono, tuning, HPSS, chroma."""
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tuning = C.estimate_tuning_semitones(y, int(sr))
    tuning_bins = tuning * (C._BINS_PER_OCTAVE / 12.0)
    harmonic, _perc = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic, sr=sr, hop_length=C._HOP, tuning=tuning_bins,
        bins_per_octave=C._BINS_PER_OCTAVE, norm=None,
    )
    return y, chroma, int(sr), tuning


def bass_chroma_for(path: Path | None, y: np.ndarray, sr: int, tuning_bins: float, frames: int) -> np.ndarray:
    """Bass chroma: stem file when given; detect_chords's fallback is the harmony low end."""
    if path is not None and Path(path).is_file():
        try:
            loaded, _sr2 = librosa.load(str(path), sr=sr, mono=True)
            if loaded.size >= int(sr * 0.2):
                bc = librosa.feature.chroma_cqt(
                    y=loaded, sr=sr, hop_length=C._HOP, fmin=librosa.note_to_hz("C2"), n_octaves=3,
                    tuning=tuning_bins, bins_per_octave=C._BINS_PER_OCTAVE, norm=None,
                )
                if bc.shape[1] >= frames:
                    return bc[:, :frames]
                return np.pad(bc, ((0, 0), (0, frames - bc.shape[1])))
        except Exception:
            pass
    bc = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=C._HOP, fmin=librosa.note_to_hz("C2"), n_octaves=3,
        tuning=tuning_bins, bins_per_octave=C._BINS_PER_OCTAVE, norm=None,
    )
    if bc.shape[1] < frames:
        bc = np.pad(bc, ((0, 0), (0, frames - bc.shape[1])))
    return bc[:, :frames]


def seventh_relaxed(ch: np.ndarray, index: int) -> str:
    chord = _TEMPLATE_CHORDS[index]
    if chord is None or chord.quality not in {"maj7", "mMaj7", "m7", "7"}:
        return ""
    root = _TEMPLATE_ROOTS[index]
    third_interval = 3 if chord.quality in {"mMaj7", "m7"} else 4
    seventh_interval = 10 if chord.quality in {"m7", "7"} else 11
    third = float(ch[(root + third_interval) % 12])
    seventh = float(ch[(root + seventh_interval) % 12])
    root_energy = float(ch[root])
    parts = []
    if root_energy > 1e-8 and seventh >= 0.32 * root_energy:
        parts.append(f"strong_color({seventh:.2f}>={0.32 * root_energy:.2f})")
    if (
        root_energy > 1e-8 and third >= 0.45 * root_energy and third > 1e-8
        and seventh >= 0.35 * third
    ):
        parts.append(f"voiced(3 {third:.2f},7/3 {seventh / max(third, 1e-9):.2f})")
    return ";".join(parts)


def bare_fifth_relaxed(ch: np.ndarray, index: int) -> str:
    chord = _TEMPLATE_CHORDS[index]
    if chord is None or chord.quality != "5":
        return ""
    root = _TEMPLATE_ROOTS[index]
    fifth = float(ch[(root + 7) % 12])
    major_third = float(ch[(root + 4) % 12])
    minor_third = float(ch[(root + 3) % 12])
    root_energy = float(ch[root])
    if root_energy > 1e-8 and fifth >= 0.35 * root_energy and major_third < 0.25 * root_energy and minor_third < 0.25 * root_energy:
        return "bare"
    return ""


def mirror(
    audio_path: Path,
    *,
    beats,
    bpm,
    provisional_key: str | None = None,
    beats_per_measure: int = 4,
    downbeats=None,
    bass_path: Path | None = None,
    keep_top: int = 6,
) -> dict | None:
    """Replicate detect_chords with full per-window instrumentation (read-only)."""
    path = Path(audio_path)
    if not path.is_file():
        return None
    y, chroma, sr, tuning = load_like_product(path)
    tuning_bins = tuning * (C._BINS_PER_OCTAVE / 12.0)
    bc = bass_chroma_for(bass_path, y, sr, tuning_bins, chroma.shape[1])
    duration = float(y.size) / float(sr)

    boundaries, tolerance = _boundaries(beats, duration)
    frames = librosa.time_to_frames(np.asarray(boundaries), sr=sr, hop_length=C._HOP)
    frames = np.clip(frames, 0, chroma.shape[1] - 1)

    windows = []
    for index in range(len(boundaries) - 1):
        start_frame, end_frame = int(frames[index]), int(frames[index + 1])
        if end_frame <= start_frame:
            continue
        vector = chroma[:, start_frame:end_frame].mean(axis=1)
        bass_vector = bc[:, start_frame:end_frame].mean(axis=1)
        start_sample = max(0, int(boundaries[index] * sr))
        end_sample = min(y.size, int(boundaries[index + 1] * sr))
        segment = y[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0
        windows.append((boundaries[index], boundaries[index + 1], vector, bass_vector, rms))
    if not windows:
        return None

    if provisional_key is None:
        from audio_to_tab.key import provisional_key as est_key

        chroma_mean = np.mean([w[2] for w in windows], axis=0)
        bass_mean = np.mean([w[3] for w in windows], axis=0)
        provisional_key, _score = est_key(chroma_mean, bass_mean)

    matrix = np.stack([w[2] for w in windows])
    bass_matrix = np.stack([w[3] for w in windows])
    norms = np.linalg.norm(matrix, axis=1)
    rmses = np.asarray([w[4] for w in windows], dtype=np.float64)
    silent = (norms < _SILENCE_NORM) | (rmses < 1e-4)
    normalised = np.divide(
        matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] > 1e-9
    )
    scores = normalised @ _TEMPLATE_TABLE.T
    scores = (scores - scores.mean(axis=1, keepdims=True)) / (
        scores.std(axis=1, keepdims=True) + 1e-9
    )
    for index in range(len(windows)):
        if not silent[index]:
            scores[index] += _bass_bonus(bass_matrix[index])
    key_prior = _key_prior(provisional_key)
    qual_prior = np.stack([_quality_prior_for_window(normalised[i]) for i in range(len(windows))])
    for index in range(len(windows)):
        if silent[index]:
            continue
        scores[index] -= qual_prior[index]
        scores[index] -= key_prior
    scores[silent] = 0.0

    meter = max(1, int(beats_per_measure or 4))
    downbeat_times = tuple(downbeats or ())
    penalties = np.full(len(windows), C._CHANGE_PENALTY + C._OFF_DOWNBEAT_PENALTY, dtype=np.float64)
    for index, (start, _e, _v, _b, _r) in enumerate(windows):
        if _is_downbeat(start, downbeats=downbeat_times, beats=beats, meter=meter, tolerance=tolerance):
            penalties[index] = C._CHANGE_PENALTY - C._ON_DOWNBEAT_BONUS
    path_indices = _viterbi(scores, penalties)
    beat_sec = _beat_seconds(boundaries, bpm)

    events, energies = [], []
    rows = []
    for index, state in enumerate(path_indices):
        raw_start, raw_end, _vector, bass_vector, rms = windows[index]
        start = _snap_to_beat(raw_start, boundaries[1:-1], tolerance)
        end = _snap_to_beat(raw_end, boundaries[1:-1], tolerance)
        start = max(0.0, min(duration, start))
        end = max(start, min(duration, end))
        if silent[index]:
            symbol, confidence = "N", 1.0
        else:
            symbol = _slash_bass(_TEMPLATE_NAMES[state], bass_vector)
            confidence = _window_confidence(scores[index])
        simple, medium = _decorate(symbol)
        events.append(
            ChordEvent(start=start, end=end, symbol=symbol, confidence=confidence,
                       simple=simple, medium=medium)
        )
        energies.append(rms)
        if not silent[index]:
            ch = normalised[index]
            order = np.argsort(scores[index])[::-1][:keep_top]
            best_acoustic = int(np.argmax(normalised[index] @ _TEMPLATE_TABLE.T))
            rows.append(
                {
                    "window": [float(raw_start), float(raw_end)],
                    "state": int(state),
                    "chosen": _TEMPLATE_NAMES[state],
                    "root_index": PITCHES.index(root_of(_TEMPLATE_NAMES[state]))
                    if root_of(_TEMPLATE_NAMES[state]) in PITCHES
                    else -1,
                    "argmax_bin": PITCHES[int(np.argmax(ch))],
                    "chroma": {PITCHES[i]: round(float(ch[i]), 3) for i in range(12)},
                    "acoustic_best": _TEMPLATE_NAMES[best_acoustic],
                    "final": round(float(scores[index][state]), 3),
                    "q_prior_chosen": round(float(qual_prior[index][state]), 3),
                    "k_prior_chosen": round(float(key_prior[state]), 3),
                    "q_prior_acoustic": round(float(qual_prior[index][best_acoustic]), 3),
                    "k_prior_acoustic": round(float(key_prior[best_acoustic]), 3),
                    "confidence": round(float(confidence), 3),
                    "viterbi_penalty": round(float(penalties[index]), 3),
                    "top": [
                        {
                            "state": _TEMPLATE_NAMES[int(o)],
                            "z": round(float(scores[index][o]), 3),
                            "relax": seventh_relaxed(ch, int(o)) or bare_fifth_relaxed(ch, int(o)),
                        }
                        for o in order
                    ],
                    "chosen_relax": seventh_relaxed(ch, int(state)) or bare_fifth_relaxed(ch, int(state)),
                }
            )

    absorbed = C._absorb_short_runs(events, energies, beat_sec=beat_sec)
    merged = _merge_events(absorbed)
    return {
        "events": merged,
        "windows": rows,
        "provisional_key": provisional_key,
        "tolerance": float(tolerance),
        "beat_sec": float(beat_sec),
        "tuning_semitones": float(tuning),
    }


def events_equal(a: list[ChordEvent], b: list[ChordEvent]) -> str:
    if len(a) != len(b):
        return f"length {len(a)} vs {len(b)}"
    for x, y_ in zip(a, b):
        if x.symbol != y_.symbol or abs(x.start - y_.start) > 1e-6 or abs(x.end - y_.end) > 1e-6:
            return f"{x.start:.2f}-{x.end:.2f} {x.symbol} vs {y_.start:.2f}-{y_.end:.2f} {y_.symbol}"
    return ""


def beat_grid_from_diag(audio: Path, diag_path: Path | None, duration: float):
    from audio_to_tab.analysis import build_beat_grid

    if diag_path is None:
        return build_beat_grid(audio, None, duration_sec=duration), None
    from audio_to_tab.metronome import load_metronome_diagnostics

    diag = load_metronome_diagnostics(diag_path)
    return build_beat_grid(audio, diag, duration_sec=duration), diag


def summaries(events, beat_sec, meter, duration, key) -> dict:
    pitched = [e for e in events if e.symbol.upper() != "N"]
    quals = [quality_of(e.symbol) for e in pitched]
    roots = [root_of(e.symbol) for e in pitched]
    seq_symbols = [net_of(e.symbol) for e in pitched]
    changes = sum(1 for a, b in zip(seq_symbols, seq_symbols[1:]) if a != b)
    root_changes = sum(1 for a, b in zip(roots, roots[1:]) if a != b)
    extended = sum(1 for q in quals if q in EXTENDED)
    bar_sec = max(1e-9, beat_sec * max(1, meter))
    mode_minor = bool(key) and str(key).endswith("m") and not str(key).lower().endswith("maj")
    tonic = PITCH.index(str(key)[:-1]) if key and str(key)[:-1] in PITCH else None
    table = MINOR if mode_minor else MAJOR
    nondiat = 0
    if tonic is not None:
        for r, q in zip(roots, quals):
            if r not in PITCH:
                nondiat += 1
                continue
            degree = (PITCH.index(r) - tonic) % 12
            expected = table.get(degree)
            if expected is None or (q != expected and q not in {"7", "maj7", "m7"}):
                nondiat += 1
    short = sum(1 for e in pitched if e.end - e.start < beat_sec - 1e-9)
    return {
        "segments": len(pitched),
        "segs_per_bar": round(len(pitched) / (duration / bar_sec), 3),
        "symbol_changes": changes,
        "symbol_change_rate": round(changes / max(1, len(seq_symbols) - 1), 3),
        "root_changes": root_changes,
        "root_change_rate": round(root_changes / max(1, len(roots) - 1), 3),
        "distinct_roots": len(set(roots)),
        "top_roots": dict(sorted(Counter(roots).items(), key=lambda kv: -kv[1])[:8]),
        "extended_share": round(extended / max(1, len(quals)), 3),
        "quality_hist": dict(sorted(Counter(quals).items(), key=lambda kv: -kv[1])),
        "nondiat": round(nondiat / max(1, len(pitched)), 3),
        "under_one_beat": round(short / max(1, len(pitched)), 3),
        "key": key,
    }


def run_summary(mirror_out, duration: float, meter: int) -> dict:
    grid = mirror_out["events"]
    return summaries(grid, mirror_out["beat_sec"], meter, duration, mirror_out["provisional_key"])


def window_aggregates(mirror_out) -> dict:
    rows = [r for r in mirror_out["windows"]]
    agree_acoustic = sum(
        1 for r in rows if r["chosen"].split("/")[0] == r["acoustic_best"].split("/")[0]
    )
    root_is_argmax = sum(
        1 for r in rows
        if r["root_index"] >= 0 and r["root_index"] == PITCH.index(r["argmax_bin"])
    )
    return {
        "windows": len(rows),
        "final_equals_acoustic": round(agree_acoustic / max(1, len(rows)), 3),
        "final_root_is_loudest_bin": round(root_is_argmax / max(1, len(rows)), 3),
    }


def synth() -> dict:
    SR = 22050
    G2, D3 = 103.826, 155.563  # G#2, D#3

    def tone(partial_spec, duration=3.0):
        t = np.arange(int(SR * duration)) / SR
        y = np.zeros_like(t)
        for f, a in partial_spec:
            y += a * np.sin(2 * np.pi * f * t)
        if y.max() > 0:
            y *= 0.9 / max(1e-9, abs(y).max())
        return y

    # 50f1 profile places chroma bins by pitch class (07's measured medians):
    # min3 B2=123.47, maj3 C3=130.81, fifth D#3=155.56, maj7 G3=196.00
    specs = {
        "g5_clean": [(G2, 0.25), (D3, 0.18), (G2 * 2, 0.08), (D3 * 2, 0.06)],
        "g5_dist": [(G2, 0.25), (D3, 0.18), (G2 * 2, 0.10), (D3 * 2, 0.08),
                    (G2 * 5, 0.03), (D3 * 5, 0.025), (G2 * 3, 0.04), (G2 * 7, 0.02)],
        "g5_50f1_profile": [
            (G2, 1.00), (G2 * 2, 0.20), (D3, 0.26), (D3 * 2, 0.10),
            (123.47, 0.50),   # B  (min3 of G#)
            (130.81, 0.34),   # C  (maj3 of G#)
            (196.00, 0.79),   # G  (maj7 of G#)
        ],
    }
    beats = tuple(i * 0.5 for i in range(13))

    results = {}
    for name, spec in specs.items():
        wav = TMP / f"{name}.wav"
        sf.write(wav, tone(spec), SR)
        ref = detect_chords(wav, beats=beats, bpm=120.0, provisional_key="G#m")
        mir = mirror(wav, beats=beats, bpm=120.0, provisional_key="G#m")
        ref2 = detect_chords(wav, beats=beats, bpm=120.0)
        results[name] = {
            "with_key_G#m": [net_of(e.symbol) for e in (ref or [])][:12],
            "with_key_auto": [net_of(e.symbol) for e in (ref2 or [])][:12],
            "auto_key": mir["provisional_key"] if mir else None,
            "windows": mir["windows"][:4] if mir else [],
        }
    # pad-conflict experiments: G#5 distorted + C-major pad at growing levels
    base = tone(specs["g5_dist"])
    t = np.arange(base.size) / SR
    for level in (0.05, 0.15, 0.30):
        pad = (
            np.sin(2 * np.pi * 130.81 * t)   # C3
            + 0.8 * np.sin(2 * np.pi * 164.81 * t)   # E3
            + 0.7 * np.sin(2 * np.pi * 196.00 * t)   # G3
        )
        pad = pad / max(1e-9, float(np.abs(pad).max())) * level
        mixed = base + pad
        mixed *= 0.9 / max(1e-9, abs(mixed).max())
        wav = TMP / f"g5_pad{level}.wav"
        sf.write(wav, mixed.astype(np.float32), SR)
        ref = detect_chords(wav, beats=beats, bpm=120.0)
        mir = mirror(wav, beats=beats, bpm=120.0)
        results[f"g5_pad{level}"] = {
            "labels": [net_of(e.symbol) for e in (ref or [])][:12],
            "auto_key": mir["provisional_key"] if mir else None,
            "window0_top": mir["windows"][0]["top"] if mir else [],
        }
    # identical audio for 8 bars: stability control
    long_y = np.concatenate([base] * 8)
    wav = TMP / "g5_repeat.wav"
    sf.write(wav, long_y.astype(np.float32), SR)
    beats_long = tuple(i * 0.5 for i in range(int(long_y.size / SR / 0.5) + 2))
    ref = detect_chords(wav, beats=beats_long, bpm=120.0)
    labels = [net_of(e.symbol) for e in (ref or [])]
    results["g5_repeat8"] = {
        "n": len(labels),
        "distinct": len(set(labels)),
        "hist": dict(sorted(Counter(labels).items(), key=lambda kv: -kv[1])[:6]),
    }
    return results


def real() -> dict:
    runs_root = REPO / "data" / "ui_runs"
    waltz = runs_root / "46b84e7b-dbb5-4c25-817a-53c4cd85b02d"
    report: dict = {}

    if (waltz / "analysis.json").is_file():
        analysis = json.loads((waltz / "analysis.json").read_text())
        beats = np.asarray(analysis.get("beats") or [], dtype=float)
        downbeats = np.asarray(analysis.get("downbeats") or [], dtype=float)
        bpm = float(analysis.get("bpm") or 0.0)
        meter = int(analysis.get("beats_per_measure") or 4)
        duration = float(analysis.get("duration_sec") or 0.0)
        from audio_to_tab.key import provisional_key_from_audio

        key, _ = provisional_key_from_audio(waltz / "source.mp3")
        print(f"waltz: bpm={bpm} meter={meter} beats={len(beats)} key_fresh={key} duration={duration:.1f}")
        mir = mirror(
            waltz / "source.mp3", beats=beats, bpm=bpm, provisional_key=key,
            beats_per_measure=meter, downbeats=tuple(downbeats), bass_path=waltz / "bass.wav",
        )
        ref = detect_chords(
            waltz / "source.mp3", beats=beats, bpm=bpm, provisional_key=key,
            beats_per_measure=meter, downbeats=tuple(downbeats), bass_path=waltz / "bass.wav",
        )
        diff = events_equal(ref, mir["events"]) if mir else "mirror None"
        print(f"waltz parity: {'EXACT' if not diff else ''} {diff}")
        report["waltz"] = {
            "parity": diff,
            "summary": run_summary(mir, duration, meter),
            "windows_n": len(mir["windows"]),
        }
        report["waltz"]["failing"] = failing_windows(mir, limit=8)

    # metal runs: rebuild harmony sum + fresh beat grid exactly like a new desktop analyse
    for run_id, label in (
        ("50f1b1f5-2f38-41ea-8f2a-f31e63fddd29", "50f1_metal"),
        ("9a18a73f-193c-4058-a96c-dfb490e565cc", "9a18_band"),
    ):
        run = runs_root / run_id
        if not run.is_dir():
            continue
        harmony = TMP / f"{label}_harmony.wav"
        _temp, sr_h = sum_stems([run / "guitar.wav", run / "piano.wav", run / "other.wav"], harmony)
        if harmony not in TMP or not harmony.is_file():
            print(f"{label}: harmony sum failed")
            continue
        import soundfile as _sf

        info = _sf.info(str(harmony))
        duration = float(info.frames) / float(info.samplerate)
        diag_out = TMP / f"{label}_metro.wav"
        from audio_to_tab.metronome import generate_metronome_stem, write_metronome_diagnostics

        result = generate_metronome_stem(run / "current_mix.wav", diag_out, source="source")
        if result is None:
            print(f"{label}: metronome failed")
            continue
        diag_path = TMP / f"{label}_metronome_diagnostics.json"
        write_metronome_diagnostics(diag_path, result)
        grid, _diag = beat_grid_from_diag(run / "current_mix.wav", diag_path, duration)
        print(
            f"{label}: bpm={grid.bpm} meter={grid.beats_per_measure} beats={len(grid.beats)} "
            f"downbeats={len(grid.downbeats)} duration={duration:.1f}"
        )
        from audio_to_tab.key import provisional_key_from_audio

        key, _ = provisional_key_from_audio(harmony)
        print(f"{label}: fresh provisional key={key}")
        bpm = float(grid.bpm) if grid.bpm and grid.bpm > 0 else None
        mir = mirror(
            harmony, beats=grid.beats, bpm=bpm, provisional_key=key,
            beats_per_measure=int(grid.beats_per_measure or 4), downbeats=tuple(grid.downbeats),
            bass_path=run / "bass.wav",
        )
        report[label] = {
            "bpm": grid.bpm,
            "meter": int(grid.beats_per_measure or 4),
            "beats": list(map(float, grid.beats[:5])) + ["..."],
            "summary": run_summary(mir, duration, int(grid.beats_per_measure or 4)),
            "windows_n": len(mir["windows"]),
            "window_aggregates": window_aggregates(mir),
        }
        report[label]["failing"] = failing_windows(mir, limit=10)

    return report


def sum_stems(paths, dest: Path, sr: int = 22050) -> tuple[Path, int] | None:
    """guitar+piano+other sum, peak-normalised — the product's _write_harmony_sum recipe."""
    pieces = []
    sample_rate = 0
    for p in paths:
        if not p.is_file():
            continue
        y, s = librosa.load(str(p), sr=None, mono=True)
        if sample_rate == 0:
            sample_rate = s
        elif s != sample_rate and y.size:
            y = librosa.resample(y, orig_sr=s, target_sr=sample_rate)
        if y.size:
            pieces.append(y)
    if not pieces or sample_rate <= 0:
        return None
    length = max(x.size for x in pieces)
    stacked = np.zeros(length, dtype=np.float64)
    for piece in pieces:
        stacked[: piece.size] += piece
    peak = float(np.max(np.abs(stacked))) if stacked.size else 0.0
    if peak > 0:
        stacked *= 0.99 / peak
    sf.write(str(dest), stacked.astype(np.float32), sample_rate, subtype="PCM_16")
    return dest, sample_rate


def failing_windows(mir, limit: int = 8) -> list[dict]:
    rows = [r for r in mir["windows"]]
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if quality_of(r["chosen"]) in EXTENDED else 1,
            0 if r["root_index"] != PITCH.index(r["argmax_bin"]) else 1,
            -abs(float(r["final"])),
        ),
    )
    return ranked[:limit]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    payload: dict = {}
    if mode == "parity":
        SR = 22050
        waltz = REPO / "data" / "ui_runs" / "46b84e7b-dbb5-4c25-817a-53c4cd85b02d"
        y = np.zeros(int(SR * 4))
        t = np.arange(y.size) / SR
        y += 0.2 * np.sin(2 * np.pi * 261.63 * t) + 0.1 * np.sin(2 * np.pi * 329.63 * t) + 0.08 * np.sin(2 * np.pi * 392.0 * t)
        y[int(SR * 2):] += 0.2 * np.sin(2 * np.pi * 226.31 * t[int(SR * 2):])  # D3
        wav = TMP / "parity.wav"
        sf.write(wav, y.astype(np.float32), SR)
        beats = tuple(i * 0.5 for i in range(9))
        ref = detect_chords(wav, beats=beats, bpm=120.0, provisional_key=None)
        mir = mirror(wav, beats=beats, bpm=120.0, provisional_key=None)
        diff = events_equal(ref, mir["events"]) if mir else "mirror None"
        print(f"synthetic parity: {'EXACT' if not diff else 'DIFF: ' + diff}")
        if (waltz / "analysis.json").is_file():
            analysis = json.loads((waltz / "analysis.json").read_text())
            beats_w = np.asarray(analysis.get("beats") or [], dtype=float)
            meter = int(analysis.get("beats_per_measure") or 4)
            bpm_w = float(analysis.get("bpm") or 0.0)
            downbeats_w = tuple(float(x) for x in analysis.get("downbeats") or [])
            from audio_to_tab.key import provisional_key_from_audio

            key, _ = provisional_key_from_audio(waltz / "source.mp3")
            ref = detect_chords(
                waltz / "source.mp3", beats=beats_w, bpm=bpm_w, provisional_key=key,
                beats_per_measure=meter, downbeats=downbeats_w, bass_path=waltz / "bass.wav",
            )
            mir = mirror(
                waltz / "source.mp3", beats=beats_w, bpm=bpm_w, provisional_key=key,
                beats_per_measure=meter, downbeats=downbeats_w, bass_path=waltz / "bass.wav",
            )
            diff = events_equal(ref, mir["events"]) if mir else "mirror None"
            print(f"waltz parity: {'EXACT' if not diff else 'DIFF: ' + diff}")
        return
    if mode == "synth":
        payload = synth()
    elif mode == "real":
        payload = real()
    out = OUT_DIR / f"diagnosis_{mode}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:4000])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
