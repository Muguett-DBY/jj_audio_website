"""Scratch driver: root vs quality drift on powerchords. Does not edit src/."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audio_to_tab.chord_symbols import PITCHES, parse
from audio_to_tab.chords import (
    _BINS_PER_OCTAVE,
    _HOP,
    _TEMPLATE_NAMES,
    _TEMPLATE_TABLE,
    _boundaries,
    _key_prior,
    _quality_prior_for_window,
    detect_chords,
    estimate_tuning_semitones,
)

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SR = 22050
SUFFIXES = (
    "maj7",
    "m7b5",
    "mMaj7",
    "dim7",
    "sus4",
    "sus2",
    "m7",
    "m6",
    "dim",
    "aug",
    "7",
    "6",
    "5",
    "m",
)


def quality_of(symbol: str) -> str:
    head = symbol.split("/", 1)[0]
    for suffix in SUFFIXES:
        if head.endswith(suffix) and len(head) > len(suffix):
            return suffix
    return "maj"


def root_of(symbol: str) -> str | None:
    chord = parse(symbol)
    return None if chord is None else chord.root


def write_power(
    path: Path,
    root_hz: float,
    *,
    sec: float = 8.0,
    distort: bool = False,
) -> None:
    t = np.arange(int(SR * sec), dtype=np.float64) / SR
    fifth = root_hz * (3.0 / 2.0)
    y = 0.28 * np.sin(2 * np.pi * root_hz * t) + 0.20 * np.sin(2 * np.pi * fifth * t)
    y += 0.10 * np.sin(2 * np.pi * root_hz * 2 * t)
    y += 0.08 * np.sin(2 * np.pi * fifth * 2 * t)
    if distort:
        # 5th harmonic ≈ major third two octaves up; keeps third/root mid-band.
        y += 0.06 * np.sin(2 * np.pi * root_hz * 5 * t)
        y = np.tanh(2.4 * y)
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(path, (0.9 * y / peak).astype(np.float32), SR)


def write_power_progression(path: Path, roots_hz: list[float], chord_sec: float = 2.0) -> None:
    chunks = []
    for root_hz in roots_hz:
        t = np.arange(int(SR * chord_sec), dtype=np.float64) / SR
        fifth = root_hz * 1.5
        y = 0.28 * np.sin(2 * np.pi * root_hz * t) + 0.20 * np.sin(2 * np.pi * fifth * t)
        y += 0.08 * np.sin(2 * np.pi * root_hz * 2 * t)
        chunks.append(y)
    y = np.concatenate(chunks)
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(path, (0.9 * y / peak).astype(np.float32), SR)


def window_chroma(path: Path, beats: list[float] | tuple[float, ...]):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    tuning = estimate_tuning_semitones(y, sr)
    harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=_HOP,
        tuning=tuning * (_BINS_PER_OCTAVE / 12.0),
        bins_per_octave=_BINS_PER_OCTAVE,
        norm=None,
    )
    duration = float(y.size) / float(sr)
    boundaries, _tol = _boundaries(beats, duration)
    frames = np.clip(
        librosa.time_to_frames(np.asarray(boundaries), sr=sr, hop_length=_HOP),
        0,
        chroma.shape[1] - 1,
    )
    windows = []
    for index in range(len(boundaries) - 1):
        a, b = int(frames[index]), int(frames[index + 1])
        if b <= a:
            continue
        vector = chroma[:, a:b].mean(axis=1)
        windows.append((boundaries[index], boundaries[index + 1], vector))
    return windows, duration


def score_window(vector: np.ndarray, provisional_key: str | None):
    norm = float(np.linalg.norm(vector)) + 1e-12
    normalised = vector / norm
    raw = normalised @ _TEMPLATE_TABLE.T
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    q = _quality_prior_for_window(normalised)
    k = _key_prior(provisional_key)
    final = z - q - k
    order = np.argsort(final)[::-1]
    top = []
    for index in order[:8]:
        name = _TEMPLATE_NAMES[index]
        top.append(
            {
                "symbol": name,
                "z": round(float(z[index]), 3),
                "q": round(float(q[index]), 3),
                "k": round(float(k[index]), 3),
                "final": round(float(final[index]), 3),
            }
        )
    return normalised, top


def ratios_for(vector: np.ndarray, root_pc: int) -> dict[str, float]:
    root = float(vector[root_pc]) + 1e-9
    return {
        "maj3": round(float(vector[(root_pc + 4) % 12]) / root, 3),
        "min3": round(float(vector[(root_pc + 3) % 12]) / root, 3),
        "fifth": round(float(vector[(root_pc + 7) % 12]) / root, 3),
        "maj7": round(float(vector[(root_pc + 11) % 12]) / root, 3),
        "ninth": round(float(vector[(root_pc + 2) % 12]) / root, 3),
    }


def summarize_events(events) -> dict:
    roots = [root_of(event.symbol) for event in events]
    qualities = [quality_of(event.symbol) for event in events]
    root_changes = sum(
        1 for index in range(1, len(roots)) if roots[index] != roots[index - 1]
    )
    quality_changes = sum(
        1
        for index in range(1, len(events))
        if quality_of(events[index].symbol) != quality_of(events[index - 1].symbol)
        and root_of(events[index].symbol) == root_of(events[index - 1].symbol)
    )
    return {
        "n": len(events),
        "symbols": [event.symbol for event in events],
        "root_counts": dict(Counter(roots)),
        "quality_counts": dict(Counter(qualities)),
        "root_changes": root_changes,
        "same_root_quality_changes": quality_changes,
        "first": events[0].symbol if events else None,
        "unique_roots": sorted({root for root in roots if root}),
    }


def run_case(label: str, path: Path, bpm: float, provisional_key: str | None) -> dict:
    duration = librosa.get_duration(path=str(path))
    beat_sec = 60.0 / bpm
    beats = tuple(round(index * beat_sec, 6) for index in range(int(duration / beat_sec) + 1))
    events = detect_chords(
        path,
        beats=beats,
        bpm=bpm,
        provisional_key=provisional_key,
        beats_per_measure=4,
    )
    windows, _ = window_chroma(path, beats)
    trail = []
    for start, end, vector in windows[:12]:
        normalised, top = score_window(vector, provisional_key)
        winner = top[0]["symbol"]
        chord = parse(winner)
        root_pc = PITCHES.index(chord.root) if chord else 0
        trail.append(
            {
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "top": top[:5],
                "ratios_vs_winner_root": ratios_for(normalised, root_pc),
            }
        )
    return {
        "label": label,
        "path": str(path),
        "provisional_key": provisional_key,
        "summary": summarize_events(events),
        "first_windows": trail,
    }


def real_mix_case() -> dict:
    bench = json.loads(
        (REPO / "docs/chord-recognition/bench/band_mix_before.json").read_text()
    )
    rid = "50f1b1f5-2f38-41ea-8f2a-f31e63fddd29"
    meta = bench[rid]
    run = REPO / "data" / "ui_runs" / rid
    rows = {}
    for stem in ("current_mix.wav", "guitar.wav", "rhythm_guitar.wav"):
        path = run / stem
        if not path.is_file():
            continue
        events = detect_chords(
            path,
            beats=meta["beats"],
            bpm=float(meta["bpm"]),
            beats_per_measure=4,
            provisional_key="D#m",
        )
        summary = summarize_events(events)
        # first 20 symbols + root sequence stability over time
        symbols = [event.symbol for event in events]
        roots = [root_of(symbol) for symbol in symbols]
        first_root = roots[0] if roots else None
        same_as_first = sum(1 for root in roots if root == first_root)
        rows[stem] = {
            "summary": summary,
            "first_20": symbols[:20],
            "fraction_root_equals_first": round(same_as_first / max(1, len(roots)), 3),
            "n_unique_roots": len({root for root in roots if root}),
        }
    return {"run": rid, "stems": rows}


def main() -> None:
    gs_hz = 103.83  # G#2
    clean = OUT / "gs_power_clean.wav"
    dist = OUT / "gs_power_distort.wav"
    prog = OUT / "gs_as_cs_power_prog.wav"
    write_power(clean, gs_hz, sec=8.0, distort=False)
    write_power(dist, gs_hz, sec=8.0, distort=True)
    # G#5 → A#5 → C#5 → G#5 (powerchord progression, roots should move with music)
    write_power_progression(
        prog,
        [103.83, 116.54, 138.59, 103.83],
        chord_sec=2.0,
    )

    report = {
        "held_gsharp_clean_key_None": run_case("held_gsharp_clean", clean, 120.0, None),
        "held_gsharp_clean_key_Gsm": run_case("held_gsharp_clean_Gsm", clean, 120.0, "G#m"),
        "held_gsharp_distort_key_Gsm": run_case("held_gsharp_distort", dist, 120.0, "G#m"),
        "held_gsharp_distort_key_C": run_case("held_gsharp_distort_C", dist, 120.0, "C"),
        "progression_powers": run_case("prog_powers", prog, 120.0, "G#m"),
        "real_50f1": real_mix_case(),
        "note_on_add9": (
            "DETECTED_QUALITIES has no add9; chart-level symbols cannot emit Cmaj7add9. "
            "User-facing 'add9' is either a misremembered maj7/sus2/9-like label or a "
            "different product surface."
        ),
    }
    out = OUT / "diagnosis_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)
    for key in (
        "held_gsharp_clean_key_None",
        "held_gsharp_clean_key_Gsm",
        "held_gsharp_distort_key_Gsm",
        "held_gsharp_distort_key_C",
        "progression_powers",
    ):
        summary = report[key]["summary"]
        print(
            key,
            "first=",
            summary["first"],
            "roots=",
            summary["root_counts"],
            "qual=",
            summary["quality_counts"],
            "root_changes=",
            summary["root_changes"],
            "same_root_q_changes=",
            summary["same_root_quality_changes"],
        )
    for stem, payload in report["real_50f1"]["stems"].items():
        print(
            "50f1",
            stem,
            "first=",
            payload["summary"]["first"],
            "frac_first_root=",
            payload["fraction_root_equals_first"],
            "unique_roots=",
            payload["n_unique_roots"],
            "qual=",
            payload["summary"]["quality_counts"],
            "first20=",
            payload["first_20"][:8],
        )


if __name__ == "__main__":
    main()
