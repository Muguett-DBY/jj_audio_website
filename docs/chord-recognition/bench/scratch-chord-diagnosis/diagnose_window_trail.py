"""Per-window trail on 50f1 + Viterbi vs greedy. Read-only w.r.t. src/."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

from audio_to_tab.chord_symbols import PITCHES, parse
from audio_to_tab.chords import (
    _BINS_PER_OCTAVE,
    _CHANGE_PENALTY,
    _HOP,
    _OFF_DOWNBEAT_PENALTY,
    _ON_DOWNBEAT_BONUS,
    _SILENCE_NORM,
    _TEMPLATE_NAMES,
    _TEMPLATE_TABLE,
    _boundaries,
    _is_downbeat,
    _key_prior,
    _quality_prior_for_window,
    _viterbi,
    detect_chords,
    estimate_tuning_semitones,
)

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SR = 22050


def quality_of(symbol: str) -> str:
    head = symbol.split("/", 1)[0]
    for suffix in (
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
    ):
        if head.endswith(suffix) and len(head) > len(suffix):
            return suffix
    return "maj"


def root_of(symbol: str) -> str | None:
    chord = parse(symbol)
    return None if chord is None else chord.root


def ratios(vector: np.ndarray, root_pc: int) -> dict[str, float]:
    root = float(vector[root_pc]) + 1e-9
    return {
        "maj3": round(float(vector[(root_pc + 4) % 12]) / root, 3),
        "min3": round(float(vector[(root_pc + 3) % 12]) / root, 3),
        "fifth": round(float(vector[(root_pc + 7) % 12]) / root, 3),
        "maj7": round(float(vector[(root_pc + 11) % 12]) / root, 3),
        "ninth": round(float(vector[(root_pc + 2) % 12]) / root, 3),
        "root_abs": round(float(vector[root_pc]), 4),
    }


def analyze_file(audio_path: Path, beats, bpm: float, provisional_key: str | None) -> dict:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
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
    boundaries, tolerance = _boundaries(beats, duration)
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
        start_sample = max(0, int(boundaries[index] * sr))
        end_sample = min(y.size, int(boundaries[index + 1] * sr))
        segment = y[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0
        windows.append((boundaries[index], boundaries[index + 1], vector, rms))

    matrix = np.stack([vector for _, _, vector, _ in windows])
    norms = np.linalg.norm(matrix, axis=1)
    rmses = np.asarray([rms for _, _, _, rms in windows], dtype=np.float64)
    silent = (norms < _SILENCE_NORM) | (rmses < 1e-4)
    normalised = np.divide(
        matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] > 1e-9
    )
    scores = normalised @ _TEMPLATE_TABLE.T
    scores = (scores - scores.mean(axis=1, keepdims=True)) / (
        scores.std(axis=1, keepdims=True) + 1e-9
    )
    key_prior = _key_prior(provisional_key)
    for index in range(len(windows)):
        if silent[index]:
            continue
        scores[index] -= _quality_prior_for_window(normalised[index])
        scores[index] -= key_prior
    scores[silent] = 0.0

    penalties = np.full(len(windows), _CHANGE_PENALTY + _OFF_DOWNBEAT_PENALTY, dtype=np.float64)
    for index, (start, _end, _vector, _rms) in enumerate(windows):
        if _is_downbeat(
            start,
            downbeats=(),
            beats=beats,
            meter=4,
            tolerance=tolerance,
        ):
            penalties[index] = _CHANGE_PENALTY - _ON_DOWNBEAT_BONUS

    greedy = [int(np.argmax(row)) for row in scores]
    viterbi_path = _viterbi(scores, penalties)

    trail = []
    for index in range(min(20, len(windows))):
        start, end, _vector, _rms = windows[index]
        order = np.argsort(scores[index])[::-1]
        top = []
        for state in order[:6]:
            name = _TEMPLATE_NAMES[state]
            top.append(
                {
                    "symbol": name,
                    "final": round(float(scores[index, state]), 3),
                }
            )
        g_sym = _TEMPLATE_NAMES[greedy[index]]
        v_sym = _TEMPLATE_NAMES[viterbi_path[index]]
        g_root = root_of(g_sym)
        root_pc = PITCHES.index(g_root) if g_root else 0
        # also ratios vs G# (pc 8) — user example root
        trail.append(
            {
                "i": index,
                "t": round(float(start), 3),
                "greedy": g_sym,
                "viterbi": v_sym,
                "viterbi_differs": g_sym != v_sym,
                "top": top,
                "ratios_vs_greedy_root": ratios(normalised[index], root_pc),
                "ratios_vs_Gsharp": ratios(normalised[index], 8),
                "ratios_vs_C": ratios(normalised[index], 0),
            }
        )

    greedy_roots = [root_of(_TEMPLATE_NAMES[i]) for i in greedy]
    viterbi_roots = [root_of(_TEMPLATE_NAMES[i]) for i in viterbi_path]
    greedy_q = [quality_of(_TEMPLATE_NAMES[i]) for i in greedy]
    viterbi_q = [quality_of(_TEMPLATE_NAMES[i]) for i in viterbi_path]
    differ = sum(1 for a, b in zip(greedy, viterbi_path) if a != b)

    # how often does a G#* template appear in top-3 final?
    gsharp_in_top3 = 0
    for index in range(len(windows)):
        if silent[index]:
            continue
        order = np.argsort(scores[index])[::-1][:3]
        if any(_TEMPLATE_NAMES[state].startswith("G#") for state in order):
            gsharp_in_top3 += 1

    # wrong-key experiment: force key C vs D#m on same scores chain via detect_chords
    events_dsm = detect_chords(
        audio_path, beats=beats, bpm=bpm, provisional_key="D#m", beats_per_measure=4
    )
    events_c = detect_chords(
        audio_path, beats=beats, bpm=bpm, provisional_key="C", beats_per_measure=4
    )
    events_none = detect_chords(
        audio_path, beats=beats, bpm=bpm, provisional_key=None, beats_per_measure=4
    )

    def root_hist(events):
        return dict(Counter(root_of(e.symbol) for e in events))

    def qual_hist(events):
        return dict(Counter(quality_of(e.symbol) for e in events))

    return {
        "path": str(audio_path),
        "n_windows": len(windows),
        "greedy_vs_viterbi_differ_frac": round(differ / max(1, len(windows)), 3),
        "greedy_root_hist": dict(Counter(greedy_roots)),
        "viterbi_root_hist": dict(Counter(viterbi_roots)),
        "greedy_qual_hist": dict(Counter(greedy_q)),
        "viterbi_qual_hist": dict(Counter(viterbi_q)),
        "gsharp_in_top3_frac": round(gsharp_in_top3 / max(1, len(windows) - int(silent.sum())), 3),
        "first_20_windows": trail,
        "detect_key_Dsm": {
            "first": events_dsm[0].symbol if events_dsm else None,
            "roots": root_hist(events_dsm),
            "qual": qual_hist(events_dsm),
            "n": len(events_dsm),
        },
        "detect_key_C": {
            "first": events_c[0].symbol if events_c else None,
            "roots": root_hist(events_c),
            "qual": qual_hist(events_c),
            "n": len(events_c),
        },
        "detect_key_None": {
            "first": events_none[0].symbol if events_none else None,
            "roots": root_hist(events_none),
            "qual": qual_hist(events_none),
            "n": len(events_none),
            "estimated_note": "provisional_key=None lets detect_chords estimate from chroma",
        },
    }


def main() -> None:
    bench = json.loads(
        (REPO / "docs/chord-recognition/bench/band_mix_before.json").read_text()
    )
    rid = "50f1b1f5-2f38-41ea-8f2a-f31e63fddd29"
    meta = bench[rid]
    run = REPO / "data" / "ui_runs" / rid
    report = {}
    for stem in ("current_mix.wav", "guitar.wav"):
        report[stem] = analyze_file(
            run / stem,
            meta["beats"],
            float(meta["bpm"]),
            "D#m",
        )
        print(
            stem,
            "greedy!=viterbi",
            report[stem]["greedy_vs_viterbi_differ_frac"],
            "gsharp_top3",
            report[stem]["gsharp_in_top3_frac"],
            "viterbi_roots_top",
            sorted(report[stem]["viterbi_root_hist"].items(), key=lambda x: -x[1])[:5],
            "keyC_first",
            report[stem]["detect_key_C"]["first"],
            "keyDsm_first",
            report[stem]["detect_key_Dsm"]["first"],
        )
        for row in report[stem]["first_20_windows"][:8]:
            print(
                " ",
                row["t"],
                "g=",
                row["greedy"],
                "v=",
                row["viterbi"],
                "diff=",
                row["viterbi_differs"],
                "vsG#",
                row["ratios_vs_Gsharp"],
                "vsC",
                row["ratios_vs_C"],
            )

    # Cross-root confusion on dense chroma: how often C* beats G#* acoustically
    # when both thirds are loud vs G#.
    out = OUT / "window_trail_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
