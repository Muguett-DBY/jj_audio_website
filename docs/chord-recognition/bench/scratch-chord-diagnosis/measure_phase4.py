"""Measure Phase-4 scorer candidates against synth floors + 50f1 root retention.

Does not modify src/. Copies the candidate decode logic inline for A/B.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "docs/chord-recognition/bench"))

from audio_to_tab.chords import (  # noqa: E402
    _CHANGE_PENALTY,
    _OFF_DOWNBEAT_PENALTY,
    _ON_DOWNBEAT_BONUS,
    _SILENCE_NORM,
    _TEMPLATE_NAMES,
    _TEMPLATE_ROOTS,
    _TEMPLATE_TABLE,
    _boundaries,
    _is_downbeat,
    _key_prior,
    _quality_prior_for_window,
    _viterbi,
    detect_chords,
    estimate_tuning_semitones,
)
from score_chords import root_stability  # noqa: E402

import librosa  # noqa: E402
import synth_bench as bench  # noqa: E402

SR = 22050


def viterbi_root_aware(
    scores: np.ndarray,
    frame_penalty: np.ndarray,
    *,
    root_extra: float,
) -> list[int]:
    frames, states = scores.shape
    if frames == 0:
        return []
    roots = np.asarray(_TEMPLATE_ROOTS, dtype=np.int16)
    dp = scores[0].copy()
    back = np.zeros((frames, states), dtype=np.int16)
    for index in range(1, frames):
        base = float(frame_penalty[index])
        order = np.argsort(dp)[::-1]
        best = int(order[0])
        second = int(order[1]) if states > 1 else best
        best_value = float(dp[best])
        second_value = float(dp[second])
        for state in range(states):
            stay = float(dp[state])
            if state == best:
                prev = second
                prev_value = second_value
            else:
                prev = best
                prev_value = best_value
            extra = root_extra if int(roots[prev]) != int(roots[state]) else 0.0
            other = prev_value - base - extra
            if stay >= other:
                back[index, state] = state
                dp[state] = scores[index, state] + stay
            else:
                back[index, state] = prev
                dp[state] = scores[index, state] + other
    path = np.zeros(frames, dtype=np.int16)
    path[-1] = int(np.argmax(dp))
    for index in range(frames - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return [int(value) for value in path]


def key_prior_scaled(key: str | None, *, outside: float, diatonic: float) -> np.ndarray:
    prior = _key_prior(key)
    # Rebuild with new magnitudes from the sign pattern of the stock prior.
    scaled = np.zeros_like(prior)
    scaled[prior > 0] = outside
    scaled[prior < 0] = -abs(diatonic)
    return scaled


def decode_file(
    path: Path,
    beats,
    bpm: float,
    provisional_key: str | None,
    *,
    root_extra: float,
    key_outside: float,
    key_diatonic: float,
    meter: int = 4,
):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    tuning = estimate_tuning_semitones(y, sr)
    harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=512,
        tuning=tuning * (36 / 12.0),
        bins_per_octave=36,
        norm=None,
    )
    duration = float(y.size) / float(sr)
    boundaries, tolerance = _boundaries(beats, duration)
    frames = np.clip(
        librosa.time_to_frames(np.asarray(boundaries), sr=sr, hop_length=512),
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
    matrix = np.stack([v for _, _, v, _ in windows])
    norms = np.linalg.norm(matrix, axis=1)
    rmses = np.asarray([r for _, _, _, r in windows], dtype=np.float64)
    silent = (norms < _SILENCE_NORM) | (rmses < 1e-4)
    normalised = np.divide(
        matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] > 1e-9
    )
    scores = normalised @ _TEMPLATE_TABLE.T
    scores = (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)
    key_prior = key_prior_scaled(provisional_key, outside=key_outside, diatonic=key_diatonic)
    for index in range(len(windows)):
        if silent[index]:
            continue
        scores[index] -= _quality_prior_for_window(normalised[index])
        scores[index] -= key_prior
    scores[silent] = 0.0
    penalties = np.full(len(windows), _CHANGE_PENALTY + _OFF_DOWNBEAT_PENALTY, dtype=np.float64)
    for index, (start, _e, _v, _r) in enumerate(windows):
        if _is_downbeat(start, downbeats=(), beats=beats, meter=meter, tolerance=tolerance):
            penalties[index] = _CHANGE_PENALTY - _ON_DOWNBEAT_BONUS
    if root_extra > 0:
        path = viterbi_root_aware(scores, penalties, root_extra=root_extra)
    else:
        path = _viterbi(scores, penalties)
    # Merge identical neighbours only (absorb skipped for speed).
    symbols = []
    starts = []
    ends = []
    for index, state in enumerate(path):
        if silent[index]:
            symbol = "N"
        else:
            symbol = _TEMPLATE_NAMES[state]
        start, end, _v, _r = windows[index]
        if symbols and symbols[-1] == symbol:
            ends[-1] = end
        else:
            symbols.append(symbol)
            starts.append(start)
            ends.append(end)

    class E:
        def __init__(self, start, end, symbol):
            self.start = start
            self.end = end
            self.symbol = symbol

    return [E(s, e, sym) for s, e, sym in zip(starts, ends, symbols)]


def synth_floor(cfg: dict) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="p4-"))
    rows = []
    for mode_name, hard in (("clean", False), ("realistic", True)):
        for name, progression in bench.PROGRESSIONS.items():
            wav = tmp / f"{name}-{mode_name}.wav"
            sf.write(wav, bench.render_track(progression, hard=hard), bench.SR)
            ref_iv, ref_lab = bench.ground_truth(progression)
            gt_beats = np.arange(len(progression) * bench.BEATS_PER_BAR + 1) * bench.BEAT_SEC
            # Use production detect when cfg is baseline; else custom decode then score via symbols.
            if cfg.get("use_detect"):
                events = detect_chords(wav, beats=gt_beats, bpm=bench.BPM, provisional_key="C")
                scored = bench.score(ref_iv, ref_lab, *bench.est_from_events(events))
            else:
                events = decode_file(
                    wav,
                    gt_beats,
                    bench.BPM,
                    "C",
                    root_extra=cfg["root_extra"],
                    key_outside=cfg["key_outside"],
                    key_diatonic=cfg["key_diatonic"],
                )
                scored = bench.score(ref_iv, ref_lab, *bench.est_from_events(events))
            rows.append({"mode": mode_name, "score": scored})

    def mean(mode: str, key: str) -> float:
        values = [float(row["score"].get(key, 0.0)) for row in rows if row["mode"] == mode]
        return float(np.mean(values)) if values else 0.0

    return {
        "clean_root": mean("clean", "root"),
        "clean_majmin": mean("clean", "majmin"),
        "clean_sevenths": mean("clean", "sevenths"),
        "realistic_root": mean("realistic", "root"),
        "realistic_majmin": mean("realistic", "majmin"),
        "realistic_sevenths": mean("realistic", "sevenths"),
    }


def real_50f1(cfg: dict) -> dict:
    saved = json.loads((REPO / "docs/chord-recognition/bench/band_mix_before.json").read_text())
    rid = "50f1b1f5-2f38-41ea-8f2a-f31e63fddd29"
    pin = saved[rid]
    run = REPO / "data" / "ui_runs" / rid
    out = {}
    for stem in ("current_mix.wav", "guitar.wav"):
        path = run / stem
        if not path.is_file():
            continue
        if cfg.get("use_detect"):
            events = detect_chords(
                path,
                beats=pin["beats"],
                bpm=float(pin["bpm"]),
                provisional_key="D#m",
            )
        else:
            events = decode_file(
                path,
                pin["beats"],
                float(pin["bpm"]),
                "D#m",
                root_extra=cfg["root_extra"],
                key_outside=cfg["key_outside"],
                key_diatonic=cfg["key_diatonic"],
            )
        stab = root_stability(events)
        out[stem] = {
            "n": len(events),
            "fraction_root_equals_first": stab["fraction_root_equals_first"],
            "n_unique_roots": stab["n_unique_roots"],
            "mMaj7_share": stab["mMaj7_share"],
            "power_share": stab["power_share"],
        }
    return out


def floors_ok(s: dict) -> bool:
    return (
        s["clean_root"] >= 0.995
        and s["clean_majmin"] >= 0.995
        and s["realistic_majmin"] >= 0.990
        and s["realistic_sevenths"] >= 0.900
    )


def main() -> None:
    configs = [
        {"name": "baseline_detect", "use_detect": True},
        {
            "name": "root_extra_0.25",
            "root_extra": 0.25,
            "key_outside": 0.10,
            "key_diatonic": 0.20,
        },
        {
            "name": "root_extra_0.40",
            "root_extra": 0.40,
            "key_outside": 0.10,
            "key_diatonic": 0.20,
        },
        {
            "name": "key_stronger",
            "root_extra": 0.0,
            "key_outside": 0.20,
            "key_diatonic": 0.30,
        },
        {
            "name": "root_0.25_key_stronger",
            "root_extra": 0.25,
            "key_outside": 0.18,
            "key_diatonic": 0.28,
        },
        {
            "name": "root_0.35_key_mild",
            "root_extra": 0.35,
            "key_outside": 0.15,
            "key_diatonic": 0.25,
        },
    ]
    report = {}
    for cfg in configs:
        print("==", cfg["name"])
        synth = synth_floor(cfg)
        real = real_50f1(cfg)
        ok = floors_ok(synth)
        report[cfg["name"]] = {"synth": synth, "floors_ok": ok, "real_50f1": real, "cfg": cfg}
        print("  floors_ok", ok, synth)
        print("  50f1", real)
    out = Path(__file__).with_name("phase4_measure.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
