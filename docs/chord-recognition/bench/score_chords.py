"""Score a chord detector against labeled audio.

    .venv311/bin/python docs/chord-recognition/bench/score_chords.py \
        --audio DIR --annotations DIR --out scores.json

    --baseline writes docs/chord-recognition/bench/baseline.json from the
    synthetic bench (and real-run proxies when data/ui_runs is present).
    Nothing here is imported by the product.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import librosa  # noqa: E402
import mir_eval  # noqa: E402
from mir_eval import chord as mir_chord  # noqa: E402
from mir_eval import util as mir_util  # noqa: E402

from audio_to_tab.chords import detect_chords  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SCORE_KEYS = (
    "root",
    "majmin",
    "sevenths",
    "mirex",
    "wrong_quality_right_root",
    "segs_per_bar",
    "share_events_under_one_beat",
    "median_event_sec",
    "meter_correct",
)

_QUALITY_TO_MIREVAL = {
    "maj": "maj",
    "min": "min",
    "7": "7",
    "maj7": "maj7",
    "m7": "min7",
    "sus4": "sus4",
    "sus2": "sus2",
    "dim": "dim",
    "aug": "aug",
    "6": "maj6",
    "m6": "min6",
    "m7b5": "hdim7",
    "dim7": "dim7",
    "mMaj7": "minmaj7",
    "5": "5",
    "maj9": "maj7",
    "7b9": "7",
    "add9": "maj",
    "9": "7",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _symbol_to_mireval(symbol: str) -> str:
    text = (symbol or "").strip()
    if not text or text.upper() == "N":
        return "N"
    try:
        from audio_to_tab.chord_symbols import parse

        chord = parse(text)
    except Exception:
        chord = None
    if chord is None:
        head = text.split("/", 1)[0]
        return head if ":" in head else f"{head}:maj"
    quality = _QUALITY_TO_MIREVAL.get(chord.quality, chord.quality)
    return f"{chord.root}:{quality}"


def _load_lab(path: Path) -> tuple[np.ndarray, list[str]]:
    intervals, labels = mir_eval.io.load_labeled_intervals(str(path))
    return np.asarray(intervals, dtype=float), [str(label) for label in labels]


def _load_beats(path: Path | None) -> tuple[list[float], float | None, int | None]:
    if path is None or not path.is_file():
        return [], None, None
    raw = json.loads(path.read_text(encoding="utf-8"))
    beats = [float(value) for value in raw.get("beats") or []]
    bpm = raw.get("bpm")
    meter = raw.get("expected_meter", raw.get("beats_per_measure"))
    return beats, (float(bpm) if bpm else None), (int(meter) if meter else None)


def _align(ref_iv: np.ndarray, ref_lab: list[str], est_iv: np.ndarray, est_lab: list[str]):
    est_iv, est_lab = mir_util.adjust_intervals(
        est_iv,
        est_lab,
        float(ref_iv.min()),
        float(ref_iv.max()),
        mir_chord.NO_CHORD,
        mir_chord.NO_CHORD,
    )
    intervals, ref_aligned, est_aligned = mir_util.merge_labeled_intervals(
        ref_iv, ref_lab, est_iv, est_lab
    )
    durations = mir_util.intervals_to_durations(intervals)
    return durations, list(ref_aligned), list(est_aligned)


def _wrong_quality_right_root(durations: np.ndarray, ref_lab: list[str], est_lab: list[str]) -> float:
    root_cmp = np.asarray(mir_chord.root(ref_lab, est_lab), dtype=float)
    maj_cmp = np.asarray(mir_chord.majmin(ref_lab, est_lab), dtype=float)
    valid = (root_cmp >= 0) & (maj_cmp >= 0)
    weight = float(np.sum(durations[valid])) if np.any(valid) else 0.0
    if weight <= 0:
        return 0.0
    wrong = valid & (root_cmp == 1.0) & (maj_cmp == 0.0)
    return float(np.sum(durations[wrong]) / weight)


def _score_track(
    audio: Path,
    lab: Path,
    *,
    beats_path: Path | None,
) -> dict:
    ref_iv, ref_lab = _load_lab(lab)
    beats, bpm, expected_meter = _load_beats(beats_path)
    events = detect_chords(audio, beats=beats, bpm=bpm)
    if events:
        est_iv = np.asarray([[event.start, event.end] for event in events], dtype=float)
        est_lab = [_symbol_to_mireval(event.symbol) for event in events]
    else:
        est_iv = np.zeros((0, 2), dtype=float)
        est_lab = []
    if len(est_iv) == 0:
        scores = {"root": 0.0, "majmin": 0.0, "sevenths": 0.0, "mirex": 0.0}
        wrong = 0.0
    else:
        raw = mir_chord.evaluate(ref_iv, ref_lab, est_iv, est_lab)
        scores = {key: float(raw[key]) for key in ("root", "majmin", "sevenths", "mirex")}
        durations, ref_aligned, est_aligned = _align(ref_iv, ref_lab, est_iv, est_lab)
        wrong = _wrong_quality_right_root(durations, ref_aligned, est_aligned)

    durations_est = [float(event.end - event.start) for event in events if event.symbol.upper() != "N"]
    if beats and len(beats) >= 2:
        beat_sec = float(np.median(np.diff(sorted(beats))))
    elif bpm and bpm > 0:
        beat_sec = 60.0 / float(bpm)
    else:
        beat_sec = 0.5
    meter = expected_meter or 4
    span = float(ref_iv.max() - ref_iv.min()) if len(ref_iv) else 0.0
    bar_sec = beat_sec * max(1, meter)
    bars = span / bar_sec if bar_sec > 0 else 0.0
    pitched = [event for event in events if event.symbol.upper() != "N"]
    short = [event for event in pitched if (event.end - event.start) < beat_sec - 1e-9]
    median = float(np.median(durations_est)) if durations_est else 0.0
    meter_correct: bool | None = None
    if expected_meter is not None:
        meter_correct = _meter_matches(audio, beats, bpm, expected_meter)
    return {
        "track": audio.stem,
        "root": scores["root"],
        "majmin": scores["majmin"],
        "sevenths": scores["sevenths"],
        "mirex": scores["mirex"],
        "wrong_quality_right_root": wrong,
        "segs_per_bar": (len(pitched) / bars) if bars > 0 else 0.0,
        "share_events_under_one_beat": (len(short) / len(pitched)) if pitched else 0.0,
        "median_event_sec": median,
        "meter_correct": meter_correct,
        "segments": len(pitched),
    }


def _meter_matches(audio: Path, beats: list[float], bpm: float | None, expected: int) -> bool:
    try:
        import soundfile as sf

        from audio_to_tab.metronome import detect_beats_per_measure
    except Exception:
        return False
    try:
        audio_data, sr = sf.read(str(audio), always_2d=True)
    except Exception:
        y, sr = librosa.load(str(audio), sr=22050, mono=True)
        audio_data = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    mono = np.mean(audio_data, axis=1)
    times = np.asarray(beats if beats else [0.0], dtype=np.float64)
    if times.size < 8:
        return False
    detected = detect_beats_per_measure(times, mono, int(sr), float(bpm or 120.0))
    return int(detected) == int(expected)


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(values)) if values else 0.0


def score_directories(audio_dir: Path, annotations_dir: Path) -> dict:
    audio_dir = Path(audio_dir)
    annotations_dir = Path(annotations_dir)
    rows: list[dict] = []
    for lab in sorted(annotations_dir.glob("*.lab")):
        audio = next(
            (audio_dir / f"{lab.stem}{suffix}" for suffix in AUDIO_SUFFIXES if (audio_dir / f"{lab.stem}{suffix}").is_file()),
            None,
        )
        if audio is None:
            continue
        beats_path = annotations_dir / f"{lab.stem}.beats.json"
        rows.append(_score_track(audio, lab, beats_path=beats_path if beats_path.is_file() else None))
    meter_flags = [row["meter_correct"] for row in rows if isinstance(row["meter_correct"], bool)]
    payload = {
        "root": _mean(rows, "root"),
        "majmin": _mean(rows, "majmin"),
        "sevenths": _mean(rows, "sevenths"),
        "mirex": _mean(rows, "mirex"),
        "wrong_quality_right_root": _mean(rows, "wrong_quality_right_root"),
        "segs_per_bar": _mean(rows, "segs_per_bar"),
        "share_events_under_one_beat": _mean(rows, "share_events_under_one_beat"),
        "median_event_sec": _mean(rows, "median_event_sec"),
        "meter_correct": (float(np.mean(meter_flags)) if meter_flags else None),
        "git_sha": git_sha(),
        "librosa_version": librosa.__version__,
        "tracks": rows,
    }
    return payload


def _synthetic_baseline() -> dict:
    import soundfile as sf

    import synth_bench as bench

    tmp = Path(tempfile.mkdtemp(prefix="chord-baseline-"))
    rows = []
    for mode_name, hard in (("clean", False), ("realistic", True)):
        for name, progression in bench.PROGRESSIONS.items():
            wav = tmp / f"{name}-{mode_name}.wav"
            sf.write(wav, bench.render_track(progression, hard=hard), bench.SR)
            ref_iv, ref_lab = bench.ground_truth(progression)
            gt_beats = np.arange(len(progression) * bench.BEATS_PER_BAR + 1) * bench.BEAT_SEC
            events = detect_chords(wav, beats=gt_beats, bpm=bench.BPM)
            scored = bench.score(ref_iv, ref_lab, *bench.est_from_events(events))
            rows.append({"track": f"{name}-{mode_name}", "mode": mode_name, "score": scored})

    def mean(mode: str, key: str) -> float:
        values = [float(row["score"].get(key, 0.0)) for row in rows if row["mode"] == mode]
        return float(np.mean(values)) if values else 0.0

    return {
        "root_wcsr_clean": mean("clean", "root"),
        "majmin_wcsr_clean": mean("clean", "majmin"),
        "majmin_wcsr_realistic": mean("realistic", "majmin"),
        "sevenths_wcsr_clean": mean("clean", "sevenths"),
        "sevenths_wcsr_realistic": mean("realistic", "sevenths"),
        "root_wcsr_realistic": mean("realistic", "root"),
        "mirex_wcsr_clean": mean("clean", "mirex"),
        "tracks": rows,
    }


def _quality_of(symbol: str) -> str:
    head = symbol.split("/", 1)[0]
    for suffix in ("maj7", "m7b5", "mMaj7", "dim7", "sus4", "sus2", "m7", "m6", "dim", "aug", "7", "6", "5", "m"):
        if head.endswith(suffix) and len(head) > len(suffix):
            return suffix
    return "maj"


def _root_of(symbol: str) -> str | None:
    try:
        from audio_to_tab.chord_symbols import parse

        chord = parse(symbol)
    except Exception:
        return None
    return None if chord is None else chord.root


def root_stability(events) -> dict:
    """Phase-1 FM2 metrics: how much the labelled root wanders after the first event."""
    roots = [_root_of(event.symbol) for event in events]
    qualities = [_quality_of(event.symbol) for event in events]
    total = max(1, len(roots))
    first = roots[0] if roots else None
    same_as_first = sum(1 for root in roots if root == first and first is not None)
    unique = {root for root in roots if root}
    root_changes = sum(1 for index in range(1, len(roots)) if roots[index] != roots[index - 1])
    same_root_quality_changes = sum(
        1
        for index in range(1, len(events))
        if roots[index] == roots[index - 1]
        and qualities[index] != qualities[index - 1]
        and roots[index] is not None
    )
    return {
        "fraction_root_equals_first": round(same_as_first / total, 4),
        "n_unique_roots": len(unique),
        "root_changes": root_changes,
        "same_root_quality_changes": same_root_quality_changes,
        "first_symbol": events[0].symbol if events else None,
        "mMaj7_share": round(qualities.count("mMaj7") / total, 4),
        "power_share": round(qualities.count("5") / total, 4),
    }


def _proxy_runs() -> dict:
    """Unlabelled real-run proxies. Reads audio; does not write inside data/ui_runs."""
    runs_root = REPO / "data" / "ui_runs"
    wanted = {
        "46b84e7b-dbb5-4c25-817a-53c4cd85b02d": {"role": "waltz", "meter": 3},
        "50f1b1f5-2f38-41ea-8f2a-f31e63fddd29": {"role": "band", "meter": 4},
        "9a18a73f-193c-4058-a96c-dfb490e565cc": {"role": "band", "meter": 4},
        "ad6e7c9f-df76-411e-a700-4bd2b3d65d30": {"role": "band", "meter": 4},
    }
    saved = BENCH / "band_mix_before.json"
    saved_beats = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
    pitches = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    major = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
    minor = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}
    rows = []
    if not runs_root.is_dir():
        return {"available": False, "runs": []}
    for run_id, meta in wanted.items():
        run = runs_root / run_id
        analysis_path = run / "analysis.json"
        # Prefer current_mix / guitar when present so proxies match diagnosis pins.
        audio_candidates = (
            run / "current_mix.wav",
            run / "source.mp3",
            run / "guitar.wav",
        )
        source = next((path for path in audio_candidates if path.is_file()), None)
        if not analysis_path.is_file() or source is None:
            continue
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        pin = saved_beats.get(run_id) or {}
        beats = [float(value) for value in (pin.get("beats") or analysis.get("beats") or [])]
        bpm = float(pin.get("bpm") or analysis.get("bpm") or 0.0)
        meter = int(analysis.get("beats_per_measure") or 4)
        duration = float(pin.get("duration") or analysis.get("duration_sec") or 0.0)
        key = analysis.get("key") or ""
        events = detect_chords(source, beats=beats, bpm=bpm)
        beat_sec = 60.0 / bpm if bpm > 0 else 0.5
        bar_sec = beat_sec * max(1, meter)
        bars = duration / bar_sec if bar_sec > 0 else 0.0
        qualities = [_quality_of(event.symbol) for event in events]
        total = max(1, len(events))
        mode_minor = str(key).endswith("m")
        tonic_name = str(key)[:-1] if mode_minor else str(key)
        tonic = pitches.index(tonic_name) if tonic_name in pitches else None
        table = minor if mode_minor else major
        non = 0
        if tonic is not None:
            for event in events:
                root_name = event.symbol.split("/", 1)[0]
                quality = _quality_of(event.symbol)
                for suffix in ("maj7", "m7b5", "mMaj7", "dim7", "sus4", "sus2", "m7", "m6", "dim", "aug", "7", "6", "5", "m"):
                    if root_name.endswith(suffix) and len(root_name) > len(suffix):
                        root_name = root_name[: -len(suffix)]
                        break
                if root_name not in pitches:
                    non += 1
                    continue
                degree = (pitches.index(root_name) - tonic) % 12
                expected = table.get(degree)
                if expected is None or not (expected == quality or quality in {"7", "maj7", "m7"}):
                    non += 1
        short = sum(1 for event in events if (event.end - event.start) < beat_sec - 1e-9)
        stored_meter_correct = int(meter) == int(meta["meter"])
        stability = root_stability(events)
        guitar_stability = None
        guitar = run / "guitar.wav"
        if guitar.is_file() and guitar.resolve() != source.resolve():
            guitar_events = detect_chords(guitar, beats=beats, bpm=bpm)
            guitar_stability = root_stability(guitar_events)
        rows.append(
            {
                "run": run_id,
                "role": meta["role"],
                "audio": source.name,
                "stored_beats_per_measure": meter,
                "meter_correct": stored_meter_correct,
                "segments": len(events),
                "segs_per_bar": (len(events) / bars) if bars else 0.0,
                "maj7_share": qualities.count("maj7") / total,
                "mMaj7_share": stability["mMaj7_share"],
                "power_share": stability["power_share"],
                "sus4_share": qualities.count("sus4") / total,
                "non_diatonic_share": non / total,
                "share_events_under_one_beat": short / total,
                "fraction_root_equals_first": stability["fraction_root_equals_first"],
                "n_unique_roots": stability["n_unique_roots"],
                "root_changes": stability["root_changes"],
                "same_root_quality_changes": stability["same_root_quality_changes"],
                "first_symbol": stability["first_symbol"],
                "guitar_root_stability": guitar_stability,
                "key": key,
            }
        )
    return {"available": bool(rows), "runs": rows}


def write_baseline(out: Path) -> dict:
    synthetic = _synthetic_baseline()
    proxy = _proxy_runs()
    payload = {
        "git_sha": git_sha(),
        "librosa_version": librosa.__version__,
        "synthetic": synthetic,
        "proxy": proxy,
        # Flat copies of the audit §5 point values this snapshot is checked against.
        "root_wcsr_clean": synthetic["root_wcsr_clean"],
        "majmin_wcsr_clean": synthetic["majmin_wcsr_clean"],
        "majmin_wcsr_realistic": synthetic["majmin_wcsr_realistic"],
        "sevenths_wcsr_realistic": synthetic["sevenths_wcsr_realistic"],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Score chord labels with mir_eval")
    parser.add_argument("--audio", type=Path, help="Directory of audio files")
    parser.add_argument("--annotations", type=Path, help="Directory of .lab files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true", help="Write the Phase 0 baseline snapshot")
    args = parser.parse_args()
    if args.baseline:
        payload = write_baseline(args.out)
        synth = payload["synthetic"]
        print(
            f"baseline clean root={synth['root_wcsr_clean']:.3f} "
            f"majmin={synth['majmin_wcsr_clean']:.3f} "
            f"realistic majmin={synth['majmin_wcsr_realistic']:.3f} "
            f"realistic sevenths={synth['sevenths_wcsr_realistic']:.3f}"
        )
        print(f"wrote {args.out}")
        return
    if args.audio is None or args.annotations is None:
        parser.error("--audio and --annotations are required unless --baseline is set")
    payload = score_directories(args.audio, args.annotations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
