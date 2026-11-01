"""Real-track diagnostics for the chord stage.

No source changes. Reads an existing analysed run under data/ui_runs and reports
what the chord stage does with different harmony inputs (full mix vs stems) and
with a shifted beat grid. There is no ground truth here, so the numbers are
proxies: diatonic fit, quality mix, over-segmentation, and grid sensitivity.

    .venv311/bin/python docs/chord-recognition/bench/real_track_diag.py RUN_DIR
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from audio_to_tab.chord_symbols import parse as parse_chord  # noqa: E402
from audio_to_tab.chords import detect_chords  # noqa: E402

PITCH = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR = {0: "", 2: "m", 4: "m", 5: "", 7: "", 9: "m", 11: "dim"}
MINOR = {0: "m", 2: "dim", 3: "", 5: "m", 7: "m", 8: "", 10: ""}


def split_symbol(symbol: str):
    chord = parse_chord(symbol)
    if chord is None:
        return "N", "N"
    return chord.root, chord.quality


def diatonic_fraction(events, key: str | None) -> float:
    if not key:
        return float("nan")
    mode_minor = key.endswith("m")
    tonic = PITCH.index(key[:-1] if mode_minor else key)
    table = MINOR if mode_minor else MAJOR
    total = 0.0
    good = 0.0
    for e in events:
        dur = e.end - e.start
        root, quality = split_symbol(e.symbol)
        if root not in PITCH:
            total += dur
            continue
        degree = (PITCH.index(root) - tonic) % 12
        expected = table.get(degree)
        total += dur
        if expected is not None and (expected == quality or quality in {"7", "maj7", "m7"}):
            good += dur
    return good / total if total else float("nan")


def root_stability(events) -> dict:
    roots = [split_symbol(event.symbol)[0] for event in events]
    qualities = [split_symbol(event.symbol)[1] or "maj" for event in events]
    total = max(1, len(roots))
    first = roots[0] if roots else None
    same = sum(1 for root in roots if root == first and first not in {None, "N"})
    unique = {root for root in roots if root not in {None, "N"}}
    return {
        "fraction_root_equals_first": round(same / total, 4),
        "n_unique_roots": len(unique),
        "root_changes": sum(
            1 for index in range(1, len(roots)) if roots[index] != roots[index - 1]
        ),
        "first_symbol": events[0].symbol if events else None,
        "mMaj7_share": round(qualities.count("mMaj7") / total, 4),
        "power_share": round(qualities.count("5") / total, 4),
    }


def summarise(name: str, events, key: str | None, bar_sec: float, duration: float):
    segs = len(events)
    quality = Counter(split_symbol(e.symbol)[1] or "maj" for e in events)
    stability = root_stability(events)
    print(
        f"  {name:26s} segs={segs:4d} segs/bar={segs / max(duration / bar_sec, 1e-9):5.2f} "
        f"diatonic={diatonic_fraction(events, key):5.2f} "
        f"root1={stability['fraction_root_equals_first']:.3f} "
        f"roots={stability['n_unique_roots']} "
        f"vocab={dict(sorted(quality.items(), key=lambda kv: -kv[1]))}"
    )
    return {
        "segs": segs,
        "diatonic": diatonic_fraction(events, key),
        "quality": dict(quality),
        "root_stability": stability,
    }


def mix_stems(paths: list[Path], out: Path, sr: int = 44100) -> Path | None:
    import librosa

    ys = []
    for p in paths:
        if p.is_file():
            y, _ = librosa.load(str(p), sr=sr, mono=True)
            ys.append(y)
    if not ys:
        return None
    n = min(len(y) for y in ys)
    y = sum(v[:n] for v in ys) / len(ys)
    sf.write(out, y.astype(np.float32), sr)
    return out


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else
               "data/ui_runs/46b84e7b-dbb5-4c25-817a-53c4cd85b02d")
    analysis = json.loads((run / "analysis.json").read_text())
    diag = json.loads((run / "metronome_diagnostics.json").read_text())
    beats = np.asarray(analysis["beats"], dtype=float)
    key = analysis.get("key")
    bar_sec = 60.0 / float(analysis["bpm"]) * int(analysis.get("beats_per_measure") or 4)
    duration = float(analysis["duration_sec"])

    print(f"run={run.name} key={key} bpm={analysis['bpm']} meter={analysis.get('beats_per_measure')} "
          f"beats={len(beats)} bar={bar_sec:.2f}s duration={duration:.1f}s")
    print(f"metronome: source={diag.get('source')} body_start={diag.get('body_start_sec'):.3f} "
          f"downbeat={diag.get('downbeat_sec')}")

    tmp = Path(tempfile.mkdtemp(prefix="realchord-"))
    inputs = {}
    src = run / "source.mp3"
    if src.is_file():
        inputs["full mix (what ran)"] = src
    if (run / "guitar.wav").is_file():
        inputs["guitar stem"] = run / "guitar.wav"
    if (run / "piano.wav").is_file():
        inputs["piano stem"] = run / "piano.wav"
    if (run / "bass.wav").is_file():
        inputs["bass stem"] = run / "bass.wav"
    harm = mix_stems([run / "guitar.wav", run / "piano.wav", run / "bass.wav"], tmp / "harmony.wav")
    if harm:
        inputs["guitar+piano+bass"] = harm
    novoc = mix_stems([run / "guitar.wav", run / "piano.wav", run / "bass.wav", run / "drums.wav"],
                      tmp / "novoc.wav")
    if novoc:
        inputs["no vocals (+drums)"] = novoc

    print("\n-- harmony input A/B (same beat grid)")
    rows = {}
    for name, path in inputs.items():
        ev = detect_chords(path, beats=beats, bpm=analysis["bpm"])
        rows[name] = summarise(name, ev, key, bar_sec, duration)

    print("\n-- beat-grid sensitivity (full mix)")
    for shift_beats in (-0.5, -0.25, 0.25, 0.5):
        shifted = beats + shift_beats * (60.0 / float(analysis["bpm"]))
        shifted = shifted[(shifted >= 0) & (shifted < duration)]
        ev = detect_chords(src, beats=shifted, bpm=analysis["bpm"])
        summarise(f"beats {shift_beats:+.2f} beat", ev, key, bar_sec, duration)

    print("\n-- vocabulary sanity vs key (full mix)")
    events = detect_chords(src, beats=beats, bpm=analysis["bpm"])
    outside = [e.symbol for e in events if not _is_diatonic(e.symbol, key)]
    print(f"  non-diatonic chord events: {len(outside)}/{len(events)} "
          f"({100.0 * len(outside) / max(len(events), 1):.0f}%)")
    print(f"  examples: {Counter(outside).most_common(12)}")

    out = Path(__file__).with_name("real_track_diag.json")
    out.write_text(json.dumps({"run": str(run), "key": key, "inputs": rows}, indent=2))
    print(f"\nwrote {out}")


def _is_diatonic(symbol: str, key: str | None) -> bool:
    if not key:
        return False
    mode_minor = key.endswith("m")
    tonic = PITCH.index(key[:-1] if mode_minor else key)
    root, quality = split_symbol(symbol)
    if root not in PITCH:
        return False
    degree = (PITCH.index(root) - tonic) % 12
    table = MINOR if mode_minor else MAJOR
    expected = table.get(degree)
    if expected is None:
        return False
    return expected == quality or quality in {"7", "maj7", "m7"}


if __name__ == "__main__":
    main()
