# Powerchord accuracy diagnosis

**Date:** 2026-10-28. Template stack only. No new model.

## 1. Path

1. Desktop writes `harmony_sum.wav` from guitar+piano+other (`ui/isolate_jobs.py` `_write_harmony_sum`, names at line 1416). Bass is not in that sum. `bass_path` is slash-only (`chords.py` `detect_chords`).
2. HPSS + tuned CQT chroma, hop 512 (`detect_chords`).
3. Dot with 15×12 L2 templates (`_build_templates`, `_QUALITY_TONES`). Quality `5` is root+fifth only.
4. Z-score per window, subtract `_quality_prior_for_window` and `_key_prior` (`detect_chords` around the score loop).
5. `_viterbi` with change penalty 0.95 off the downbeat and 0.10 on it. Do not retune those.
6. `_absorb_short_runs` folds only under half a beat. One-beat absorb is blocked (clean root 0.896).
7. UI shows `displaySymbol` at the chart level, which keeps `C5` (`chord_symbols.level_symbol`). Demo note stays in the practice sheet.

## 2. Why a powerchord became a triad

Measured on a synthetic C2+G2 (no third), provisional key C:

| Symbol | Acoustic z | Quality prior | Key prior | Final |
|---|---|---|---|---|
| C5 | **3.15** | 0.70 | 0.00 | 2.45 |
| C | 2.36 | 0.00 | **−0.20** | **2.56** |

`C5` wins the template match by 0.79σ. It loses because `"5"` is penalised 0.70 (`_QUALITY_PRIOR`) and the tonic triad gets a diatonic bonus the fifth does not (`_MAJOR_DIATONIC` has no `"5"`). Net: **C**, not C5. A distorted powerchord with a quiet major-third partial (E/root ≈ 0.09) still lost the same way. A real C major triad (E/root ≈ 0.72) keeps C; C5’s final score there is 1.58 vs C at 3.09.

## 3. Real mixes (saved beats, after the bare-fifth relaxation)

Before maj7 is `bench/after_metrics.json`. After is `bench/powerchord_after_metrics.json`.

| Case | maj7 before → after | sus4 after | power `5` after | segs/bar | key |
|---|---|---|---|---|---|
| 50f1 metal mix | 0.174 → **0.157** | 0.014 | **0.043** | 0.821 | D#m |
| 9a18 | 0.543 → 0.543 | 0.000 | 0.000 | 1.640 | D# |
| ad6e | 0.435 → **0.421** | 0.000 | 0.031 | 1.746 | Am |
| waltz source | 0.168 → 0.167 | 0.007 | 0.073 | 2.732 | Dm |

50f1 is the powerchord-like song. Only 4% of events became `5`. The rest still have a third in the chroma (distortion partials, other instruments, vocals), so the bare-fifth gate does not fire and maj7 stays the wrong quality. 9a18 did not move.

## 4. Causes

| Rank | Cause | Evidence | Class |
|---|---|---|---|
| 1 | `"5"` prior 0.70 plus no diatonic slot, so a bare fifth loses to the triad by ~0.11σ | Synthetic table above | **FIXABLE-NOW** (shipped: relax only when both thirds < 0.25× root and the fifth ≥ 0.35× root) |
| 2 | Distorted / mixed powerchords still put energy on the major third, so maj/maj7 templates stay legal | 50f1 power share 0.043 after the fix; E partial on the distorted synthetic was already 0.09× root | **NEEDS-MODEL** (a fifth-vs-third timbre model). Do not raise the 0.25 cutoff without a new floor measurement |
| 3 | maj7 relaxation is an OR and real triads cross 0.32× root | Prior session: AND dropped realistic sevenths to 0.889. 9a18 maj7 stayed 0.543 | **BLOCKED-BY-FLOOR** |
| 4 | Chops (segs/bar) | One-beat and same-root absorb both scored clean root 0.896. Waltz still 2.73 | **BLOCKED-BY-FLOOR** |

Rejected and not retried: bass bonus > 0, on-downbeat penalty 0.70, seventh cutoff 0.45, flat maj7 prior 0.55.

## 5. Lever result

Only lever shipped: bare-fifth prior relaxation in `_quality_prior_for_window`. Constant `"5": 0.70` is unchanged for windows that have a third.

Synthetic gate (`synth_bench.py`, GT beats) after the change: clean root/majmin/sevenths **1.000 / 1.000 / 1.000**; realistic **1.000 / 1.000 / 0.979**. Floors hold.

Further powerchord accuracy on a full distorted mix needs a model that can call a fifth a fifth while a harmonic third is present. That is outside this stack.
