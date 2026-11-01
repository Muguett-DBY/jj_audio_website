# Distorted powerchords and leftover maj7

**Date:** 2026-10-28. Measured on the current detector (bare-fifth relaxation already shipped). No product code change in this pass.

## Synthetic distorted fifth

`/tmp/dist_power_strong.wav`: C3+G3 plus octaves and the 5th harmonic (E5). Detected **C5**. Ratios vs that root: major third **0.122**, fifth **0.622**, major seventh **0.058**. The third is still under the 0.25× root bare-fifth gate, so this “distorted” tone is not the blocked case.

## Real windows (saved beats, ratios vs the labelled root)

| Set | n | med min3/root | med maj3/root | med fifth/root | med maj7/root | maj7 ≥ 0.32 and ≥ 0.8× fifth |
|---|---|---|---|---|---|---|
| 50f1 `mMaj7` | 48 | 0.504 | 0.343 | **0.262** | **0.792** | **48/48** |
| 50f1 `maj7` | 11 | 0.540 | 0.734 | 0.448 | 0.874 | 10/11 |
| 50f1 `5` | 3 | 0.147 | 0.109 | 1.591 | 0.310 | 0/3 |
| 9a18 `maj7` | 126 | 0.326 | 0.375 | 0.735 | **0.816** | seventh is as loud as the fifth |

50f1’s main error is not a quiet-third powerchord. 48 of 70 events are `mMaj7`. On those windows the minor third is about half the root and the major seventh is louder than the fifth. The fifth is the weak bin.

## Candidate levers

| Lever | What the windows do | Class |
|---|---|---|
| Prefer `5` when max(third)/root is 0.25–0.45 and third < 0.75× fifth | 50f1 maj7: **0/11**. 50f1 mMaj7 third is ~0.50, outside the band. 9a18 maj7: **62/126**, but those windows have maj7/root 0.816, so they are sevenths, not powerchords. Shipping this would relabel real sevenths as fifths. | **BLOCKED** — measured, not shipped |
| Refuse maj7 when the seventh is loud vs the root but weak vs the fifth | 50f1 mMaj7 and maj7 have the seventh **stronger** than the fifth (48/48 and 10/11). The gate does not see them. | **BLOCKED** — would be a no-op on the failure |
| AND the seventh relaxation, one-beat absorb, bass bonus, cutoff 0.45 | Already failed the synthetic floor | **BLOCKED-BY-FLOOR** |
| Separate a distorted fifth from a voiced third+seventh | Chroma at the chosen root already contains both a third and a seventh at triad strength. Templates are matching that vector. | **NEEDS-MODEL** |

## Result

No lever in the allowed set moves 50f1 power share above 0.043 or 9a18 maj7 below 0.543 without relabelling windows whose seventh bin is ~0.8× the root. The template stack cannot call those powerchords. A free neural chord model, or a different front-end than 12-bin chroma, is what is left.
