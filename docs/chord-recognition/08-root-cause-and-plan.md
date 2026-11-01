# 08 — Root-cause diagnosis and plan (root-right then wander)

**Date:** 2026-10-28.  
**Scope:** Template chord detector only (`src/audio_to_tab/chords.py`). Diagnosis pass changed
no product code; execution of Phases 1–3 landed afterward (see **Execution status** below).  
**Inputs:** this diagnosis; prior docs `00`, `01`, `06`, `07`; scratch drivers under  
`docs/chord-recognition/bench/scratch-chord-diagnosis/` (wavs + JSON; not product code).  
**Baseline tests:** `python -m pytest tests/ -q -k chord` → **65 passed**, 1092 deselected, 0 skips unexpected.

## Execution status

| Phase | Status | What landed |
|---|---|---|
| 1 Metrics + copy | Done | `root_stability` in `bench/score_chords.py` + `real_track_diag.py`; demo note mentions root drift; no `add9` detector state documented in `_MEDIUM` comment |
| 2 Display vocabulary | Done | Medium collapses `maj7`/`mMaj7`→triad; default UI level is `medium` (chart/advanced still full) |
| 3 Harmony source | Done | Chord harmony sum is guitar(+piano) only — kind `guitar_piano_sum` / `guitar_sum`; `other` excluded |
| 4 Scorer levers | **Measured, not shipped** | Root-change Viterbi (+0.25σ) + mild key prior held synth floors but **cut 50f1 guitar first-root retention 0.335→0.157**. Reverted. Evidence: `bench/phase4_after_ship_candidate.json`, `bench/scratch-chord-diagnosis/phase4_measure.json` |
| 5 Neural model | **Opt-in shipped** | Vendored BTC MIT code under `src/audio_to_tab/btc/`; weights SHA256-pinned via `chord_neural.ensure_chord_model`; enable with `ATT_CHORD_MODEL=1` (default remains templates). Training-audio provenance undocumented — keep off for redistribution-sensitive builds |

---

## Executive summary

**Why does the detector get the first root roughly right, then wander?**

On **clean or lightly distorted powerchords** it does **not** wander: a held synthetic G♯5
stays `G#5` for the whole clip, with or without a wrong provisional key. A clean G♯→A♯→C♯→G♯
power progression tracks roots correctly and keeps quality `5`. The bare-fifth prior
relaxation from `06` works when thirds are truly quiet (maj3/min3 ≪ 0.25× root).

On **real full mixes** (and still badly on guitar stems), the failure is different and worse
than “quality drift on a stable root”:

1. **The root itself wanders.** On metal mix `50f1` (`current_mix.wav`, saved beats, key
   `D#m`), only **5.7%** of merged events share the first event’s root; **11 unique roots**
   appear. Qualities are dominated by `mMaj7` (45/70) and `maj7` (11/70), not by `5` (2/70).
2. **Acoustic scoring already wanders before Viterbi.** Per-window greedy argmax on the same
   mix already has a wild root histogram (A♯ 152 / G♯ 110 windows among 331). Viterbi
   disagrees with greedy on **26%** of windows but does not invent the chaos.
3. **Extended qualities win because the chroma is dense**, not because of a first-window
   privilege. Windows labelled `mMaj7`/`maj7` show third bins ~0.3–0.9× root and maj7 bins
   often ≥0.8× root (confirms `07`). The seventh prior then **relaxes** (OR gate), so
   `maj7`/`mMaj7` are almost free. There is **no `add9` detector state** — the user’s
   “CMaj7add9” cannot be emitted; closest real emissions are `maj7` / `mMaj7` / `sus2`,
   sometimes with a loud ninth bin that the ear reads as “add9”.
4. **Key / transposition error is not the driver.** Forcing provisional key `C` vs `D#m`
   changes the first label (`EmMaj7` vs `FmMaj7`) and shuffles counts slightly; the mix still
   collapses to A♯/G♯-heavy `mMaj7` spam. A held G♯5 with key forced to `C` still exits as
   `G#5`.

Plain English: the chart looks “root-right then lost” because early bars sometimes land on a
plausible root by chance or because the intro is thinner, while later beat windows on a
distorted multi-instrument mix look like many competing jazz voicings to a 12-bin template
matcher. The decoder faithfully follows those windows.

---

## Decision-chain map

Pipeline entry: `analysis.run_song_pipeline` → `provisional_key_from_audio(harmony)` →
`detect_chords(harmony, beats, bpm, provisional_key, beats_per_measure, downbeats, bass_path)`.

```
audio (harmony_sum or mix)
  → load 22.05 kHz mono
  → estimate_tuning_semitones
  → HPSS → harmonic
  → chroma_cqt (hop 512, bins_per_octave 36, tuning applied, norm=None)
  → mean chroma per beat window (+ separate bass chroma for slash only)
  → L2-normalise window → z-score(dot with 15×12 weighted templates)
  → − quality prior (window-adaptive) − key prior (+ bass bonus, currently 0)
  → approx 2-best Viterbi with metrical change penalties
  → slash bass, confidence, absorb <0.5 beat, merge identical neighbours
  → ChordEvent.symbol (+ simple/medium via level_symbol)
```

### Stage knobs (current values that can flip a label)

| Stage | Knob | Current value | Effect |
|---|---|---|---|
| Source | Harmony file | guitar+piano+other sum, else mix (`isolate_jobs`) | Vocals/drums/bleed densify chroma |
| Source | `bass_path` | Slash only; `_BASS_ROOT_BONUS = 0` | Cannot rename root |
| Chroma | `sr` / hop / BPO | 22050 / 512 / 36 | ~23 ms frames; beat mean ~0.4–0.7 s |
| Chroma | HPSS | always | Removes some percussion; not distortion thirds |
| Chroma | Tuning | estimated, applied | Mis-tune smears bins across roots |
| Templates | Qualities | 15 incl. `5`, `maj7`, `mMaj7`, sus, dims | Large vocabulary; no `add9`/`maj9` |
| Templates | Tone weights | root 1.0, 3rd 0.90, 5th 0.70, 7th 0.55, … | Seventh still influential when present |
| Scores | Z-norm | per window mean/std | Priors are in σ units |
| Quality prior | Base `_QUALITY_PRIOR` | `5`: **0.70**, `maj7`: **0.55**, `mMaj7`: **0.50**, `sus4`/`sus2`: 0.50, `m7`: 0.15, maj/min/7: 0.0 | Bare fifth starts behind triad by design |
| Quality prior | Seventh relax | if seventh ≥ **0.32× root** **OR** (third ≥0.45×root and seventh ≥0.35×third) → prior min 0.05 | Overtone/mix sevenths often pass OR |
| Quality prior | Bare-fifth relax | fifth ≥0.35×root **and** both thirds **&lt;0.25×root** → `"5"` prior 0.05 | Shipped in `06`; rare on full mixes |
| Key prior | Diatonic | −0.20 if (degree, quality) in `_MAJOR/_MINOR_DIATONIC` | `"5"` **not** in diatonic tables |
| Key prior | Outside | +0.10 | Weak vs ~2σ acoustic margins |
| Silence | norms / RMS | `_SILENCE_NORM=1e-4`, RMS&lt;1e-4 → `N` | |
| Viterbi | Off-downbeat | `_CHANGE_PENALTY(0.8)+_OFF_DOWNBEAT(0.15)=0.95` | Prefers stay |
| Viterbi | On-downbeat | `0.8−0.70=0.10` | Easy change on bar |
| Absorb | Short runs | only &lt; **0.5×** beat; same-root one-beat absorb **blocked** (clean root 0.896) | Chops remain |
| Display | `level_symbol` | chart keeps detector symbol; simple/medium collapse extensions | Cannot invent `add9` |
| Symbols | `DETECTED_QUALITIES` | no add9/maj9 | `parse_render` rejects them as detector output |

Note: `01-current-system-audit.md` describes an **older** detector (no HPSS, no key prior,
binary templates, tiny change penalty). Prefer this table and `06`/`07` over `01` for knobs.

### What chord tests actually assert (`tests/test_chords.py` + related)

Happy path / contracts, not real-mix WCSR:

- Symbol normalisation, snap-to-beat, merge, silence/missing paths, progress callback.
- Synthetic clean progression detects expected roots/qualities.
- Maj7 prior: overtone-only seventh prefers triad; sounded maj7 stays.
- **Powerchord without a third stays a fifth** (bare-fifth gate).
- Bass non-chord tone does not rename root; bass third may slash.
- Absorb: sub-half-beat flicker merges; same-root quality flicker under one beat **stays**;
  different root inside a beat kept.
- Confidence not constant; silence → no chords / `N` behaviour.

They do **not** assert: real-mix root stability, powerchord share on distorted mixes, or
absence of `mMaj7` spam.

---

## Diagnosed failure modes (ranked)

### FM1 — Dense real-mix chroma makes many roots and jazz qualities win acoustically  
**Confidence: high**

**Mechanism:** Distortion + multi-instrument harmony put energy on thirds, sevenths, and
often ninths relative to many candidate roots. Weighted templates for `mMaj7`/`maj7` match
that vector; the seventh OR-relax drops their prior to ~0.05σ.

**Evidence (Phase 1):**

- `50f1` mix merged events: `mMaj7` 45, `maj7` 11, `5` only 2 (`diagnosis_report.json`).
- Window 0 greedy top: `FmMaj7` 2.182 vs `EmMaj7` 2.168 (margin **0.014σ**); ratios vs F:
  min3 0.863, maj7 1.185, fifth **0.229** — fifth is the weak bin (matches `07`).
- Greedy quality hist before Viterbi: `mMaj7` **223 / 331** windows.
- Median top1−top2 margin on first 20 mix windows ≈ **0.23σ**; several windows &lt;0.02σ.

**Explains:** “Goes everywhere with occasional right chord”; extended labels on “simple”
material; user’s jazz-looking symbols (not literal `add9`).

**vs 06/07:** Confirms `07` (voiced third+seventh, weak fifth). Extends it: **root** competition
is as bad as quality (see FM2).

---

### FM2 — Root wander after the first event (not quality-only drift)  
**Confidence: high**

**Mechanism:** When many roots score within ~0.2σ, beat-to-beat winners jump. Merge collapses
runs but preserves root changes. User hears “first root ok, then wrong roots”.

**Evidence:**

- Mix: `fraction_root_equals_first = 0.057`, `unique_roots = 11`, first20 starts
  `FmMaj7, D#maj7, G5/D, G5, F#mMaj7, …`.
- Guitar stem better but still bad: first-root fraction **0.335**, unique roots 11,
  first20 `Cm7, Asus4, D#7, G#maj7, EmMaj7, …` — here a `G#maj7` appears mid-clip while
  neighbouring roots are C/D♯/E (the “G♯ power → wrong colour / wrong neighbour” story).
- Clean held G♯5: `root_changes=0`, `same_root_quality_changes=0` → symptom is
  **timbral/mix**, not a general “first privileged then decoder drifts” bug.

**Explains:** Primary user wording (“after that it just goes everywhere”).

**vs 06/07:** **New emphasis.** `06`/`07` focused on quality `5` vs triad/maj7; they understated
cross-root flipping on full mixes.

---

### FM3 — Seventh prior OR-relax + maj7/mMaj7 in the template set  
**Confidence: high**

**Mechanism:** `_quality_prior_for_window` sets maj7-family prior to ≤0.05 when
`seventh ≥ 0.32× root` **OR** voiced-third path. Real mix windows routinely clear the first
clause (`07`: 48/48 `mMaj7` windows on 50f1). Templates include those qualities with seventh
weight 0.55.

**Evidence:** Prior code path + `07` ratios + Phase-1 greedy `mMaj7` dominance. Synthetic
AND-relax previously dropped realistic sevenths WCSR to **0.889** (blocked in `06`).

**Explains:** Why add9/maj7-*like* labels appear on powerchord-like songs.

**vs 06/07:** Confirms; still **BLOCKED-BY-FLOOR** for simple AND tightening.

---

### FM4 — `"5"` structurally disadvantaged when thirds are present  
**Confidence: high (already shipped partial fix)**

**Mechanism:** Base prior 0.70; `"5"` absent from `_MAJOR/_MINOR_DIATONIC` (no −0.20). Bare-fifth
relax needs both thirds &lt;0.25× root. Distorted/mixed powerchords fail that gate (`06` power
share after fix still **0.043** on 50f1).

**Evidence:** Synthetic bare fifth now wins (`test_powerchord_without_a_third_stays_a_fifth`;
Phase-1 G♯5 z≈3.14, q=0.05, final≈3.09 vs triad ≈2.35). Real mix `5` share stays tiny.

**Explains:** Powerchords labelled as triads/sevenths once any third energy exists.

**vs 06/07:** Confirms; mid-band levers in `07` measured **BLOCKED**.

---

### FM5 — Harmony source is often the full mix / dense stems  
**Confidence: medium–high**

**Mechanism:** Chord stage scores whatever `harmony_path` is. Mix includes drums bleed,
vocals, saturated guitars → denser chroma than a mono powerchord DI.

**Evidence:** Same beats/key: mix first-root stability 0.057 vs guitar 0.335; mix `mMaj7`
share higher. `00`/`01` already documented fold/`source.mp3` fallthrough; this run used
`current_mix.wav` and `guitar.wav` explicitly.

**Explains:** Gap between “simple G♯ power” mental model and detector input.

---

### FM6 — Viterbi / absorb only partially shape the chaos  
**Confidence: medium**

**Mechanism:** Off-downbeat stay penalty 0.95σ helps when margins are small, but with
greedy already wrong, smoothing preserves wrong roots. Absorb cannot merge one-beat
same-root quality flips (blocked: clean root 0.896).

**Evidence:** Greedy≠Viterbi on 26% (mix) / 37% (guitar) of windows; Viterbi quality hist
still `mMaj7`-dominated. Not the root cause of wander.

---

## What is NOT the cause (eliminated)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Quality drifts while root stays fixed | **Rejected as primary** | Mix first-root retention 5.7%; 11 roots |
| First window specially initialised / anchored | **Rejected** | No first-only bonus in code; held G♯5 stable all windows; window 0 already contested (0.014σ) |
| Wrong key / transposition shifts everything after chord 1 | **Rejected as primary** | Key `C` vs `D#m` vs `None`: same A♯/`mMaj7` regime; held G♯5 + key `C` → still `G#5` |
| Detector emits `add9` / `CMaj7add9` | **Impossible today** | Not in `DETECTED_QUALITIES` / templates; `parse_render` rejects |
| Clean powerchords always mislabelled extended | **Rejected after `06` fix** | Phase-1 clean & distort synth → `G#5`; progression all `5` |
| Viterbi alone creates wander | **Rejected** | Greedy histograms already chaotic |
| Bass path renaming roots | **Rejected** | Bonus 0; slash-only; tests lock this |

---

## Hypotheses still open

| Open | Why unsettled | Experiment to settle |
|---|---|---|
| H-O1: User’s specific “G♯ → CMaj7add9” track | No hand GT for that song; 50f1 is a stand-in metal mix | Ask for the file or annotate 30 s of 50f1 / user’s clip with expected roots |
| H-O2: How often “first root right” is real vs luck | No GT; first mix label is `FmMaj7` under key `D#m` | Hand-label first 8 bars of 2–3 songs; measure first-root accuracy vs later |
| H-O3: Tuning error contribution on detuned live takes | Held synth was in tune | Detune synth ±30–50¢ and re-score root stability |
| H-O4: Whether `harmony_sum` (guitar+piano+other) beats guitar-only for metal | Only mix vs guitar compared here | Re-run `detect_chords` on written `harmony_sum.wav` when present in a run |
| H-O5: Neural free model ceiling | Out of scope for template levers | Spike BTC / similar on same 50f1 beats; compare root accuracy only |

---

## Fix options (proposals only — do not ship from this doc)

### Against FM1 / FM3 (dense chroma → maj7/mMaj7)

| Option | Expected effect | Blast radius | Complexity | Depends on |
|---|---|---|---|---|
| A1. Restrict chart vocabulary (hide maj7/mMaj7 at simple/medium; optional “pop” detect set) | Fewer scary symbols; roots may still wander | Advanced users lose colour | Low (symbols / detect list) | Product decision on levels |
| A2. Stronger seventh gate (AND, or raise 0.32) | Cut false maj7 | Realistic sevenths floor (&lt;0.90 before) | Low | New synth floor campaign |
| A3. Prefer triad/`5` when fifth ≫ seventh or fifth ≫ third | Help power-like windows | May hurt jazz repertoire | Medium | Careful bench on 9a18/ad6e |
| A4. Timbre / HPSS / multi-f0 front-end beyond 12-bin | Separate partial thirds from voiced thirds | Large; new failure modes | High | Research spike |
| A5. Free neural chord model (BTC-class) | Likely best root+quality jump | License/deps; still needs beats | High | User OK on model |

### Against FM2 (root wander)

| Option | Expected effect | Blast radius | Complexity | Depends on |
|---|---|---|---|---|
| B1. Stronger key prior / tonic stickiness | Fewer remote roots | Wrong-key songs get worse | Low–med | Better provisional key |
| B2. Higher change penalty / root-change vs quality-change costs | Fewer root flips | Late changes sluggish; synthetic root risk | Med | Synth floors |
| B3. Two-stage decode: root then quality | Stabilize roots first | Needs redesign of templates/priors | High | — |
| B4. Harmony source: guitar-only or no-vocals for metal profiles | Cleaner chroma | Miss piano harmony | Med | Source selection UX |

### Against FM4 / FM5 (powerchords / source)

| Option | Expected effect | Blast radius | Complexity | Depends on |
|---|---|---|---|---|
| C1. Raise bare-fifth third cutoff (e.g. 0.25→0.40) | More `5` on mild distortion | Real triads → `5` | Low | **Rejected without new floor** in `07` |
| C2. Always prefer guitar stem when present for chord stage | Better on metal | Worse on band piano songs | Med | Product |
| C3. Demo/WIP copy (already present) | Sets expectation | No accuracy gain | Done | — |

### Against FM6 (smoothing)

| Option | Expected effect | Blast radius | Complexity | Depends on |
|---|---|---|---|---|
| D1. Same-root quality absorb ≥1 beat | Fewer maj↔maj7 flickers | Clean root WCSR cratered before | Low | **BLOCKED** unless new evidence |
| D2. On-downbeat penalty retune | Bar-aligned changes | Previously hurt slash fixture | Low | **BLOCKED** at 0.70 bonus path |

---

## Validation strategy

1. **Keep synthetic floors** (`bench/synth_bench.py` GT beats): clean root/majmin ≥0.995;
   realistic majmin ≥0.990; sevenths ≥0.900. Any prior/absorb change that breaks these is a
   no-ship (locked by history in `06`).
2. **Powerchord unit:** `test_powerchord_without_a_third_stays_a_fifth` must stay green;
   add (when fixing) distorted-with-voiced-third cases only with explicit expected behaviour.
3. **Real-mix pins:** re-score `docs/chord-recognition/bench/band_mix_before.json` tracks
   (50f1, 9a18, ad6e, waltz) for: maj7/mMaj7 share, `5` share, segs/bar, **new:**
   `fraction_root_equals_first` and `n_unique_roots` (Phase-1 metrics).
4. **Scratch regression:** re-run
   `docs/chord-recognition/bench/scratch-chord-diagnosis/diagnose_root_drift.py` —
   held G♯5 must remain `G#5`; progression roots must track.
5. **Test files likely touched when implementing:** `tests/test_chords.py` (priors, absorb,
   vocabulary); possibly `tests/test_chord_symbols.py` if display levels change; **not** a
   substitute for real-mix JSON pins.
6. **Do not claim WCSR** on real music until ≥5 tracks are hand-annotated (`00`).

---

## Phased implementation order

Each phase should be planned and measured alone. **No phase is approved for coding by this
document alone.**

### Phase 1 — Lowest risk, highest evidence (product / measurement)
- Add root-stability metrics to the existing real-mix bench harness (first-root retention,
  unique roots, greedy vs Viterbi disagree rate).
- Confirm / document that chart cannot show `add9`; align demo copy with “roots may drift on
  dense mixes” if product wants clearer WIP language.
- Optional: prefer guitar (or harmony_sum) explicitly in diagnostics so users/dev see source.

**Why first:** zero scorer risk; makes later fixes measurable against FM2.

### Phase 2 — Vocabulary / display without touching priors
- Map advanced→medium→simple so `mMaj7`/`maj7` collapse more aggressively for default chart
  level; keep detector states for advanced.
- Measure: does user-perceived “wander” drop when qualities collapse but roots still jump?
  If yes, still need Phase 3/4 for roots.

### Phase 3 — Harmony source selection
- Prefer guitar-only or guitar+piano for chord stage when metal / high-distortion profile;
  keep piano+other for jazz/pop profiles.
- Re-score 50f1/9a18; expect higher first-root retention on guitar (Phase-1 saw 0.335 vs 0.057).

### Phase 4 — Scorer changes only with floor gates
- Candidates: mild key-prior strengthening; root-vs-quality transition costs; carefully
  designed fifth-vs-seventh preference **without** repeating blocked mid-band/`AND` levers
  unless floors are re-baselined.
- Ship only if synth floors hold **and** 50f1 first-root retention rises without exploding
  segs/bar on waltz.

### Phase 5 — Model path (if product accepts)
- Free neural chord scorer behind `model_dir`, template fallback unchanged.
- Success metric: root accuracy on annotated clips; maj7 spam secondary.

---

## Open questions for the user

1. **How prominent is the chord chart?** Is “fewer wrong scary symbols” (display collapse)
   enough for demo, or do you need correct roots on metal powerchord songs?
2. **Willingness to add a free neural model** (deps, disk, license diligence) vs staying
   template-only?
3. **Can you share the exact clip** that showed “G♯ power → CMaj7add9”, or annotate 8 bars
   of 50f1 / similar, so Phase-1 open hypotheses H-O1/H-O2 close?
4. **Default harmony source:** guitar-only for isolation jobs that look like rock/metal?
5. **Simple vs advanced vocabulary:** should the default chart ever show `maj7` / `mMaj7`?

---

## Appendix A — Phase-1 baseline commands and artifacts

```text
PYTHONPATH=src:. .venv311/bin/python -m pytest tests/ -q -k chord
# 65 passed, 1092 deselected

PYTHONPATH=src:. .venv311/bin/python \
  docs/chord-recognition/bench/scratch-chord-diagnosis/diagnose_root_drift.py
PYTHONPATH=src:. .venv311/bin/python \
  docs/chord-recognition/bench/scratch-chord-diagnosis/diagnose_window_trail.py
```

Artifacts (scratch only):

- `…/scratch-chord-diagnosis/diagnosis_report.json`
- `…/scratch-chord-diagnosis/window_trail_report.json`
- `gs_power_clean.wav`, `gs_power_distort.wav`, `gs_as_cs_power_prog.wav`

## Appendix B — Answers to the four concrete questions

1. **Root stable, quality drifts?** No on real mixes — **root wanders** (5.7% first-root
   retention on 50f1 mix). On clean powerchords, both stay put.
2. **Why add9/maj7 on powerchord-like material?** Dense chroma + maj7 templates + seventh
   OR-relax; `add9` is not a detector label. Trace: chroma → z-scores → prior 0.05 on
   maj7-family → Viterbi follows.
3. **First chord privileged?** Not by initialization. Early windows can be slightly less
   contested by chance; there is no special anchor for the rest.
4. **Transposition/key errors?** Minor label nudges only; not the cause of post-first wander.

## Appendix C — Confirm vs new vs prior docs

| Finding | Status |
|---|---|
| `"5"` prior 0.70 + no diatonic slot | Confirm `06` (relax shipped) |
| Distorted mix thirds block bare-fifth gate | Confirm `06`/`07` |
| 50f1 `mMaj7` with strong seventh, weak fifth | Confirm `07` |
| Mid-band / weak-seventh levers blocked | Confirm `07` |
| Root wander as dominant user symptom | **New (this doc)** |
| Greedy already chaotic (Viterbi secondary) | **New** |
| `add9` impossible; user symbol is approximate | **New** |
| Key force ≠ wholesale transpose | **New measurement** |
| Clean G♯5 / progression OK after bare-fifth fix | Confirm `06` + remeasured |
