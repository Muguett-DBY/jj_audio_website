# Chord diagnosis worklog (scratch, uncommitted)

Task: user symptom — "root of first chord is usually right; after that labels go everywhere;
G# power chord → CMaj7add9-ish". Prompt: diagnose root cause (NO fixes), then write
docs/chord-recognition/08-root-cause-and-plan.md.

## Checklist
- [x] Read AGENTS.md / PROJECT_TREE.md
- [x] Read docs/chord-recognition 00, 01, 06, 07
- [ ] Read 02, 03, 04, 05 (skim for deltas)
- [ ] Read src/audio_to_tab/chords.py end to end
- [ ] Read src/audio_to_tab/chord_symbols.py (vocabulary, level_symbol)
- [ ] Read chord path in analysis.py + ui/isolate_jobs.py (_run_analysis_stage, _write_harmony_sum)
- [ ] Baseline: pytest -k chord
- [ ] Inventory bench/ artifacts + data/ui_runs (50f1, 9a18, ad6e, waltz)
- [ ] Reproduce G#5 symptom with evidence trail (chroma/scores/priors/Viterbi/final symbol)
- [ ] Rank failure modes; separate confirms of 06/07 vs new
- [ ] State eliminated causes
- [ ] Write docs/chord-recognition/08-root-cause-and-plan.md

## Findings (running log)

(doc 00) Decoder 0.997 WCSR on clean synthetic → decoder not the bottleneck. Real mixes: maj7
40-66% of events, 43-56% non-diatonic, segs/bar 0.85-2.64. (doc 01, 2026-10-25)
(doc 06, 2026-10-28) Current path: harmony_sum = guitar+piano+other (bass excluded), HPSS+tuned
CQT, 15x12 templates, z-scored priors (_quality_prior_for_window, _key_prior), Viterbi 0.95
off-downbeat / 0.10 on-downbeat, _absorb_short_runs < half beat. Powerchord lost to triad because
"5" prior 0.70 + no "5" diatonic slot. Shipped: bare-fifth relaxation (thirds < 0.25x root AND
fifth >= 0.35x root). Result: 50f1 maj7 0.174→0.157, power "5" only 0.043.
(doc 07, 2026-10-28) 50f1 main error is mMaj7 (48/70 events); med maj7/root 0.792 (seventh louder
than fifth). All template levers BLOCKED without relabelling real sevenths. Verdict: needs neural
model or non-12-bin front end.

## Key open questions for code read
1. Current _QUALITY_TONES vocabulary — 15 qualities: which? Is add9 present? What does the UI
   actually show (displaySymbol / level_symbol)?
2. _key_prior — how is provisional key computed and how strongly does it shape which qualities win?
3. Viterbi penalty asymmetry (0.95 off / 0.10 on downbeat) — could first-chord anchoring come
   from stream initialization rather than "first chord privileged"?
4. Is harmony_sum now the default input? Deviations between docs and code.
