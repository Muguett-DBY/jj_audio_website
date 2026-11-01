# PROJECT_TREE

## How to use this file

Agents (including Continue / local LLMs): read this before searching the repo. Pick a path from the table, then open that file.
Humans: map of where desktop vs website vs shared engine live.
How to run: `DESKTOP.md` (local Streamlit / installer) or `DEPLOY.md` (hosted compose). Not this file.
If a path is not listed, it was not on disk when this was written (see **Last verified**).

## Two products

Both products import `src/audio_to_tab/` (isolate, YouTube ingest, tab PDF pipeline). They do **not** share UI or job runners.

1. **Desktop (local only)** — Streamlit `ui/app.py` behind `packaging/launcher.py`. Boot page is **Home** (`ui/pages/home.py`); then Audio Isolation (separate/mix), YouTube to MP3 (save file only), Tab PDF (demo), Settings. Serial isolate jobs in-process (`ui/isolate_jobs.py`). No Docker.
   - **Lite / Pro** here means Streamlit **interface mode** (`ui_mode` in sidebar): mainly shapes **Audio Isolation** (and Tab PDF Pro gates). Lite shows goal outcome cards and auto-profiles speed/device/guitar from RAM/GPU (`audio_to_tab/hardware.py`, `prefer_roformer` gated on accelerator); Pro exposes every stem/engine/diagnostic control. **YouTube to MP3 ignores Lite/Pro.** Wiring: `ui/app.py`, `ui/isolate_state.py`, `ui/pages/isolate.py` / `tab_pdf.py`. Testers: `DESKTOP.md`.
2. **Website (hosted)** — FastAPI `backend/` + React `web/` + Caddy. Jobs via Postgres/Redis/MinIO + arq worker (`docker-compose.yml`). Streamlit is **not** in that stack.
   - Hosted processing modes Auto / `lite` / fast_cpu / etc. live in `backend/capabilities.py`. **That “lite” is not desktop Interface Lite.** Operators: `DEPLOY.md`.

Shared: isolate engine, mixer math (`src/audio_to_tab/mixer.py`), ingest, tab pipeline.
Not shared: Streamlit pages vs React pages; desktop disk queue vs API `JobManager`; desktop iframe mixer vs website Web Audio mixer (same wake/background-play policy in both). The desktop mixer also shows a synced picture (YouTube or a local source video) and can resize stem waveforms while that picture is visible. The website mixer is audio-only.

## Guitar isolation status (important context)

The engine supports several guitar isolation backends, behind optional extras
(`[demucs]`, `[roformer]`, `[separator]`, `[scnet]`) and SHA256-pinned urllib downloads (no `huggingface_hub`):

- **HTDemucs-6s** (stock) and **htdemucs-6s-guitar-ft** in-process (`separate.py`).
- **BS-RoFormer-SW** (6-stem) and **MelBand-RoFormer Guitar** specialist (`roformer.py`).
- **SCNet** (`guitar_scnet`) — optional 4-stem MUSDB18 model; guitar content is in `other` (`scnet.py`). Opt-in, not the default first-stage separator.
- Post-processing: **fold-other** (incl. adaptive fold / bleed gate / ensemble paths — see `tests/test_guitar_improvements.py`), **bass-bleed diagnostics + opt-in HPF mitigation**, **low-end recovery**, **guitar pre-refine vs refined variants**, **lead/rhythm split** (not default).

The **Tab PDF path (`pipeline.py`) and the Isolate path (`isolate.py`)** are both wired for Demucs/RoFormer/refine/debleed/restore; SCNet is isolate/CLI-facing. Desktop UI (`tab_pdf.py`, `isolate.py`) and website/API (`backend/contracts.py`, `main.py`, `jobs/runner.py`) expose the guitar option fields they support.

## If you need to change X, start here

| Job | Start here |
|-----|------------|
| Isolation UI (desktop) | `ui/pages/isolate.py` (page: Home\|mix\|+ strip, Queue below), `ui/isolate_state.py` (Lite outcomes / Pro stems / progress / mode helpers), `ui/opened_audio.py` (OS open-with queue), `ui/app.py` (Lite↔Pro sidebar + overlay), `ui/stem_icons.py` (outcome/stem icons). **Separate/mix** — not download-only. |
| Desktop Home / UI craft (Phase 1.5) | Current Home: `ui/pages/home.py` (clean Home by default; `AUDIO_TOOLS_UI_V2=0` keeps the old page). Shell chrome is shared across desktop pages from `ui/app.py` + `ui/design_tokens.py` (navy `#192542`, blue `#2A4467`, amber `#F6AD49`; do not retint `ui/theme_presets.py`). Component: `ui/home_component/` (`make home-build`). Measurements: `docs/plans/moises-parity/10-MOISES-UI-CRAFT.md` §13. Tasks: `11-DESKTOP-UI-REDESIGN.md` §12. |
| Desktop Lite↔Pro mode | `ui/app.py` (sidebar `UI_MODE_*`, `load_ui_mode` / `write_ui_mode`), `ui/isolate_state.py` (`is_pro_mode`, `DEFAULT_UI_MODE`). Branching: Isolation `ui/pages/isolate.py`; Tab PDF Pro gates in `ui/pages/tab_pdf.py`. YouTube to MP3 does not read Lite/Pro. |
| Lite outcome cards | `ui/isolate_state.py` (`OUTCOME_CARDS`, `resolve_outcome_card`, `outcome_card_for_options`), icons `ui/stem_icons.py`, render in `ui/pages/isolate.py` |
| Desktop Lite hardware auto | `src/audio_to_tab/hardware.py` (`lite_auto_choice`, `lite_auto_speed_id`, `lite_accelerator_available`, Detected/Using captions; GPU-first when CUDA/MPS available; MPS ≥12 GB). Wired in `ui/pages/isolate.py`; guitar preference via `prefer_roformer` in `ui/isolate_state.py`. Hosted Auto is separate: `backend/capabilities.py` |
| Global loading overlay | `ui/common.py` (`should_show_global_loading` — **nav-only**, not jobs), inject/clear in `ui/app.py`. Job progress stays on isolate status strip. Tests: `tests/test_global_loading.py` |
| Job retry / failed strip | `ui/isolate_jobs.py` (`requeue_job`, status JSON on disk), `ui/isolate_state.py` (`should_show_failed_job`, dismiss TTL), strip UI in `ui/pages/isolate.py` |
| Bundled Satoshi font | `ui/satoshi_font.py` + `ui/fonts/` (woff2); injected from `ui/app.py` (no CDN). Website copy: `web/public/fonts/` |
| Guitar fix-up post-separation (desktop Mixer) | `ui/guitar_fixup.py` (low-end recovery, pre/refined switch), `ui/pages/isolate.py` |
| Isolation UI (website) | `web/src/pages/IsolatePage.tsx`, `web/src/api.ts`, `web/src/trackOptions.ts`, isolate routes in `backend/main.py`. SPA is **file upload only** (no YouTube URL/search). API `POST /v1/jobs` can take `youtube_url` only when `allow_youtube` is on (`backend/config.py`); default is off. |
| Guitar stem engine | `src/audio_to_tab/isolate.py` (`IsolateConfig`, `separate_stems`, fold, bass-bleed, low-end recovery, pre/refined), `src/audio_to_tab/separate.py` (Demucs / guitar-ft), `src/audio_to_tab/roformer.py`, `src/audio_to_tab/scnet.py` |
| SCNet backend | `src/audio_to_tab/scnet.py`; dispatch in `isolate.py` / CLI `cli/isolate.py`; tests `tests/test_scnet.py` |
| Live mixer (desktop iframe) | Edit `ui/stem_mixer_component/frontend/src/main.ts` + `pan.ts` + `metronomeClicks.ts` + `style.css`, then `make mixer-build`. Pan law is `pan.ts` (equal-power, −100..100), copied in `web/src/mixer/pan.ts`. Live click-track Accent/rate/sound on the metronome row (hot-swap; no full stem reload). **Background play:** `installWakeHooks` recovers on visible/pageshow only — does **not** soft-pause on document hide. Loader: `ui/stem_mixer_component/__init__.py` (`youtube_video_id`, `local_video_url`, `video_offset_sec`, `hide_youtube_video`). Preview encode: `ui/media.py`. Downloads: build mix **on Save**, not on every mute/solo or pan move. |
| Practice player / analysis (desktop) | Engine: `src/audio_to_tab/analysis.py` + `chords.py` / `chord_symbols.py` / `chart.py` / `key.py` / `sections.py` / `lyrics.py`; worker stages in `ui/isolate_jobs.py`; tab artifact via `tab_generate.py`; player state `ui/project.py` (`project.json`); panel + loop/count-in in `ui/stem_mixer_component/frontend/src/main.ts` / `analysis.ts`. Outputs per run: `analysis.json`, `analysis_diagnostics.json`, `chart.musicxml`, `chart.txt`, `tab/transcription.{tab,pdf,mid}`, `tab/tab.json`, `project.json`. Chord `confidence` is the top-two softmax gap. |
| Region picker (desktop iframe) | Edit `ui/region_picker_component/frontend/src/main.ts` + `style.css`, then `make region-picker-build`. Loader: `ui/region_picker_component/__init__.py`. Wired in `ui/pages/isolate.py` (`_render_region_controls`) |
| Mix tabs (desktop iframe) | Edit `ui/mix_tabs_component/frontend/src/main.ts` + `style.css`, then `make mix-tabs-build`. Loader: `ui/mix_tabs_component/__init__.py`. Moises-style Home\|mix\|+ strip on Isolate |
| Mixer picture (desktop) | YouTube iframe or local `source_video` beside the stems, synced to mixer time. Wiring: `ui/pages/isolate.py`, `ui/isolate_state.py` (`mixer_youtube_video_id`, `mixer_local_video_path`, `prefer_local_mixer_picture`), offset `ui/isolate_jobs.py` (`video_offset_sec_for_run`), preview file `download_youtube_preview_video` in `src/audio_to_tab/ingest.py`. In the iframe: move/resize, snaps left/right/above/below, free-float (`videoLayout` in `sessionStorage`). Local file is tried first and can fall back to the YouTube id. |
| Stem wave height (desktop, picture visible) | Drag `#btn-resize-stems` on `#stems` while a picture is showing (not hidden). `stemWaveHeight` default 52, clamp 32–120, `sessionStorage` key `audiotools_stem_mixer_stem_wave`. CSS `--stem-wave-height` on `.waveform-svg` (metro waves scale with it). Handle hides with the picture. Does not change seek, mute/solo, or video layout. Tests: `tests/test_stem_mixer_component.py`. |
| Live mixer (website) | `web/src/mixer/engine.ts`, `web/src/mixer/pan.ts`, `web/src/components/StemMixer.tsx` — same wake/background-play policy as desktop; Media Session + `keepPlaying` across blur (blur is not Pause). Pan uses the same law as desktop `pan.ts`. No picture. Tests: `web/src/mixer/engine.test.ts`, `web/src/mixer/pan.test.ts` |
| Metronome click stem | `src/audio_to_tab/metronome.py` (adaptive tempo curve + local `beat_track`; `click_times_1x` / optional `bpm_curve` diagnostics; rebake). Desktop attach/wiring: `ui/pages/isolate.py`. Live options: mixer `metronomeClicks.ts`. Tests: `tests/test_metronome.py` |
| YouTube ingest | `src/audio_to_tab/ingest.py` (shared: `download_youtube_audio`, `download_youtube_preview_video`, `search_youtube_videos`, `is_youtube_url`, `format_youtube_duration`). Surfaces: `ui/pages/isolate.py`, `ui/pages/youtube_audio.py`, `ui/pages/tab_pdf.py`. Hosted SPA has **no** YouTube UI. `POST /v1/jobs` accepts `youtube_url` only if `allow_youtube` (`backend/config.py`, `backend/main.py`); default off. `POST /v1/isolate/jobs` is upload-based. Tests: `tests/test_ingest.py` |
| YouTube to MP3 (desktop) | **Save file only** (not separate/mix). `ui/pages/youtube_audio.py` (paste/search → format → native folder save). Export: `ui/desktop_export.py` (`EXPORT_FORMATS`, `choose_export_dir`, `export_mix_to_folder`, `open_path_in_os`). Caveats: picker may sit behind pywebview; Linux needs zenity; overwrite same name; auto-reveal folder. Nav: `ui/app.py`. Freeze: `packaging/audio_tools.spec` (`ui.pages.youtube_audio`). Tester copy: `DESKTOP.md`. Tests: `tests/test_ui_pages.py`, nav order in `tests/test_desktop_paths.py` |
| Isolate job queue (desktop) | `ui/isolate_jobs.py` (serial worker, status on disk, `requeue_job`), queue UI in `ui/pages/isolate.py`. Notifications: `ui/desktop_notify.py`. Export/download: `ui/desktop_export.py` |
| Isolate jobs (API/worker) | `backend/jobs/manager.py`, `backend/jobs/runner.py` (`separate_stems`), `backend/worker.py`, `POST /v1/jobs` and `POST /v1/isolate/jobs` in `backend/main.py`. Single-flight: `backend/jobs/single_flight.py` |
| Tab PDF | Engine: `src/audio_to_tab/pipeline.py` (model/refine/debleed/restore), `transcribe.py`, `tab_generate.py`, `pdf_render.py`. Desktop: `ui/pages/tab_pdf.py` (Lite hides engine/advanced; Pro exposes them). Website: `web/src/pages/TabPage.tsx` + tab job in `backend/jobs/runner.py` |
| Desktop packaging / launcher | `packaging/launcher.py`, `packaging/audio_tools.spec`, freeze helpers in `src/audio_to_tab/edition.py`. Runbook: `DESKTOP.md`. CI: `.github/workflows/desktop-release.yml` |
| CI (pytest / ruff) | `.github/workflows/ci.yml` (push/PR). Distinct from desktop release packaging. |
| Agent / planning docs | `AGENTS.md` (agent rules), `PLANNING.md` (planning notes), `docs/issues.md` (I-xxx history), `docs/ui-issues.md` (I-600+ UI register; many rows are `implemented-in-code` history — verify code before treating as open gaps) |
| Moises mixer research | `docs/research/moises/REFERENCE.md` (agent handoff) + `docs/research/moises/FULL-TEARDOWN.md` (byte-level deep pass: shell vs hosted studio, tab system, IPC, capture/plugin stacks) + `docs/research/moises/COMPETITIVE-INSPIRATION-PLAN.md` (build authority: priority, interaction, design constraints). Same notes split in that folder: `README.md`, `mixer-ux.md`, `architecture.md`, `gaps-vs-audio-tools.md`, `evidence-index.md`. Static REA on Moises.app 2.0.3. No Moises source. |
| Tests for isolation / guitar / desktop UX | `tests/test_isolate.py`, `test_isolate_jobs.py`, `test_isolate_state.py`, `test_global_loading.py`, `test_hardware.py`, `test_scnet.py`, `test_guitar_ft_weights.py`, `test_guitar_backups.py`, `test_guitar_fixup.py`, `test_guitar_improvements.py`, `test_pipeline_guitar.py`, `test_ingest.py`, `test_mixer.py`, `test_metronome.py`, `test_stem_mixer_component.py`, `test_region_picker_component.py`, `test_mix_tabs_component.py`, `test_ui_pages.py`, `test_cli_smoke.py` |

Lead/rhythm split (`src/audio_to_tab/lead_rhythm.py`) exists; it is **not** the default isolate path.

## Tree

### Root files

- `Makefile` — install, tester, pytest, Streamlit UI, compose-adjacent web/backend targets, mixer-build, region-picker-build, mix-tabs-build, desktop pkg.
- `pyproject.toml` — package `audio-to-tab-pdf`, Python `>=3.10,<3.13` (3.10–3.12), extras `dev` / `eval` / `demucs` / `roformer` / `separator` / `scnet` / `desktop`.
- `requirements.txt` — desktop-only dev pins (Streamlit, yt-dlp, onnxruntime, eval/dev extras; hosted backend deps live in `pyproject.toml` `[project.dependencies]`).
- `requirements-demucs.txt` — Demucs + torch on top of `requirements.txt`.
- `Dockerfile` — API/worker image (ffmpeg, `src/`, `backend/`, prewarm script).
- `Dockerfile.web` — SPA build + Caddy.
- `docker-compose.yml` — hosted stack: Caddy, API, worker, Postgres, Redis, MinIO.
- `docker-compose.lite.yml` — single API, sqlite, local disk (no worker/Postgres/Redis/MinIO).
- `Caddyfile` — TLS; `/v1/*` → api:8000; SPA from `/srv`.
- `README.md` — project overview.
- `AGENTS.md` — agent rules (venv, commands, what not to commit). Read with this file.
- `PLANNING.md` — planning notes; cited by `docs/issues.md`.
- `DESKTOP.md` — local tester / installer runbook (Isolation Home\|mix\|+, YouTube to MP3 save-to-folder, Tab PDF).
- `DEPLOY.md` — compose production runbook.
- `LICENSE` — MIT.
- `.env.example` — hosted env template (copy to `.env`; do not commit `.env`).
- `.env.lite.example` — lite compose env template.
- `.python-version` — `3.11`.
- `.gitattributes` — LF normalization.
- `.gitignore` — venvs, `.env`, wavs, eval lead/rhythm clips and local notes (`eval/lead_rhythm/clips/`, `out/`, `manifest.json`, `RESEARCH.md`, `RESULTS.md`), internal docs (`CONTEXT_TREE.md`, `docs/oracle-free-memory-spike.md`).
- `PROJECT_TREE.md` — this file (tracked). Canonical in-repo tree. (`CONTEXT_TREE.md` is a gitignored internal name, not present in a clean clone.)
- `.opencode/plans/fix-plan-2026-09-02.md` — tracked historical desktop fix plan (not a runbook; product docs are `DESKTOP.md` / `DEPLOY.md`).
- `.superpowers/` — local agent scratch; gitignored. Not product code.
- `docs/` — `issues.md` (I-xxx history; I-500 isolation audit); `ui-issues.md` (I-600+ UI/UX register; does not replace `issues.md`); `legal/` (Privacy/Terms/Refund/Cookies drafts + SDK/notices); `compliance-checklist.md` (C01–C20).

### `src/`

Shared engine. Import as `audio_to_tab`.

- `audio_to_tab/__init__.py` — package marker / version.
- `audio_to_tab/isolate.py` — Demucs/RoFormer/SCNet multi-stem isolate, region trim, `--segment` clamp, fold-other, presence / bass-bleed / guitar-stem-quality diagnostics, opt-in bass-bleed HPF mitigation, low-end recovery (`apply_guitar_low_end_recovery`, `apply_low_end_restore`, `LowEndRecoveryDiagnostics`), guitar pre-refine vs refined variants (`GUITAR_PREREFINE_NAME` / `GUITAR_REFINED_NAME`, `switch_guitar_stem_variant`), melband refine, checkpoint/resume. Bleed-gate / adaptive fold / ensemble covered in tests `test_guitar_improvements.py`.
- `audio_to_tab/separate.py` — Demucs subprocess / frozen in-process; `separate_guitar_stem` (model/guitar-ft/roformer/refine/debleed/restore); guitar-ft weights download + SHA256 verify (`run_demucs_guitar_ft_inprocess`).
- `audio_to_tab/roformer.py` — BS-RoFormer-SW (6-stem) + MelBand-RoFormer Guitar specialist; urllib + SHA256 downloads; bs-roformer-infer / audio-separator backends; `run_guitar_refine` (residual-aware).
- `audio_to_tab/scnet.py` — optional SCNet (`guitar_scnet`) 4-stem MUSDB18 backend; urllib + SHA256 checkpoint; `.[scnet]` extra. Not the default first-stage separator.
- `audio_to_tab/ingest.py` — file normalize + YouTube download/search (`yt-dlp`: `download_youtube_audio`, `download_youtube_preview_video` for the mixer picture, `search_youtube_videos`, `is_youtube_url`, `format_youtube_duration`).
- `audio_to_tab/mixer.py` — stem mix / waveform helpers used by desktop downloads; metronome sorts last and starts muted. Equal-power `pan_matrix` / `apply_pan` / `clamp_pan` match `pan.ts`; `mix_stems_to_wav(..., pans=...)` bakes pan into the desktop mix (center is bit-identical).
- `audio_to_tab/hardware.py` — RAM/GPU/chip probe (`HostProbe.cpu_brand`, Apple Silicon label e.g. `Apple M2 Pro`); desktop speed recommendations (**GPU-first** for CUDA / eligible MPS); MPS gated at ≥12 GB; Lite helpers (`lite_accelerator_available`, `lite_auto_choice`, Detected/Using captions).
- `audio_to_tab/edition.py` — Windows CPU vs NVIDIA freeze flavor.
- `audio_to_tab/lead_rhythm.py` — opt-in lead/rhythm split on the guitar stem (not product default).
- `audio_to_tab/pipeline.py` — audio → tab PDF end-to-end (`PipelineConfig` incl. model, guitar_checkpoint, guitar_refine, demucs_segment/jobs, fold_other_mode, low_end_restore_db, sub_bass_debleed).
- `audio_to_tab/transcribe.py` — Basic Pitch audio → MIDI.
- `audio_to_tab/tab_generate.py` — MIDI → ASCII tab; `tab_measure_spans` + `tab_document_to_dict` feed the mixer Tab panel (`tab/tab.json`).
- `audio_to_tab/pdf_render.py` — ASCII/MIDI → PDF (ReportLab).
- `audio_to_tab/tempo.py` — tempo from audio/MIDI.
- `audio_to_tab/metronome.py` — adaptive (“floating”) metronome: one onset envelope → global + curve-driven `beat_track` (librosa 0.11); gate keeps quiet regular intros; accents use `body_start_sec`; `click_times_1x` is the mixer/rebake contract; optional `bpm_curve` in diagnostics; `MetronomeRenderOptions` / `rebake_metronome_artifact` (accent, rate, sound).
- `audio_to_tab/analysis.py` — song analysis schema/cache/orchestrator (`AnalysisResult`, chord/section/lyric spans, optional `chart`, `analysis_cache_key`). `SCHEMA_VERSION` stays 1. Order: beats → provisional key → chords → key → sections → lyrics → chart.
- `audio_to_tab/key.py` — Krumhansl provisional key (bass breaks relative-major ties) and a duration-weighted chord histogram.
- `audio_to_tab/chord_symbols.py` — frozen `Chord` parse, render, transpose, capo shape, display level, chord tones.
- `audio_to_tab/chords.py` — tuned HPSS CQT chroma, weighted templates, z-scored priors, beat-synchronous Viterbi. `confidence` is the softmax top-two gap. Optional `model_dir` / `ATT_CHORD_MODEL=1` opts into BTC.
- `audio_to_tab/chord_neural.py` — SHA256-pinned BTC maj/min download + inference; template fallback when off/unavailable.
- `audio_to_tab/btc/` — vendored MIT BTC model (jayg996/BTC-ISMIR19); weights not bundled.
- `audio_to_tab/chart.py` — `ChartDocument` from an analysis and its beat grid. Continuation cells are empty.
- `audio_to_tab/export_musicxml.py` — chart to a MusicXML 3.1 subset (harmony + lyrics), stdlib only.
- `audio_to_tab/export_text.py` — chart to ChordPro-ish text.
- `audio_to_tab/sections.py` — librosa self-similarity/novelty sections, minimum 8 s.
- `audio_to_tab/lyrics.py` — `faster-whisper` small int8 from the pinned local model dir.
- `audio_to_tab/rhythm.py` — beat-grid quantize for tab events.
- `audio_to_tab/midi_cleanup.py` — post-pitch MIDI cleanup.
- `audio_to_tab/tuning.py` — fretboard / tuning helpers.
- `audio_to_tab/subprocess_util.py` — subprocess kwargs for desktop/CLI.
- `audio_to_tab/cli/__init__.py` — CLI package marker.
- `audio_to_tab/cli/isolate.py` — CLI isolate (Demucs/RoFormer/SCNet, `--guitar-checkpoint`, `--guitar-refine`, `--fold-other-mode`, `--sub-bass-debleed`, `--low-end-restore-db`, `--low-end-recovery`, `--two-pass`, lead/rhythm thresholds).
- `audio_to_tab/cli/pipeline.py` — CLI full pipeline.
- `audio_to_tab/cli/transcribe.py` — CLI transcribe.
- `audio_to_tab/cli/tab2pdf.py` — CLI tab/MIDI → PDF.
- `audio_to_tab/cli/mid2tab.py` — CLI MIDI → ASCII tab.

### `ui/`

Desktop Streamlit only.

- `__init__.py` — empty package marker.
- `app.py` — multipage router (Home default, then Audio Isolation, YouTube to MP3, Tab PDF, Settings); Lite↔Pro sidebar (`UI_MODE_*`, persist to disk; mainly Isolation/Tab PDF); nav-only global loading overlay (`should_show_global_loading`); injects Satoshi via `satoshi_font.py`.
- `pages/home.py` — boot screen. Launch cards for Learn a song, YouTube save, and Tab PDF. Recents and Library list local isolate runs (`list_recent_runs`); library rows show `BPM · Key · Sections`, a Practice action for analysed runs, and Analyse for older runs (`analyse_existing_run`).
- `pages/isolate.py` — Audio Isolation (**separate/mix**). Sticky Home\|mix\|+ strip; Home form (upload/YouTube search, outcomes/stems, section); mix tab = live mixer (optional synced picture) + Downloads (export on Save); Queue below. Lite: outcome cards + hardware Detected/Using + auto speed/device/guitar; Pro: Speed / Engine / This computer / stem checkboxes. Status strip owns running/failed jobs (retry, dismiss).
- `pages/youtube_audio.py` — YouTube to MP3 (**save file only**, not separate/mix). Paste or search a public URL, pick format (default MP3), native folder save via `choose_export_dir`. Pure helper `save_youtube_audio_to_folder`. Ignores Lite/Pro. Caveats: see `desktop_export.py` / `DESKTOP.md`.
- `pages/tab_pdf.py` — Tab PDF demo page. Lite: simpler convert path (hides engine/advanced). Pro: guitar refine, guitar-ft, low-end restore, sub-bass de-bleed, Advanced transcription. Upload cap caption from `ui/common.py` (`max_upload_caption`).
- `pages/__init__.py` — empty package marker.
- `guitar_fixup.py` — post-separation guitar stem fix-up for the mix-tab Mixer (low-end recovery, pre/refined switch, diagnostics).
- `isolate_jobs.py` — serial Demucs/RoFormer worker; job status JSON under app data dir; `requeue_job`, `separation_in_progress`, `format_job_error`; `video_offset_sec_for_run` (region start for mixer picture sync). Pipeline stages include `analysis` (`analysis.json` + `analysis_diagnostics.json` + meta summary) and `tab` (`tab/`); both are warnings, not job failures. `analyse_existing_run` is the Home repair action for older runs.
- `project.py` — mutable per-run player state in `project.json` (mixer volumes/pan/mute/solo/master, rate/semitones/loop/count-in, relative analysis/tab paths). `load_project` / `save_project` (atomic, write-on-change), `seed_mixer`, `update_from_mixer`, `load_analysis_for_run`, `load_tab_for_run`.
- `isolate_state.py` — Lite outcome cards, Pro track options, progress stages, UI mode helpers (`is_pro_mode`, persist settings), failed-strip TTL/dismiss, guitar selection (`prefer_roformer` for Lite CPU-only hosts), Home/mix shell tab helpers, mixer picture helpers (`mixer_youtube_video_id`, `mixer_local_video_path`, `prefer_local_mixer_picture`).
- `stem_icons.py` — Lucide-style line-art icons for Lite/Pro stem and outcome tiles.
- `satoshi_font.py` — `@font-face` CSS for bundled Satoshi (used by `app.py`).
- `fonts/` — Satoshi `.woff2` + `satoshi.css` + `satoshi/README.txt` (local, no CDN).
- `desktop_export.py` — save to a chosen folder + reveal in OS file manager (Isolate Downloads and YouTube to MP3). `EXPORT_FORMATS`, `convert_audio`, `choose_export_dir`, `export_mix_to_folder`. Docstring lists picker/overwrite/reveal caveats.
- `desktop_notify.py` — best-effort OS notifications for isolate job completion (no extra pip deps).
- `common.py` — run listing, uploads, data dir, edition labels, `max_upload_mb` / `max_upload_caption`, `should_show_global_loading` (nav-only).
- `media.py` — Streamlit media URLs, preview encode, mix cleanup.
- `icon.png` — window / tab icon.
- `stem_mixer_component/__init__.py` — iframe component; serves `frontend/build/` when `mixer_build_is_complete`. Args include `youtube_video_id`, `local_video_url`, `video_offset_sec`, `hide_youtube_video`, plus practice state: `analysis`, `tab`, `initial_loop`, `count_in_bars`, `initial_rate`, `initial_semitones`, `initial_pan`.
- `stem_mixer_component/frontend/src/main.ts` — live mixer (Web Audio) source; metronome row live controls; wake hooks keep playback across app/window hide (soft-pause only on dead/interrupted context). Picture: YouTube or local video, move/resize, snaps, free-float. Stem wave drag (`stemWaveHeight`, `#btn-resize-stems`) only while the picture is visible. Practice panel: Chords/Sections/Lyrics/Tab, A/B loop, count-in, keyboard `L`/`C`.
- `stem_mixer_component/frontend/src/analysis.ts` — pure practice helpers (chord/section/lyric/measure selection, loop helpers, chord-shape/diagram geometry).
- `stem_mixer_component/frontend/src/pan.ts` — equal-power pan law (−100..100). Same text as `web/src/mixer/pan.ts`. The Python export uses the same matrix (`audio_to_tab.mixer.pan_matrix`).
- `stem_mixer_component/frontend/src/metronomeClicks.ts` — click synthesis / rate / peaks (parity with Python metronome render).
- `stem_mixer_component/frontend/src/style.css` — mixer styles (metro toolbar, video frame, `--stem-wave-height`).
- `stem_mixer_component/frontend/src/vite-env.d.ts` — Vite typings for the mixer iframe.
- `stem_mixer_component/frontend/package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html` — Vite build for the iframe.
- `region_picker_component/__init__.py` — waveform region iframe; serves `frontend/build/`.
- `region_picker_component/frontend/src/main.ts` — wavesurfer + Regions plugin source.
- `region_picker_component/frontend/src/style.css` — region picker styles.
- `region_picker_component/frontend/package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html` — Vite build.
- `mix_tabs_component/__init__.py` — Moises-style Home|mix|+ strip; serves `frontend/build/`.
- `mix_tabs_component/frontend/src/main.ts` — tab strip UI source.
- `mix_tabs_component/frontend/src/style.css` — tab strip styles.
- `mix_tabs_component/frontend/package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html` — Vite build.

### `web/`

Hosted SPA (Vite + React). Dev: `make web` (proxies API). Paths below are repo-rooted.

- `web/index.html` — SPA shell; icons `/favicon.ico`, `/favicon-32x32.png`, `/apple-touch-icon.png`; Satoshi from `/fonts/satoshi.css`.
- `web/package.json` / `web/package-lock.json` / `web/vite.config.ts` / `web/tsconfig.json` — build; Vitest for mixer engine and `App.test.ts`.
- `web/netlify.toml`, `web/vercel.json` — SPA fallback hosting.
- `web/.env.example` — optional `VITE_API_BASE_URL` (same-origin Caddy leaves unset).
- `web/public/fonts/` — Satoshi `.woff2` + `satoshi.css`.
- `web/public/favicon.ico`, `web/public/favicon-32x32.png`, `web/public/apple-touch-icon.png`, `web/public/icon-512.png` — site icons.
- `web/src/main.tsx` — React mount.
- `web/src/App.tsx` — routes `/login`, `/isolate`, `/tab`; catch-all `*` → isolate. Shell sign-in / sign-out.
- `web/src/App.test.ts` — source-level routing checks (`/login` → `LoginPage`, post-login navigate to `/isolate`).
- `web/src/api.ts` — REST/WebSocket client for jobs, uploads, stems. Capabilities include `allow_youtube` (SPA isolate UI does not offer a YouTube field).
- `web/src/auth.tsx` — session / JWT.
- `web/src/styles.css` — global styles.
- `web/src/trackOptions.ts` — stem / track option labels shared by isolate UI.
- `web/src/vite-env.d.ts` — Vite typings.
- `web/src/pages/IsolatePage.tsx` — website isolation UI (file upload).
- `web/src/pages/TabPage.tsx` — website tab PDF UI.
- `web/src/pages/LoginPage.tsx` — login form; routed at `/login`; after sign-in navigates to `/isolate`.
- `web/src/components/RegionPicker.tsx` — wavesurfer region trim for section mode.
- `web/src/components/StemMixer.tsx` — website mixer UI (audio only; blur does not pause).
- `web/src/components/JobProgress.tsx` — job status display.
- `web/src/components/ProcessingModeSelect.tsx` — auto/fast/balanced from capabilities API.
- `web/src/components/ProcessingModeSelect.test.ts` — Vitest for processing-mode select.
- `web/src/components/PracticeSheet.tsx` — practice sheets (chords/sections/lyrics/tab).
- `web/src/components/PracticeSheet.test.tsx` — Vitest for practice sheet.
- `web/src/components/ThemeSelector.tsx` — theme/accent color picker.
- `web/src/components/ColorPicker.tsx` — custom accent color swatches.
- `web/src/components/ColorSwatch.tsx` — individual color swatch component.
- `web/src/components/CookieBanner.tsx` — cookie consent banner.
- `web/src/components/SiteFooter.tsx` — page footer.
- `web/src/components/StemMixer.layout.test.ts` — layout smoke test for StemMixer.
- `web/src/mixer/analysis.ts` — chord/loop helpers (`clampLoop`, `formatChordSymbol`).
- `web/src/mixer/analysis.test.ts` — Vitest for analysis helpers.
- `web/src/mixer/pan.ts` — equal-power pan law (−100..100), copied from desktop.
- `web/src/mixer/pan.test.ts` — Vitest for pan law.
- `web/src/mixer/practice.ts` — practice rate/key management (SoundTouch).
- `web/src/mixer/practice.test.ts` — Vitest for practice management.
- `web/src/mixer/engine.ts` — Web Audio mix engine (same background-play / wake policy as desktop iframe; Media Session + `keepPlaying`).
- `web/src/mixer/engine.test.ts` — mixer unit tests (incl. hide-while-playing does not soft-pause; `keepPlaying` / Media Session).
- `web/src/theme/ThemeContext.tsx` — global theme/accent context for the SPA.

### `backend/`

Hosted API.

- `__init__.py` — package marker.
- `main.py` — FastAPI routes: auth, upload, `POST /v1/jobs` (optional `youtube_url` when `allow_youtube`), `POST /v1/isolate/jobs`, tab jobs (incl. guitar model/refine/debleed/restore fields), stem/mix download, WS events.
- `config.py` — `ATT_*` settings. `allow_youtube` defaults false.
- `contracts.py` — Pydantic request/response models (incl. `youtube_url` on job create, `low_end_restore_db`, `sub_bass_debleed`, guitar model/refine fields).
- `auth.py` — JWT, bootstrap admin, demo session.
- `models.py` — SQLAlchemy User / Job / Upload.
- `db.py` — engine + sessions.
- `storage.py` — local disk or S3/MinIO.
- `events.py` — in-process / Redis job progress.
- `capabilities.py` — hosted SPA processing modes (Auto / fast_cpu / balanced / high_gpu / lite). Distinct from desktop Streamlit Lite UI mode.
- `limits.py` — upload size/type checks.
- `worker.py` — arq worker entry.
- `jobs/__init__.py` — jobs package marker.
- `jobs/manager.py` — persist jobs, artifacts, cancel (guitar option fields; optional `youtube_url` on `POST /v1/jobs`).
- `jobs/runner.py` — runs `separate_stems` / `run_pipeline` (guitar options; YouTube source when the job has `youtube_url`).
- `jobs/single_flight.py` — one heavy job at a time on small hosts.

### `packaging/`

Desktop freeze + installers. How-to: `DESKTOP.md`.

- `launcher.py` — native window over local Streamlit (dev or frozen).
- `audio_tools.spec` — PyInstaller onedir (hiddenimports include `ui.pages.isolate`, `ui.pages.youtube_audio`, `ui.pages.tab_pdf`).
- `bundle_ffmpeg.py` — download ffmpeg into `packaging/ffmpeg/` (gitignored binaries).
- `macos_signing.py` — Developer ID + notarization helpers.
- `make_app.sh` — wrap onedir → `AudioTools.app`.
- `make_pkg.sh` — macOS `.pkg` (preferred installer).
- `make_dmg.sh` — macOS `.dmg` (fallback).
- `make_windows_installer.ps1` — Inno Setup CPU/CUDA Setup.exe.
- `AudioTools.iss` — Inno script.
- `Install AudioTools.command` — DMG double-click copy to `/Applications`.
- `entitlements.plist`, `pkg_component.plist` — macOS signing / pkg metadata.
- `pkg_scripts/postinstall` — launch app after `.pkg` install.
- `generate_icons.sh` — rebuild icons from `icon-source.png` (square master) with a macOS squircle alpha mask.
- `icon-source.png` — unmasked 1024×1024 square art (edit this, then re-run `generate_icons.sh`).
- `icon.png` / `icon.icns` / `icon.ico` — generated app icons (squircle-masked PNG/ICNS; ICO for Windows).

### `tests/`

- `conftest.py` — puts `src/` and repo root on `sys.path`.
- `backend_test_utils.py` — rebind backend settings without reloading ORM.
- `test_isolate.py` — isolate engine (Demucs mocked), including `--segment` clamp, fold, bass-bleed.
- `test_isolate_jobs.py` — desktop serial queue / remove / error formatting / requeue.
- `test_isolate_state.py` — Lite outcomes, Pro stems, progress helpers, UI mode, failed-strip TTL; `prefer_roformer` guitar defaults.
- `test_global_loading.py` — nav-only overlay gate (`should_show_global_loading`).
- `test_scnet.py` — SCNet constants, availability, dispatch.
- `test_guitar_improvements.py` — spectral bleed gate, adaptive fold gain, guitar ensemble.
- `test_guitar_ft_weights.py` — guitar-ft weights SHA256 / download helpers.
- `test_guitar_backups.py` — guitar pre-refine vs refined variants.
- `test_guitar_fixup.py` — `ui/guitar_fixup.py` low-end recovery + variant switch.
- `test_pipeline_guitar.py` — tab pipeline guitar options (model/refine/debleed/restore).
- `test_cli_smoke.py` — five CLI entrypoints + `pyproject` scripts.
- `test_ui_pages.py` — Streamlit page pure helpers (incl. YouTube to MP3 save helper).
- `test_ingest.py` — YouTube ingest (network opt-in).
- `test_mixer.py` — mix helpers, pan matrix/export (center bit-identical, hard left/right, mute).
- `test_metronome.py` — adaptive metronome (curve helpers, vocal-intro pulse, steady-tempo regression, 48 kHz, talking-intro gate) + render/rebake / `bpm_curve` diagnostics.
- `test_analysis.py`, `test_analysis_orchestrator.py` — analysis schema/cache and stage fallbacks.
- `test_chords.py`, `test_chord_symbols.py`, `test_chord_neural.py`, `test_chord_bench.py`, `test_chart.py`, `test_export_musicxml.py`, `test_key.py`, `test_sections.py`, `test_lyrics.py` — per-stage analysis engines, the chord bench scorer, and chart export.
- `test_tab_generate.py` — tab document / measure spans.
- `test_project.py` — `project.json` round-trip, clamping, write-on-change, seeding, artifact loading.
- `test_home.py` — Home analysis line, Practice/Analyse actions, `analyse_existing_run`.
- `test_stem_mixer_component.py` — desktop mixer build packaging, live metronome wiring, picture layout, and stem-wave height smoke.
- `test_region_picker_component.py` — desktop region picker build packaging.
- `test_mix_tabs_component.py` — desktop mix-tabs build packaging.
- `test_media.py` — preview / cleanup.
- `test_ui_common.py` — run list / delete / upload caption helpers.
- `test_desktop_paths.py` — launcher paths, freeze helpers, Streamlit about/shell assertions, YouTube to MP3 nav order + `ui.pages.youtube_audio` freeze import.
- `test_desktop_export.py` — `ui/desktop_export.py`.
- `test_desktop_notify.py` — `ui/desktop_notify.py`.
- `test_hardware.py` — RAM/GPU/chip probe, desktop recommend, Lite Detected/Using captions, MPS gate.
- `test_edition.py` — CPU vs CUDA edition.
- `test_macos_signing.py` — signing helpers (no codesign I/O).
- `test_api.py` — FastAPI smoke (incl. guitar option fields).
- `test_web_security.py` — auth, CORS, upload caps.
- `test_capabilities.py` — hosted processing-mode API (not desktop Lite UI).
- `test_single_flight.py` — API job gate.
- `test_lead_rhythm.py` — lead/rhythm post-process.
- `test_tab.py` — tab / tempo / MIDI cleanup.
- `test_subprocess_util.py` — subprocess helpers.

### `scripts/`

- `dev.sh` — macOS/Linux: `tester`, `install`, `ui`, `backend`, `test`, … (no make required).
- `dev.ps1` — Windows equivalent.
- `prewarm.py` — pull Demucs / Basic Pitch weights (API first-request); integrates `is_roformer_backend_available`.

### `eval/`

Offline scoring. Gitignored under `eval/lead_rhythm/`: `clips/`, `out/`, `manifest.json`, `RESEARCH.md`, `RESULTS.md`. Those may exist locally and are not in a clean clone. `eval/results.json` is also gitignored.

- `eval/__init__.py`, `eval/lead_rhythm/__init__.py` — empty package markers.
- `generate_fixtures.py` — synthetic MIDI/WAV fixtures.
- `score_transcription.py` — mir_eval vs ground-truth MIDI.
- `fixtures/manifest.json` — fixture list; `*.mid` + allowed `*.wav` here.
- `lead_rhythm/README.md` — how to score lead/rhythm locally.
- `lead_rhythm/score_lead_rhythm.py` — scorer.
- `lead_rhythm/score_guitar_stage1.py` — scores stage-1 guitar stems across output dirs via `analyze_guitar_stem_quality` (low/mid/high share, competitor overlap, optional SI-SDR + high-end preservation).
- `lead_rhythm/make_synthetic_clips.py` — synthetic clips (output gitignored).
- `lead_rhythm/manifest.example.json` — example manifest.

### `.github/`

- `workflows/ci.yml` — push/PR pytest + ruff on `src/`.
- `workflows/desktop-release.yml` — Windows CPU/CUDA Setup + macOS arm64/x64 `.pkg`.
- Dependabot is off; dependency bumps are manual.

### `.streamlit/`

- `config.toml` — local Streamlit: `gatherUsageStats = false`, `toolbarMode = "minimal"`, `headless = true`, `address = "127.0.0.1"`, `fileWatcherType = "none"`, `runOnSave = false`, `maxUploadSize = 500`, CORS + XSRF on. `credentials.toml` is gitignored.

### `docs/`

- `issues.md` — canonical I-xxx design-decision / issue history (CI and `PLANNING.md` cite it). Includes reserved historical IDs and the 2026-09-08 isolation performance/functional audit (`I-500`–`I-509`).
- `ui-issues.md` — UI/UX-only issue register (`I-600`+). Does not replace `issues.md`.
- `research/moises/REFERENCE.md` — single agent handoff for the Moises.app 2.0.3 REA pass (stack, IPC, mixer literals, P0/P1/P2 gaps, Evidence IDs). Same folder also has `FULL-TEARDOWN.md` (deep static pass; corrects the "local mixer UI" inference — shell renders chrome only, studio is hosted) and the topic split (`README.md`, `mixer-ux.md`, `architecture.md`, `gaps-vs-audio-tools.md`, `evidence-index.md`). No extracted Moises source.
- `plans/moises-parity/` — portable Moises-parity program (2026-10-19, UI pass 2026-10-20): `README.md` (index), `01-MOISES-SURFACE.md` (2.0.3 + 2.0.18 runtime surface + UI craft pass), `02-CURRENT-STATE.md` (code inventory + 2026-10-20 delta), `03-FEATURE-MATRIX.md` (gap/decision + experience rows), `04-SCOPE-CUT.md` (build order), `05-ROADMAP.md` (phases/gates incl. P1.5 UI / P1.6 hardening), `06-PHASE-1-PLAN.md` (task plan + verification log), `07-BUILD-PROMPT.md` (Phase 1 historical prompt + Phase 1.5 UI prompt), `08-EVIDENCE-LOG.md` (audit trail + UI captures + Phase 1 verification), `09-SKILL-WORKFLOW.md` (skill order), `10-MOISES-UI-CRAFT.md` (measured Moises UI, current audit, Tyrian Nocturne direction, tokens/screens), `11-DESKTOP-UI-REDESIGN.md` (Phase 1.5 file-level plan, UI-0–UI-9, gates). `_work/` holds the raw research. Not a GSD `.planning/` workspace.
- `oracle-free-memory-spike.md` — gitignored name (not in public git; may be absent locally).

### Do not edit (generated / vendor)

One line each; do not hand-edit:

- `node_modules/`, `ui/stem_mixer_component/frontend/node_modules/`, `ui/region_picker_component/frontend/node_modules/`, `ui/mix_tabs_component/frontend/node_modules/`
- `.venv/`, `.venv311/`, `.venv312/`, `.venv-desktop/`, `.venv-desktop-x64/`, `.venv-*/`
- `web/dist/`
- `ui/stem_mixer_component/frontend/build/assets/*` (hashed Vite output)
- `ui/region_picker_component/frontend/build/assets/*` (hashed Vite output)
- `ui/mix_tabs_component/frontend/build/assets/*` (hashed Vite output)
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `audio_to_tab_pdf.egg-info/`
- `dist/`, `build/`, `.build/`, `packaging/ffmpeg/`
- `data/`, `output/` (local runs)

## Do not edit unless

- Mixer iframe **build**: `ui/stem_mixer_component/frontend/build/` is committed so testers need no Node. Change `frontend/src/` (`main.ts`, `metronomeClicks.ts`, `style.css`) then `make mixer-build`. Do not rewrite hashed `assets/*` by hand.
- Region picker iframe **build**: `ui/region_picker_component/frontend/build/` is committed the same way. Change `frontend/src/` then `make region-picker-build`.
- Mix tabs iframe **build**: `ui/mix_tabs_component/frontend/build/` is committed the same way. Change `frontend/src/` then `make mix-tabs-build`.
- Desktop fonts: prefer editing `ui/satoshi_font.py` / replacing woff2 under `ui/fonts/`; do not reintroduce a CDN `<link>` for Satoshi.
- Env: copy `.env.example` / `.env.lite.example` / `web/.env.example` → `.env`. Never commit `.env`.
- Do not commit wavs (except `eval/fixtures/*.wav`), Demucs/torch weights, or `packaging/ffmpeg/` binaries.
- `CONTEXT_TREE.md` (and `docs/oracle-free-memory-spike.md`) are gitignored; `PROJECT_TREE.md` is tracked and is the canonical in-repo tree.

## Last verified

2026-09-30 — desktop mixer picture (YouTube or local `source_video`, move/resize, snaps, free-float) and stem-wave height while that picture is visible; website `/login` route + `App.test.ts`; favicons under `web/public/`; `POST /v1/jobs` `youtube_url` behind `allow_youtube` (SPA isolate stays upload-only); `download_youtube_preview_video`. Prior 2026-09-18 pass: metronome adaptive tempo map; mixer background play (desktop `main.ts` + web `engine.ts` / `engine.test.ts`).
