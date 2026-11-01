# Web readiness audit (2026-10-29)

Read-only audit: is the hosted web product (`backend/` + `web/` + compose) ready to launch with feature parity with the desktop app (`ui/`)? Evidence is `file:line`. Nothing was modified except this file. Web checks run per the audit prompt: `npx tsc --noEmit` (clean, exit 0) and `npm test` (65/65 pass); `npm install`, `pytest`, and builds were NOT run.

## Verdict

**READY WITH GAPS.** The hosted stack is deployable and coherent today: production guards are real (`backend/config.py:84-97` refuses weak secrets/`*` CORS/auth-off in prod), upload validation is magic-byte-backed (`backend/limits.py:45-67`), anonymous demo is rate-limited (`backend/main.py:237-238`), the web mixer matches the desktop wake/background-play invariant byte-for-byte (`web/src/mixer/engine.ts:292-306` ≡ `ui/stem_mixer_component/frontend/src/main.ts:324-338`), and the SPA's mixer has the I-605/I-607/I-609/I-622/I-623 fixes implemented in code. What keeps it from "basically the exact same as desktop": (1) the production image cannot run the desktop's high-quality guitar engines — RoFormer/separator extras are not installed (`Dockerfile:13`), while the SPA advertises them (`web/src/pages/IsolatePage.tsx:42`); (2) mixer practice features that exist on desktop are absent on web (live metronome controls, A/B loop, count-in); (3) documented product splits (no YouTube in the SPA, no mixer picture, no mix-tabs strip, audio-only mixer) mean "same features" in the engine, not the same surface; (4) a public unmetered demo will slowly fill Postgres/MinIO with no cleanup tooling (`DEPLOY.md:34`). Verdict: launch a small upload-based demo today; not yet ready as a public unmetered "same as desktop" product.

## P0 launch blockers

**P0-1 — Hosted image cannot run the guitar engines the hosted UI advertises.**
Evidence: `Dockerfile:13` installs `-e ".[dev,eval]"` (pytest/ruff/mir_eval) plus `requirements-demucs.txt` (stock Demucs + torch, `requirements-demucs.txt:1-5`) — it never installs the `roformer` / `separator` / `scnet` extras (`pyproject.toml:45-47`). `scripts/prewarm.py:44-51` explicitly logs "RoFormer backend not installed; Full band preset will fail until extras are present". Meanwhile the SPA's guitar-engine copy says "Downloads ~700 MB BS-RoFormer-SW weights…" (`web/src/pages/IsolatePage.tsx:42`) and `backend/jobs/runner.py:169-189` forwards guitar refine/checkpoint/fold fields from the job model. Desktop Pro exposes these engines (`ui/pages/tab_pdf.py`, `ui/pages/isolate.py`); a hosted user selecting the advertised preset gets a runtime failure instead of "basically the same as desktop". Fix direction: add `.[roformer]`/`.[separator]` to the image (and scnet if offered) or gate those hosted Track options until the extras exist.

**P0-2 — No cleanup path for anonymous demo data (documented, unfixed).**
Evidence: `DEPLOY.md:34` — "Anonymous users, jobs, and uploads accumulate in Postgres and object storage over time and should be pruned periodically by the operator (documented operational follow-up; no automated cleanup in this stack yet)." Full compose persists `pg_data`, `minio_data`, `app_data` (`docker-compose.yml:44,63,69,96,119-121`). On a public domain with `ATT_DEMO_MODE=true`, visitors mint a row-per-session and 25 MB uploads each; disk/DB fill is the failure mode. Fix direction: operator cron pruning script or TTL task before public launch.

## P1 parity gaps (desktop has it, web doesn't)

- **Live metronome controls.** Desktop has hot-swap accent/rate/sound click synthesis (`ui/stem_mixer_component/frontend/src/metronomeClicks.ts`, PROJECT_TREE.md:50,169). The web engine only plays the recorded metronome stem, default-muted (`web/src/mixer/engine.ts:20`, karaoke preset keeps it unsoloed at `engine.ts:128`); there is no click-synthesis module under `web/src/mixer/`. The Python-side `click_times_1x` contract exists (`audio_to_tab/metronome.py`), but re-rendering clicks at new rates is desktop-only.
- **A/B loop + count-in.** Desktop mixer schedules count-in and keeps transport math for a loop region (`ui/stem_mixer_component/frontend/src/main.ts:363-377`, `count_in_bars` in `ui/stem_mixer_component/__init__.py` per PROJECT_TREE.md:165; keyboard `L`/`C`). Web: `clampLoop` exists but is dead code — defined (`web/src/mixer/analysis.ts:124-135`), tested nowhere in UI, never imported by any component (grep: only `analysis.ts:125`). `engine.ts` `currentTime()` has no loop math and no count-in scheduling.
- **Mixer picture (YouTube/local video) + wave-height drag.** Desktop-only, documented product split, not a forgotten widget ("website mixer is audio-only", `PROJECT_TREE.md:20`; `_mixer_and_downloads_fragment` I-607 territory). Stem wave resize depends on the picture, so it falls with it.
- **Mix-tabs Home|mix|+ strip.** Desktop-only (`ui/mix_tabs_component/`); web stacks form + queue + mixer on one page — documented split (`docs/ui-issues.md:56`).
- **YouTube ingest in the SPA.** Deliberate: SPA is file-upload-only (`PROJECT_TREE.md:47,58`); the API can accept `youtube_url` only when `ATT_ALLOW_YOUTUBE` is on (`backend/main.py:362`, default off `backend/config.py:44`). Desktop has the full paste/search/save flow (`ui/pages/youtube_audio.py`).

Verified present on web (desktop behaviors ported, not gaps): Master fader (`web/src/components/StemMixer.tsx:82,280`), "Download current mix" built on an explicit button not on every fader move (`StemMixer.tsx:508-511`, I-605/I-607 direction), Practice panel for chords/sections/lyrics/tab (`web/src/components/PracticeSheet.tsx`), practice speed & key via SoundTouch (`StemMixer.tsx:288-313`), per-stem downloads above the mixer (`web/src/pages/IsolatePage.tsx:695-701`), seek, region trim picker (`web/src/components/RegionPicker.tsx`), over-cap Start guards and Tab-watch fix (I-622, implemented per register), "Low RAM (60 s)" mode label (I-609, implemented), Auto-first processing select (I-509, implemented).

## P2 nice-to-haves / verified-in-code

- **Web health is green right now, but ungated.** CI runs pytest+ruff over `src/audio_to_tab` only (`.github/workflows/ci.yml:69-94`); `Dockerfile.web:5` builds with `npm run build` (esbuild, no typecheck) and `web/tsconfig.json` is `strict` with `noUnusedLocals/noUnusedParameters`. Today `npx tsc --noEmit` is clean and `npm test` passes 65/65 (verified), so nothing is red — add `tsc --noEmit` + `npm test` to CI or `Dockerfile.web` so it can't regress silently.
- **`PROJECT_TREE.md` is stale for `web/src`.** Missing entries: `components/PracticeSheet.tsx` (+`.test.tsx`), `ThemeSelector.tsx`, `ColorPicker.tsx`, `ColorSwatch.tsx`, `CookieBanner.tsx`, `SiteFooter.tsx`, `StemMixer.layout.test.ts`, `mixer/practice.ts(+test)`, `mixer/analysis.ts(+test)`, `pages/isolatePractice.ts`, `theme/ThemeContext.tsx`, `legal/`. AGENTS.md requires keeping it updated when files are added.
- **Prod image ships dev extras.** `Dockerfile:13` installs `.[dev,eval]` (pytest, ruff, httpx, mir_eval) into the API/worker image — bloat and a larger surface; slim to what `backend` + `jobs` import plus the model extras.
- **MinIO default credentials.** `docker-compose.yml:34-36` / `.env.example:17-18` default `minioadmin`/`minioadmin`. MinIO isn't publicly exposed in compose, but override `ATT_S3_*` on a public host (DEPLOY.md env checklist already says this).
- **Anonymous-session quota multiplier is bounded but real.** `POST /v1/auth/session` is rate-limited 20/min per IP (`backend/main.py:237-238`), but each minted session gets its own per-user upload/job quotas (`_rate_key`, `backend/main.py:159-163`). 20 fresh identities per minute per IP on a demo box is worth capping harder or metering.
- **Known-issue register cross-check:** every row I traced in `docs/ui-issues.md` (I-600–I-623) and the I-500–I-510 rows in `docs/issues.md` matched `implemented-in-code` claims in actual code — no row needed re-filing as an open gap.

## Parity matrix

Desktop feature | Web status | Evidence
---|---|---
Audio Isolation (separate/mix) | ✅ present | `web/src/pages/IsolatePage.tsx`; `POST /v1/isolate/jobs` `backend/main.py:409`
Processing modes fast/balanced/high/lite | ✅ present | `backend/capabilities.py:19-26,130-141`; `web/src/components/ProcessingModeSelect.tsx` (Auto first, I-509)
Device options (cuda/mps≥12GB/cpu) | ✅ present | `backend/capabilities.py:97-105`; capabilities API `web/src/api.ts:145-147`
Guitar engines (guitar-ft/refine/groups) | ⚠️ partial | fields wired `backend/jobs/runner.py:169-189`; copy at `IsolatePage.tsx:42`; **image lacks roformer/separator extras** (`Dockerfile:13` → P0-1)
YouTube to MP3 (save only) | ❌ missing (documented split) | SPA upload-only `PROJECT_TREE.md:47`; API gate `backend/main.py:362`
Tab PDF export | ✅ present | `web/src/pages/TabPage.tsx:137-142` (PDF+MIDI)
Tab PDF honesty warning | ✅ implemented | I-600 implemented-in-code (`docs/ui-issues.md:24`)
Region trim picker | ✅ present | `web/src/components/RegionPicker.tsx`; desktop `ui/region_picker_component/`
Metronome click stem | ⚠️ partial | web plays recorded stem, default muted (`engine.ts:20,128`); no live accent/rate/sound (desktop `metronomeClicks.ts`)
Mixer mute/solo/volume/pan/master | ✅ present | `StemMixer.tsx:82` master; pan law copied (`web/src/mixer/pan.ts` ≡ `ui/.../pan.ts`)
Background playback / wake hooks | ✅ present | `engine.ts:292-306` ≡ desktop `main.ts:324-338`; Media Session + keepPlaying (I-618)
"Download current mix" on Save-style button | ✅ present | `StemMixer.tsx:508-511` (I-605/I-607 direction implemented)
Practice panel (chords/sections/lyrics/tab) | ✅ present | `PracticeSheet.tsx`; fed by `practiceFromJob` (`IsolatePage.tsx:396`)
Practice speed & key | ✅ present | `StemMixer.tsx:288-313`, `web/src/mixer/practice.ts`
A/B loop + count-in | ❌ missing | `clampLoop` dead code (`analysis.ts:124-135`); no engine loop math/count-in
Mixer picture + stem-wave resize | ❌ missing (documented split) | "website mixer is audio-only", `PROJECT_TREE.md:20`
Mix-tabs Home\|mix\|+ strip | ❌ missing (documented split) | web is one stacked page (`docs/ui-issues.md:56`)
Job queue + retry/failed display | ✅ present | `web/src/pages/IsolatePage.tsx` job list + `JobProgress.tsx`, `backend/main.py:479-525`
Login / sign-out / anonymous session | ✅ present | `web/src/pages/LoginPage.tsx`, `/login` route (`web/src/App.tsx`), I-601 implemented
Over-cap submit guards | ✅ implemented | I-622 implemented-in-code (`docs/ui-issues.md:74`)
Dark mode / theme | ✅ web-only extra | `ThemeSelector.tsx`, `theme/ThemeContext.tsx` (desktop theme is a Streamlit menu)
Legal pages / cookie banner / footer | ✅ web-only extra | `web/src/legal/`, `CookieBanner.tsx`, `SiteFooter.tsx`, `docs/legal/`

## Cost / feasibility facts (why GPU matters for "same as desktop")

- Caps are hard-coded per mode: fast_cpu **90 s**, lite **60 s**, high-quality quality tiers beyond that need `max_job_duration_sec` (default **300 s**, `backend/capabilities.py:36-38`, `backend/config.py:65`).
- Hosted Auto: lite host (single-flight or <16 GB RAM) → `lite`; CUDA → `balanced`; MPS ≥12 GB → `balanced`; otherwise `fast_cpu` (`backend/capabilities.py:112-141`). High (GPU) mode requires CUDA and is disabled otherwise (`capabilities.py:125-127,189-193`).
- Worker runs **one job at a time** by design (`backend/worker.py:32 max_jobs = 1`). Full-clip isolate on CPU wants **16–32 GB RAM** per `DEPLOY.md:12`; single-flight lite compose also 503s the second job (`DEPLOY.md:90`).
- Measured reference for CPU hosting: Oracle Always Free 2 OCPU/12 GB is *below* guidance; a synthetic-clip run peaked ~2 GiB RSS for ≤120 s — pass for short clips, not dense mixes (`DEPLOY.md:78-80`). DEPLOY.md quotes a ~16–24 GB x86 VPS at roughly ¥2,500–3,500/month for concurrent testers (`DEPLOY.md:94`).
- Implication for a public free tier: on CPU-only hosting the product is a 60–90 s README demo; full-song, high-quality, Guitar-specialist isolation is where GPU or paying users come in. The 60 s `lite` quota is exactly what makes an unmetered public demo survivable on small hosts.
- Progress UX on weak browsers: WS with 2 s polling fallback (`web/src/api.ts:155-204`).

## What I did NOT check

- The Python test suite was not run (stop condition; CI covers pytest+ruff, see `.github/workflows/ci.yml`).
- No end-to-end compose bring-up: TLS/ACME issuance, model weight downloads over the network, prewarm wall-clock, `docker compose up -d --build` success were not executed by this audit.
- No browser E2E of the SPA flows (login → upload → job → mixer); `npm test`'s jsdom coverage of `StemMixer.layout.test.ts` was run but not exercised in a real browser.
- Desktop runtime (Streamlit app itself) was not launched; parity reads are of code, not of the running desktop app.
- `web/vercel.json` / `web/netlify.toml` split-hosting paths were read in DEPLOY.md but not deployed.
- RoFormer-runtime failure behavior under the missing extras was inferred from `scripts/prewarm.py:44-51` copy; the exact HTTP/error surfaced to a hosted user for a failed roformer isolate job was not traced end-to-end.
