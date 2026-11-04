# Rust feasibility study

**Date:** 2026-11-04
**Repo state:** `main` @ `36d9222` (v0.1.5)
**Question:** Can this project be Rust? Can most of it become Rust so it is more efficient and loads faster?

---

## TL;DR

**Yes, but the win is not where you'd expect.**

- The slow part of this app is **neural-network inference** (Demucs / RoFormer / SCNet / Basic Pitch / Whisper), and that is already **native C++/CUDA** underneath Python. Rewriting the Python glue in Rust will **not** make a 3-minute song separate much faster by itself.
- What a Rust rewrite *does* fix is the part users actually feel: **installer size, cold start, RAM, crash isolation, and distribution**. Today the desktop app is a ~300 MB PyInstaller bundle on top of a ~5 GB venv, boots a Streamlit server inside a pywebview window, and imports Torch/Streamlit before the first paint. Tauri 2 + Rust can get the shell to roughly 10–40 MB and sub-second start.
- As of late 2026 the missing Rust pieces **exist**: `demucs-rs` (HTDemucs v4 in pure Rust/Burn), `demucs-onnx` (Rust ONNX implementation of v3/v4 checkpoints), `basic-pitch-rs` (parity-tested Basic Pitch port), `beat-this` (Rust beat tracker), `typst` + `fretwork` (publication-quality guitar tab PDF), `symphonia`/`rubato`/`rustfft` (decode/resample/DSP), `axum`/`sqlx`/`apalis` (server).
- Realistic split: **~70–80 % of the Python source can move to Rust**, but **~80 % of runtime stays in native inference** no matter what language the glue is. The honest target is "Rust core + native models", not "100 % Rust".
- The two optional separation engines that differentiate this app — **BS-RoFormer-SW and SCNet** — have **no Rust port today**. They are the reason to keep a Python sidecar (or do an ONNX export project) rather than delete Python outright.

**Recommendation:** don't do a big-bang rewrite. Do a staged migration:

1. **Tauri 2 desktop shell** reusing the existing React UI, Python kept as a sidecar (weeks, not months).
2. **Rust audio core** — ingest, DSP, analysis, Basic Pitch, tab generation, PDF — replacing librosa/numpy/scipy/soundfile/reportlab/pretty_midi.
3. **Rust separation** via `demucs-rs`/`demucs-onnx` for the default HTDemucs path; keep Python only for RoFormer/SCNet/Whisper/BTC until each is proven exportable.
4. **Server rewrite** (`axum` + `sqlx` + `apalis`) is optional and lowest value; FastAPI is not the bottleneck.

---

## 1. What the project actually is (measured, not guessed)

| Layer | Stack | Size |
|---|---|---|
| Audio/ML pipeline | Python 3.10–3.12, numpy, librosa, scipy, soundfile, pretty_midi, reportlab, PyTorch, Demucs, BS-RoFormer, SCNet, Basic Pitch (ONNX), faster-whisper, yt-dlp | `src/audio_to_tab` ≈ 13.5k LOC |
| Desktop UI | Streamlit app + 5 custom React/TS components (mixer, region picker, home, mix tabs) | `ui/` ≈ 15k Python LOC + ≈ 6k TS |
| Hosted web | FastAPI + SQLAlchemy + Postgres/SQLite + Redis/arq + S3 + JWT; React 18 + Vite + Web Audio + SoundTouch | `backend/` ≈ 2.5k LOC, 15 endpoints + 1 WebSocket; `web/src` ≈ 7k TS |
| Desktop packaging | PyInstaller onedir → Inno Setup (Windows) / pkg (macOS), pywebview native window, bundled ffmpeg | installer "~300 MB class" (README), venv ~5 GB |
| Tests | pytest | 62 files / ≈18k LOC |

Total Python ≈ 32.4k LOC, frontend TS ≈ 13.3k LOC.

The dependency census confirms the shape:

```
soundfile 24x, librosa 19x, torch 19x, numpy 18x, demucs 7x,
scipy 7x, bs_roformer 6x, pretty_midi 5x, basic_pitch 4x,
yt_dlp 4x, reportlab 3x, faster_whisper 2x
```

`torch` is imported in 19 places, but mostly to *run* models, move them to a device, or `torch.load` a checkpoint. That is orchestration, not computation.

---

## 2. Where the time actually goes

Per-song work today:

| Stage | What runs | Where the CPU/GPU time goes | Python's role |
|---|---|---|---|
| Ingest / ffmpeg | subprocess ffmpeg | native codec | orchestration |
| Stem separation | Demucs / BS-RoFormer / SCNet | **PyTorch C++/CUDA kernels** | config + file I/O |
| Transcription | Basic Pitch | **ONNX Runtime C++ kernels** (or TFLite/CoreML) | wrapper |
| Chords/key/tempo/beats | librosa CQT/HPSS/Viterbi | numpy C loops + BLAS | real Python/Numpy compute, but seconds |
| Tab assignment | `tab_generate.py`, `rhythm.py` | pure Python | **portable, no native deps** |
| PDF | reportlab | C-accelerated drawing | pure Python layout |
| Lyrics | faster-whisper | **CTranslate2 C++** | wrapper |

So the app is a **native-inference program with a Python control plane**. A Rust control plane is a legitimate goal (distribution, startup, safety), but "efficiency" claims must be separated from "speed of the slow stage".

One nuance: on CPU, replacing PyTorch with ONNX Runtime / `tract` / `burn` can be **faster per song** because of better threading and no Python/GIL overhead, but the model math dominates. Expect single-digit-to-30 % gains on separation, not 5×. The big measurable wins are cold start (seconds → sub-second) and memory (hundreds of MB → tens of MB for the shell).

---

## 3. Piece-by-piece feasibility

Legend: **A** = straightforward Rust port; **B** = feasible with a known crate, parity work needed; **C** = no mature Rust path, keep Python or invest in ONNX/Burn export.

| Component (current) | Rust option | Maturity (Nov 2026) | Effort | Verdict |
|---|---|---|---|---|
| Decode MP3/AAC/FLAC/WAV/OGG/MP4/MKV (`soundfile`+ffmpeg) | `symphonia` 0.6 (pure Rust, SIMD), `lofty` for tags | High | S | **A** — drop `soundfile`; keep ffmpeg for exotic/video |
| Resample (`librosa`/soxr) | `rubato` 5.x (SIMD, async) | High | S | **A** |
| STFT/CQT/HPSS/chroma/onset (`librosa`, `scipy`) | `rustfft`, `realfft`, `spectrum-analyzer`, `fundsp`; hand-port CQT/HPSS | Medium (primitives high, music-specific algorithms DIY) | M–L | **B** — biggest hand-porting surface |
| Tempo/beats (`librosa.beat`) | `beat-this` crate (Beat This! port, MIT, `rten`/`ort`, symphonia+rubato) or `aubio` bindings (**GPL-3**, license trap) | Medium | M | **B** — `beat-this` is the clean path |
| Chord templates + Viterbi (`chords.py`) | plain Rust (`ndarray` optional) | n/a | M | **A** — ~670 LOC of pure logic |
| Neural chords (BTC `.pt`) | export to ONNX → `ort`, or port to Burn | Low (no known port) | M–L | **C/B** — ONNX export is the pragmatic route |
| Basic Pitch (`basic_pitch` + ONNX) | `basic-pitch-rs` (Apache-2.0, parity-tested against Python, bundled `nmp.onnx`, `tract` default or `ort`, WASM) | New (Oct 2026) but designed exactly for this | S–M | **B** — strongest single win; test against fixtures |
| Demucs HTDemucs v4 / 6s | `demucs-rs` (Burn, `htdemucs`/`_6s`/`_ft`, Metal/Vulkan/WebGPU, WASM) **or** `demucs-onnx` (Rust + ONNX Runtime, v3/v4 checkpoints, CPU/CUDA/TensorRT/DirectML/CoreML) | New (2026), active | M–L | **B** — default path can leave PyTorch |
| guitar-ft checkpoint (`htdemucs_6s_guitar_ft`, custom `.pt`) | export that checkpoint to ONNX and use `demucs-onnx`; or keep PyTorch | Unknown export | M | **C** until export is proven |
| BS-RoFormer-SW / MelBand RoFormer (`bs_roformer_infer`) | none | — | XL | **C** — keep Python sidecar |
| SCNet (`scnet.py`) | none | — | XL | **C** — keep Python sidecar |
| Whisper lyrics (`faster-whisper`/CTranslate2) | `whisper-rs` 0.16 (whisper.cpp), `whisper-cpp-plus` (streaming/VAD) | High | M | **B** — quality parity check needed |
| MIDI read/write (`pretty_midi`) | `midly`, `midi-file` | High | S | **A** |
| Tab fret assignment + ASCII (`tab_generate.py`) | port directly; evaluate `guitar-tab-generator` crate | Medium | M | **A/B** |
| PDF tab (`reportlab`) | `typst` 0.15 + `fretwork` 0.4 Typst package (publication-quality tab: rhythm stems, bends, slurs, palm mutes) | Medium-high | M | **B** — better output, new template |
| Zip/artifacts | `zip` crate | High | S | **A** |
| YouTube (`yt-dlp`) | no equivalent | — | — | **C** — keep `yt-dlp` as a bundled sidecar binary; it is already a subprocess |
| Desktop shell (pywebview + Streamlit server) | **Tauri 2** + existing React UI | High | M | **A** — biggest UX win |
| Desktop UI (Streamlit, ~15k LOC) | delete; reuse `web/` React app with Tauri plugins | High | L | **A** strategically, L effort |
| Server (FastAPI, 15 endpoints + WS) | `axum` + `tokio` + `sqlx` + `apalis` + `tower-governor` + `argon2` + `jsonwebtoken` + `aws-sdk-s3` | High | M | **B** — cheap, low value |
| Job queue (`arq`/Redis) | `apalis` (Redis/Postgres backends, retries, unique jobs, leases) | High | S–M | **B** |
| DB (`SQLAlchemy`/Alembic) | `sqlx` migrations | High | S | **A** |
| Auth (JWT, bcrypt) | `jsonwebtoken`, `argon2` | High | S | **A** |

### Estimated code that can move

- **Directly portable (A):** ingest/normalize, resample, mixer math, MIDI, tab/rhythm/quantize, chord symbols, key profiles, export text/MusicXML, PDF layout, zip/storage, server/auth/DB/jobs.
- **Portable with parity work (B):** CQT/chroma/HPSS/onset, beat tracking, Basic Pitch integration, Demucs path, Typst templates, Whisper.
- **Must stay Python or become a research project (C):** BS-RoFormer, SCNet, the custom guitar-ft checkpoint, BTC neural chords, `yt-dlp`.

That lands at **≈70–80 % of source LOC** and **≈50–60 % of runtime code paths** realistically in Rust in a first serious pass. The remaining Python shrinks from a 32k-LOC app to a **narrow ML sidecar** (a few files + PyTorch wheels), which is a much better shape even if it never reaches zero.

---

## 4. The three architectures

### Option A — Full Rust (no Python at all)

Replace every model with a Rust-runnable graph, including RoFormer and SCNet.

- **Pros:** one toolchain, one binary, smallest install, no venv, no PyInstaller, no GIL.
- **Cons:** RoFormer/SCNet/BTC have no Rust implementations. You would be exporting complex attention models to ONNX and maintaining them, plus re-validating every accuracy heuristic. High risk to the thing users care about (separation quality).
- **Verdict:** not justified today.

### Option B — Rust core + thin Python ML sidecar (**recommended**)

```
Tauri 2 shell (React UI)
   │  commands / HTTP over localhost or stdio
   ▼
Rust core  ── ingest, DSP, analysis, Basic Pitch, tab, PDF, job control
   │
   ├─ demucs-rs / demucs-onnx  (default separation)
   └─ Python sidecar (optional engines only: RoFormer, SCNet, guitar-ft, BTC)
         └─ spawned on demand, JSON-RPC, frozen with PyInstaller as a single optional component
```

- **Pros:** gets 80 % of the benefits; keeps the differentiating engines; sidecar can be optional (small download without it); clean failure isolation.
- **Cons:** two runtimes to ship; sidecar still large when installed.

### Option C — Keep Python, swap only the desktop shell (Tauri + pywebview replacement)

- **Pros:** 3–6 weeks; immediate startup/size/RAM win; no ML risk.
- **Cons:** Torch still ships; startup still imports Python for the pipeline; doesn't remove the 5 GB dev/runtime footprint.

**Recommended sequence is C → B**, i.e. ship the shell first, then move the core inward. Option A only if the optional engines are declared out of scope.

---

## 5. What "loads nicely" would actually look like

| Metric | Today (PyInstaller + Streamlit + pywebview) | Tauri 2 + Rust core (est.) |
|---|---|---|
| Installer (no models) | ~300 MB class | 15–40 MB (shell + ORT + Rust) |
| Models on disk | Demucs 84 MB, guitar-ft 330 MB, RoFormer/SCNet GB-class | same models, downloaded on demand |
| Cold start to first paint | several seconds (Streamlit server + Python imports) | < 1 s (Tauri + static frontend) |
| Idle RAM (shell) | hundreds of MB | 30–80 MB typical Tauri |
| First separation | PyTorch load + inference | ORT/Burn session load + inference (comparable, sometimes faster on CPU) |
| Crash blast radius | one Python process | UI survives; worker crash is recoverable |

Caveats: Tauri uses the OS WebView (WebView2 on Windows, WKWebView on macOS, WebKitGTK on Linux), so rendering is less uniform than Electron/Chromium; some users still need the WebView2 runtime on old Windows 10. And model weights dominate the download once optional engines are installed — no shell rewrite changes that.

---

## 6. Hard problems and traps

1. **Accuracy parity is the real project.** `docs/chord-recognition/` and `docs/plans/moeses-parity/` show a lot of tuned heuristics and benchmark fixtures. Porting DSP changes numbers; the 62 test files / 18k LOC of tests are the safety net. Budget for golden-file comparison (MIDI notes, chord spans, tab output), not just "it compiles".
2. **ONNX Runtime packaging.** `ort` is production-grade, but on Windows you must ship `onnxruntime.dll` next to the exe (Windows also has an older system copy). Use `load-dynamic` + a pinned runtime, and test on a clean machine.
3. **Licensing.**
   - `aubio` Rust bindings are **GPL-3.0**; this repo is MIT. Avoid unless you accept GPL or use `beat-this`/own DSP.
   - Rubber Band is GPL; the web mixer deliberately uses `@soundtouchjs/audio-worklet` (MPL-2.0). Preserve that choice; a Rust time-stretch should use a permissive crate (e.g. Signalsmith Stretch bindings) or stay in Web Audio.
   - Meta Demucs **weights** are research-purpose; the code is MIT. The guitar-ft fine-tune is Apache-2.0 but does not change the base weights' terms. Keep NOTICES current.
   - `yt-dlp` is Unlicense but changes constantly; keep it as an updatable sidecar, not vendored source.
4. **Model export is where schedules die.** Exporting the custom `guitar_htdemucs_6s.pt`, BTC, BS-RoFormer, and SCNet graphs to ONNX is not guaranteed. Prototype each export before committing to "no Python".
5. **Two UIs, one product.** The repo already has a React app for the web and a Streamlit app for desktop. The migration's biggest simplification is **deleting the Streamlit UI** and pointing Tauri at the React app. That is also the largest UI work item (~15k Python LOC + desktop-only features: native file picker, notifications, Open With, "save audio to folder").
6. **Job cancellation.** Python's current cancel story is weak (README: a Demucs child can outlive Streamlit Stop). Rust + Tokio gives real cancellation, but only if inference is in-process and the runtime supports it; a Python sidecar reproduces the current limitation.
7. **GPU matrix.** PyTorch already covers CUDA and Apple MPS. `ort` covers CUDA/TensorRT/DirectML/CoreML; `burn` covers Metal/Vulkan/WebGPU. Verify per-platform before promising speedups.

---

## 7. Suggested migration plan

**Phase 0 — contracts (1–2 weeks).**
Freeze `analysis.json` schema, job/artifact contracts, and golden fixtures. Add a parity harness that runs old vs new on `eval/fixtures/*` and `docs/chord-recognition/bench/*` and diffs notes/chords/tabs.

**Phase 1 — Tauri shell (3–6 weeks).**
Replace pywebview + Streamlit with Tauri 2 hosting the existing `web/` React app. Keep the Python backend as a sidecar. Ship. Measure installer, start, RAM. This is the risk-free win and validates the UX.

**Phase 2 — Rust audio core (8–14 weeks).**
`symphonia` + `rubato` + `rustfft` + `beat-this` + ported chord/key/tab/rhythm logic + `basic-pitch-rs` + `midly` + `typst`/`fretwork` PDF. At the end, a solo-guitar tab can be produced with **zero Python**.

**Phase 3 — Rust separation (4–8 weeks).**
Integrate `demucs-onnx` (or `demucs-rs`) for `htdemucs` / `htdemucs_6s`. Export and validate guitar-ft. Keep the Python sidecar for RoFormer/SCNet/BTC/guitar-ft until each is proven.

**Phase 4 — optional consolidation.**
Rewrite the server in `axum`/`sqlx`/`apalis` only if ops cost or consistency justifies it. Move Whisper to `whisper-rs` if lyrics quality matches. Otherwise stop at Phase 3 — that is already "most of it in Rust".

**Total:** roughly 4–6 months for an experienced Rust developer, 6–12 months solo/part-time, with Phase 1 shippable much earlier.

---

## 8. Bottom line

- **Can it be Rust?** Yes — the architecture is unusually friendly to it because the heavy lifting is already native and model formats (ONNX, safetensors, WAV) are language-neutral.
- **Can most of it be Rust?** Yes, ~70–80 % of the source and the entire "boring" surface (I/O, DSP, MIDI, tab, PDF, server). The last 20 % is the model zoo, and that is better isolated than deleted.
- **Will it be more efficient?** The shell will be dramatically lighter and faster to start; per-song inference will be roughly the same unless you also change runtime (PyTorch → ONNX Runtime/Burn), where modest gains are plausible.
- **Should you do it?** If the pain is install size, startup, RAM, and shipping complexity: **yes, start with Tauri + Rust core**. If the pain is separation accuracy or GPU coverage: **no, that is a model problem, not a language problem**.

---

## References (checked 2026-11-04)

- `demucs-rs` — HTDemucs v4 in Rust/Burn, native + WASM + VST3: <https://github.com/nikhilunni/demucs-rs>
- `demucs-onnx` — Demucs v3/v4 on ONNX Runtime, Rust/C++/Java/C#/JS implementations: <https://github.com/HRSadeghi/demucs-onnx>
- `basic-pitch-rs` — parity-tested Rust Basic Pitch, bundled model, tract/ort, WASM: <https://github.com/AThevon/basic-pitch-rs>
- `beat-this` crate — Rust port of Beat This!, `rten`/`ort`, symphonia+rubato: <https://crates.io/crates/beat-this>
- `ort` — Rust ONNX Runtime bindings (2.0 RC line): <https://ort.pyke.io/>
- `whisper-rs` — whisper.cpp bindings: <https://crates.io/crates/whisper-rs>
- `symphonia` 0.6 — pure-Rust decoding/demuxing: <https://crates.io/crates/symphonia>
- `rubato` 5.x — resampling: <https://crates.io/crates/rubato>
- `typst` 0.15 / `typst-pdf`: <https://crates.io/crates/typst>
- `fretwork` 0.4 — publication-quality guitar tab in Typst: <https://typst.app/universe/package/fretwork>
- `guitar-tab-generator` crate (evaluate before adopting): <https://docs.rs/guitar-tab-generator>
- Tauri 2 sizing/startup comparisons: <https://newstack.dev/compare/tauri-vs-electron>, <https://prod0.dev/compare/electron-vs-tauri>
- `axum` + `sqlx` + `apalis`: <https://docs.rs/apalis>, <https://encore.dev/articles/add-postgres-to-axum>
- `aubio` Rust bindings (GPL-3.0): <https://docs.rs/aubio>
- Spotify Basic Pitch (model/runtime facts): <https://github.com/spotify/basic-pitch>
- Meta Demucs (weights/licensing): <https://github.com/facebookresearch/demucs>
