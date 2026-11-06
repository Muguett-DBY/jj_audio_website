# Contributing

Thanks for helping. This repo is two products on one engine:

- **Desktop** — Streamlit in `ui/` (local mixer / isolation / YouTube to MP3 / Tab PDF)
- **Website** — FastAPI `backend/` + React `web/`
- **Shared** — `src/audio_to_tab/` (import as `audio_to_tab`)

Desktop **Interface Lite** is not the hosted processing mode named `lite`. Do not unify them.

## Find work

| Looking for | Filter |
|---|---|
| Newcomer-friendly | [good first issue](https://github.com/notjj1234/jj_audio_website/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) |
| Open to outside help | [help wanted](https://github.com/notjj1234/jj_audio_website/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) |

Comment on the issue before starting so we do not double up. Prefer fixing an existing issue over opening a large feature without discussion.

## Setup

Python **3.10–3.12** (prefer 3.11). From a clone:

```bash
make install
make test
```

Windows: `.\scripts\dev.ps1 install` then `.\scripts\dev.ps1 test`.

Useful local surfaces:

| What | Command |
|---|---|
| Desktop UI | `make ui` → http://127.0.0.1:8501 |
| API | `make backend` → :8000 |
| Website SPA | `make web` → :5173 (proxies `/v1` → :8000) |
| Lint (engine only) | `ruff check src/audio_to_tab` |

More: [README.md](README.md), [DESKTOP.md](DESKTOP.md), [AGENTS.md](AGENTS.md), [PROJECT_TREE.md](PROJECT_TREE.md).

## Before you open a PR

1. Repro the issue (or say what you could not repro).
2. Keep the diff tight — one issue per PR when possible.
3. Run the tests that cover your change (`make test`, or a focused `pytest` path).
4. If you edit a committed Streamlit iframe frontend, rebuild it:
   - Home → `make home-build`
   - Stem mixer → `make mixer-build`
   - Region picker → `make region-picker-build`
   - Mix tabs → `make mix-tabs-build`
5. Do not commit `.env`, `data/`, model weights, or hand-edited hashed files under `*/frontend/build/assets/`.
6. Use the PR template. Link the issue.

## Issue labels we use for contributors

- **`good first issue`** — small, clear steps, safe for a first PR
- **`help wanted`** — we want outside help; may still need product context
- **`bug` / `enhancement`** — type of change

If an issue says it is *not* a good-first issue, believe that and treat it as `help wanted` only.
