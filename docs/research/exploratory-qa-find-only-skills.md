# Skills for find-only app testing → GitHub issues

Research date: 2026-11-05.  
Scope: off-the-shelf Agent Skills (Cursor / Claude Code / Codex-compatible `SKILL.md`) that help an agent **find** product bugs and produce **issue-ready reports**, without fixing code.  
Companion to: [`cursor-agent-testing-skills.md`](./cursor-agent-testing-skills.md) (broader Cursor testing stack, Playwright CLI, a11y).

Primary sources only for install/claims: [Cursor Skills docs](https://cursor.com/docs/skills), vendor GitHub `SKILL.md` / skills.sh pages cited below.

## Verdict (what to add)

For your goal (*tester finds problems → you post GitHub issues*), adopt this **pair**:

| Priority | Skill | Install | Role |
|---|---|---|---|
| **1 — explore** | **`dogfood`** ([vercel-labs/agent-browser](https://github.com/vercel-labs/agent-browser)) | `npx skills add https://github.com/vercel-labs/agent-browser --skill dogfood` (pick Cursor) | Live-app bug hunt; screenshots/video; repro steps; **no fix loop** |
| **1b — alternate explore** | **`browser-uat-analyst`** ([microsoft/cat-agent-skills](https://github.com/microsoft/cat-agent-skills)) | `npx skills add https://github.com/microsoft/cat-agent-skills --skill browser-uat-analyst` | Evidence-first UAT; every claim needs a screenshot; sceptical analyst voice |
| **2 — file issues** | **`qa`** ([mattpocock/skills](https://github.com/mattpocock/skills)) | `npx skills add https://github.com/mattpocock/skills --skill qa` | Turns findings (or your verbal bugs) into durable GitHub issues; explores code for **context only**, not fixes |

Optional third: **`github-issue-writer`** ([tomlord1122/tomtom-skill](https://www.skills.sh/tomlord1122/tomtom-skill/github-issue-writer)) if you want a dedicated `gh issue create` formatter after dogfood.

**Do not rely on Cursor built-ins alone** for this job. `/review`, `/review-bugbot`, `/review-security` are **diff/code** reviewers ([Cursor docs](https://cursor.com/docs/skills)), not product explorers.

Cursor loads project + user skills from `.cursor/skills/`, `.agents/skills/`, and also Claude/Codex dirs (`.claude/skills/`, `.codex/skills/`, `~/…`) ([Cursor docs](https://cursor.com/docs/skills)).

## Ranked candidates

Fit score = how well the skill matches **find-only + issue board**, for this Streamlit/web repo (browser at `:8501` / Vite).

| Rank | Skill | Source | Find-only? | Issue-ready? | Cursor install | Fit |
|---:|---|---|---|---|---|---:|
| 1 | `dogfood` | [vercel-labs/agent-browser `skill-data/dogfood`](https://github.com/vercel-labs/agent-browser/blob/main/skill-data/dogfood/SKILL.md); [agent-browser skills](https://agent-browser.dev/skills) | Yes (explore + report; no fix) | Hand-off report; pair with `qa` / `gh` | `npx skills add … --skill dogfood` | **5** |
| 2 | `browser-uat-analyst` | [skills.sh microsoft/cat-agent-skills](https://www.skills.sh/microsoft/cat-agent-skills/browser-uat-analyst) | Yes (evidence-first UAT) | Structured observed states | `npx skills add … --skill browser-uat-analyst` | **5** |
| 3 | `qa` (Matt Pocock) | [skills.sh mattpocock/skills/qa](https://www.skills.sh/mattpocock/skills/qa) | Partial (user-described + light explore) | Explicitly files GitHub issues | `npx skills add … --skill qa` | **4** (best **filing** skill) |
| 4 | `exploratory-testing` (agentic-dev-team) | [bdfinst/agentic-dev-team](https://github.com/bdfinst/agentic-dev-team/blob/main/plugins/dev-team/skills/exploratory-testing/SKILL.md) | Yes (charter, probe budget, **never fix**) | Report + triage handoff | Copy plugin skill / marketplace | **4** |
| 5 | qualiow `/qa-explore*` | [willcoliveira/qualiow-exploratory-testing-skills](https://github.com/willcoliveira/qualiow-exploratory-testing-skills) | Yes (SBTM-style explore) | Executive bug reports | Claude plugin marketplace; skills are portable `SKILL.md` | **4** (heavier; Claude-first packaging) |
| 6 | `qa-generating-bug-reports` | [wizeline/sdlc-agents](https://www.skills.sh/wizeline/sdlc-agents/qa-generating-bug-reports) | N/A (formatter) | Standardizes bug report shape | skills.sh add | **3** (use after explore) |
| 7 | `qa-reporter` | [fikyanggra/skills](https://www.skills.sh/fikyanggra/skills/qa-reporter) | Report-only (no execute) | Issue packages; no submit unless asked | skills.sh add | **3** |
| 8 | Firegoat `ui-test` | [skills.sh firegoat-ai/skills/ui-test](https://www.skills.sh/firegoat-ai/skills/ui-test) | Yes (try to break) | Results merge; not GitHub-native | skills.sh add | **3** |
| 9 | e2e.com bug bash | [e2e.com/docs/bug-bash](https://e2e.com/docs/bug-bash); `npx skills add tester-army/e2e` | Explore yes | Confirms bugs via **failing tests** | skills CLI | **2** (pulls toward automation, not issue board) |
| 10 | `qa-explore` plugin | [victoraguilarsantamariadev/qa-explore](https://github.com/victoraguilarsantamariadev/qa-explore) | Explore + file tracker issues | Strong | Claude plugin | **2** for you — also has **`qa-fix`** (auto-fix path) |
| — | `bug-hunt` (agentops) | [skills.sh boshu2/agentops/bug-hunt](https://www.skills.sh/boshu2/agentops/bug-hunt) | Audit mode finds; investigation designs **fix** | Beads/issues | skills.sh | **1** for find-only (fix-oriented) |
| — | Cursor `/review*` | [cursor.com/docs/skills](https://cursor.com/docs/skills) | Code review | PR comments | Built-in | **1** for product UAT |

## What Cursor does *not* ship

Built-in skills listed in [Cursor docs](https://cursor.com/docs/skills) cover automate, review, create-skill/rule, shell, etc. **There is no built-in exploratory QA / dogfood / bug-bash skill.** You add community or first-party Microsoft/Vercel skills, or write a thin project skill.

## Installed in this workspace (2026-11-05)

Skills live under project `.cursor/skills/` (and mirrored in `.agents/skills/`; both dirs are
**gitignored**) and were also copied to `~/.cursor/skills/` for Cursor discovery:

| Skill | Notes |
|---|---|
| `dogfood` | From vercel-labs/agent-browser `skill-data/dogfood` (not in the thin CLI catalog) |
| `agent-browser` | Discovery stub → `agent-browser skills get …` |
| `browser-uat-analyst` | microsoft/cat-agent-skills |
| `github-issue-writer` | tomlord1122/tomtom-skill (scripts patched off Tom’s absolute paths) |
| `qa` | Local skill matching skills.sh mattpocock/qa intent (upstream `--skill qa` not in public repo catalog) |
| `exploratory-qa` | Thin find-only orchestrator for this product |

CLI: `agent-browser` 0.27.0 + Chrome for Testing installed via `agent-browser install`.

### Re-install commands

```bash
# Explore live UI (Streamlit :8501 or web :5173)
npx skills add https://github.com/vercel-labs/agent-browser --skill agent-browser -y
npx skills add https://github.com/microsoft/cat-agent-skills --skill browser-uat-analyst -y
npx skills add https://github.com/tomlord1122/tomtom-skill --skill github-issue-writer -y
# dogfood: copy from skill-data/dogfood (skills CLI catalog only lists agent-browser)

npm install -g agent-browser && agent-browser install
```

Session contract (paste into the chat even with skills loaded):

- Find-only: no code edits, no commits, no “quick fixes”
- Output: issue drafts (title, area, severity, steps, expected/actual, evidence paths)
- Prefer `make tester` / `DESKTOP.md` for desktop; do not invent UI backdoors
- Optional: `gh issue create` only when you explicitly say so

## Optional: thin project skill

If off-the-shelf skills drift toward fixing (skill router → `systematic-debugging`), add a project skill  
`.cursor/skills/exploratory-qa/SKILL.md` that:

1. Loads `dogfood` / browser UAT posture  
2. Hard-forbids edits and `systematic-debugging`  
3. Maps areas (Home, Isolation, mixer, YouTube, Tab PDF, Lite/Pro) from `DESKTOP.md`  
4. Emits GitHub issue markdown only  

That is the gap called out in [`cursor-agent-testing-skills.md`](./cursor-agent-testing-skills.md): no stock skill fully encodes *this* product’s surfaces.

## Caveats for Audio Tools

- Streamlit UI is browser-reachable at `127.0.0.1:8501`; native installer / pywebview shell still needs a human ([prior research](./cursor-agent-testing-skills.md)).
- `dogfood` prefers not reading app source; your repo rules often load `graphify` first — for pure UAT, tell the agent to treat graphify as optional orientation only, not the bug source.
- Skills that **codify Playwright tests** or **open fix MRs** (`qa-explore`, e2e bug bash) are useful later; they fight a pure “issues board” workflow.

## Sources

- [Cursor Agent Skills](https://cursor.com/docs/skills)
- [vercel-labs/agent-browser dogfood SKILL.md](https://github.com/vercel-labs/agent-browser/blob/main/skill-data/dogfood/SKILL.md)
- [agent-browser.dev/skills](https://agent-browser.dev/skills)
- [microsoft/cat-agent-skills browser-uat-analyst](https://www.skills.sh/microsoft/cat-agent-skills/browser-uat-analyst)
- [mattpocock/skills qa](https://www.skills.sh/mattpocock/skills/qa)
- [bdfinst agentic-dev-team exploratory-testing](https://github.com/bdfinst/agentic-dev-team/blob/main/plugins/dev-team/skills/exploratory-testing/SKILL.md)
- [willcoliveira/qualiow-exploratory-testing-skills](https://github.com/willcoliveira/qualiow-exploratory-testing-skills)
- [e2e.com bug bashes](https://e2e.com/docs/bug-bash)
