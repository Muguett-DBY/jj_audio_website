# Cursor agent skills for app testing and exhaustive issue finding

Research date: 2026-11-04. Scope: Cursor (not Claude Code), Agent Skills (`SKILL.md`), MCP
browser tooling, and whole-repo audit skills. Claims are tied to the page or repo that owns
them; marketplace listicles were used only to discover candidates, never as evidence.

## Verdict

- **Cursor has native Agent Skills since 2.4 (2026-01-22), and they are the same `SKILL.md`
  folders Claude Code uses.** Cursor also loads `~/.claude/skills/` and `.claude/skills/`
  directly, so the existing Claude shelf is already visible in Cursor. No conversion step.
- **Subagents exist in Cursor too, since the same 2.4 release.** `.cursor/agents/` and
  `.claude/agents/` are both read; the delegation tool is called Task. Skills that say
  "dispatch a subagent" work. Skills that depend on Claude-only frontmatter (`context: fork`,
  `agent`, `model`) silently lose that behavior.
- **The best off-the-shelf app tester is Microsoft's `playwright-cli` + its official skill,
  not Playwright MCP.** It is a CLI plus a `SKILL.md`, installs into `.agents/skills/`
  (which Cursor reads), and avoids loading MCP tool schemas and accessibility trees into
  context. Adopt this first.
- **Cursor's built-in Browser is the zero-setup exploratory pass.** It already drives pages,
  reads console/network, and takes screenshots. Use it for "poke the app"; use Playwright for
  anything repeatable or asserted.
- **For accessibility, adopt Chrome DevTools MCP plus its official `a11y-debugging` skill.**
  It has Lighthouse, the accessibility tree, console issue audits, and keyboard/focus
  guidance. No credible standalone axe MCP was found.
- **For visual regression, write 20 lines of Playwright.** Playwright's `toHaveScreenshot()`
  is the actual engine. The one third-party skill found is a 1-star wrapper around it; not
  worth the dependency.
- **Nothing good exists for driving a Streamlit desktop app or its native pywebview shell.**
  The Streamlit server is reachable at `127.0.0.1:8501`, so browser tools can test the UI.
  The native window, installer, and single-instance lock cannot be driven by any skill found.
- **For exhaustive issue finding, use the three static sweeps you already have
  (`code-quality-audit`, `ponytail-audit`, Cloudflare `security-audit`) plus
  `workersio/wio` for test-gap and edge-case discovery.** No skill drives a running app and
  enumerates every broken flow; that one has to be written.
- **`security-audit` is the only genuinely exhaustive, coverage-led skill in the local set.**
  It uses a coverage ledger, isolated hunters, adversarial verifiers, and schema-validated
  findings. The two `code-*review*` skills are diff-only and should not be used for this job.
- **Trail of Bits' `property-based-testing` fits this repo's Python/audio math.**
  Its `mutation-testing` skill does not: it targets mewt/muton, and mewt supports
  C++, Go, JS/TS, Ruby, Rust, Solidity, Move, DAML — not Python.
- **Correct one common belief: oversized skills are not skipped by Cursor.** Cursor warns in
  `/skills` when name > 64 chars, description > 1,024 chars, name/folder mismatch, or body
  over about 5,000 tokens, but still loads them. Invalid YAML is the real failure mode.

## Cursor skill support

### Native support and version

Agent Skills support shipped in **Cursor 2.4 on 2026-01-22**. The 2.4 changelog says: "Cursor
now supports Agent Skills in the editor and CLI. Agents can discover and apply skills when
domain-specific knowledge and workflows are relevant. You can also invoke a skill using the
slash command menu." ([Cursor 2.4 changelog](https://cursor.com/changelog/2-4)) The docs page
is now the canonical reference ([cursor.com/docs/skills](https://cursor.com/docs/skills)).

Subagents shipped in the same release, including default subagents for codebase research,
terminal commands, and parallel work streams. ([Cursor 2.4
changelog](https://cursor.com/changelog/2-4))

### Where skills live

| Location | Scope |
|---|---|
| `.agents/skills/` | Project |
| `.cursor/skills/` | Project |
| `~/.agents/skills/` | User (global) |
| `~/.cursor/skills/` | User (global) |
| `.claude/skills/`, `.codex/skills/` | Project, Claude/Codex compatibility |
| `~/.claude/skills/`, `~/.codex/skills/` | User, Claude/Codex compatibility |

Source: [Skill directories](https://cursor.com/docs/skills#skill-directories). Cursor walks the
skills root recursively, including nested project directories such as
`apps/web/.cursor/skills/`, and scopes nested skills to files under that directory
([Nested skill directories](https://cursor.com/docs/skills#nested-skill-directories)).

### Required frontmatter and limits

Cursor requires exactly `name` and `description`:

| Field | Required | Constraint |
|---|---|---|
| `name` | Yes | Lowercase letters, numbers, hyphens; must match the parent folder name |
| `description` | Yes | Used by the agent to decide relevance |
| `paths` | No | Glob or comma-separated globs; scopes the skill to matching files |
| `disable-model-invocation` | No | `true` makes it slash-only |
| `icon`, `color` | No | Custom Mode badge styling |
| `metadata` | No | Arbitrary key/value |

Source: [Frontmatter fields](https://cursor.com/docs/skills#frontmatter-fields).

The numeric limits come from the Agent Skills spec and Cursor's CLI warning logic, not from
the table above:

- `name`: max 64 characters, lowercase alphanumeric plus hyphens, no leading/trailing hyphen,
  no consecutive hyphens, must match the folder. ([Agent Skills
  specification](https://agentskills.io/specification))
- `description`: max 1,024 characters. ([Agent Skills
  specification](https://agentskills.io/specification))
- Body: under about 5,000 tokens recommended; keep `SKILL.md` under 500 lines and move detail
  into `references/`. ([Progressive
  disclosure](https://agentskills.io/specification#progressive-disclosure))
- Cursor's `/skills` shows a yellow warning count for: name over 64 characters or not
  lowercase-hyphen form, frontmatter `name` not matching the folder, description over 1,024
  characters, and a body over about 5,000 tokens. It lists what to fix; it does **not** refuse
  to load the skill. ([Cursor CLI
  changelog](https://cursor.com/docs/cli/changelog))

### Auto-invocation vs explicit invocation

- **Automatic.** On startup Cursor discovers skills and presents them to the agent. "The agent
  is presented with available skills and decides when they are relevant based on context."
  ([How skills work](https://cursor.com/docs/skills#how-skills-work))
- **Explicit.** Type `/` in Agent chat and pick the skill. A skill invoked this way attaches to
  one message. Press Option+Enter (Mac) / Alt+Enter (Windows) to keep it on for the session as
  a Custom Mode. ([How skills work](https://cursor.com/docs/skills#how-skills-work))
- **Not `@`.** Cursor's `@` mention applies to files and rules, not skills. ([Rules, manual
  application](https://cursor.com/docs/rules#rule-anatomy))
- **Opting out of auto.** `disable-model-invocation: true` makes a skill behave like a
  traditional slash command. ([Disabling automatic
  invocation](https://cursor.com/docs/skills#disabling-automatic-invocation))

### Do Claude Code skills load as-is?

**Directory-wise, yes.** Cursor explicitly loads `.claude/skills/` and `~/.claude/skills/`
for compatibility ([Skill directories](https://cursor.com/docs/skills#skill-directories)).
A Claude skill folder dropped in either location appears in Cursor without edits.

**Frontmatter-wise, only the spec fields survive.** Claude Code's docs are explicit that
`argument-hint`, `user-invocable`, `allowed-tools`, `disallowed-tools`, `model`, `context:
fork`, `agent`, `hooks`, and similar fields are Claude Code extensions. Outside Claude Code,
only `name`, `description`, `license`, `compatibility`, `metadata`, and `allowed-tools` are
usable. ([Using skill frontmatter outside Claude
Code](https://code.claude.com/docs/en/skills#using-skill-frontmatter-outside-claude-code))
Cursor's own table lists only `name`, `description`, `paths`, `disable-model-invocation`,
`icon`, `color`, `metadata`, so the Claude extensions are ignored rather than rejected.

Consequences for porting:

- `context: fork` does not fork. The skill body runs inline in the main context. A long audit
  skill that expected a clean subagent window will eat the main window.
- `agent: Explore` / `agent: Plan` do nothing.
- `allowed-tools` does not pre-approve anything. Cursor still asks for tool approval.
- `argument-hint` and `user-invocable` do nothing. Cursor's `/` menu still lists the skill.
- Claude plugin-only installs are not one of the four compatibility directories. A skill that
  exists only in Claude's plugin cache is **not** visible in Cursor; copy the skill folder.

**Tool-name compatibility.** Cursor has a Task tool for subagents ([Cursor
subagents](https://cursor.com/docs/subagents)), so text like "dispatch a subagent" or "use the
Task tool" maps. Claude-only tools do not: `SlashCommand` has no Cursor equivalent, and
Claude Code's `Skill` tool is not how Cursor loads skills. Skills that call plugin commands
(`/plugin`, `/goal`) or read `$HOME/.claude/get-shit-done/...` are not portable. MCP tool
names are also rewritten: Cursor exposes MCP tools with a server prefix (reported as
`mcp_<server>_<tool>`), not Claude's `mcp__server__tool`. ([Cursor forum, MCP tool name
prefix](https://forum.cursor.com/t/in-cursor-version-0-47-8-agent-mode-incorrectly-prefixes-mcp-tool-names-with-mcp-servername-causing-tools-to-fail/72778))
Skills that hard-code Claude's MCP spelling need a rename pass.

### Does Cursor support subagents?

**Yes.** Cursor has custom subagents in `.cursor/agents/` and `~/.cursor/agents/`, plus
`.claude/agents/` and `.codex/agents/` compatibility. Each subagent has its own context
window, can run foreground or background, and is launched through the Task tool. Custom
subagents use YAML frontmatter (`name`, `description`, `model`, `readonly`, `is_background`).
([Cursor subagents](https://cursor.com/docs/subagents))

Since Cursor 2.5, subagents can launch child subagents within a nesting limit. ([Subagents
FAQ](https://cursor.com/docs/subagents#can-subagents-launch-other-subagents)) So the
orchestrator/hunter/verifier pattern that Cloudflare's `security-audit` depends on is
supported, provided the skill says "subagent"/"Task" rather than naming Claude's `Agent` tool.

### Installation mechanics

There is no `cursor skill install` command. Discovery is directory-based, so installation is
"put the folder where Cursor walks."

1. **Pick a scope.** Project: `.cursor/skills/<name>/SKILL.md` or `.agents/skills/<name>/`.
   Global: `~/.cursor/skills/<name>/` or `~/.agents/skills/<name>/`. ([Skill
   directories](https://cursor.com/docs/skills#skill-directories))
2. **Copy or symlink the whole skill folder**, not just `SKILL.md`. `references/`, `scripts/`,
   and `assets/` are resolved relative to the skill root. Cursor's docs do not document
   symlink behavior; the ecosystem CLI uses symlinks by default, with `--copy` as the
   fallback. ([skills CLI installation
   methods](https://github.com/vercel-labs/skills/blob/main/docs/commands.md#installation-methods))
   Copy if you want a frozen version, symlink if you want `git pull` to update it.
3. **Use the CLI where it supports Cursor.** `npx skills add <owner/repo> -a cursor` installs
   into `.agents/skills/` for a project or `~/.cursor/skills/` with `-g`. The CLI's agent table
   lists Cursor exactly that way. ([skills CLI supported
   agents](https://github.com/vercel-labs/skills/blob/main/docs/agents.md))
4. **Open Customize → Skills to confirm.** Cursor does not import a GitHub repo on its own.
   Repo-level distribution requires a Cursor plugin with
   `.cursor-plugin/marketplace.json`; otherwise it is a manual/CLI copy. ([Installing skills
   from a repository](https://cursor.com/docs/skills#installing-skills-from-a-repository))
5. **Restart or reopen the Agent panel** if a newly added skill does not appear. Cursor
   discovers at startup; unlike Claude Code it is not documented to hot-reload skill
   directories. ([How skills work](https://cursor.com/docs/skills#how-skills-work))

For MCP-based tools, the shape Cursor expects is a `mcpServers` object in `.cursor/mcp.json`
(project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    },
    "chrome-devtools": {
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@latest", "--no-usage-statistics"]
    }
  }
}
```

`type: "stdio"` is documented for stdio servers, but the official examples omit it and Cursor
accepts the short form. `env`, `envFile`, `url`, `headers`, and `${env:NAME}` /
`${workspaceFolder}` interpolation are supported. ([Cursor MCP
configuration](https://cursor.com/docs/mcp#using-mcpjson))

### Gotchas measured on this machine

- **One local skill has invalid YAML and will never auto-trigger.** `mem-search/SKILL.md` has
  an unquoted `description:` containing `three steps: search, timeline, ...`. YAML reads the
  second colon as a mapping and fails. Cursor will show the skill without a usable
  description; Claude Code behaves the same. Quote the whole description.
- **Seven local skills exceed the ~5,000-token body recommendation.** Measured `SKILL.md`
  sizes: `graphify` 48 KB, `prompt-master` 33 KB, `humanizer` 32 KB, `diagram-design` 31 KB,
  `apple-design` 28 KB, `security-audit` 22 KB, `code-review-and-quality` 21 KB. Cursor loads
  them but flags them in `/skills`. Move detail into `references/` if the warning matters.
- **Claude-only fields are silently dropped.** `impeccable` and `unslop` use `argument-hint`
  and `user-invocable`; Cursor ignores both. No error, no behavior.
- **`disable-model-invocation: true` is honored by Cursor** (it is in Cursor's own frontmatter
  table), so `i-have-adhd` stays slash-only in both tools.
- **Personal vs synced.** Only `~/.cursor/skills/` syncs to Cloud Agents. `~/.agents/skills/`
  and `~/.claude/skills/` stay local. ([Use personal skills with Cloud
  Agents](https://cursor.com/docs/skills#use-personal-skills-with-cloud-agents))

## Rigorous app testing

| Name | Source | What it actually does | Cursor-compatible? | Install step | Verdict |
|---|---|---|---|---|---|
| Cursor Browser (built-in) | [cursor.com/docs/agent/tools/browser](https://cursor.com/docs/agent/tools/browser) | Drives pages: navigate, click, type, scroll, screenshot; reads console logs and network traffic; session persists per workspace. No external install. | Native | Nothing | **Adopt** as the exploratory pass and quick repro tool |
| `playwright-cli` + `playwright-cli` skill | [microsoft/playwright-cli](https://github.com/microsoft/playwright-cli) | CLI that keeps a browser session alive across commands: `open`, `goto`, `snapshot`, `click <ref>`, `fill`, `find`, `eval`, `screenshot`, `state-save`, tabs, network mocking, tracing. The skill is the command reference; snapshots are text, not pixels. | Yes. Pure shell commands + `SKILL.md`; no Claude-only tools. | `npm i -g @playwright/cli@latest` then `playwright-cli install --skills=agents` (writes `.agents/skills/playwright-cli`, which Cursor reads). `npx playwright install chrome`. | **Adopt.** Best fit for token-efficient E2E and exploratory runs |
| Playwright MCP | [microsoft/playwright-mcp](https://github.com/microsoft/playwright-mcp) | MCP server exposing Playwright over the accessibility tree: navigate, snapshot, click, type, console, network, screenshots. Maintained daily (last push 2026-11-03, ~39k stars). | Yes. Cursor is listed as a supported client; standard `mcpServers` config. | Add to `.cursor/mcp.json` with `npx @playwright/mcp@latest` | **Consider** only for long-lived, stateful agent loops. Microsoft itself recommends CLI+skills for coding agents |
| Chrome DevTools MCP + `chrome-devtools` / `a11y-debugging` skills | [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) | Controls live Chrome: input automation, snapshots, console with source maps, network requests, performance traces, `lighthouse_audit`, CSS inspection, heap snapshots. Official skills cover general debugging and WCAG-oriented a11y checks (semantic HTML, ARIA, focus, tap targets, contrast). Active (last push 2026-11-04, ~55k stars). | Yes. Cursor has an official client config. Skills reference bare MCP tool names (`take_snapshot`, `lighthouse_audit`); Cursor prefixes them, so expect to adjust names. | Add `chrome-devtools` to `.cursor/mcp.json` with `npx -y chrome-devtools-mcp@latest`; copy `skills/a11y-debugging` and `skills/chrome-devtools` into `~/.cursor/skills/`. Requires Chrome stable. | **Adopt** for accessibility and performance. This is the only credible WCAG/Lighthouse path found |
| Anthropic `webapp-testing` | [anthropics/skills · webapp-testing](https://github.com/anthropics/skills/tree/main/skills/webapp-testing) | Python Playwright toolkit. Ships `scripts/with_server.py` to start one or more servers, then write sync Playwright scripts. Covers reconnaissance-then-action, screenshots, console logs, element discovery. | Yes. It is plain Python + `SKILL.md`; no Claude-only tools. Needs Python Playwright installed. | Copy `skills/webapp-testing/` into `.cursor/skills/`; `pip install playwright && playwright install chromium` | **Consider.** Useful if you prefer Python over the Node CLI; overlaps playwright-cli |
| Playwright Test Agents (planner / generator / healer) | [playwright.dev/docs/test-agents](https://playwright.dev/docs/test-agents) | Official agent loop: planner explores the app and writes a Markdown test plan; generator turns it into Playwright tests; healer runs and repairs them. Installed with `npx playwright init-agents --loop=...`. | **Partial.** Generated loops exist for VS Code, Claude Code, Codex, and OpenCode — not Cursor. The Markdown agent definitions can be adapted, but there is no supported Cursor target. | `npx playwright init-agents --loop=claude` or `--loop=vscode`, then port the generated definitions into `.cursor/agents/` | **Consider** only if you want generated Playwright suites; the Cursor path is manual |
| `pixel-perfect` visual regression skill | [maxrihter/claude-skill-visual-regression](https://github.com/maxrihter/claude-skill-visual-regression) | Wraps Playwright's `toHaveScreenshot()`: baseline capture, multi-viewport, diff report, masking, CI workflows. | Yes, mechanically. But 1 star, last push 2026-07-09, single author. | Copy skill folder into `.cursor/skills/` | **Skip.** Use Playwright's own `toHaveScreenshot()` ([official docs](https://playwright.dev/docs/test-snapshots)); the skill adds no engine |
| `qa-skills` (exploratory-testing, automated-e2e-testing) | [fishzjp/qa-skills](https://github.com/fishzjp/qa-skills) | 12-skill QA framework: requirement analysis, risk/test strategy, charter-driven exploratory sessions, Playwright E2E with page objects, API testing, bug analysis, regression, flaky-test governance. Active (last push 2026-11-04, 60 stars). | Yes. Plain `SKILL.md`; install script accepts `--target ~/.cursor/skills`. Content and most headings are Chinese; skills depend on a shared `core/` folder. | `npx skills add fishzjp/qa-skills --skill '*' -a cursor`, or `./install.sh --target ~/.cursor/skills` | **Consider** for the exploratory-testing charter and bug-report schema. Language and `core/` coupling are real costs |
| `browser-use` | [browser-use/browser-use](https://github.com/browser-use/browser-use) | Python library for building AI browser agents, plus a hosted cloud. Not a skill, not an MCP server in the Cursor sense. | Not as a skill. Would be a Python dependency you write against. | N/A | **Skip.** Redundant with Playwright and Cursor Browser for this app |
| Puppeteer MCP | [modelcontextprotocol/servers-archived](https://github.com/modelcontextprotocol/servers-archived) | Archived reference server. The active MCP repo lists Puppeteer under "Archived". | N/A | N/A | **Skip.** Archived |
| Streamlit `AppTest` | [Streamlit App testing](https://docs.streamlit.io/develop/api-reference/app-testing) | Headless test harness that runs the Streamlit script directly, simulates widget input, and inspects rendered elements. Not browser automation. | N/A (pytest, not a skill) | `pytest` + `streamlit.testing.v1.AppTest` | **Adopt as harness**, not as an agent skill. Complements browser tests |
| `@axe-core/playwright` | [Playwright accessibility testing](https://playwright.dev/docs/accessibility-testing) | Runs axe-core inside Playwright tests and asserts on violations. Official Playwright guide; Deque's engine. | N/A (test dependency) | `npm i -D @axe-core/playwright`, add a spec | **Adopt** for repeatable a11y assertions; pair with Chrome DevTools for manual/keyboard checks |

### Notes that matter for this app

- **Streamlit desktop is browser-testable at the server, not the window.** `packaging/launcher.py`
  starts Streamlit on `127.0.0.1` with `DEFAULT_PORT = 8501` and fallbacks 8501–8505, writes a
  lock file with the port, and shows it in a pywebview window. Playwright/Chrome can attach to
  `http://127.0.0.1:<port>` while the app runs; the native window chrome cannot be driven.
  Read the lock file rather than hard-coding 8501.
- **Streamlit selectors are generated.** Prefer `data-testid` attributes (`stButton`,
  `stTextInput`, etc.) and visible text over CSS structure. Streamlit reruns the whole script
  on interaction, so waits must target the resulting element, not a fixed timeout.
- **The three mix-tab/mixer/region-picker components are iframes.** Playwright's frame APIs
  handle them. Cursor's built-in Browser may not reach inside them reliably; that is unverified
  and should be treated as a Playwright-only path.
- **Accessibility caveat from the axe docs:** automated scans catch only some issues. Playwright
  explicitly recommends combining automated testing with manual assessment. ([Playwright
  accessibility testing](https://playwright.dev/docs/accessibility-testing))
- **Electron support is irrelevant here** — the desktop app is Streamlit in pywebview, not
  Electron. Playwright's experimental `_electron` API would only matter if the shell changed.
  ([Playwright Electron](https://playwright.dev/docs/api/class-electron))

## Exhaustive issue finding

| Name | Source | What it actually covers | Cursor-compatible? | Install step | Verdict |
|---|---|---|---|---|---|
| `code-quality-audit` (local) | Local `~/.config/opencode/skills/code-quality-audit/SKILL.md`; no upstream repo declared | Whole-repo maintainability: dead code, unused UI components, duplicate logic, over-complexity, legacy leftovers, redundant queries/API calls, abandoned files, tech debt. Requires evidence and reachability before calling something dead; folds in `knip` and `jscpd` when present. Static only. | Yes. Only `name` + `description`; no Claude-only tools. | Already local; copy or symlink into `.cursor/skills/` | **Adopt** as the maintainability sweep |
| `ponytail-audit` (local) | [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) | Whole-repo quality audit: bugs, security holes, behavior under real load, risky code without tests, slow paths, and what to delete/merge/split. One-shot report, changes nothing. Ranks findings, explains in plain English, states assumed load and what was not read. Static only. | Yes. Plain `SKILL.md`; the repo advertises 20 agents including Cursor. | Copy `skills/ponytail-audit/` into `.cursor/skills/`, or `npx skills add DietrichGebert/ponytail --skill ponytail-audit -a cursor` | **Adopt** as the correctness/scale sweep. Complements `code-quality-audit` |
| `security-audit` (Cloudflare) | [cloudflare/security-audit-skill](https://github.com/cloudflare/security-audit-skill) | Six phases: reconnaissance and coverage ledger, coverage-led hunting with isolated hunters, adversarial candidate validation by fresh verifiers, schema-validated `findings.json`, independent record verification, target-neutral reports. The only local skill that tries to prove coverage rather than sample. | Yes, with one caveat: it is written agent-neutrally (parent / Task tool / research / general agents) and Cursor has those. It does not use `context: fork`. | Clone the repo; copy the whole `skills/security-audit/` folder (all `.md`, `report-schema.json`, `validate-*.cjs`) into `.cursor/skills/security-audit/` | **Adopt** for security. Heavy; run as a deliberate audit, not per change |
| `code-review-and-quality` (local) | Local `~/.config/opencode/skills/code-review-and-quality/SKILL.md` | Five-axis review of a **change**: correctness, readability, architecture, security, performance. Approval standard, quality gates. Reads the diff and its immediate context. | Yes | Already local | **Skip for this job.** It is change-scoped, not repo-scoped |
| `code-review` (local) | Local, from mattpocock/skills lineage | Two-axis review of the diff between `HEAD` and a fixed point: Standards (repo conventions + Fowler smell baseline) and Spec (does it match the issue). Runs both as parallel subagents. | Yes, and it benefits from Cursor's Task tool. | Already local | **Skip for this job.** Diff-only by design |
| `ponytail-review` (local) | [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) | Reviews a change: logic, safety, load behavior, test coverage of risky code, speed, necessity. Reads callers and tests, not only the diff. | Yes | Already local | **Skip for this job.** Change-scoped |
| `workersio/wio` | [workersio/skills](https://github.com/workersio/skills) | One skill, five modes: `scan` (highest-value tests to add), `test` (write one focused test), `workload` (realistic adversarial sessions/traffic), `review` (KEEP/REDO/REMOVE), `doctor` (weak assertions, over-mocking, flakes, CI blind spots). Explicitly targets behavior and regression value, not coverage. | Yes. README lists Cursor; core content is plain Markdown. | `npx skills add workersio/skills -a cursor` | **Adopt** for edge-case and test-gap discovery. Pairs with pytest and vitest |
| Trail of Bits `property-based-testing` | [trailofbits/skills · property-based-testing](https://github.com/trailofbits/skills/tree/main/plugins/property-based-testing) | Property catalog (roundtrip, inverse, oracle, idempotence, invariant, commutativity), Hypothesis/fast-check/proptest guidance, tautology detection, shrinking triage, refactoring-for-testability. | Yes. `allowed-tools` is ignored by Cursor, but the body is plain instructions. | Copy the skill folder into `.cursor/skills/property-based-testing/`; install `hypothesis` | **Adopt** for audio math, timing, parsers, and serialization in `src/audio_to_tab/` |
| Trail of Bits `sharp-edges` | [trailofbits/skills · sharp-edges](https://github.com/trailofbits/skills/tree/main/plugins/sharp-edges) | Finds error-prone APIs, dangerous configuration defaults, and footguns: "secure by default" / "pit of success" review of interfaces and config schemas. | Partial. The body routes to a `sharp-edges-analyzer` agent that Cursor will not have; the four-phase workflow itself is usable inline. | Copy folder; ignore the agent reference | **Consider** for `backend/config.py` and user-facing job options |
| Trail of Bits `mutation-testing` | [trailofbits/skills · mutation-testing](https://github.com/trailofbits/skills/tree/main/plugins/mutation-testing) | Configures mewt/muton campaigns, analyzes surviving mutants, classifies equivalent mutants, hunts bugs in weakly tested code. | Yes mechanically, but **wrong toolchain**: mewt supports C++, DAML, Go, JS/TS, Ruby, Rust, Solidity, Move — no Python. ([mewt README](https://github.com/trailofbits/mewt)) | N/A for this repo | **Skip.** Use `mutmut` or `cosmic-ray` manually; no maintained Python mutation skill was found |
| Trail of Bits `audit-context-building` | [trailofbits/skills · audit-context-building](https://github.com/trailofbits/skills/tree/main/plugins/audit-context-building) | Builds a function-by-function dossier of assumptions, guarantees, and dependencies before hunting bugs. Strong idea: separate understanding from verdicts. | **No as shipped.** It routes through Claude plugin commands (`/audit-context-building:audit-context`) and named agents. | N/A | **Skip** unless you rewrite the routing for Cursor subagents |
| `fishzjp/qa-skills` (`bug-analysis`, `test-strategy`, `regression-testing`) | [fishzjp/qa-skills](https://github.com/fishzjp/qa-skills) | QA process skills: risk model, ten-axis test-type decision matrix, bug root-cause/impact analysis, regression scope selection. | Yes, with the same Chinese-language and `core/` coupling caveats as above. | `npx skills add fishzjp/qa-skills --skill '*' -a cursor` | **Consider** for the bug-analysis and regression-scope templates |
| `gsd-audit-fix` / `gsd-audit-uat` (local Claude skills) | Local `~/.claude/skills/gsd-*` | Audit-to-fix pipeline and cross-phase UAT audit. | **No.** They read `$HOME/.claude/get-shit-done/workflows/*` and `.planning/phases/*`; neither exists in Cursor's world. | N/A | **Skip** |
| Indexes: VoltAgent/awesome-agent-skills, skills.sh | [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills), [skills.sh](https://skills.sh) | Discovery only. Useful for finding candidates; not evidence of quality or Cursor compatibility. | N/A | N/A | **Use for discovery only** |

### What "exhaustive" means in practice

Only `security-audit` earns the word. It is coverage-led: it writes a ledger of units, assigns
hunters, uses critics to find uncovered units, and validates findings against a schema. Its own
README describes the six phases and the three verdicts (`confirmed`, `needs_validation`,
`rejected`). ([cloudflare/security-audit-skill
README](https://github.com/cloudflare/security-audit-skill))

`code-quality-audit` and `ponytail-audit` are thorough but sampling-based: they read the repo,
follow entry points, and report what they find, with explicit "what I did not read" limits.
Neither drives a running app.

`workersio/wio` is the closest thing to "find the bugs the tests miss" because it generates
adversarial workloads and judges whether an assertion would catch a real regression. It still
works at the test-suite level, not "click every control and report what breaks."

## Gaps no existing skill fills

These are the things you would have to write for this app. Each is a real gap, not a nice-to-have.

1. **A Cursor-native "run and verify" skill.** Claude Code has bundled `/run`, `/verify`, and
   `/run-skill-generator` that infer how to launch an app and then drive it. Cursor has no
   equivalent. ([Claude Code bundled skills](https://code.claude.com/docs/en/skills#run-and-verify-your-app))
   For this repo the recipe is knowable and should be recorded: `.venv311`, `make backend`,
   `make web`, `make ui`, `ATT_REQUIRE_AUTH=false`, the 8501–8505 port fallback and lock file,
   and the "restart Streamlit after editing `ui/`" quirk. This is the single highest-value
   skill to write.
2. **A Streamlit-driving recipe.** No skill knows that Streamlit reruns the script on every
   interaction, that widgets need `data-testid` selectors, that custom components live in
   iframes, or that `/_stcore/health` is the readiness probe. A 100-line skill with a
   `scripts/wait_for_streamlit.py` would make every browser tool reliable on this app.
3. **Desktop-shell and installer testing.** No skill can drive the pywebview window, the
   single-instance lock, the ffmpeg bundle, `edition.txt`, or the "onedir must not contain
   `backend/`, `web/`, or `.env*`" release assertions. That is a packaging smoke-test skill
   someone has to write; it will be shell + assertions, not browser automation.
4. **Cross-surface UAT.** The desktop vs hosted differences are product rules, not generic QA:
   Desktop Lite must not clamp audio length; hosted `lite` caps at 60 s; desktop isolation is
   serial; YouTube to MP3 is download-only. A UAT skill needs this matrix or it will "verify"
   the wrong behavior. ([AGENTS.md](../../AGENTS.md), `docs/issues.md` I-500, `docs/ui-issues.md` I-609)
5. **Bug reproduction from a user report against a live instance.** No skill covers: take a
   report, find the job ID, pull logs/artifacts, reproduce against the running surface, and
   write a minimal failing case. The pieces exist (`systematic-debugging`, Playwright, browser
   console/network); the workflow does not.
6. **Exhaustive *behavioral* defect enumeration.** Nothing drives every route/control/state and
   reports broken flows, missing error handling, empty states, and dead ends. `code-quality-audit`
   is static; `ponytail-audit` is static; `wio` is test-level. This is the biggest gap in the
   "find every issue" job.
7. **Python mutation testing.** `mewt` has no Python target. A Python mutation skill would wrap
   `mutmut` or `cosmic-ray`, scope mutants to changed modules, and triage survivors. Nothing
   maintained exists.
8. **Streamlit-specific accessibility.** Generic a11y skills audit the DOM Streamlit generates,
   including its iframes and canvas-based mixer component. No skill knows which Streamlit
   widgets produce which ARIA gaps, or how to test the mixer's keyboard-only path.
9. **FastAPI contract fuzzing.** No skill wraps Schemathesis or Hypothesis-driven API testing for
   the `backend/` routes. `wio` can choose "contract" as a strategy, but the OpenAPI-driven
   harness is yours.
10. **A coverage ledger for non-security defects.** `security-audit` proves coverage for security.
    Nothing does the same for functional defects: routes exercised, states visited, error paths
    triggered, surfaces compared.

## Sources

Cursor:

- https://cursor.com/docs/skills
- https://cursor.com/docs/skills#skill-directories
- https://cursor.com/docs/skills#frontmatter-fields
- https://cursor.com/docs/skills#how-skills-work
- https://cursor.com/docs/skills#disabling-automatic-invocation
- https://cursor.com/docs/skills#nested-skill-directories
- https://cursor.com/changelog/2-4
- https://cursor.com/docs/subagents
- https://cursor.com/docs/subagents#can-subagents-launch-other-subagents
- https://cursor.com/docs/mcp
- https://cursor.com/docs/rules
- https://cursor.com/docs/agent/tools/browser
- https://cursor.com/docs/cli/changelog
- https://cursor.com/docs/release-notes
- https://cursor.com/help/customization/skills
- https://forum.cursor.com/t/in-cursor-version-0-47-8-agent-mode-incorrectly-prefixes-mcp-tool-names-with-mcp-servername-causing-tools-to-fail/72778

Agent Skills standard and Claude Code:

- https://agentskills.io/specification
- https://agentskills.io/specification#progressive-disclosure
- https://code.claude.com/docs/en/skills
- https://code.claude.com/docs/en/skills#using-skill-frontmatter-outside-claude-code
- https://code.claude.com/docs/en/skills#run-and-verify-your-app
- https://github.com/anthropics/skills
- https://github.com/anthropics/skills/tree/main/skills/webapp-testing
- https://github.com/anthropics/skills/blob/main/spec/agent-skills-spec.md

Browser and testing tools:

- https://github.com/microsoft/playwright-cli
- https://github.com/microsoft/playwright-cli/blob/main/skills/playwright-cli/SKILL.md
- https://github.com/microsoft/playwright-cli/blob/main/skillCheck.js
- https://github.com/microsoft/playwright-mcp
- https://github.com/ChromeDevTools/chrome-devtools-mcp
- https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/docs/tool-reference.md
- https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/skills/a11y-debugging/SKILL.md
- https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/skills/chrome-devtools/SKILL.md
- https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/docs/client-configurations.md
- https://playwright.dev/docs/test-agents
- https://playwright.dev/docs/test-snapshots
- https://playwright.dev/docs/accessibility-testing
- https://playwright.dev/docs/api/class-electron
- https://docs.streamlit.io/develop/api-reference/app-testing
- https://github.com/browser-use/browser-use
- https://github.com/modelcontextprotocol/servers
- https://github.com/modelcontextprotocol/servers-archived

Audit and issue-finding skills:

- https://github.com/cloudflare/security-audit-skill
- https://github.com/DietrichGebert/ponytail
- https://github.com/DietrichGebert/ponytail/blob/main/skills/ponytail-audit/SKILL.md
- https://github.com/workersio/skills
- https://github.com/trailofbits/skills
- https://github.com/trailofbits/skills/tree/main/plugins/mutation-testing
- https://github.com/trailofbits/skills/tree/main/plugins/property-based-testing
- https://github.com/trailofbits/skills/tree/main/plugins/sharp-edges
- https://github.com/trailofbits/skills/tree/main/plugins/audit-context-building
- https://github.com/trailofbits/mewt
- https://github.com/fishzjp/qa-skills
- https://github.com/maxrihter/claude-skill-visual-regression

Installation mechanics:

- https://github.com/vercel-labs/skills
- https://github.com/vercel-labs/skills/blob/main/docs/agents.md
- https://github.com/vercel-labs/skills/blob/main/docs/commands.md

Local files read:

- `~/.config/opencode/skills/code-quality-audit/SKILL.md`
- `~/.config/opencode/skills/ponytail-audit/SKILL.md`
- `~/.config/opencode/skills/security-audit/SKILL.md`
- `~/.config/opencode/skills/code-review-and-quality/SKILL.md`
- `~/.config/opencode/skills/code-review/SKILL.md`
- `~/.config/opencode/skills/ponytail-review/SKILL.md`
- `~/.claude/skills/gsd-audit-fix/SKILL.md`
- `packaging/launcher.py`
