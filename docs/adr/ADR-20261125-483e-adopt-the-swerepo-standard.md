---
id: ADR-20261125-483e
title: Adopt the SWERepo standard
status: ACCEPTED            # PROPOSED | ACCEPTED | SUPERSEDED | REJECTED
date: 2026-11-25
scopes: [".swerepo.toml", ".github/", "AGENTS.md", "CLAUDE.md", "GEMINI.md", "docs/", "SPEC.md", "SECURITY.md", "pyproject.toml"]
invariants: []              # SeamSpec rule ids that enforce it, if any
supersedes: null
superseded_by: null
---

# ADR-20261125-483e: Adopt the SWERepo standard

## Context

The repository had no CI, no linter configuration, no written process for changes, and nothing telling coding agents how to work here. Work is done with Claude Code and Antigravity, so the rules for changes need to be written down where agents read them and checked by CI, not kept in someone's head. SWERepo (`swerepo init`) sets a repository up to one standard and keeps what it writes up to date (`swerepo update`), without replacing existing files.

## Decision

1. **Components:** `docs`, `agents`, `ci`, `security` and `github` are adopted; `seamspec` (locked rules) isn't, for now.
2. **How changes are made** (in `AGENTS.md`, for people and agents alike): one branch and one pull request per change; a progress entry in `docs/progress/`; an ADR for a change to the design, the process (CI included) or the runtime dependencies; `SPEC.md` updated when behaviour changes; no personal data or secrets in the change.
3. **Editors:** Claude Code and Antigravity. `CLAUDE.md` and `GEMINI.md` point to `AGENTS.md`, so the rules live in one place.
4. **CI** (`.github/workflows/swerepo-ci.yml`): one required check, `ci-ok`, which passes only when every job did:
   - **lint:** ruff on the Python files each change touches. The codebase has never been linted, so it moves to ruff file by file instead of in one large reformatting change;
   - **tests** on every Python version `requires-python` allows (3.11 to 3.14; ADR-20261125-2401), and macOS on pushes to `main`;
   - **workflow audit** (zizmor) and **dependency review**, which fails a pull request that adds a dependency with known vulnerabilities;
   - **Copilot's review:** `ci-ok` waits for it.
5. **GitHub, as code** (`.github/swerepo/`), in **solo mode**: changes only through pull requests, `ci-ok` required, Copilot's review requested and its threads resolved before a merge, squash merges only, no approving review needed. The settings are applied after this merges, shown first and confirmed (`swerepo init --only github`). Dependabot proposes updates once a release is a week old.
6. **Security:** `SECURITY.md` points to GitHub's private vulnerability reporting; `.gitignore` gains a block for secrets and local files.
7. **The pull-request rules start as warnings** (`[pr] enforce = false`), as for an existing repository. `swerepo check --base main` shows them locally.
8. **The governance job and the privacy workflow wait for a released SWERepo.** They run those rules and the privacy scan in CI, but SWERepo isn't on PyPI yet, and installing it from its GitHub branch would put an unpinned, moving dependency into CI. Once SWERepo 0.1.0 is released, `swerepo update` adds both, pinned to that release. Until then the privacy scan runs locally (`swerepo privacy`, or the pre-push hook from `swerepo hook install`).

**Alternatives considered:**
- **Writing the process and CI by hand.** It's the same work without the checks that keep it current, and every repository drifts differently.
- **Full ruff on every file now.** CI would be red until one large reformatting change landed, which clutters `git blame` and mixes style with real changes.
- **The governance job now, installed from SWERepo's `main` branch.** An unpinned dependency in CI that can change under this repository; waiting for the release is safer.

## Enforced Invariants

None yet. `ci-ok` enforces the CI jobs once the ruleset requires it; the pull-request rules are warnings until `[pr] enforce` is turned on.

## Consequences

- Every change goes through a pull request with green CI and Copilot's review, once the ruleset is applied.
- Agents read the same rules and commands as people, in `AGENTS.md`.
- Files touched by a change are brought up to ruff's standard as part of that change.
- To revisit: add the governance job and the privacy workflow after SWERepo 0.1.0 (`swerepo update`), turn the pull-request rules into failures (`swerepo update --enforce-pr-rules`) once the team is used to them, and decide on SeamSpec.
