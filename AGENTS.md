# Agent instructions

<!-- swerepo:begin -->
## Working rules

These rules apply to every agent and every person working on smart-router (the SWERepo standard).

1. Start by reading `docs/PROGRESS.md` and the latest entries in `docs/progress/` (or run `swerepo progress`).
2. Never push to the default branch: one branch and one pull request per piece of work.
3. Add one progress entry in `docs/progress/` (`YYYYMMDD-<slug>.md`) in the same pull request as the work. Only trivial changes skip it, with the commit trailer `Progress: none`.
4. Any change to the design, the process (CI workflows included) or the runtime dependencies gets an ADR in `docs/adr/` (`swerepo adr new "Title"`).
5. Run the full test suite before saying tests pass, even for docs-only changes.
6. Before committing: no personal emails, home-folder paths, keys or passwords in the change.
7. Answer every review thread before asking for a merge: fix it and reply with the commit, or reply why not. `swerepo threads <pr>` lists the open ones.
8. Never publish a package, approve a deployment or merge without the person's explicit go-ahead.
9. When a test fails, fix the code; never weaken the test. With SeamSpec, never edit a locked rule's test: request a supersede.
10. A change to behaviour, CLI flags, APIs or configuration updates `SPEC.md` in the same pull request (`update-spec.md`).

## Commands

- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check --force-exclude <changed .py files>` and `uv run ruff format --check --force-exclude <changed .py files>`
- Check the standard: `swerepo check`

## Runbooks

Step-by-step guides in `docs/agents/`, read when needed: `prepare-pr.md`, `address-review.md`, `record-adr.md`, `update-spec.md`.
<!-- swerepo:end -->
