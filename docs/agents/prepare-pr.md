# Prepare a pull request

1. **Branch.** Work on a branch named for the change (`feat/…`, `fix/…`, `docs/…`, `chore/…`), never on the default branch.
2. **Test.** Run `uv run pytest`, `uv run ruff check --force-exclude <changed .py files>` and `uv run ruff format --check --force-exclude <changed .py files>`, even for docs-only changes. Say tests pass only after all of them pass. **No tests yet?** If pytest says "no tests collected", write a real test for the behaviour your change touches (the spec's behaviours are a good place to start). Never make the test job pass by masking it, skipping it or removing it.
3. **Progress entry.** Add `docs/progress/YYYYMMDD-<slug>.md`: what changed, the branch or pull request, the ADRs and locked rules it touches, and what comes next. Never edit earlier entries. Edit `docs/PROGRESS.md` only when Now or Next changes.
4. **ADR, if needed.** A change to the design, the process or the runtime dependencies needs one: see `record-adr.md`.
5. **Spec, if behaviour changed.** A change to behaviour, CLI flags, APIs or configuration updates `SPEC.md` in the same pull request: see `update-spec.md`. A refactor or a bug fix that restores intended behaviour doesn't.
6. **Locked rules.** If the change approves or supersedes SeamSpec rules, regenerate `docs/EARNED_SPEC.md` in the same pull request: `seamspec spec --out docs/EARNED_SPEC.md`.
7. **Check.** `swerepo check --base <default branch>` reports a missing progress entry or ADR before CI does.
8. **Privacy.** No personal emails, home-folder paths, keys or passwords in the diff.
9. **Open the pull request** with the template: what changed, the decisions it touches, the checklist. Never merge it without the person's go-ahead.
