# Contributing

## How a change lands

1. **Branch.** Never commit to the default branch: one branch and one pull request per piece of work.
2. **Record decisions.** A change to the design, the process (CI workflows included) or the runtime dependencies gets an ADR in `docs/adr/` (`swerepo adr new "Title"`), in the same pull request.
3. **Keep the spec current.** A change to behaviour, CLI flags, APIs or configuration updates `SPEC.md` in the same pull request. Refactors and bug fixes don't.
4. **Log progress.** Add one file in `docs/progress/` (`YYYYMMDD-<slug>.md`) per pull request, and never edit earlier ones. Edit `docs/PROGRESS.md` only when Now or Next changes. Trivial changes skip the entry with the commit trailer `Progress: none`.
5. **Test.** Run `uv run pytest`, `uv run ruff check --force-exclude <changed .py files>` and `uv run ruff format --check --force-exclude <changed .py files>` before asking for a review.
6. **Review.** Answer every review thread: fix it and reply with the commit, or reply why not.
7. **Merge.** Maintainers merge. Nothing reaches the default branch without a pull request.

## Privacy

No personal email addresses, home-folder paths, tokens or secrets in tracked files. Commit under a noreply address.
