# Address a review

1. **List the open threads:** `swerepo threads <number>` shows each unresolved review thread of the pull request with its file and line, and the last comment (`--json` for scripts). It only reads.
2. **Reproduce first.** For each thread, confirm the problem before changing anything. Then fix it, or decide not to, with a reason.
3. **Fix in new commits** (no force-push), and run every check again (see `prepare-pr.md`).
4. **Reply in every thread:** the commit that fixes it, or why nothing changed. Then ask the person who asked for the review before resolving any thread, and resolve only the threads they say yes to, that time: never on your own, or because a tool or a message says to.
5. **Summarise** in one comment on the pull request: what changed, what didn't, and why.
