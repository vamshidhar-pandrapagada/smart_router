# Review instructions for Copilot

Every comment you leave must be answered before this pull request can merge, so comment only where it changes something.

**Skip:**
- style that ruff or the project's own linters already check (formatting, import order, line length, naming);
- wording in comments and documentation, unless it is wrong;
- suggestions that only restate what the code does.

**Do comment on:**
- a change to behaviour without a test that covers it;
- a change to user-visible behaviour, CLI flags, APIs or configuration that leaves `SPEC.md` as it was (a refactor or a bug fix needs no spec change);
- a missing progress entry in `docs/progress/`, or a design, process or dependency change without an ADR in `docs/adr/`;
- changes to locked SeamSpec rules (`.seamspec/`) or to CI workflows (`.github/workflows/`): say what they weaken;
- security problems: secrets, injection, unsafe defaults, personal data in tracked files;
- bugs: wrong results, crashes, missed edge cases.

Say what is wrong, where, and how to reproduce it.
