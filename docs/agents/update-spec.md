# Update the spec

`SPEC.md` says what smart-router does and is for. The ADRs in `docs/adr/` say why it was decided that way, and the progress entries in `docs/progress/` say what each change did.

1. **When.** A change to behaviour, CLI flags, APIs or configuration updates the spec in the same pull request. A refactor, or a bug fix that restores what the spec already says, doesn't.
2. **What goes in.** What a user or another system can rely on: the behaviours, the commands and their flags, the API, the configuration keys and their defaults, and what is deliberately out of scope. Write it in words a user would recognise, not as a description of the code.
3. **What stays out.** How it's implemented, the reasons for a choice (an ADR), and the history of a change (a progress entry).
4. **On an existing codebase.** Don't describe everything at once. Start with the part your change touches, in the section it belongs to, and let the spec grow from there.
5. **Open questions.** Note a decision still to be made under "Open questions"; once it's made, record it in an ADR and update the spec.
6. **In the pull request.** Name the sections you changed under "Decisions and spec".
