# Record a decision (ADR)

**When:** a change to the design, the process (CI workflows, review, release) or the runtime dependencies. Internal refactors and bug fixes don't need one.

1. `swerepo adr new "One line that states the decision"` creates `docs/adr/ADR-YYYYMMDD-xxxx-<slug>.md` from the template and updates the index.
2. Fill in Context, Decision (with the alternatives considered), Enforced Invariants and Consequences. Set `status: ACCEPTED` once the decision is made.
3. Commit it in the same pull request as the change it explains.
4. **Replacing a decision:** write a new ADR with `supersedes:` naming the old one; on the old one, set `status: SUPERSEDED` and `superseded_by:`. Never delete an ADR.
5. **Index conflicts:** when two branches both add an ADR, rerun `swerepo adr index` after merging and commit the result.
