---
id: ADR-20261125-2401
title: Python 3.11 is the oldest version supported
status: ACCEPTED            # PROPOSED | ACCEPTED | SUPERSEDED | REJECTED
date: 2026-11-25
scopes: ["pyproject.toml", "uv.lock", ".github/workflows/swerepo-ci.yml"]
invariants: []              # SeamSpec rule ids that enforce it, if any
supersedes: null
superseded_by: null
---

# ADR-20261125-2401: Python 3.11 is the oldest version supported

## Context

Adding `uv.lock` let GitHub's dependency review see the exact locked versions for the first time, and it failed the pull request that sets the repository up to the SWERepo standard. `filelock` (pulled in through `huggingface-hub`, for the `providers` and `embed` extras) was locked at 4.1.0 for Python 3.10, with 17 security advisories, one of high severity. The fixed releases (4.2.1 and later) need Python 3.11, and two advisories cover every release up to 4.2.0, so no `filelock` that runs on Python 3.10 is free of them. Python 3.10 reached end of life in October 2026 and gets no security fixes either.

## Decision

1. `requires-python` is `>=3.11`.
2. `uv.lock` is resolved for Python 3.11 and later only, so `filelock` is locked at 4.3.1.
3. CI tests Python 3.11 to 3.14 (`swerepo update --only ci` follows `requires-python`).

**Alternatives considered:**
- **Keep Python 3.10 and allow these advisories in dependency review.** It weakens a security check for every pull request, and the high-severity issue would still reach anyone on 3.10.
- **Pin an older `filelock` for Python 3.10.** Every release that runs on 3.10 is still covered by at least two advisories.

## Enforced Invariants

None yet. The CI matrix follows `requires-python`, and dependency review fails a pull request that locks a vulnerable version.

## Consequences

- Python 3.10 users must upgrade to use this package.
- Python 3.10-only backports (`exceptiongroup`, `tomli`, `async-timeout`) leave the lockfile.
- When Python 3.11 reaches end of life (October 2027), revisit the oldest version supported.
