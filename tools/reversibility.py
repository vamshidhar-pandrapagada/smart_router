"""Irreversibility inference for tools the gateway does not annotate.

`ToolSpec.irreversible` used to default to False, which meant a gateway that advertised
`git.push` without saying anything got the read-only treatment: no dry-run, no mandatory
judge, no approval. That is fail-open on a safety gate.

The field is now tri-state -- True, False, or None for "not declared" -- and an
undeclared tool falls back to this policy. The default for a name matching nothing is
IRREVERSIBLE, because over-gating an unknown tool costs a dry-run and a judge call, while
under-gating one costs a real side effect that cannot be undone.

Inference is a safety net, not a substitute for declaration. `ToolRegistry.audit()` lists
every tool whose class was guessed so an operator can annotate them properly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Verbs that change something outside the router.
IRREVERSIBLE_VERBS = (
    "send", "post", "publish", "delete", "remove", "drop", "purge", "push", "commit",
    "merge", "rebase", "revert", "tag", "release", "deploy", "create", "add", "insert",
    "update", "patch", "put", "write", "upload", "rename", "move", "archive", "restore",
    "invite", "assign", "approve", "reject", "close", "resolve", "pay", "purchase",
    "charge", "refund", "transfer", "schedule", "cancel", "grant", "revoke", "reset",
    "rotate", "execute", "run", "trigger", "start", "stop", "restart", "scale",
)

#: Verbs that only observe.
READ_ONLY_VERBS = (
    "get", "list", "search", "read", "find", "query", "fetch", "show", "view",
    "describe", "count", "preview", "check", "status", "diff", "log", "history",
    "branch", "blame", "inspect", "tail", "head", "stat",
    "resolve_id", "lookup", "summarize", "analyze", "validate",
)


def tokens(name: str) -> list[str]:
    """`git_commit` -> ['git', 'commit']; `sharepoint.search_items` -> [...].

    Earlier this extracted a single trailing 'leaf' by splitting on `.`, `/` and `__`
    only. That missed the single-underscore convention `mcp-server-git` actually uses --
    `git_commit` stayed whole, matched no verb, and fell through to the conservative
    default along with `git_log`, `git_status` and `git_diff`. Safe, but it forced
    dry-runs and judge calls on every read, which destroys the cheap path.
    """
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


@dataclass
class IrreversibilityPolicy:
    irreversible_verbs: tuple[str, ...] = IRREVERSIBLE_VERBS
    read_only_verbs: tuple[str, ...] = READ_ONLY_VERBS
    #: Fail-safe. A tool matching neither list is gated as if it changed the world.
    default_when_unknown: bool = True
    #: Names to force either way, when inference gets a specific tool wrong.
    overrides: dict[str, bool] = field(default_factory=dict)

    def infer(self, tool_name: str) -> tuple[bool, str]:
        """Scan every token, exact matches only, mutating verbs win.

        Precedence is deliberate: `git_create_branch` must be irreversible even though
        'branch' reads like an observation, so any mutating token decides. Exact matching
        keeps `get_deployment_status` read-only -- 'deployment' is not 'deploy'.
        """
        if tool_name in self.overrides:
            return self.overrides[tool_name], "override"
        parts = tokens(tool_name)
        irreversible = set(self.irreversible_verbs)
        read_only = set(self.read_only_verbs)

        for token in parts:
            if token in irreversible:
                return True, f"mutating verb {token!r}"
        for token in parts:
            if token in read_only:
                return False, f"read-only verb {token!r}"
        return self.default_when_unknown, "unrecognized verb — defaulted conservatively"


DEFAULT_POLICY = IrreversibilityPolicy()
