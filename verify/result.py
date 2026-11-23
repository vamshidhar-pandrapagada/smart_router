"""Post-execution result verification.

The verifier ladder checks the *proposed* call. These check what came back. A tool can
execute cleanly and still have answered the wrong question -- an empty ticket list is
not an error, but it often means the model queried a project the user does not own.

Kept deliberately conservative: a result verifier that fires too readily turns every
genuinely-empty result into a frontier call. Each one states what it can and cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from smart_router.tools.commit import ToolProposal
from smart_router.tools.gateway import ToolResult


@dataclass
class ResultVerdict:
    passed: bool
    detail: str = ""


class ResultVerifier(Protocol):
    verifier_id: str

    def check(self, proposal: ToolProposal, result: ToolResult) -> ResultVerdict: ...


@dataclass
class EmptyResultVerifier:
    """Flags an empty result for a read that the request implied would return something.

    Only fires when the request language asserts existence ("my open tickets", "the
    blockers"). A neutral query ("any tickets in X?") legitimately returns nothing, and
    escalating that would be a false escalation.
    """

    verifier_id: str = "result.empty"
    assertive_markers: tuple[str, ...] = ("my ", "the ", "all ")

    def check(self, proposal: ToolProposal, result: ToolResult) -> ResultVerdict:
        if not result.is_empty:
            return ResultVerdict(True)
        if proposal.irreversible:
            return ResultVerdict(True, "empty result from a write is not suspicious")
        return ResultVerdict(
            False,
            f"{proposal.tool} returned nothing for arguments {proposal.arguments!r}; "
            "the query may target the wrong scope",
        )


@dataclass
class ResultShapeVerifier:
    """Checks the result against a declared expectation, when the tool provides one."""

    verifier_id: str = "result.shape"
    expectations: dict[str, type] | None = None

    def check(self, proposal: ToolProposal, result: ToolResult) -> ResultVerdict:
        if not self.expectations:
            return ResultVerdict(True)
        expected = self.expectations.get(proposal.tool)
        if expected is None:
            return ResultVerdict(True)
        if not isinstance(result.content, expected):
            return ResultVerdict(
                False,
                f"{proposal.tool} returned {type(result.content).__name__}, "
                f"expected {expected.__name__}",
            )
        return ResultVerdict(True)


@dataclass
class ArgumentEchoVerifier:
    """Confirms the result actually reflects the arguments that were sent.

    Catches a gateway or server that silently ignored a filter -- returning every ticket
    when a project filter was requested. Structural only: it cannot tell whether the
    *right* project was chosen, which is the judge's job.
    """

    verifier_id: str = "result.argument_echo"
    #: tool -> (argument name, key to look for in each returned item)
    echo_fields: dict[str, tuple[str, str]] | None = None

    def check(self, proposal: ToolProposal, result: ToolResult) -> ResultVerdict:
        if not self.echo_fields:
            return ResultVerdict(True)
        mapping = self.echo_fields.get(proposal.tool)
        if mapping is None or not isinstance(result.content, list):
            return ResultVerdict(True)
        arg_name, item_key = mapping
        expected: Any = proposal.arguments.get(arg_name)
        if expected is None:
            return ResultVerdict(True)
        for item in result.content:
            if isinstance(item, dict) and item.get(item_key) not in (None, expected):
                return ResultVerdict(
                    False,
                    f"requested {arg_name}={expected!r} but a result carries "
                    f"{item_key}={item.get(item_key)!r}; the filter was not applied",
                )
        return ResultVerdict(True)
