"""Two-phase commit gate (plan SSE): propose -> verify -> commit.

The router must never let a model's output reach a live connector unchecked. Tool calls
are therefore split so escalation remains possible when tier 1 is wrong:

    propose   parse the model's output into a typed proposal. Nothing executes.
    verify    registry checks, then the verifier ladder, then (irreversible only) a
              gateway dry-run. Still nothing executes.
    commit    the gateway executes, under an idempotency key.

Once `commit` has run the side effect exists and no amount of escalation undoes it,
which is why everything expensive happens before it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from smart_router.schemas.routing import RiskClass
from smart_router.schemas.verification import TERMINAL_CATEGORIES, FailureCategory
from smart_router.tools.errors import (
    ToolAuthError,
    ToolError,
    ToolNotFoundError,
    ToolSemanticError,
    ToolTransientError,
)
from smart_router.tools.gateway import MCPGateway, ToolResult
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.tools.schema import validate_schema


@dataclass
class ToolProposal:
    """A parsed, not-yet-executed tool call."""

    tool: str
    arguments: dict[str, Any]
    spec: ToolSpec | None
    raw: str
    request_id: UUID | None = None

    @property
    def irreversible(self) -> bool:
        return bool(self.spec and self.spec.is_irreversible())

    @property
    def risk_class(self) -> RiskClass:
        if self.irreversible:
            return RiskClass.IRREVERSIBLE_WRITE
        return RiskClass.READ_ONLY

    def idempotency_key(self) -> str:
        """Stable across retries of the same logical call.

        Derived from the request and the exact arguments, so a retry after a transient
        fault cannot double-execute, while a genuinely different call gets a new key.
        """
        payload = json.dumps(
            {"rid": str(self.request_id), "tool": self.tool, "args": self.arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass
class CommitOutcome:
    committed: bool
    result: ToolResult | None = None
    category: FailureCategory | None = None
    detail: str = ""
    dry_run_result: ToolResult | None = None
    attempts: int = 0

    @property
    def should_escalate(self) -> bool:
        return self.category in (
            FailureCategory.TOOL_SEMANTIC,
            FailureCategory.TOOL_ARG_INVALID,
            FailureCategory.TOOL_NOT_FOUND,
            FailureCategory.TOOL_RESULT_SUSPECT,
        )

    @property
    def should_surface(self) -> bool:
        """Terminal: neither retry nor escalate. A human has to act."""
        return self.category in TERMINAL_CATEGORIES


class ParseError(ValueError):
    pass


def propose(raw: str, registry: ToolRegistry, request_id: UUID | None = None) -> ToolProposal:
    """Parse model output into a proposal. Never executes anything."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"output is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or "tool" not in payload:
        raise ParseError("output is not a tool-call envelope")
    tool = payload["tool"]
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ParseError("arguments is not an object")
    return ToolProposal(
        tool=tool if isinstance(tool, str) else str(tool),
        arguments=arguments,
        spec=registry.get(tool) if isinstance(tool, str) else None,
        raw=raw,
        request_id=request_id,
    )


def validate(proposal: ToolProposal) -> tuple[bool, FailureCategory | None, str]:
    """Registry-level checks. Cheap, deterministic, and before any network call.

    Walks the connector's declared schema recursively, so a nested error -- a wrong field
    name inside a SharePoint filter, an invented operator, a value of the wrong type three
    levels down -- is caught here rather than by the connector.
    """
    spec = proposal.spec
    if spec is None:
        return False, FailureCategory.TOOL_NOT_FOUND, f"tool {proposal.tool!r} not in registry"
    ok, detail = validate_schema(proposal.arguments, spec.schema_for_arguments())
    if not ok:
        return False, FailureCategory.TOOL_ARG_INVALID, detail
    return True, None, "valid"


@dataclass
class CommitGate:
    gateway: MCPGateway
    registry: ToolRegistry
    #: Transient tool faults retry the *tool*, not the model, and never escalate.
    max_tool_retries: int = 1
    #: Dry-run irreversible calls first where the gateway supports it.
    dry_run_irreversible: bool = True
    #: Irreversible calls that cannot be dry-run and have no human approval are refused
    #: rather than executed optimistically.
    require_approval_for_irreversible: bool = False
    result_verifiers: list = field(default_factory=list)

    def commit(
        self, proposal: ToolProposal, *, approved: bool = False, timeout: float | None = None
    ) -> CommitOutcome:
        ok, category, detail = validate(proposal)
        if not ok:
            return CommitOutcome(False, category=category, detail=detail)

        spec = proposal.spec
        assert spec is not None
        key = proposal.idempotency_key()
        outcome = CommitOutcome(False)

        if spec.is_irreversible():
            if self.require_approval_for_irreversible and not approved:
                return CommitOutcome(
                    False,
                    category=FailureCategory.APPROVAL_REQUIRED,
                    detail="irreversible tool requires explicit approval",
                )
            if self.dry_run_irreversible and spec.supports_dry_run:
                try:
                    outcome.dry_run_result = self.gateway.call(
                        proposal.tool, proposal.arguments,
                        idempotency_key=key, dry_run=True, timeout=timeout,
                    )
                except ToolError as exc:
                    return self._from_exception(exc, outcome)

        for attempt in range(self.max_tool_retries + 1):
            outcome.attempts = attempt + 1
            try:
                result = self.gateway.call(
                    proposal.tool, proposal.arguments,
                    idempotency_key=key, dry_run=False, timeout=timeout,
                )
            except ToolTransientError as exc:
                if attempt < self.max_tool_retries:
                    continue
                return self._from_exception(exc, outcome)
            except ToolError as exc:
                return self._from_exception(exc, outcome)

            # Executed cleanly. The result itself may still be wrong for the request --
            # an empty ticket list is not an error, but it may mean the wrong project
            # was queried.
            for verifier in self.result_verifiers:
                verdict = verifier.check(proposal, result)
                if not verdict.passed:
                    outcome.result = result
                    outcome.category = FailureCategory.TOOL_RESULT_SUSPECT
                    outcome.detail = verdict.detail
                    return outcome

            outcome.committed = True
            outcome.result = result
            outcome.detail = "committed"
            return outcome

        return outcome

    @staticmethod
    def _from_exception(exc: ToolError, outcome: CommitOutcome) -> CommitOutcome:
        if isinstance(exc, ToolAuthError):
            outcome.category = FailureCategory.TOOL_AUTH
        elif isinstance(exc, ToolTransientError):
            outcome.category = FailureCategory.TOOL_TRANSIENT
        elif isinstance(exc, ToolNotFoundError):
            outcome.category = FailureCategory.TOOL_NOT_FOUND
        elif isinstance(exc, ToolSemanticError):
            outcome.category = FailureCategory.TOOL_SEMANTIC
        else:
            outcome.category = FailureCategory.TOOL_SEMANTIC
        outcome.detail = str(exc)
        outcome.committed = False
        return outcome
