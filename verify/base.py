"""Verifier protocol and the payload verifiers inspect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from smart_router.gateway.base import ModelResponse
from smart_router.schemas.routing import RiskClass
from smart_router.schemas.verification import VerifierClass, VerifierResult


@dataclass
class VerificationContext:
    """Everything a verifier may look at."""

    query: str
    response: ModelResponse
    risk_class: RiskClass = RiskClass.READ_ONLY
    #: Source material the answer must stay grounded in (retrieved docs, tool output).
    grounding: Sequence[str] = field(default_factory=tuple)
    tool_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    verifier_id: str
    verifier_class: VerifierClass

    def check(self, ctx: VerificationContext) -> VerifierResult: ...


def passed(verifier_id: str, cls: VerifierClass, **kw: Any) -> VerifierResult:
    return VerifierResult(verifier_id=verifier_id, verifier_class=cls, passed=True, **kw)
