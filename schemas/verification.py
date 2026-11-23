"""Verification contracts (plan SS2.5).

The failure taxonomy is deliberately NOT pramana's TYPE_I..TYPE_V, which classify
epistemic contamination in a POMDP and do not map onto router failures.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VerifierClass(str, Enum):
    FREE = "FREE"
    CHEAP = "CHEAP"
    EXPENSIVE = "EXPENSIVE"


class FailureCategory(str, Enum):
    # Transient infrastructure. These retry the SLM once; they never escalate.
    TIMEOUT = "TIMEOUT"
    PROVIDER_FAULT = "PROVIDER_FAULT"
    # Structural. Largely preventable by constrained decoding (the P set).
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_ARG_INVALID = "TOOL_ARG_INVALID"
    TRUNCATION = "TRUNCATION"
    # Tool execution, as reported by the MCP gateway. Distinct from the provider
    # categories above: TIMEOUT is the LLM endpoint, TOOL_TRANSIENT is the MCP server.
    TOOL_TRANSIENT = "TOOL_TRANSIENT"
    TOOL_SEMANTIC = "TOOL_SEMANTIC"
    TOOL_AUTH = "TOOL_AUTH"
    #: An irreversible call refused because no human approved it. Not a model failure:
    #: a frontier model would propose the same call and be refused the same way.
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    #: Executed cleanly but the result looks wrong for the request.
    TOOL_RESULT_SUSPECT = "TOOL_RESULT_SUSPECT"
    # Semantic. The residue (R) plus what no verifier sees (U).
    WRONG_TOOL_SELECTED = "WRONG_TOOL_SELECTED"
    UNGROUNDED_CLAIM = "UNGROUNDED_CLAIM"
    REFUSAL = "REFUSAL"
    SILENT_SEMANTIC = "SILENT_SEMANTIC"


#: Categories that are infrastructure noise rather than capability failures.
#: Escalating these spends frontier money on a problem the small model never had.
TRANSIENT_CATEGORIES = frozenset({
    FailureCategory.TIMEOUT,
    FailureCategory.PROVIDER_FAULT,
    FailureCategory.TOOL_TRANSIENT,
})

#: Neither retried nor escalated. A frontier model cannot supply a missing credential,
#: so spending a frontier call to rediscover that delays the only useful outcome:
#: surfacing the problem to a human.
TERMINAL_CATEGORIES = frozenset({FailureCategory.TOOL_AUTH, FailureCategory.APPROVAL_REQUIRED})


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifier_id: str
    verifier_class: VerifierClass
    passed: bool
    category: FailureCategory | None = None
    detail: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    #: Would grammar/schema-constrained decoding have prevented this failure?
    #: Feeds the P/R split that the Phase 1 kill gate turns on.
    preventable_by_construction: bool = False


class VerificationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    results: list[VerifierResult] = Field(default_factory=list)
    escalate: bool = False
    retry_small: bool = False
    first_failure: FailureCategory | None = None

    @property
    def passed(self) -> bool:
        return not self.escalate and not self.retry_small

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results)

    def judged(self) -> bool:
        return any(r.verifier_class is VerifierClass.EXPENSIVE for r in self.results)
