"""End-to-end request trace. The unit the evaluation harness consumes."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from smart_router.schemas.routing import RoutingDecision, RouteTier
from smart_router.schemas.verification import VerificationOutcome


class ModelCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    tier: RouteTier
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class RequestTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    tenant_id: str
    session_id: str | None = None
    query: str
    decision: RoutingDecision
    calls: list[ModelCall] = Field(default_factory=list)
    verification: VerificationOutcome | None = None
    escalated: bool = False
    small_retried: bool = False
    final_output: str | None = None
    committed: bool = False

    #: Tool execution, when a commit gate is wired. `committed` above means the response
    #: was returned to the caller; `tool_committed` means a side effect actually happened.
    tool_name: str | None = None
    tool_committed: bool = False
    tool_attempts: int = 0
    #: First tool failure seen on this request. Set even when a later escalation
    #: recovered and committed -- the pair (tool_failure, tool_committed) distinguishes
    #: "failed" from "failed then recovered".
    tool_failure: str | None = None
    tool_dry_run: bool = False
    #: Terminal failures (auth) are surfaced to a human rather than escalated.
    surfaced_to_human: bool = False

    @property
    def total_cost_usd(self) -> float:
        base = sum(c.cost_usd for c in self.calls)
        return base + (self.verification.cost_usd if self.verification else 0.0)

    @property
    def total_latency_ms(self) -> float:
        base = sum(c.latency_ms for c in self.calls) + self.decision.decision_latency_ms
        return base + (self.verification.latency_ms if self.verification else 0.0)

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prompt tokens served from the provider prefix cache."""
        total = sum(c.prompt_tokens for c in self.calls)
        if total == 0:
            return 0.0
        return sum(c.cached_prompt_tokens for c in self.calls) / total
