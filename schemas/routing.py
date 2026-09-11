"""Routing contracts (plan SS2.5)."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RouteTier(str, Enum):
    CACHE = "CACHE"
    SMALL = "SMALL"
    FRONTIER = "FRONTIER"


class RouteReason(str, Enum):
    CACHE_HIT = "CACHE_HIT"
    POLICY_FORCED = "POLICY_FORCED"
    PREDICTED_HARD = "PREDICTED_HARD"
    OPTIMISTIC = "OPTIMISTIC"
    VERIFIER_ESCALATION = "VERIFIER_ESCALATION"


class RiskClass(str, Enum):
    """Coarse route classes carrying `cost_of_being_wrong` (open question 2).

    Three classes, not per-endpoint values: the number is a business input and
    nobody will maintain a per-endpoint table.
    """

    READ_ONLY = "READ_ONLY"
    REVERSIBLE_WRITE = "REVERSIBLE_WRITE"
    IRREVERSIBLE_WRITE = "IRREVERSIBLE_WRITE"


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    tenant_id: str
    tier: RouteTier
    reason: RouteReason
    selected_model: str
    risk_class: RiskClass = RiskClass.READ_ONLY
    p_small_succeeds: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_used: float | None = None
    cost_of_being_wrong_usd: float | None = None
    #: Pinned rule snapshot in force for this request. Logged so a request can be
    #: replayed against exactly the vectors and rules that were live.
    snapshot_version: str | None = None
    injected_rule_ids: list[str] = Field(default_factory=list)
    injected_token_count: int = 0
    decision_latency_ms: float = 0.0
