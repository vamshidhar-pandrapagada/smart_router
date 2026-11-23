"""Deterministic policy gate -- layer 1, ~0ms (plan SSA.1).

Config lookup, not prediction. This is where "being wrong is unacceptable regardless of
predicted difficulty" is enforced: data classification, irreversible side effects,
tenant tier, explicit override.

Note the intent classifier below. Open question 7 in the plan: for routes that always
invoke the judge, latency break-even sits near a 70% judge-pass rate, so classifying
irreversible *intent* from the query -- rather than waiting to inspect the proposed tool
call -- lets those go straight to frontier and skip both the SLM and the judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from smart_router.schemas.routing import RiskClass

#: Default cost_of_being_wrong per risk class (open question 2). Three coarse classes,
#: not per-endpoint values -- the number is a business input and a per-endpoint table
#: will not be maintained. These are placeholders; the business owns the real ones.
DEFAULT_COST_OF_BEING_WRONG_USD: dict[RiskClass, float] = {
    RiskClass.READ_ONLY: 0.05,
    RiskClass.REVERSIBLE_WRITE: 2.00,
    RiskClass.IRREVERSIBLE_WRITE: 50.00,
}

# Two alternations. The first is unconditionally irreversible; the second is a
# standing-configuration verb followed, within a few words, by a recurrence term.
# The gap matters: real requests say "set me up a daily digest", not "set up a daily
# digest", and an adjacency-only pattern silently classifies those READ_ONLY.
_IRREVERSIBLE_INTENT = re.compile(
    r"\b(send|post|publish|delete|remove|purchase|pay|transfer|deploy)\b"
    r"|\b(set|create|schedule|configure|add)\b(?:\s+\S+){0,4}?"
    r"\s+\b(daily|weekly|nightly|hourly|recurring|standing|cron)\b",
    re.IGNORECASE,
)
_WRITE_INTENT = re.compile(r"\b(create|update|draft|add|assign|move|rename)\b", re.IGNORECASE)


@dataclass
class PolicyDecision:
    force_frontier: bool
    risk_class: RiskClass
    cost_of_being_wrong_usd: float
    reason: str


@dataclass
class PolicyGate:
    #: Tenants whose traffic never touches the small model.
    frontier_only_tenants: frozenset[str] = frozenset()
    #: Substrings that mark restricted material.
    sensitive_markers: tuple[str, ...] = ("confidential", "restricted", "[secret]")
    #: Send irreversible-intent queries straight to frontier without attempting the SLM.
    #: Enable once Phase 1 reports a judge-pass rate below ~70% on those routes.
    frontier_on_irreversible_intent: bool = False
    cost_of_being_wrong: dict[RiskClass, float] = field(
        default_factory=lambda: dict(DEFAULT_COST_OF_BEING_WRONG_USD)
    )

    def classify_risk(self, query: str) -> RiskClass:
        if _IRREVERSIBLE_INTENT.search(query):
            return RiskClass.IRREVERSIBLE_WRITE
        if _WRITE_INTENT.search(query):
            return RiskClass.REVERSIBLE_WRITE
        return RiskClass.READ_ONLY

    def evaluate(
        self, query: str, *, tenant_id: str, override_frontier: bool = False
    ) -> PolicyDecision:
        risk = self.classify_risk(query)
        cost = self.cost_of_being_wrong[risk]

        if override_frontier:
            return PolicyDecision(True, risk, cost, "explicit override")
        if tenant_id in self.frontier_only_tenants:
            return PolicyDecision(True, risk, cost, f"tenant {tenant_id} is frontier-only")

        lowered = query.lower()
        for marker in self.sensitive_markers:
            if marker in lowered:
                return PolicyDecision(True, risk, cost, f"sensitive marker {marker!r}")

        if self.frontier_on_irreversible_intent and risk is RiskClass.IRREVERSIBLE_WRITE:
            return PolicyDecision(True, risk, cost, "irreversible intent")

        return PolicyDecision(False, risk, cost, "no policy constraint")
