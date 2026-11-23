"""Suppression gate (plan SSC.2)."""

from __future__ import annotations

from dataclasses import dataclass

from smart_router.memory.failures import FailureCluster
from smart_router.schemas.verification import TRANSIENT_CATEGORIES, FailureCategory


@dataclass
class GateDecision:
    admit: bool
    reason: str


@dataclass
class SuppressionGate:
    """Decides whether a failure cluster deserves a candidate rule.

    Four ways a cluster is rejected, in order of how often they fire in practice:

    1. Transient category. A 503 teaches nothing generalizable.
    2. Too few *distinct* queries. One query failing twenty times is one failure.
    3. Mostly preventable by construction. If a decoding constraint would have removed
       these, fix the decoder -- a rule about it is strictly worse: it costs tokens on
       every request, can be ignored by the model, and needs a lifecycle.
    4. Category is not rule-shaped. A refusal is a policy matter, not an invariant.
    """

    min_distinct_queries: int = 5
    #: Above this share of preventable failures, the cluster is a decoder bug.
    max_preventable_share: float = 0.5
    excluded_categories: frozenset[FailureCategory] = frozenset(
        TRANSIENT_CATEGORIES | {FailureCategory.REFUSAL}
    )

    def evaluate(self, cluster: FailureCluster) -> GateDecision:
        if cluster.category in self.excluded_categories:
            return GateDecision(False, f"category {cluster.category.value} is not rule-shaped")
        distinct = cluster.distinct_queries
        if distinct < self.min_distinct_queries:
            return GateDecision(
                False, f"only {distinct} distinct queries, need {self.min_distinct_queries}"
            )
        share = cluster.preventable_share
        if share > self.max_preventable_share:
            return GateDecision(
                False,
                f"{share:.0%} preventable by construction -- fix the decoder, not the prompt",
            )
        return GateDecision(True, f"{distinct} distinct queries, {share:.0%} preventable")
