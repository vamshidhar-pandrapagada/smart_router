"""Routing threshold, derived rather than hand-tuned (plan SSA.3).

Attempt the small model when the expected total cost of doing so is lower than routing
straight to frontier:

    P(fail) * (cost_slm + cost_frontier + cost_of_being_wrong)  <  cost_frontier

Everything there is measurable except `cost_of_being_wrong`, which is a business input
and differs per route -- a wrong summary and a wrong database write are not the same
number. A hand-tuned global 0.85 confidence cutoff silently assumes they are.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    cost_slm_usd: float
    cost_frontier_usd: float
    #: Verification spend on the optimistic path, paid whether or not it escalates.
    cost_verification_usd: float = 0.0

    def max_acceptable_failure_prob(self, cost_of_being_wrong_usd: float) -> float:
        """Highest P(fail) at which attempting the small model still pays."""
        attempt_overhead = self.cost_slm_usd + self.cost_verification_usd
        denominator = attempt_overhead + self.cost_frontier_usd + cost_of_being_wrong_usd
        if denominator <= 0:
            return 0.0
        # Attempting costs `attempt_overhead` even on success, so it must be recovered.
        numerator = self.cost_frontier_usd - attempt_overhead
        if numerator <= 0:
            return 0.0
        return min(1.0, numerator / denominator)

    def min_success_prob(self, cost_of_being_wrong_usd: float) -> float:
        """Threshold the difficulty classifier is compared against."""
        return 1.0 - self.max_acceptable_failure_prob(cost_of_being_wrong_usd)

    def expected_cost(self, p_fail: float, cost_of_being_wrong_usd: float) -> float:
        attempt = self.cost_slm_usd + self.cost_verification_usd
        return attempt + p_fail * (self.cost_frontier_usd + cost_of_being_wrong_usd)


#: Illustrative defaults matching gateway.model_config.REGISTRY at ~3k in / 500 out.
DEFAULT_COST_MODEL = CostModel(
    cost_slm_usd=0.0009,
    cost_frontier_usd=0.0825,
    cost_verification_usd=0.0,
)
