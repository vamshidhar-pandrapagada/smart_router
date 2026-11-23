"""Rule scoring and state transitions (plan SSC.1, SSC.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_router.schemas.rule import RouterRule, RuleState

Z_95 = 1.959963984540054


def wilson_lower_bound(successes: int, trials: int, z: float = Z_95) -> float:
    """Lower bound of the Wilson score interval.

    A raw ratio calls a rule excellent at 3-for-3. This does not.

    NOTE ON WHAT IS BEING ESTIMATED. The only label available cheaply is *verifier
    passed*, and the verifier cannot see silent semantic failure -- which is why the
    ladder has a judge and an offline audit at all. So this is a tight interval around
    "this rule makes output pass our checks", not "this rule makes output correct". The
    gap between those is the label bias, and it must be measured by an adjudicated audit
    sample (see `LabelBiasEstimate`) and carried explicitly, not assumed to be zero.
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


@dataclass
class LabelBiasEstimate:
    """Measured divergence between verifier-pass and adjudicated correctness."""

    audited: int
    verifier_passed: int
    actually_correct: int

    @property
    def bias(self) -> float:
        """How much the proxy label overstates correctness. Zero means unbiased."""
        if self.verifier_passed == 0:
            return 0.0
        return (self.verifier_passed - self.actually_correct) / self.verifier_passed

    def corrected(self, wilson_lb: float) -> float:
        return max(0.0, wilson_lb * (1.0 - self.bias))


@dataclass
class LifecyclePolicy:
    """`Proposed -> Shadow -> Promoted -> Measured -> Demoted/Retired`."""

    promote_lower_bound: float = 0.70
    demote_lower_bound: float = 0.50
    min_trials_to_demote: int = 20
    #: A rule that fires where it does not apply costs tokens on every unrelated query
    #: and can actively mislead. Win rate alone never surfaces this.
    max_false_trigger_rate: float = 0.30
    require_human_review_for_global: bool = True

    def score(self, rule: RouterRule, bias: LabelBiasEstimate | None = None) -> float:
        lb = wilson_lower_bound(rule.success_count, rule.application_count)
        return bias.corrected(lb) if bias else lb

    def false_trigger_rate(self, rule: RouterRule) -> float:
        total = rule.application_count + rule.false_trigger_count
        return rule.false_trigger_count / total if total else 0.0

    def evaluate(
        self, rule: RouterRule, bias: LabelBiasEstimate | None = None
    ) -> tuple[RuleState, str]:
        from smart_router.schemas.rule import GLOBAL_SCOPE

        lb = self.score(rule, bias)
        ftr = self.false_trigger_rate(rule)

        if rule.state is RuleState.QUARANTINED:
            return RuleState.QUARANTINED, rule.quarantine_reason or "quarantined"

        if ftr > self.max_false_trigger_rate and rule.application_count > 0:
            return RuleState.DEMOTED, f"false-trigger rate {ftr:.0%}"

        if rule.state in (RuleState.PROPOSED, RuleState.SHADOW):
            if lb < self.promote_lower_bound:
                return RuleState.SHADOW, f"Wilson LB {lb:.2f} < {self.promote_lower_bound}"
            if (
                self.require_human_review_for_global
                and rule.tenant_id == GLOBAL_SCOPE
                and not rule.human_reviewed
            ):
                return RuleState.SHADOW, "GLOBAL scope awaits human review"
            return RuleState.PROMOTED, f"Wilson LB {lb:.2f}"

        if rule.state is RuleState.PROMOTED:
            if rule.application_count >= self.min_trials_to_demote and lb < self.demote_lower_bound:
                return RuleState.DEMOTED, f"Wilson LB decayed to {lb:.2f}"
            return RuleState.PROMOTED, f"Wilson LB {lb:.2f}"

        return rule.state, "unchanged"
