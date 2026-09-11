"""Counterfactual replay harness -- CI/CD for prompts (plan SSC.3).

A candidate rule is never served on synthesis. It must pass, out of band:

* **Resolution** -- replaying the failing traces with the rule injected makes the small
  model pass the verifier.
* **Regression** -- replaying a golden set of previously-passing queries shows no
  regression *beyond noise at a stated confidence*. "Zero regression" is too strict a
  bar: as the golden set grows, nothing would ever promote.
* **Set-level** -- rules are validated alone but deployed together. The candidate is
  re-tested alongside the rules it will actually be retrieved with, because two
  individually-good rules can conflict or dilute each other.

Every result is stamped with the small-model checkpoint it was gathered against. A model
upgrade invalidates the evidence and forces re-validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from smart_router.schemas.rule import RouterRule

#: (query, rules) -> did the small model pass verification?
ReplayFn = Callable[[str, Sequence[RouterRule]], bool]


@dataclass
class GoldenCase:
    query: str
    tenant_id: str = "default"
    #: Intent cluster this case belongs to; regression is judged per cluster.
    intent: str = "default"


@dataclass
class ReplayReport:
    rule_id: str
    validated_against_model: str
    resolution_passed: int = 0
    resolution_total: int = 0
    regression_baseline_passed: int = 0
    regression_with_rule_passed: int = 0
    regression_total: int = 0
    set_level_passed: int = 0
    set_level_total: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def resolution_rate(self) -> float:
        return self.resolution_passed / self.resolution_total if self.resolution_total else 0.0

    @property
    def regression_delta(self) -> int:
        """Negative means the rule broke previously-passing queries."""
        return self.regression_with_rule_passed - self.regression_baseline_passed


@dataclass
class ReplayHarness:
    replay: ReplayFn
    golden: list[GoldenCase] = field(default_factory=list)
    #: Share of the cluster's failures the rule must actually fix.
    min_resolution_rate: float = 0.6
    #: Confidence for the regression bound. A delta inside the noise band passes.
    regression_z: float = 1.96
    #: Absolute floor on the band, as a fraction of the golden set. Without this, a
    #: golden set that passes 100% at baseline has zero binomial variance, so a single
    #: regressed case rejects the rule -- silently restoring the "zero regression" bar
    #: that blocks all promotion as the set grows. Replay also has non-binomial noise
    #: (sampling temperature, provider variance) that the binomial term cannot see.
    min_regression_tolerance: float = 0.01

    def _noise_band(self, n: int, p: float) -> float:
        """Half-width of the acceptable regression band, in cases."""
        if n == 0:
            return 0.0
        binomial = self.regression_z * math.sqrt(max(p * (1 - p), 0.0) * n)
        return max(binomial, self.min_regression_tolerance * n)

    def run(
        self,
        rule: RouterRule,
        failing_queries: Sequence[str],
        *,
        model: str,
        co_retrieved: Sequence[RouterRule] = (),
    ) -> ReplayReport:
        report = ReplayReport(rule_id=rule.rule_id, validated_against_model=model)

        # 1. Resolution: does the rule fix what it was distilled from?
        for q in failing_queries:
            report.resolution_total += 1
            if self.replay(q, [rule]):
                report.resolution_passed += 1

        # 2. Regression: does it break anything that already worked?
        for case in self.golden:
            report.regression_total += 1
            if self.replay(case.query, []):
                report.regression_baseline_passed += 1
            if self.replay(case.query, [rule]):
                report.regression_with_rule_passed += 1

        # 3. Set-level: does it survive alongside the rules it ships with?
        if co_retrieved:
            combined = [*co_retrieved, rule]
            for q in failing_queries:
                report.set_level_total += 1
                if self.replay(q, combined):
                    report.set_level_passed += 1

        return report

    def verdict(self, report: ReplayReport) -> tuple[bool, str]:
        if report.resolution_total == 0:
            return False, "no failing traces to replay"
        if report.resolution_rate < self.min_resolution_rate:
            return False, (
                f"resolution {report.resolution_rate:.0%} < {self.min_resolution_rate:.0%}"
            )

        n = report.regression_total
        if n:
            p = report.regression_baseline_passed / n
            band = self._noise_band(n, p)
            if report.regression_delta < -band:
                return False, (
                    f"regression {report.regression_delta} cases exceeds the "
                    f"noise band of {band:.1f}"
                )

        if report.set_level_total:
            set_rate = report.set_level_passed / report.set_level_total
            if set_rate < self.min_resolution_rate:
                return False, (
                    f"passes alone but only {set_rate:.0%} alongside co-retrieved rules"
                )

        return True, (
            f"resolution {report.resolution_rate:.0%}, regression delta "
            f"{report.regression_delta:+d}"
        )
