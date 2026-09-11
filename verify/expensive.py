"""LLM judge -- the expensive tier (plan SS2.3).

Never universal. At ~2s and ~15x the cost of the SLM call it verifies, a judge on every
request erases both the latency and the cost case. `JudgeSampler` decides when it fires:

* a small random sample, which *measures* the system, and
* unconditionally on irreversible routes, which *protects* it.

Both are needed. A 2% random sample would miss a bad standing-job creation 49 times out
of 50; risk routing alone gives you no unbiased estimate of what the cheaper tiers miss.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable

from smart_router.gateway.base import ModelResponse
from smart_router.schemas.routing import RiskClass
from smart_router.schemas.verification import FailureCategory, VerifierClass, VerifierResult
from smart_router.verify.base import VerificationContext

JUDGE_SYSTEM = (
    "You audit another model's proposed action against the user's request. "
    "Report ONLY substantive mismatches: values that contradict what was asked, "
    "scope wider than requested, or missing constraints. Ignore style. "
    'Reply as JSON: {"verdict": "PASS"|"FAIL", "findings": [string]}'
)


class JudgeSampler:
    def __init__(
        self,
        *,
        random_rate: float = 0.02,
        always_judge: frozenset[RiskClass] = frozenset({RiskClass.IRREVERSIBLE_WRITE}),
        cost_of_being_wrong_threshold_usd: float | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.random_rate = random_rate
        self.always_judge = always_judge
        self.cost_of_being_wrong_threshold_usd = cost_of_being_wrong_threshold_usd
        self._rng = rng or random.Random()

    def should_judge(
        self, risk_class: RiskClass, cost_of_being_wrong_usd: float | None = None
    ) -> tuple[bool, str]:
        if risk_class in self.always_judge:
            return True, f"risk_class={risk_class.value}"
        threshold = self.cost_of_being_wrong_threshold_usd
        if (
            threshold is not None
            and cost_of_being_wrong_usd is not None
            and cost_of_being_wrong_usd >= threshold
        ):
            return True, f"cost_of_being_wrong={cost_of_being_wrong_usd:.2f}"
        if self._rng.random() < self.random_rate:
            return True, "random_sample"
        return False, "not_sampled"


class LLMJudge:
    verifier_id = "expensive.judge"
    verifier_class = VerifierClass.EXPENSIVE

    def __init__(
        self,
        complete: Callable[[list[dict[str, str]]], ModelResponse],
        *,
        cost_usd: float = 0.014,
    ) -> None:
        self._complete = complete
        self._cost_usd = cost_usd

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{ctx.query}\n\n"
                    f"PROPOSED ACTION:\n{ctx.response.text}"
                ),
            },
        ]
        resp = self._complete(messages)
        ms = (time.perf_counter() - t0) * 1000
        try:
            payload: Any = json.loads(resp.text)
            verdict = str(payload.get("verdict", "")).upper()
            findings = payload.get("findings") or []
        except (json.JSONDecodeError, AttributeError):
            # A judge that cannot answer in its own schema is not evidence of a failure.
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.EXPENSIVE,
                passed=True, detail="judge response unparseable; treated as no-finding",
                cost_usd=self._cost_usd, latency_ms=ms,
            )
        if verdict == "FAIL":
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.EXPENSIVE,
                passed=False, category=FailureCategory.SILENT_SEMANTIC,
                detail="; ".join(str(f) for f in findings) or "judge returned FAIL",
                cost_usd=self._cost_usd, latency_ms=ms,
            )
        return VerifierResult(
            verifier_id=self.verifier_id, verifier_class=VerifierClass.EXPENSIVE,
            passed=True, cost_usd=self._cost_usd, latency_ms=ms,
        )
