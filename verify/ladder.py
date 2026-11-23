"""Verifier ladder -- ordering, short-circuit, and the judge gate (plan SSB, SS2.3)."""

from __future__ import annotations

from uuid import UUID

from smart_router.schemas.routing import RiskClass
from smart_router.schemas.verification import (
    TRANSIENT_CATEGORIES,
    VerificationOutcome,
    VerifierClass,
    VerifierResult,
)
from smart_router.verify.base import VerificationContext, Verifier
from smart_router.verify.expensive import JudgeSampler


class VerifierLadder:
    """Runs free, then cheap, then (gated) expensive verifiers.

    Short-circuits on the first failure: there is no value in paying for a judge to
    confirm what a schema validator already rejected.
    """

    def __init__(
        self,
        *,
        free: list[Verifier] | None = None,
        cheap: list[Verifier] | None = None,
        judge: Verifier | None = None,
        sampler: JudgeSampler | None = None,
    ) -> None:
        self.free = free or []
        self.cheap = cheap or []
        self.judge = judge
        self.sampler = sampler or JudgeSampler()

    def run(
        self,
        request_id: UUID,
        ctx: VerificationContext,
        *,
        risk_class: RiskClass = RiskClass.READ_ONLY,
        cost_of_being_wrong_usd: float | None = None,
    ) -> VerificationOutcome:
        outcome = VerificationOutcome(request_id=request_id)

        for tier in (self.free, self.cheap):
            for verifier in tier:
                available = getattr(verifier, "available", None)
                if available is not None and not available(ctx):
                    continue
                result = verifier.check(ctx)
                outcome.results.append(result)
                if not result.passed:
                    return self._finish(outcome, result)

        if self.judge is not None:
            should, reason = self.sampler.should_judge(risk_class, cost_of_being_wrong_usd)
            if should:
                result = self.judge.check(ctx)
                result = result.model_copy(
                    update={"detail": f"[{reason}] {result.detail or ''}".strip()}
                )
                outcome.results.append(result)
                if not result.passed:
                    return self._finish(outcome, result)

        return outcome

    @staticmethod
    def _finish(outcome: VerificationOutcome, failure: VerifierResult) -> VerificationOutcome:
        outcome.first_failure = failure.category
        # Transient infrastructure faults retry the small model once; they never escalate.
        # Sending a 503 to a frontier model spends real money on a problem the small
        # model never had (plan SS1.2).
        if failure.category in TRANSIENT_CATEGORIES:
            outcome.retry_small = True
        else:
            outcome.escalate = True
        return outcome


def decomposition(outcomes: list[VerificationOutcome]) -> dict[str, int]:
    """Phase 1 failure decomposition: |D|, |P|, |R| over detected failures.

    |F| and |U| additionally require benchmark ground truth and are computed by the
    evaluation harness, not here -- production traffic does not carry correct answers.
    """
    detected = [o for o in outcomes if o.first_failure is not None]
    preventable = 0
    for o in detected:
        failing = next((r for r in o.results if not r.passed), None)
        if failing is not None and failing.preventable_by_construction:
            preventable += 1
    return {
        "detected": len(detected),
        "preventable_by_construction": preventable,
        "residue": len(detected) - preventable,
        "judged": sum(1 for o in outcomes if o.judged()),
    }
