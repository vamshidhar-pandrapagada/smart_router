"""Cheap verifiers -- probabilistic, ~$0.002 (plan SSB).

These catch models that are *uncertain*. They are blind to models that are confidently
wrong, which is the failure class that actually threatens reliability -- see the
self-consistency note below.
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Callable, Sequence

from smart_router.schemas.verification import FailureCategory, VerifierClass, VerifierResult
from smart_router.verify.base import VerificationContext


class LogprobVerifier:
    """Flags low mean token logprob.

    Unavailable on managed endpoints that do not expose logprobs; the ladder skips it
    rather than silently passing (plan SSA.4 capability probe).
    """

    verifier_id = "cheap.logprob"
    verifier_class = VerifierClass.CHEAP

    def __init__(self, min_mean_logprob: float = -1.0) -> None:
        self.min_mean_logprob = min_mean_logprob

    def available(self, ctx: VerificationContext) -> bool:
        return bool(ctx.response.logprobs)

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        lps = ctx.response.logprobs or []
        ms = (time.perf_counter() - t0) * 1000
        if not lps:
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.CHEAP,
                passed=True, detail="logprobs unavailable; check skipped", latency_ms=ms,
            )
        avg = mean(lps)
        if avg < self.min_mean_logprob:
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.CHEAP,
                passed=False, category=FailureCategory.SILENT_SEMANTIC,
                detail=f"mean logprob {avg:.3f} < {self.min_mean_logprob}", latency_ms=ms,
            )
        return VerifierResult(
            verifier_id=self.verifier_id, verifier_class=VerifierClass.CHEAP,
            passed=True, detail=f"mean logprob {avg:.3f}", latency_ms=ms,
        )


class SelfConsistencyVerifier:
    """Samples the small model n times and measures agreement.

    IMPORTANT: agreement measures *stability*, not correctness. A model that misreads
    "weekday mornings" as "every day" will misread it identically on all n samples and
    score 3/3. This verifier catches wobble; it cannot catch confident error. Do not
    treat a pass here as evidence the answer is right.

    Samples are issued in parallel in production; the cost is n x the SLM call, which is
    still trivial next to a frontier call.
    """

    verifier_id = "cheap.self_consistency"
    verifier_class = VerifierClass.CHEAP

    def __init__(
        self,
        sampler: Callable[[VerificationContext], Sequence[str]],
        *,
        min_agreement: float = 0.66,
        cost_per_sample_usd: float = 0.0009,
        n: int = 3,
    ) -> None:
        self.sampler = sampler
        self.min_agreement = min_agreement
        self.cost_per_sample_usd = cost_per_sample_usd
        self.n = n

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        samples = list(self.sampler(ctx))
        cost = self.cost_per_sample_usd * len(samples)
        if not samples:
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.CHEAP,
                passed=True, detail="no samples", latency_ms=(time.perf_counter() - t0) * 1000,
            )
        target = ctx.response.text.strip()
        agree = sum(1 for s in samples if s.strip() == target)
        ratio = agree / len(samples)
        ms = (time.perf_counter() - t0) * 1000
        if ratio < self.min_agreement:
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.CHEAP,
                passed=False, category=FailureCategory.SILENT_SEMANTIC,
                detail=f"agreement {agree}/{len(samples)} below {self.min_agreement}",
                cost_usd=cost, latency_ms=ms,
            )
        return VerifierResult(
            verifier_id=self.verifier_id, verifier_class=VerifierClass.CHEAP,
            passed=True, detail=f"agreement {agree}/{len(samples)}", cost_usd=cost, latency_ms=ms,
        )
