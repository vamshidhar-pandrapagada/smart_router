"""Cost, latency and cache accounting over a set of traces.

Every number here is measured from traces, never estimated. Cost savings in particular
are reported against a control arm that shadow-routes a slice of traffic 100% to
frontier -- an estimated saving is not evidence (plan SSG).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from smart_router.schemas.routing import RouteReason, RouteTier
from smart_router.schemas.trace import RequestTrace


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


@dataclass
class RouterReport:
    requests: int
    cache_hits: int
    escalations: int
    policy_forced: int
    predicted_hard: int
    judged: int
    small_retries: int
    cost_usd: float
    cost_per_1k_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    prefix_cache_hit_ratio: float

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.requests if self.requests else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.requests if self.requests else 0.0

    def savings_vs(self, control_cost_per_1k_usd: float) -> float:
        """Fractional saving against a measured frontier-only control arm."""
        if control_cost_per_1k_usd <= 0:
            return 0.0
        return 1.0 - (self.cost_per_1k_usd / control_cost_per_1k_usd)


def report(traces: list[RequestTrace]) -> RouterReport:
    if not traces:
        return RouterReport(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    latencies = [t.total_latency_ms for t in traces]
    total_cost = sum(t.total_cost_usd for t in traces)
    prompt_tokens = sum(c.prompt_tokens for t in traces for c in t.calls)
    cached_tokens = sum(c.cached_prompt_tokens for t in traces for c in t.calls)

    return RouterReport(
        requests=len(traces),
        cache_hits=sum(1 for t in traces if t.decision.tier is RouteTier.CACHE),
        escalations=sum(1 for t in traces if t.escalated),
        policy_forced=sum(1 for t in traces if t.decision.reason is RouteReason.POLICY_FORCED),
        predicted_hard=sum(1 for t in traces if t.decision.reason is RouteReason.PREDICTED_HARD),
        judged=sum(1 for t in traces if t.verification and t.verification.judged()),
        small_retries=sum(1 for t in traces if t.small_retried),
        cost_usd=total_cost,
        cost_per_1k_usd=total_cost / len(traces) * 1000,
        p50_latency_ms=median(latencies),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        prefix_cache_hit_ratio=(cached_tokens / prompt_tokens) if prompt_tokens else 0.0,
    )


def format_report(r: RouterReport, control_cost_per_1k_usd: float | None = None) -> str:
    lines = [
        f"requests              {r.requests}",
        f"cache hit rate        {r.cache_hit_rate:.1%}  ({r.cache_hits} hits)",
        f"escalation rate       {r.escalation_rate:.1%}  ({r.escalations})",
        f"policy-forced         {r.policy_forced}",
        f"predicted-hard        {r.predicted_hard}",
        f"judged                {r.judged}",
        f"small retries         {r.small_retries}",
        f"prefix cache ratio    {r.prefix_cache_hit_ratio:.1%}",
        f"cost / 1k requests    ${r.cost_per_1k_usd:.2f}",
        f"latency p50/p95/p99   {r.p50_latency_ms:.0f} / {r.p95_latency_ms:.0f} / {r.p99_latency_ms:.0f} ms",
    ]
    if control_cost_per_1k_usd is not None:
        lines.append(
            f"savings vs control    {r.savings_vs(control_cost_per_1k_usd):.1%} "
            f"(control ${control_cost_per_1k_usd:.2f}/1k)"
        )
    return "\n".join(lines)
