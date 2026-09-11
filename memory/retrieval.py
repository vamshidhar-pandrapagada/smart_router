"""Rule retrieval under a token budget (plan SSD).

Every injected rule is prompt tokens on the *cheap* path, paid on every request whether
or not the rule changes the outcome. Three controls keep the ledger from eating the
savings it exists to produce:

* top-k with MMR, so near-duplicate rules do not consume the budget twice,
* a hard per-request token cap, lowest-confidence rules dropped first,
* a similarity floor, so a weakly-related rule is not injected merely to fill the budget.

The floor also feeds credit assignment: a rule retrieved above the floor counts as
*triggered*; one that fires and turns out not to apply is a false trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smart_router.schemas.rule import RouterRule
from smart_router.vectors import VectorSnapshot


@dataclass
class Retrieved:
    rules: list[RouterRule]
    token_count: int
    dropped_for_budget: int = 0
    dropped_for_floor: int = 0

    @property
    def rule_ids(self) -> list[str]:
        return [r.rule_id for r in self.rules]


@dataclass
class RuleRetriever:
    snapshot: VectorSnapshot[RouterRule] | None
    embedder_name: str
    k: int = 4
    max_tokens: int = 400
    min_similarity: float = 0.35
    mmr_lambda: float = 0.7

    def retrieve(self, query_vector: np.ndarray) -> Retrieved:
        if self.snapshot is None or len(self.snapshot) == 0:
            return Retrieved([], 0)

        scored = self.snapshot.search_mmr(
            query_vector, self.k, self.embedder_name, lambda_=self.mmr_lambda
        )

        kept: list[RouterRule] = []
        tokens = 0
        below_floor = budget_dropped = 0
        # Highest Wilson lower bound first, so the budget is spent on the best evidence.
        for s in sorted(scored, key=lambda s: -s.item.wilson_lower_bound):
            if s.score < self.min_similarity:
                below_floor += 1
                continue
            cost = s.item.token_cost or max(1, len(s.item.render()) // 4)
            if tokens + cost > self.max_tokens:
                budget_dropped += 1
                continue
            kept.append(s.item)
            tokens += cost

        return Retrieved(kept, tokens, budget_dropped, below_floor)


def break_even_ledger_size(
    *,
    cost_slm_usd: float,
    cost_frontier_usd: float,
    input_usd_per_mtok: float,
    tokens_per_rule: int,
    escalations_avoided_per_1k: float,
) -> float:
    """Rules at which injection stops paying for itself (plan SSD).

    Injection is paid on every request; the saving accrues only on requests where a rule
    actually prevents an escalation. This returns the ledger size -- in rules injected
    per request -- at which those two cancel.
    """
    if tokens_per_rule <= 0 or input_usd_per_mtok <= 0:
        return float("inf")
    saving_per_1k = escalations_avoided_per_1k * (cost_frontier_usd - cost_slm_usd)
    cost_per_rule_per_1k = 1000 * tokens_per_rule * input_usd_per_mtok / 1_000_000
    if cost_per_rule_per_1k <= 0:
        return float("inf")
    return saving_per_1k / cost_per_rule_per_1k
