"""Reflexion critic (plan SSC).

Adapted from pramana-ai's `ReflexionCritic`. Only the LLM path carries over -- its
`_synthesize_symbolic` branch is entirely POMDP-specific (host policy types, door
marginals) and has no analogue here.

The discipline that carries over is `leak_ground_truth=False`: the critic is shown the
*failure* -- the rejected output and the verifier's finding -- but never the frontier
model's corrected answer. Feeding the right answer into a rule that is then injected
verbatim into later prompts converts measured adaptation into answer lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from smart_router.gateway.base import ModelResponse
from smart_router.memory.failures import FailureCluster
from smart_router.memory.hygiene import AntiMemorizationFilter, scrub
from smart_router.schemas.rule import RouterRule, RuleProvenance, RuleState

CRITIC_SYSTEM = """\
You analyse a cluster of similar failures by a small language model and state the single \
generalizable constraint that would have prevented all of them.

Rules:
- State a constraint that generalizes across the whole cluster, never the answer to one case.
- Never include instance literals: ticket ids, dates, emails, URLs, specific values.
- If the failures share no generalizable cause, reply with ABSTAIN.

Reply as JSON:
{"trigger_condition": str, "causal_invariant": str, "deduction_guideline": str}
or {"abstain": true, "reason": str}"""


@dataclass
class CriticOutput:
    rule: RouterRule | None
    reason: str


class ReflexionCritic:
    def __init__(
        self,
        complete: Callable[[list[dict[str, str]]], ModelResponse],
        *,
        leak_ground_truth: bool = False,
        hygiene: AntiMemorizationFilter | None = None,
        max_examples: int = 5,
    ) -> None:
        self._complete = complete
        #: Kept as an explicit switch so an experiment can measure the difference rather
        #: than assume it. Never true in production.
        self.leak_ground_truth = leak_ground_truth
        self.hygiene = hygiene or AntiMemorizationFilter()
        self.max_examples = max_examples

    def _prompt(self, cluster: FailureCluster) -> list[dict[str, str]]:
        lines = [
            f"FAILURE CATEGORY: {cluster.category.value}",
            f"CLUSTER SIZE: {len(cluster)} failures across "
            f"{cluster.distinct_queries} distinct queries",
            "",
        ]
        for i, rec in enumerate(cluster.records[: self.max_examples], 1):
            lines += [
                f"--- example {i} ---",
                f"REQUEST: {scrub(rec.query)}",
                f"REJECTED OUTPUT: {scrub(rec.rejected_output)}",
                f"VERIFIER FINDING: {rec.detail}",
            ]
            if self.leak_ground_truth and rec.frontier_output:
                lines.append(f"CORRECTED OUTPUT: {scrub(rec.frontier_output)}")
        return [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def synthesize(
        self, cluster: FailureCluster, rule_id: str, *, embedder=None
    ) -> CriticOutput:
        import json

        resp = self._complete(self._prompt(cluster))
        try:
            payload = json.loads(resp.text)
        except json.JSONDecodeError:
            return CriticOutput(None, "critic response was not valid JSON")

        if payload.get("abstain"):
            return CriticOutput(None, f"critic abstained: {payload.get('reason', '')}")

        try:
            trigger = scrub(str(payload["trigger_condition"]))
            invariant = scrub(str(payload["causal_invariant"]))
            deduction = scrub(str(payload["deduction_guideline"]))
        except KeyError as exc:
            return CriticOutput(None, f"critic omitted {exc.args[0]}")

        rule_text = f"{trigger} {invariant} {deduction}"
        source = cluster.records[0]
        rule_vec = source_vec = None
        if embedder is not None:
            rule_vec = embedder.encode([rule_text])[0]
            source_vec = source.vector

        verdict = self.hygiene.check(
            rule_text,
            source_query=source.query,
            source_answer=source.rejected_output,
            rule_vector=rule_vec,
            source_vector=source_vec,
        )

        rule = RouterRule(
            rule_id=rule_id,
            tenant_id=cluster.tenant_id,
            trigger_condition=trigger,
            causal_invariant=invariant,
            deduction_guideline=deduction,
            provenance=RuleProvenance(
                query_fingerprints=[r.query_fingerprint for r in cluster.records],
                failure_category=cluster.category,
                verifier_trace_ids=[r.request_id for r in cluster.records],
                frontier_trace_ids=[
                    r.request_id for r in cluster.records if r.frontier_output
                ],
                authored_by=resp.model,
                cluster_size=len(cluster),
            ),
            token_cost=max(1, len(rule_text) // 4),
            state=RuleState.PROPOSED,
        )

        if not verdict.ok:
            # Kept for audit rather than discarded: a pattern of quarantines is a signal
            # that the critic prompt, not the rule, is what needs fixing.
            rule.state = RuleState.QUARANTINED
            rule.quarantine_reason = verdict.reason
            return CriticOutput(rule, f"quarantined -- {verdict.reason}")

        return CriticOutput(rule, "synthesized")
