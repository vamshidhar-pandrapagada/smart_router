"""The offline learning loop (plan SSC), as one callable.

Runs entirely off the request path. Nothing in `SmartRouter.route()` waits for any of it.

    failures -> cluster -> suppression gate -> critic -> hygiene
             -> replay (resolution + regression + set-level) -> lifecycle -> snapshot cut

The gap between a failure occurring and a rule going live is dominated by replay and the
canary window -- hours to days, not the next query. That delay is what buys correctness:
an ungated version is "next query self-corrects", and is also a system that writes rules
from noise and gets less reliable as it learns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_router.memory.critic import ReflexionCritic
from smart_router.memory.failures import FailureLog
from smart_router.memory.gate import SuppressionGate
from smart_router.memory.replay import ReplayHarness
from smart_router.memory.store import RuleStore
from smart_router.schemas.rule import RouterRule, RuleState


@dataclass
class LoopReport:
    clusters_found: int = 0
    clusters_gated_out: list[str] = field(default_factory=list)
    candidates_synthesized: int = 0
    quarantined: list[str] = field(default_factory=list)
    replay_rejected: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    held_in_shadow: list[str] = field(default_factory=list)
    #: Failure records whose cluster reached the critic. Consumed by the caller so the
    #: next pass does not re-synthesize -- and re-pay replay for -- the same failures.
    consumed_request_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"clusters={self.clusters_found} "
            f"gated_out={len(self.clusters_gated_out)} "
            f"synthesized={self.candidates_synthesized} "
            f"quarantined={len(self.quarantined)} "
            f"replay_rejected={len(self.replay_rejected)} "
            f"promoted={len(self.promoted)} "
            f"shadow={len(self.held_in_shadow)}"
        )


@dataclass
class LearningLoop:
    failure_log: FailureLog
    store: RuleStore
    critic: ReflexionCritic
    harness: ReplayHarness
    gate: SuppressionGate = field(default_factory=SuppressionGate)
    embedder: object | None = None
    similarity_threshold: float = 0.72

    def run(self, *, live_model: str, tenant_id: str | None = None) -> LoopReport:
        report = LoopReport()

        # Evidence gathered against a superseded checkpoint is not evidence (SSC.3).
        self.store.invalidate_for_model(live_model)

        clusters = self.failure_log.cluster(
            similarity_threshold=self.similarity_threshold, tenant_id=tenant_id
        )
        report.clusters_found = len(clusters)

        for cluster in clusters:
            decision = self.gate.evaluate(cluster)
            if not decision.admit:
                report.clusters_gated_out.append(decision.reason)
                continue

            out = self.critic.synthesize(
                cluster, self.store.next_rule_id(), embedder=self.embedder
            )
            # The critic has now seen this cluster, so its records are consumed whatever
            # the outcome. Clusters the gate stopped are deliberately NOT consumed: they
            # keep accumulating evidence until they are large enough to learn from.
            report.consumed_request_ids.extend(r.request_id for r in cluster.records)
            if out.rule is None:
                report.clusters_gated_out.append(out.reason)
                continue

            report.candidates_synthesized += 1
            rule: RouterRule = out.rule

            if rule.state is RuleState.QUARANTINED:
                self.store.add(rule)
                report.quarantined.append(f"{rule.rule_id}: {rule.quarantine_reason}")
                continue

            # Set-level: test the candidate alongside what it will actually ship with.
            co_retrieved = self.store.promoted(rule.tenant_id)[:3]
            replay = self.harness.run(
                rule,
                [r.query for r in cluster.records],
                model=live_model,
                co_retrieved=co_retrieved,
            )
            ok, reason = self.harness.verdict(replay)
            if not ok:
                rule.state = RuleState.SHADOW
                self.store.add(rule)
                report.replay_rejected.append(f"{rule.rule_id}: {reason}")
                continue

            # Replay evidence is the rule's first trials. It enters SHADOW, not PROMOTED:
            # promotion still requires a Wilson lower bound the replay alone rarely clears.
            rule.validated_against_model = live_model
            rule.application_count += replay.resolution_total
            rule.success_count += replay.resolution_passed
            rule.state = RuleState.SHADOW
            stored = self.store.add(rule)

            stored.wilson_lower_bound = self.store.policy.score(stored)
            new_state, why = self.store.policy.evaluate(stored)
            stored.state = new_state
            self.store.save()
            if new_state is RuleState.PROMOTED:
                report.promoted.append(f"{stored.rule_id}: {why}")
            else:
                report.held_in_shadow.append(f"{stored.rule_id}: {why}")

        return report
