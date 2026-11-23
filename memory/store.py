"""Rule store and pinned snapshots (plan SSC.4, SSD).

Adapts pramana-ai's `RuleLedgerStore`: file persistence, deduplication by trigger, and
the monotonic `issued_rule_count` high-water mark so an id is never reused after pruning.

What is added is the reason snapshots exist at all. A live-mutating rule store changes
the prompt tail on every promotion, which invalidates the provider prefix cache
fleet-wide and makes a request irreproducible after the fact. Promotions are therefore
batched into immutable, versioned snapshots: cut one, load it into each replica, serve
until the next cut. Rollback runs on a different clock -- a harmful rule is pulled
immediately, without waiting for a cut.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smart_router.memory.lifecycle import LabelBiasEstimate, LifecyclePolicy
from smart_router.schemas.rule import RouterRule, RuleState


class RuleStore:
    def __init__(self, path: Path | None = None, policy: LifecyclePolicy | None = None) -> None:
        self.path = path
        self.policy = policy or LifecyclePolicy()
        self._rules: dict[str, RouterRule] = {}
        self._issued = 0
        if path and path.exists():
            self.load()

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self._issued = int(data.get("issued_rule_count", 0))
            self._rules = {
                r["rule_id"]: RouterRule.model_validate(r) for r in data.get("rules", [])
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            # A corrupted store must not take the router down; it degrades to no memory.
            self._rules, self._issued = {}, 0

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "issued_rule_count": self._issued,
                    "rules": [r.model_dump(mode="json") for r in self._rules.values()],
                },
                indent=2,
            )
        )

    # -- lifecycle -----------------------------------------------------------

    def next_rule_id(self) -> str:
        self._issued += 1
        return f"RULE-{self._issued:03d}"

    def add(self, rule: RouterRule) -> RouterRule:
        """Insert, deduplicating by (tenant, trigger).

        Accumulated telemetry survives re-synthesis of the same trigger -- otherwise a
        re-proposed rule silently resets its own evidence.
        """
        for existing in self._rules.values():
            same_tenant = existing.tenant_id == rule.tenant_id
            same_trigger = (
                existing.trigger_condition.strip().lower()
                == rule.trigger_condition.strip().lower()
            )
            if same_tenant and same_trigger:
                rule = rule.model_copy(
                    update={
                        "rule_id": existing.rule_id,
                        "application_count": max(rule.application_count, existing.application_count),
                        "success_count": max(rule.success_count, existing.success_count),
                        "false_trigger_count": max(
                            rule.false_trigger_count, existing.false_trigger_count
                        ),
                    }
                )
                break
        self._rules[rule.rule_id] = rule
        self.save()
        return rule

    def get(self, rule_id: str) -> RouterRule | None:
        return self._rules.get(rule_id)

    def all(self) -> list[RouterRule]:
        return list(self._rules.values())

    def promoted(self, tenant_id: str | None = None) -> list[RouterRule]:
        return [
            r
            for r in self._rules.values()
            if r.is_servable() and (tenant_id is None or r.tenant_id in (tenant_id, "GLOBAL"))
        ]

    def record_outcome(self, rule_id: str, *, succeeded: bool, triggered: bool = True) -> None:
        """Credit assignment.

        `application_count` increments only when this rule's trigger actually matched.
        Crediting every co-injected rule for a pass makes inert rules look good and
        computes the Wilson bound over contaminated counts.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            return
        if not triggered:
            rule.false_trigger_count += 1
        else:
            rule.application_count += 1
            if succeeded:
                rule.success_count += 1
        rule.wilson_lower_bound = self.policy.score(rule)
        self.save()

    def reconcile(self, bias: LabelBiasEstimate | None = None) -> list[tuple[str, RuleState, str]]:
        """Apply lifecycle transitions. Returns (rule_id, new_state, reason)."""
        changes = []
        for rule in self._rules.values():
            new_state, reason = self.policy.evaluate(rule, bias)
            if new_state is not rule.state:
                rule.state = new_state
                changes.append((rule.rule_id, new_state, reason))
        if changes:
            self.save()
        return changes

    def invalidate_for_model(self, live_model: str) -> list[str]:
        """Demote rules whose replay evidence predates the live checkpoint (SSC.3)."""
        demoted = []
        for rule in self._rules.values():
            if rule.is_servable() and rule.validated_against_model not in (None, live_model):
                rule.state = RuleState.SHADOW
                demoted.append(rule.rule_id)
        if demoted:
            self.save()
        return demoted

    def pull(self, rule_id: str, reason: str) -> None:
        """Immediate removal from service. Does not wait for a snapshot cut."""
        rule = self._rules.get(rule_id)
        if rule is not None:
            rule.state = RuleState.RETIRED
            rule.quarantine_reason = reason
            self.save()


def cut_snapshot(store: RuleStore, embedder, *, tenant_id: str | None = None):
    """Freeze the promoted set into an immutable, versioned vector snapshot."""
    from smart_router.vectors import build_snapshot

    import hashlib

    rules = store.promoted(tenant_id)
    digest = hashlib.sha256(
        "|".join(sorted(f"{r.rule_id}:{r.state.value}" for r in rules)).encode()
    ).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    version = f"v{stamp}-{digest}"
    return build_snapshot(version, embedder, rules, lambda r: r.trigger_text)
