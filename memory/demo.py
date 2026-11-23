"""End-to-end reflexion loop demo. No API key required.

    python -m smart_router.memory.demo

Shows the full path from repeated failures to a promoted rule, and -- more usefully --
shows what gets rejected on the way. Most clusters never become rules.
"""

from __future__ import annotations

import json

from smart_router.embed import HashingEmbedder
from smart_router.gateway.base import ModelResponse
from smart_router.memory.critic import ReflexionCritic
from smart_router.memory.failures import FailureLog, FailureRecord, fingerprint
from smart_router.memory.gate import SuppressionGate
from smart_router.memory.loop import LearningLoop
from smart_router.memory.replay import GoldenCase, ReplayHarness
from smart_router.memory.retrieval import RuleRetriever
from smart_router.memory.store import RuleStore, cut_snapshot
from smart_router.schemas.verification import FailureCategory

EMB = HashingEmbedder(dim=256)
LIVE_MODEL = "small/default"


def failure(query, category, detail, preventable=False):
    return FailureRecord(
        request_id=f"req-{abs(hash(query)) % 100000}", tenant_id="acme", query=query,
        query_fingerprint=fingerprint(query), category=category,
        verifier_id="free.tool_call" if preventable else "expensive.judge",
        detail=detail, rejected_output='{"tool":"jira.search","arguments":{"jql":{}}}',
        vector=EMB.encode([query])[0], preventable_by_construction=preventable,
        model=LIVE_MODEL,
    )


def critic_handler(messages):
    """Stands in for a real critic."""
    body = messages[-1]["content"]
    if "TOOL_ARG_INVALID" in body:
        return ModelResponse(model="critic/mock", text=json.dumps({
            "trigger_condition": "constructing a jql argument for the jira search tool",
            "causal_invariant": "the connector types jql as a string; a structured object "
                                "is rejected before execution",
            "deduction_guideline": "serialize query conditions into JQL string syntax; "
                                   "never emit a nested object",
        }))
    return ModelResponse(model="critic/mock", text=json.dumps(
        {"abstain": True, "reason": "failures share no generalizable cause"}))


def main() -> None:
    log = FailureLog()

    # (a) A genuine recurring semantic failure -- eight distinct queries, one cause.
    for i in range(8):
        log.record(failure(
            f"find open blockers in project atlas{i} assigned to me",
            FailureCategory.TOOL_ARG_INVALID, "jql was an object, not a string"))
    # (b) One query retried repeatedly. Looks big, teaches nothing.
    for _ in range(15):
        log.record(failure("summarize my inbox", FailureCategory.UNGROUNDED_CLAIM,
                           "quoted span absent from source"))
    # (c) Transient noise.
    for i in range(9):
        log.record(failure(f"list calendar events for week {i}",
                           FailureCategory.TIMEOUT, "provider timeout"))
    # (d) Structural failures a decoder constraint would remove for free.
    for i in range(7):
        log.record(failure(f"create issue in project beta{i}",
                           FailureCategory.SCHEMA_VIOLATION, "invalid JSON", preventable=True))

    print("=" * 78)
    print(f"Failure log: {len(log)} records")
    print("=" * 78)
    for c in log.cluster():
        print(f"  {c.category.value:20} size={len(c):3}  distinct={c.distinct_queries:3}  "
              f"preventable={c.preventable_share:.0%}")

    store = RuleStore()
    loop = LearningLoop(
        failure_log=log,
        store=store,
        critic=ReflexionCritic(critic_handler, leak_ground_truth=False),
        harness=ReplayHarness(
            # Stands in for re-running the small model with the rule injected.
            replay=lambda q, rules: bool(rules) or not q.startswith("find open blockers"),
            golden=[GoldenCase(query=f"golden case {i}") for i in range(60)],
        ),
        gate=SuppressionGate(min_distinct_queries=5),
        embedder=EMB,
    )

    report = loop.run(live_model=LIVE_MODEL, tenant_id="acme")

    print("\n" + "=" * 78)
    print("Learning loop (offline — no request waits on this)")
    print("=" * 78)
    print(report.summary())
    for label, items in (
        ("gated out", report.clusters_gated_out),
        ("quarantined", report.quarantined),
        ("replay rejected", report.replay_rejected),
        ("held in shadow", report.held_in_shadow),
        ("promoted", report.promoted),
    ):
        for item in items:
            print(f"  {label:16} {item}")

    print("\n" + "=" * 78)
    print("Rule store")
    print("=" * 78)
    for r in store.all():
        print(f"  {r.rule_id}  {r.state.value:12} "
              f"LB={r.wilson_lower_bound:.2f} "
              f"{r.success_count}/{r.application_count}")
        print(f"    TRIGGER:   {r.trigger_condition}")
        print(f"    INVARIANT: {r.causal_invariant}")

    def show_snapshot(label):
        snap = cut_snapshot(store, EMB, tenant_id="acme")
        retriever = RuleRetriever(snap, EMB.name, k=3, max_tokens=400, min_similarity=0.2)
        print(f"\n{label}")
        print(f"  snapshot {snap.version}: {len(snap)} servable rule(s)")
        for probe in ("constructing a jql argument for a jira search",
                      "find open blockers in project zeta for me"):
            got = retriever.retrieve(EMB.encode([probe])[0])
            print(f"  {probe[:46]:48} -> {got.rule_ids or 'nothing'}")

    show_snapshot("Immediately after synthesis")

    # Shadow evaluation. The rule is live nowhere; it is replayed against traffic
    # offline while evidence accumulates. This is the delay that buys correctness --
    # 8-for-8 is not enough to put text into every future prompt.
    print("\n" + "=" * 78)
    print("Shadow evaluation — 25 further offline trials")
    print("=" * 78)
    rule_id = store.all()[0].rule_id
    for i in range(25):
        store.record_outcome(rule_id, succeeded=(i != 7), triggered=True)
    r = store.get(rule_id)
    print(f"  {rule_id}: {r.success_count}/{r.application_count}  "
          f"Wilson LB={r.wilson_lower_bound:.2f}")
    for rid, state, why in store.reconcile():
        print(f"  transition: {rid} -> {state.value} ({why})")

    show_snapshot("After promotion")

    print("\n" + "-" * 78)
    print("Note: HashingEmbedder is lexical, not semantic — it matches the first probe")
    print("and misses the second, which a real model would relate. Retrieval quality")
    print("here says nothing about production; install the 'embed' extra to measure it.")


if __name__ == "__main__":
    main()
