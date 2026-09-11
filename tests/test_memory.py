"""Reflexion loop invariants (plan SSC)."""

import json

import numpy as np
import pytest

from smart_router.embed import HashingEmbedder
from smart_router.gateway.base import ModelResponse
from smart_router.memory.critic import ReflexionCritic
from smart_router.memory.exemplars import Exemplar, build_exemplar_snapshot
from smart_router.memory.failures import FailureLog, FailureRecord, fingerprint
from smart_router.memory.gate import SuppressionGate
from smart_router.memory.hygiene import AntiMemorizationFilter, scrub
from smart_router.memory.lifecycle import (
    LabelBiasEstimate, LifecyclePolicy, wilson_lower_bound,
)
from smart_router.memory.loop import LearningLoop
from smart_router.memory.replay import GoldenCase, ReplayHarness
from smart_router.memory.retrieval import RuleRetriever, break_even_ledger_size
from smart_router.memory.store import RuleStore, cut_snapshot
from smart_router.schemas.rule import GLOBAL_SCOPE, RouterRule, RuleProvenance, RuleState
from smart_router.schemas.verification import FailureCategory

EMB = HashingEmbedder(dim=128)


def rec(query, category=FailureCategory.TOOL_ARG_INVALID, preventable=False, tenant="acme"):
    return FailureRecord(
        request_id=f"r-{abs(hash(query)) % 10000}", tenant_id=tenant, query=query,
        query_fingerprint=fingerprint(query), category=category,
        verifier_id="free.tool_call", detail="jql was an object",
        rejected_output='{"tool":"jira.search","arguments":{"jql":{}}}',
        vector=EMB.encode([query])[0], preventable_by_construction=preventable,
        model="small/default",
    )


def make_rule(rid="RULE-001", tenant="acme", **kw):
    return RouterRule(
        rule_id=rid, tenant_id=tenant,
        trigger_condition=kw.pop("trigger", "jira search tool call"),
        causal_invariant="the connector types jql as a string",
        deduction_guideline="serialize conditions into JQL syntax",
        provenance=RuleProvenance(
            failure_category=FailureCategory.TOOL_ARG_INVALID, authored_by="critic"
        ),
        token_cost=kw.pop("token_cost", 40), **kw,
    )


# -- Wilson and label bias -------------------------------------------------

def test_wilson_does_not_call_a_rule_excellent_at_three_for_three():
    assert wilson_lower_bound(3, 3) < 0.5
    assert wilson_lower_bound(30, 30) > 0.85


def test_label_bias_discounts_the_wilson_bound():
    unbiased = LabelBiasEstimate(audited=100, verifier_passed=80, actually_correct=80)
    biased = LabelBiasEstimate(audited=100, verifier_passed=80, actually_correct=60)
    assert unbiased.bias == 0.0
    assert biased.bias == pytest.approx(0.25)
    assert biased.corrected(0.80) < unbiased.corrected(0.80)


# -- suppression gate ------------------------------------------------------

def test_gate_rejects_a_cluster_of_one_query_seen_many_times():
    log = FailureLog()
    for _ in range(20):
        log.record(rec("find my open tickets"))
    cluster = log.cluster()[0]
    assert len(cluster) == 20 and cluster.distinct_queries == 1
    assert not SuppressionGate().evaluate(cluster).admit


def test_gate_rejects_a_cluster_that_constrained_decoding_would_fix():
    log = FailureLog()
    for i in range(8):
        log.record(rec(f"search jira project {i} for blockers", preventable=True))
    cluster = log.cluster()[0]
    d = SuppressionGate().evaluate(cluster)
    assert not d.admit and "decoder" in d.reason


def test_gate_rejects_transient_categories():
    log = FailureLog()
    for i in range(9):
        log.record(rec(f"query variant {i}", category=FailureCategory.TIMEOUT))
    assert not SuppressionGate().evaluate(log.cluster()[0]).admit


def test_gate_admits_a_recurring_semantic_cluster():
    log = FailureLog()
    for i in range(8):
        log.record(rec(f"search jira project alpha{i} for open blockers"))
    cluster = max(log.cluster(), key=lambda c: c.distinct_queries)
    assert SuppressionGate(min_distinct_queries=5).evaluate(cluster).admit


# -- hygiene ---------------------------------------------------------------

def test_hygiene_rejects_instance_literals():
    f = AntiMemorizationFilter()
    assert not f.check("always set jql for ABC-123", source_query="q", source_answer="a").ok
    assert not f.check("email bob@corp.com first", source_query="q", source_answer="a").ok


def test_hygiene_rejects_a_rule_that_copies_its_source():
    f = AntiMemorizationFilter(max_ngram_overlap=0.3)
    src = "the quick brown fox jumps over the lazy dog repeatedly today"
    assert not f.check(src, source_query=src, source_answer="").ok


def test_scrub_redacts_secrets():
    assert "sk-" not in scrub("key sk-abcd1234efgh here")


# -- critic ----------------------------------------------------------------

def _critic(payload, leak=False):
    return ReflexionCritic(
        lambda m: ModelResponse(text=json.dumps(payload), model="critic/mock"),
        leak_ground_truth=leak,
    )


def test_critic_quarantines_a_memorizing_rule():
    out = _critic({
        "trigger_condition": "jira search",
        "causal_invariant": "ticket ABC-123 needs jql as string",
        "deduction_guideline": "use jql string",
    }).synthesize(_cluster(), "RULE-001")
    assert out.rule.state is RuleState.QUARANTINED
    assert "literal" in out.rule.quarantine_reason


def _cluster():
    log = FailureLog()
    for i in range(6):
        log.record(rec(f"search jira project alpha{i} for open blockers"))
    return max(log.cluster(), key=lambda c: c.distinct_queries)


def test_critic_honours_abstain():
    out = _critic({"abstain": True, "reason": "no shared cause"}).synthesize(_cluster(), "RULE-001")
    assert out.rule is None and "abstain" in out.reason


def test_critic_is_blinded_to_the_frontier_answer_by_default():
    seen = {}

    def capture(messages):
        seen["prompt"] = messages[-1]["content"]
        return ModelResponse(
            text=json.dumps({"trigger_condition": "t", "causal_invariant": "i",
                             "deduction_guideline": "d"}),
            model="critic/mock",
        )

    cluster = _cluster()
    for r in cluster.records:
        r.frontier_output = "THE CORRECT ANSWER"
    ReflexionCritic(capture, leak_ground_truth=False).synthesize(cluster, "RULE-001")
    assert "THE CORRECT ANSWER" not in seen["prompt"]

    ReflexionCritic(capture, leak_ground_truth=True).synthesize(cluster, "RULE-002")
    assert "THE CORRECT ANSWER" in seen["prompt"]


# -- replay ----------------------------------------------------------------

def test_replay_rejects_a_rule_that_does_not_fix_its_own_failures():
    h = ReplayHarness(replay=lambda q, rules: False)
    r = h.run(make_rule(), ["q1", "q2", "q3"], model="small/default")
    ok, reason = h.verdict(r)
    assert not ok and "resolution" in reason


def test_replay_rejects_a_rule_that_breaks_the_golden_set():
    golden = [GoldenCase(query=f"golden {i}") for i in range(40)]

    def replay(q, rules):
        if q.startswith("golden"):
            return not rules      # every golden case regresses when the rule is present
        return bool(rules)

    h = ReplayHarness(replay=replay, golden=golden)
    r = h.run(make_rule(), ["f1", "f2", "f3"], model="small/default")
    ok, reason = h.verdict(r)
    assert not ok and "regression" in reason


def test_a_perfect_baseline_still_leaves_a_regression_tolerance():
    """At p=1.0 the binomial variance is zero. Without an absolute floor the bar
    silently reverts to "zero regression", which blocks all promotion."""
    h = ReplayHarness(replay=lambda q, rules: True, golden=[GoldenCase(query=f"g{i}") for i in range(200)])
    assert h._noise_band(200, 1.0) >= 2.0


def test_replay_tolerates_regression_within_the_noise_band():
    golden = [GoldenCase(query=f"golden {i}") for i in range(200)]
    state = {"n": 0}

    def replay(q, rules):
        if q.startswith("golden"):
            if rules:
                state["n"] += 1
                return state["n"] % 100 != 0   # one case in a hundred regresses
            return True
        return bool(rules)

    h = ReplayHarness(replay=replay, golden=golden)
    ok, _ = h.verdict(h.run(make_rule(), ["f1", "f2"], model="small/default"))
    assert ok


def test_replay_catches_a_rule_that_passes_alone_but_fails_in_company():
    h = ReplayHarness(replay=lambda q, rules: len(rules) == 1)
    r = h.run(make_rule(), ["f1", "f2"], model="small/default",
              co_retrieved=[make_rule("RULE-090"), make_rule("RULE-091")])
    ok, reason = h.verdict(r)
    assert not ok and "alongside" in reason


# -- lifecycle and store ---------------------------------------------------

def test_promotion_requires_more_than_a_perfect_short_run():
    policy = LifecyclePolicy()
    rule = make_rule(application_count=3, success_count=3, state=RuleState.SHADOW)
    assert policy.evaluate(rule)[0] is RuleState.SHADOW
    rule = make_rule(application_count=30, success_count=29, state=RuleState.SHADOW)
    assert policy.evaluate(rule)[0] is RuleState.PROMOTED


def test_high_false_trigger_rate_demotes_a_winning_rule():
    rule = make_rule(application_count=40, success_count=40, false_trigger_count=40,
                     state=RuleState.PROMOTED)
    state, reason = LifecyclePolicy().evaluate(rule)
    assert state is RuleState.DEMOTED and "false-trigger" in reason


def test_global_rules_wait_for_human_review():
    rule = make_rule(tenant=GLOBAL_SCOPE, application_count=50, success_count=50,
                     state=RuleState.SHADOW)
    assert LifecyclePolicy().evaluate(rule)[0] is RuleState.SHADOW
    rule.human_reviewed = True
    assert LifecyclePolicy().evaluate(rule)[0] is RuleState.PROMOTED


def test_credit_goes_only_to_rules_that_actually_triggered():
    store = RuleStore()
    store.add(make_rule("RULE-001"))
    store.record_outcome("RULE-001", succeeded=True, triggered=True)
    store.record_outcome("RULE-001", succeeded=False, triggered=False)
    r = store.get("RULE-001")
    assert r.application_count == 1 and r.success_count == 1 and r.false_trigger_count == 1


def test_model_upgrade_invalidates_promoted_rules():
    store = RuleStore()
    store.add(make_rule("RULE-001", state=RuleState.PROMOTED,
                        validated_against_model="small/v1"))
    assert store.invalidate_for_model("small/v2") == ["RULE-001"]
    assert store.get("RULE-001").state is RuleState.SHADOW


def test_store_preserves_telemetry_when_a_trigger_is_resynthesized():
    store = RuleStore()
    store.add(make_rule("RULE-001", application_count=25, success_count=20))
    store.add(make_rule("RULE-002", trigger="jira search tool call"))  # same trigger
    assert len(store.all()) == 1
    assert store.get("RULE-001").application_count == 25


def test_pull_is_immediate_and_does_not_wait_for_a_snapshot_cut():
    store = RuleStore()
    store.add(make_rule("RULE-001", state=RuleState.PROMOTED))
    assert store.promoted()
    store.pull("RULE-001", "harmful")
    assert not store.promoted()


def test_store_survives_a_corrupted_file(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text("{not json")
    assert RuleStore(p).all() == []


# -- retrieval and budget --------------------------------------------------

def test_retrieval_respects_the_token_budget():
    store = RuleStore()
    for i in range(6):
        store.add(make_rule(f"RULE-{i:03d}", trigger=f"jira search variant {i}",
                            state=RuleState.PROMOTED, token_cost=150))
    snap = cut_snapshot(store, EMB)
    r = RuleRetriever(snap, EMB.name, k=6, max_tokens=400, min_similarity=0.0)
    got = r.retrieve(EMB.encode(["jira search variant 2"])[0])
    assert got.token_count <= 400 and got.dropped_for_budget > 0


def test_retrieval_drops_weakly_related_rules_rather_than_filling_the_budget():
    store = RuleStore()
    store.add(make_rule("RULE-001", trigger="outlook calendar timezone handling",
                        state=RuleState.PROMOTED))
    snap = cut_snapshot(store, EMB)
    r = RuleRetriever(snap, EMB.name, min_similarity=0.9)
    got = r.retrieve(EMB.encode(["completely unrelated jira query"])[0])
    assert got.rules == [] and got.dropped_for_floor == 1


def test_break_even_ledger_size_is_finite():
    n = break_even_ledger_size(
        cost_slm_usd=0.0009, cost_frontier_usd=0.0825,
        input_usd_per_mtok=0.20, tokens_per_rule=100,
        escalations_avoided_per_1k=20,
    )
    assert 0 < n < 1e6


# -- exemplar control arm --------------------------------------------------

def test_exemplars_and_rules_share_the_injection_path():
    from smart_router.prompt import assemble
    ex = Exemplar(query="find blockers", verified_output='{"tool":"jira.search"}')
    p = assemble(system="s", rules=[ex], current_turn="q")
    assert "CORRECT OUTPUT" in p.volatile[0]["content"]
    assert all("CORRECT OUTPUT" not in m["content"] for m in p.prefix)


def test_exemplar_retriever_returns_the_nearest_verified_success():
    ex = [Exemplar(query="find open blockers in jira", verified_output="A"),
          Exemplar(query="what is my calendar today", verified_output="B")]
    snap = build_exemplar_snapshot("v1", EMB, ex)
    from smart_router.memory.exemplars import ExemplarRetriever
    got = ExemplarRetriever(snap, EMB.name, k=1, min_similarity=0.0).retrieve(
        EMB.encode(["find open blockers in jira"])[0]
    )
    assert got.rules[0].verified_output == "A"


def test_snapshot_versions_do_not_collide_within_the_same_second():
    """snapshot_version is what makes a request replayable; collisions are not cosmetic."""
    store = RuleStore()
    store.add(make_rule("RULE-001", state=RuleState.PROMOTED))
    a = cut_snapshot(store, EMB)
    b = cut_snapshot(store, EMB)          # same content -> same version, by design
    store.add(make_rule("RULE-002", trigger="outlook timezone", state=RuleState.PROMOTED))
    c = cut_snapshot(store, EMB)          # different content -> different version
    assert a.version == b.version
    assert c.version != a.version


def test_learning_loop_rejects_three_cluster_kinds_and_admits_one():
    """The loop's job is mostly saying no."""
    log = FailureLog()
    for i in range(8):
        log.record(rec(f"search jira project alpha{i} for open blockers"))
    for _ in range(15):
        log.record(rec("one query many times", category=FailureCategory.UNGROUNDED_CLAIM))
    for i in range(9):
        log.record(rec(f"list all my calendar events for the week of month {i}",
                       category=FailureCategory.TIMEOUT))
    # Long enough to actually cluster: the lexical embedder needs shared tokens, and a
    # singleton would be rejected for distinct-count before the preventable check runs.
    for i in range(7):
        log.record(rec(f"create an issue in project beta{i} with summary and description",
                       category=FailureCategory.SCHEMA_VIOLATION, preventable=True))

    critic = ReflexionCritic(lambda m: ModelResponse(model="c", text=json.dumps({
        "trigger_condition": "jira search jql argument",
        "causal_invariant": "the connector types jql as a string",
        "deduction_guideline": "serialize into JQL syntax",
    })))
    store = RuleStore()
    loop = LearningLoop(
        failure_log=log, store=store, critic=critic,
        harness=ReplayHarness(replay=lambda q, rules: bool(rules),
                              golden=[GoldenCase(query=f"g{i}") for i in range(40)]),
        gate=SuppressionGate(min_distinct_queries=5), embedder=EMB,
    )
    report = loop.run(live_model="small/default", tenant_id="acme")
    # Cluster *count* depends on the embedder's similarity geometry, so assert on
    # behaviour instead: each rejection kind fires, and exactly one candidate survives.
    reasons = " | ".join(report.clusters_gated_out)
    assert "distinct queries" in reasons        # one query seen many times
    assert "TIMEOUT" in reasons                 # transient category
    assert "decoder" in reasons                 # preventable by construction
    assert report.candidates_synthesized == 1


def test_a_perfect_replay_run_is_not_enough_to_promote():
    """8-for-8 does not put text into every future prompt."""
    log = FailureLog()
    for i in range(8):
        log.record(rec(f"search jira project alpha{i} for open blockers"))
    critic = ReflexionCritic(lambda m: ModelResponse(model="c", text=json.dumps({
        "trigger_condition": "jira search jql argument",
        "causal_invariant": "the connector types jql as a string",
        "deduction_guideline": "serialize into JQL syntax",
    })))
    store = RuleStore()
    LearningLoop(
        failure_log=log, store=store, critic=critic,
        harness=ReplayHarness(replay=lambda q, rules: bool(rules)),
        gate=SuppressionGate(min_distinct_queries=5), embedder=EMB,
    ).run(live_model="small/default", tenant_id="acme")

    assert store.all()[0].state is RuleState.SHADOW
    assert store.promoted() == []          # nothing servable yet

    rid = store.all()[0].rule_id
    for i in range(25):
        store.record_outcome(rid, succeeded=(i != 7), triggered=True)
    store.reconcile()
    assert store.get(rid).state is RuleState.PROMOTED
