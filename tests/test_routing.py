import json

import numpy as np
import pytest

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.base import ProviderTimeout
from smart_router.gateway.mock import MockClient, always, raises, sequence
from smart_router.routing.classifier import DifficultyClassifier, TrainingExample
from smart_router.routing.policy import PolicyGate
from smart_router.routing.router import SmartRouter
from smart_router.routing.threshold import CostModel
from smart_router.schemas.routing import RiskClass, RouteReason, RouteTier
from smart_router.verify.expensive import JudgeSampler, LLMJudge
from smart_router.verify.free import (
    JSONSchemaVerifier, ToolCallVerifier, ToolRegistry, ToolSpec,
)
from smart_router.verify.ladder import VerifierLadder
from smart_router.vectors import VectorSnapshot, build_snapshot

REGISTRY = ToolRegistry([
    ToolSpec(name="scheduler.create_job",
             arg_types={"name": "string", "schedule": "string", "send_if_empty": "boolean"},
             required=("name", "schedule"))
])


def build_router(small_handler, *, judge=None, sampler=None, policy=None, classifier=None):
    return SmartRouter(
        small=MockClient("small/default", handler=small_handler),
        frontier=MockClient("frontier/default", handler=always("frontier answer")),
        ladder=VerifierLadder(
            free=[JSONSchemaVerifier(), ToolCallVerifier(REGISTRY)],
            judge=judge, sampler=sampler or JudgeSampler(random_rate=0.0),
        ),
        embedder=HashingEmbedder(dim=64),
        policy=policy, classifier=classifier, cache=ExactCache(),
    )


# -- threshold ------------------------------------------------------------

def test_threshold_rises_with_cost_of_being_wrong():
    cm = CostModel(cost_slm_usd=0.0009, cost_frontier_usd=0.0825)
    cheap = cm.min_success_prob(0.05)
    expensive = cm.min_success_prob(50.0)
    assert expensive > cheap
    # A wrong irreversible write is costly enough to demand near-certainty.
    assert expensive > 0.99


def test_expected_cost_matches_the_documented_model():
    cm = CostModel(cost_slm_usd=0.0009, cost_frontier_usd=0.0825)
    assert cm.expected_cost(0.0, 0.0) == pytest.approx(0.0009)
    assert cm.expected_cost(1.0, 0.0) == pytest.approx(0.0834)


# -- classifier -----------------------------------------------------------

def test_classifier_abstains_below_the_label_floor():
    c = DifficultyClassifier(dim=8, min_labels=200)
    c.fit([TrainingExample(np.ones(8), True)] * 10, "hashing-8")
    assert not c.trained
    assert c.predict(np.ones(8), "hashing-8") is None


def test_classifier_learns_from_cascade_labels():
    emb = HashingEmbedder(dim=64)
    easy = emb.encode(["list my open tickets"] * 120)
    hard = emb.encode(["reconcile the quarterly ledger across subsidiaries"] * 120)
    examples = [TrainingExample(v, True) for v in easy] + [TrainingExample(v, False) for v in hard]
    c = DifficultyClassifier(dim=64, min_labels=200)
    c.fit(examples, emb.name)
    assert c.trained
    assert c.predict(emb.encode(["list my open tickets"])[0], emb.name) > 0.5
    assert c.predict(emb.encode(["reconcile the quarterly ledger across subsidiaries"])[0], emb.name) < 0.5


def test_classifier_rejects_a_foreign_embedding_space():
    c = DifficultyClassifier(dim=8, min_labels=1)
    c.fit([TrainingExample(np.ones(8), True), TrainingExample(-np.ones(8), False)], "hashing-8")
    with pytest.raises(ValueError, match="re-fit"):
        c.predict(np.ones(8), "some-other-model")


# -- policy ---------------------------------------------------------------

def test_policy_forces_frontier_on_sensitive_content():
    d = PolicyGate().evaluate("summarize the CONFIDENTIAL merger memo", tenant_id="t")
    assert d.force_frontier


def test_policy_classifies_irreversible_intent():
    p = PolicyGate()
    assert p.classify_risk("post it to #engineering") is RiskClass.IRREVERSIBLE_WRITE
    assert p.classify_risk("set up a daily digest") is RiskClass.IRREVERSIBLE_WRITE
    assert p.classify_risk("draft a reply") is RiskClass.REVERSIBLE_WRITE
    assert p.classify_risk("what is on my calendar") is RiskClass.READ_ONLY


def test_irreversible_intent_can_skip_the_cascade_entirely():
    """Open question 7: latency break-even on always-judge routes sits near 70%."""
    gate = PolicyGate(frontier_on_irreversible_intent=True)
    assert gate.evaluate("post it to #engineering", tenant_id="t").force_frontier
    assert not PolicyGate().evaluate("post it to #engineering", tenant_id="t").force_frontier


# -- router ---------------------------------------------------------------

def test_clean_response_commits_without_escalating():
    payload = json.dumps({"tool": "scheduler.create_job",
                          "arguments": {"name": "d", "schedule": "0 8 * * 1-5"}})
    trace = build_router(always(payload)).route("set up a daily digest")
    assert trace.committed and not trace.escalated
    assert trace.decision.tier is RouteTier.SMALL


def test_schema_failure_escalates_with_the_failure_context_attached():
    trace = build_router(always("{broken")).route("list my tickets")
    assert trace.escalated
    assert trace.final_output == "frontier answer"
    assert trace.decision.reason is RouteReason.VERIFIER_ESCALATION
    frontier_prompt = trace.calls[-1]
    # The frontier model must receive the rejected attempt, not a cold re-run.
    assert len(trace.calls) == 2


def test_escalation_prompt_carries_the_rejected_output():
    router = build_router(always("{broken"))
    router.route("list my tickets")
    sent = router.frontier.calls[-1][-1]["content"]
    assert "{broken" in sent and "SCHEMA_VIOLATION" in sent


def test_transient_timeout_retries_the_small_model_then_escalates():
    trace = build_router(raises(ProviderTimeout("boom"))).route("list my tickets")
    assert trace.small_retried and trace.escalated


def test_transient_timeout_that_recovers_does_not_escalate():
    payload = json.dumps({"tool": "scheduler.create_job",
                          "arguments": {"name": "d", "schedule": "0 8 * * 1-5"}})

    calls = {"n": 0}

    def flaky(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderTimeout("first attempt times out")
        return payload

    trace = build_router(flaky).route("set up a daily digest")
    assert trace.small_retried and not trace.escalated and trace.committed


def test_policy_forced_route_never_touches_the_small_model():
    router = build_router(always("{broken"))
    trace = router.route("summarize the CONFIDENTIAL memo")
    assert trace.decision.reason is RouteReason.POLICY_FORCED
    assert router.small.calls == []


def test_exact_cache_short_circuits_the_second_identical_query():
    payload = json.dumps({"tool": "scheduler.create_job",
                          "arguments": {"name": "d", "schedule": "0 8 * * 1-5"}})
    router = build_router(always(payload))
    router.route("set up a daily digest")
    trace = router.route("  SET UP a Daily Digest ")  # normalized to the same key
    assert trace.decision.tier is RouteTier.CACHE
    assert len(router.small.calls) == 1


def test_cache_is_scoped_per_tenant():
    payload = json.dumps({"tool": "scheduler.create_job",
                          "arguments": {"name": "d", "schedule": "0 8 * * 1-5"}})
    router = build_router(always(payload))
    router.route("set up a daily digest", tenant_id="acme")
    trace = router.route("set up a daily digest", tenant_id="globex")
    assert trace.decision.tier is not RouteTier.CACHE


# -- vectors --------------------------------------------------------------

def test_mmr_reduces_redundancy_versus_plain_topk():
    emb = HashingEmbedder(dim=128)
    items = ["jira jql string", "jira jql string format", "jira jql string syntax",
             "webex channel visibility", "outlook calendar timezone"]
    snap = build_snapshot("v1", emb, items, lambda s: s)
    q = emb.encode(["jira jql string"])[0]
    top = [s.item for s in snap.search(q, 3, emb.name)]
    mmr = [s.item for s in snap.search_mmr(q, 3, emb.name, lambda_=0.3)]
    assert len({t.split()[0] for t in top}) <= len({m.split()[0] for m in mmr})


def test_snapshot_rejects_a_mismatched_embedder():
    emb = HashingEmbedder(dim=32)
    snap = build_snapshot("v1", emb, ["a", "b"], lambda s: s)
    with pytest.raises(ValueError, match="re-embed"):
        snap.search(emb.encode(["a"])[0], 1, "different-model")


def test_snapshot_refuses_to_exceed_the_bruteforce_threshold():
    with pytest.raises(ValueError, match="in-process ANN"):
        VectorSnapshot("v1", "e", np.zeros((50_001, 4), np.float32), ["x"] * 50_001)


def test_intent_classifier_tolerates_words_between_verb_and_recurrence():
    """"Set me up a daily digest" must not classify READ_ONLY.

    An adjacency-only pattern ("set up a daily") misses the way people actually phrase
    this, and the whole judge gate hangs off the risk class.
    """
    p = PolicyGate()
    for q in ["Set me up a daily digest of unread mail",
              "create a daily job",
              "schedule a weekly report",
              "configure a nightly sync for me"]:
        assert p.classify_risk(q) is RiskClass.IRREVERSIBLE_WRITE, q
    assert p.classify_risk("summarize my unread Outlook mail") is RiskClass.READ_ONLY
