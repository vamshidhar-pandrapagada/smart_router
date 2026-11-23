"""A retrieved rule binds to the turn it was retrieved for, and no further.

Raised in review of the architecture article: the volatile-tail placement of rules was
argued on prefix-cache grounds, but it carries a separate *correctness* property worth
testing on its own terms — a rule that applied on turn N must not silently steer turn
N+1. The existing tests cover the mechanism (rules are not written into history, the
cacheable prefix does not change). These cover the consequence: the model's view reverts.

The distinction matters because the mechanism could stay intact while the behaviour broke
— if the assistant's answer carrying the rule's effect were replayed as history context,
or if retrieval were sticky per session rather than per request.
"""

import json

import pytest

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.mock import MockClient
from smart_router.memory.reflexion import Reflexion, ReflexionConfig
from smart_router.prompt import ConversationHistory, assemble
from smart_router.routing.router import SmartRouter
from smart_router.schemas.rule import RouterRule, RuleProvenance, RuleState
from smart_router.schemas.verification import FailureCategory
from smart_router.verify.ladder import VerifierLadder

RULE_SENTINEL = "<<injected-rules>>"


def make_rule(rule_id: str, trigger: str, deduction: str, tenant: str = "default") -> RouterRule:
    return RouterRule(
        rule_id=rule_id, tenant_id=tenant,
        trigger_condition=trigger,
        causal_invariant="the downstream consumer rejects the other shape",
        deduction_guideline=deduction,
        provenance=RuleProvenance(
            failure_category=FailureCategory.TOOL_ARG_INVALID, authored_by="critic"
        ),
        state=RuleState.PROMOTED, token_cost=30,
    )


# -- prompt level: what the model is shown -------------------------------------

def test_a_rule_shown_on_one_turn_is_absent_from_the_next():
    """Turn 1 is governed by the rule; turn 2 is not, and the history carries no trace."""
    rule = make_rule("RULE-001", "formatting a date argument", "emit ISO-8601, never a locale string")
    history = ConversationHistory()

    first = assemble(system="sys", history=history, rules=[rule], current_turn="schedule the review")
    assert any(RULE_SENTINEL in m["content"] for m in first.volatile)
    assert first.injected_rule_ids == ("RULE-001",)

    # The turn completes and is recorded. Only the clean exchange is kept.
    history.append("user", "schedule the review")
    history.append("assistant", '{"when": "2026-03-04"}')

    second = assemble(system="sys", history=history, rules=[], current_turn="who owns the review")
    rendered = " ".join(m["content"] for m in second.messages)
    assert RULE_SENTINEL not in rendered
    assert "ISO-8601" not in rendered
    assert second.injected_rule_ids == ()


def test_a_different_rule_on_the_next_turn_fully_replaces_the_first():
    """Not additive: turn 2 sees its own rule and only its own."""
    first_rule = make_rule("RULE-001", "formatting a date argument", "emit ISO-8601")
    second_rule = make_rule("RULE-002", "naming a recipient", "use the canonical room id")
    history = ConversationHistory()

    assemble(system="sys", history=history, rules=[first_rule], current_turn="schedule it")
    history.append("user", "schedule it")
    history.append("assistant", "done")

    second = assemble(system="sys", history=history, rules=[second_rule], current_turn="post it")
    rendered = " ".join(m["content"] for m in second.messages)
    assert "RULE-002" in rendered and "RULE-001" not in rendered


def test_the_cacheable_prefix_is_unchanged_by_which_rule_was_shown():
    """The property the article argued, restated as a control for the tests above:
    reverting the rule must not have cost a stable prefix."""
    history = ConversationHistory()
    history.append("user", "schedule it")
    history.append("assistant", "done")
    with_rule = assemble(system="sys", history=history,
                         rules=[make_rule("RULE-001", "t", "d")], current_turn="post it")
    without = assemble(system="sys", history=history, rules=[], current_turn="post it")
    assert with_rule.prefix_fingerprint() == without.prefix_fingerprint()


# -- router level: what the trace records --------------------------------------

@pytest.fixture
def reflexion(tmp_path):
    embedder = HashingEmbedder(dim=128)
    rx = Reflexion(
        ReflexionConfig(enabled=True, mode="rules", state_dir=tmp_path / "state",
                        min_similarity=0.25, k=2),
        embedder,
    )
    rx.rule_store.add(make_rule(
        "RULE-001",
        "scheduling a recurring digest with a cadence",
        "map the stated cadence onto the cron day-of-week field",
    ))
    rx.refresh()
    return rx, embedder


def build_router(reflexion, embedder):
    return SmartRouter(
        small=MockClient("small/default", handler=lambda messages: json.dumps(
            {"saw_rule": any(RULE_SENTINEL in m["content"] for m in messages)}
        )),
        frontier=MockClient("frontier/default", handler=lambda messages: "{}"),
        ladder=VerifierLadder(), embedder=embedder, cache=ExactCache(),
        reflexion=reflexion,
    )


def test_the_trace_records_which_rule_and_which_snapshot_steered_the_turn(reflexion):
    """Source and version live on the trace, so a reviewer can reproduce the decision
    without the rule ever being written into conversation history."""
    rx, embedder = reflexion
    router = build_router(rx, embedder)

    governed = router.route("set up a recurring digest with a weekday cadence")
    assert governed.decision.injected_rule_ids == ["RULE-001"]
    assert governed.decision.snapshot_version == rx.snapshot_version_for("default")
    assert json.loads(governed.final_output)["saw_rule"] is True

    # A request the rule does not match: no injection, and the trace says so.
    unrelated = router.route("what does the quarterly revenue chart show")
    assert unrelated.decision.injected_rule_ids == []
    assert json.loads(unrelated.final_output)["saw_rule"] is False


def test_retrieval_is_per_request_not_sticky_for_the_session(reflexion):
    """The same session, three turns, alternating relevance. Stickiness would show up as
    the third turn inheriting the first turn's rule."""
    rx, embedder = reflexion
    router = build_router(rx, embedder)
    history = ConversationHistory()
    seen = []

    for query in ("set up a recurring digest with a weekday cadence",
                  "what does the quarterly revenue chart show",
                  "who approved the quarterly revenue chart"):
        trace = router.route(query, session_id="s1", history=history)
        seen.append(bool(trace.decision.injected_rule_ids))
        history.append("user", query)
        history.append("assistant", trace.final_output or "")

    assert seen == [True, False, False]
