"""The SS2.2 invariants. These are the tests that protect the token bill."""

import pytest

from smart_router.prompt import (
    ConversationHistory,
    HistoryContaminationError,
    assemble,
)


class FakeRule:
    def __init__(self, rid):
        self.rule_id = rid
        self.trigger_condition = f"trigger {rid}"
        self.causal_invariant = f"invariant {rid}"
        self.deduction_guideline = f"deduction {rid}"


def test_rules_land_in_the_volatile_tail_not_the_prefix():
    p = assemble(system="sys", rules=[FakeRule("RULE-001")], current_turn="hello")
    prefix_text = " ".join(m["content"] for m in p.prefix)
    assert "RULE-001" not in prefix_text
    assert any("RULE-001" in m["content"] for m in p.volatile)
    # And the rule block precedes the current turn.
    assert p.volatile[-1]["content"] == "hello"


def test_prefix_is_identical_across_queries_with_different_rules():
    """Different retrieved rules must not change the cacheable prefix."""
    hist = ConversationHistory()
    hist.append("user", "turn one")
    hist.append("assistant", "answer one")
    a = assemble(system="sys", history=hist, rules=[FakeRule("RULE-001")], current_turn="q1")
    b = assemble(system="sys", history=hist, rules=[FakeRule("RULE-999")], current_turn="q2")
    assert a.prefix_fingerprint() == b.prefix_fingerprint()


def test_prefix_extends_rather_than_changes_as_turns_accumulate():
    hist = ConversationHistory()
    hist.append("user", "turn one")
    t1 = assemble(system="sys", history=hist, current_turn="q")
    hist.append("assistant", "answer one")
    hist.append("user", "turn two")
    t2 = assemble(system="sys", history=hist, current_turn="q")
    # Turn 2's chain must begin with turn 1's -- an extension, not a rewrite.
    assert t2.prefix_chain()[: len(t1.prefix_chain())] == t1.prefix_chain()


def test_history_refuses_a_rule_block():
    hist = ConversationHistory()
    with pytest.raises(HistoryContaminationError):
        hist.append("assistant", "<<injected-rules>>\n- RULE-001")


def test_history_refuses_a_non_conversation_role():
    hist = ConversationHistory()
    with pytest.raises(HistoryContaminationError):
        hist.append("system", "you are helpful")


def test_injected_token_count_is_reported():
    p = assemble(system="sys", rules=[FakeRule("RULE-001")], current_turn="q")
    assert p.injected_token_count > 0
    assert p.injected_rule_ids == ("RULE-001",)
