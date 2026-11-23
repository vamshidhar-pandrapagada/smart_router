"""End-to-end proof that block ordering decides the cache hit rate."""

from smart_router.gateway.mock import MockClient, always
from smart_router.prompt import ConversationHistory, assemble


def _run_session(build_messages, turns=5):
    """Serve `turns` turns and report the share of prompt tokens served from cache."""
    client = MockClient("small/default", handler=always("ok"))
    hist = ConversationHistory()
    prompt_tokens = cached_tokens = 0
    for i in range(turns):
        messages = build_messages(hist, f"question {i}", i)
        resp = client.complete(messages)
        prompt_tokens += resp.prompt_tokens
        cached_tokens += resp.cached_prompt_tokens
        hist.append("user", f"question {i}")
        hist.append("assistant", f"answer {i}")
    return cached_tokens / prompt_tokens


class R:
    def __init__(self, i):
        self.rule_id = f"RULE-{i:03d}"
        self.trigger_condition = f"t{i}"
        self.causal_invariant = f"i{i}"
        self.deduction_guideline = f"d{i}"


def test_correct_ordering_caches_most_of_the_prompt():
    def correct(hist, turn, i):
        # Rules differ per query -- as top-k retrieval genuinely does -- but sit in the tail.
        return assemble(
            system="s" * 400, history=hist, rules=[R(i)], current_turn=turn
        ).messages

    assert _run_session(correct) > 0.5


def test_volatile_block_first_destroys_the_cache():
    def wrong(hist, turn, i):
        # The same content, with the per-query rule block moved into the system prefix.
        rules = {"role": "system", "content": f"RULES:\nRULE-{i:03d}"}
        tail = assemble(system="s" * 400, history=hist, current_turn=turn).messages
        return [rules, *tail]

    assert _run_session(wrong) == 0.0


def test_correct_ordering_beats_wrong_ordering_on_the_same_content():
    def correct(hist, turn, i):
        return assemble(system="s" * 400, history=hist, rules=[R(i)], current_turn=turn).messages

    def wrong(hist, turn, i):
        rules = {"role": "system", "content": f"RULES:\nRULE-{i:03d}"}
        return [rules, *assemble(system="s" * 400, history=hist, current_turn=turn).messages]

    assert _run_session(correct) > _run_session(wrong)


def test_cascading_fragments_the_prefix_cache_across_tiers():
    """Each tier warms its own prefix cache, so a cascade gets less reuse than either
    model alone would.

    An escalated request pays *uncached* frontier input, because the frontier model has
    not seen this session's prefix -- whereas under frontier-only routing that same
    prompt would have been mostly cached. Escalation is therefore more expensive than
    the naive "SLM cost + frontier cost" model implies.
    """
    small = MockClient("small/default", handler=always("ok"))
    frontier = MockClient("frontier/default", handler=always("ok"))
    solo = MockClient("frontier/default", handler=always("ok"))

    hist = ConversationHistory()
    cascade_prompt = cascade_cached = solo_prompt = solo_cached = 0

    for i in range(6):
        messages = assemble(system="s" * 400, history=hist, current_turn=f"q{i}").messages
        # Cascade: small every turn, frontier on the two that "escalate".
        r = small.complete(messages)
        cascade_prompt += r.prompt_tokens
        cascade_cached += r.cached_prompt_tokens
        if i in (2, 5):
            r = frontier.complete(messages)
            cascade_prompt += r.prompt_tokens
            cascade_cached += r.cached_prompt_tokens
        # Control: the same traffic, one model.
        r = solo.complete(messages)
        solo_prompt += r.prompt_tokens
        solo_cached += r.cached_prompt_tokens
        hist.append("user", f"q{i}")
        hist.append("assistant", "ok")

    assert (cascade_cached / cascade_prompt) < (solo_cached / solo_prompt)
