"""Well-formed calls that name the wrong entity (verify/entities.py).

Raised in review: schema validation catches malformed structure, not wrong intent. Every
payload below passes schema validation — right types, all required fields — and some of
them would still fire a destructive write against the wrong record.
"""

import json

import pytest

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.base import ModelResponse
from smart_router.gateway.mock import MockClient, always
from smart_router.prompt import ConversationHistory
from smart_router.routing.router import SmartRouter
from smart_router.schemas.verification import FailureCategory
from smart_router.tools.commit import CommitGate, propose, validate
from smart_router.tools.gateway import MockMCPGateway
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.verify.base import VerificationContext
from smart_router.verify.entities import EntityGroundingVerifier, extract_entities
from smart_router.verify.ladder import VerifierLadder

CANCEL = ToolSpec("orders_cancel", {"order_id": "string", "reason": "string"},
                  required=("order_id",), irreversible=True)
NOTE = ToolSpec("orders_note", {"order_id": "string", "body": "string"},
                required=("order_id", "body"), irreversible=False)
REG = ToolRegistry({CANCEL.name: CANCEL, NOTE.name: NOTE})


def ctx(text, query, grounding=()):
    return VerificationContext(query=query, response=ModelResponse(text=text, model="m"),
                               grounding=grounding)


def call(tool, **args):
    return json.dumps({"tool": tool, "arguments": args})


# -- the premise: schema validation cannot see this ------------------------

def test_a_transposed_order_id_passes_every_schema_check():
    payload = call("orders_cancel", order_id="4417")       # user said 4471
    ok, category, _ = validate(propose(payload, REG))
    assert ok and category is None


# -- what the verifier catches ---------------------------------------------

def test_an_identifier_from_nowhere_is_flagged():
    result = EntityGroundingVerifier().check(
        ctx(call("orders_cancel", order_id="4417"), "cancel order 4471")
    )
    assert not result.passed
    assert result.category is FailureCategory.UNGROUNDED_ENTITY
    assert "4417" in result.detail
    # No decoder constraint prevents a valid reference to the wrong entity.
    assert result.preventable_by_construction is False


def test_the_identifier_the_user_actually_said_passes():
    assert EntityGroundingVerifier().check(
        ctx(call("orders_cancel", order_id="4471"), "cancel order 4471")
    ).passed


def test_an_identifier_returned_by_an_earlier_tool_is_grounded():
    """"Cancel my latest order" — the id came from a lookup, not from the user."""
    assert EntityGroundingVerifier().check(
        ctx(call("orders_cancel", order_id="99812"), "cancel my latest order",
            grounding=('[{"order_id": "99812", "status": "open"}]',))
    ).passed


def test_nested_and_listed_identifiers_are_checked():
    payload = json.dumps({"tool": "x", "arguments": {
        "filter": {"account_id": "AC9912"}, "tickets": ["ATL-14", "ATL-99"]}})
    found = dict(extract_entities(json.loads(payload)["arguments"]))
    assert found["filter.account_id"] == "AC9912"
    assert found["tickets[1]"] == "ATL-99"


def test_free_text_fields_are_not_treated_as_entities():
    """A note body may legitimately contain numbers the conversation never mentioned."""
    assert EntityGroundingVerifier().check(
        ctx(call("orders_note", order_id="4471", body="refund 250 processed, ref 88231"),
            "add a note to order 4471")
    ).passed


@pytest.mark.parametrize("value,is_id", [
    ("4471", True), ("ATL-14", True), ("AC9912", True),
    ("deadbeefcafe", True),
    ("next Tuesday", False), ("open", False), ("12", False), ("", False),
])
def test_identifier_shapes(value, is_id):
    from smart_router.verify.entities import looks_like_identifier
    assert looks_like_identifier(value) is is_id


# -- the stated limit ------------------------------------------------------

def test_it_cannot_catch_the_wrong_one_of_two_mentioned_identifiers():
    """Both ids are grounded, so this passes. That case needs the judge or confirmation —
    the verifier narrows the problem, it does not close it."""
    assert EntityGroundingVerifier().check(
        ctx(call("orders_cancel", order_id="4471"),
            "order 4471 shipped but cancel 9902")
    ).passed


# -- through the router ----------------------------------------------------

def test_the_router_grounds_against_conversation_history_and_blocks_the_write():
    gw = MockMCPGateway(specs=[CANCEL], handlers={"orders_cancel": lambda a, d: {"ok": True}})
    reg = ToolRegistry.from_gateway(gw)
    router = SmartRouter(
        small=MockClient("small/default", handler=always(call("orders_cancel", order_id="4417"))),
        frontier=MockClient("frontier/default", handler=always(call("orders_cancel", order_id="4471"))),
        ladder=VerifierLadder(free=[EntityGroundingVerifier()]),
        embedder=HashingEmbedder(dim=64), cache=ExactCache(),
        commit_gate=CommitGate(gw, reg, dry_run_irreversible=False),
    )
    history = ConversationHistory()
    history.append("user", "show me order 4471")
    history.append("assistant", '{"order_id": "4471", "status": "open"}')

    trace = router.route("cancel it", history=history)
    assert trace.escalated
    # The small model's wrong id never reached the connector; the frontier's correct one did.
    assert [c[1]["order_id"] for c in gw.calls] == ["4471"]
