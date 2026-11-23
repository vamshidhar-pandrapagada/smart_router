"""Tool execution: gateway, commit gate, and the error taxonomy (plan SSE)."""

import json
from uuid import uuid4

import pytest

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.mock import MockClient, always
from smart_router.routing.router import SmartRouter
from smart_router.schemas.routing import RiskClass
from smart_router.schemas.verification import (
    TERMINAL_CATEGORIES, TRANSIENT_CATEGORIES, FailureCategory,
)
from smart_router.tools.commit import CommitGate, ParseError, propose, validate
from smart_router.tools.errors import (
    ToolAuthError, ToolSemanticError, ToolTransientError, classify_status,
)
from smart_router.tools.gateway import MockMCPGateway, http_status_handler
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.verify.free import JSONSchemaVerifier, ToolCallVerifier
from smart_router.verify.ladder import VerifierLadder
from smart_router.verify.result import ArgumentEchoVerifier, EmptyResultVerifier

SEARCH = ToolSpec("jira.search", {"jql": "string", "max_results": "integer"},
                  required=("jql",), server="jira")
POST = ToolSpec("webex.post", {"room": "string", "text": "string"},
                required=("room", "text"), irreversible=True, supports_dry_run=True,
                server="webex")


def gw(**handlers):
    return MockMCPGateway(specs=[SEARCH, POST], handlers=handlers)


def call(tool, **args):
    return json.dumps({"tool": tool, "arguments": args})


# -- registry is the single source of truth --------------------------------

def test_registry_derives_from_the_gateway_inventory():
    reg = ToolRegistry.from_gateway(gw())
    assert "jira.search" in reg and "webex.post" in reg


def test_prompt_block_and_verifier_share_one_registry():
    reg = ToolRegistry.from_gateway(gw())
    names_in_prompt = {t["name"] for t in reg.prompt_block()}
    assert names_in_prompt == set(reg.tools)
    # And the verifier reads the same registry object.
    r = ToolCallVerifier(reg).check(_ctx(call("jira.nope", jql="x")))
    assert r.category is FailureCategory.TOOL_NOT_FOUND


def _ctx(text):
    from smart_router.gateway.base import ModelResponse
    from smart_router.verify.base import VerificationContext
    return VerificationContext(query="q", response=ModelResponse(text=text, model="m"))


def test_envelope_schema_covers_every_tool():
    schema = ToolRegistry.from_gateway(gw()).envelope_schema()
    assert len(schema["oneOf"]) == 2


# -- error taxonomy --------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (503, ToolTransientError), (429, ToolTransientError), (504, ToolTransientError),
    (404, ToolSemanticError), (400, ToolSemanticError), (422, ToolSemanticError),
    (401, ToolAuthError), (403, ToolAuthError),
])
def test_status_classification(status, expected):
    assert classify_status(status) is expected


def test_tool_transient_is_transient_but_tool_auth_is_terminal():
    assert FailureCategory.TOOL_TRANSIENT in TRANSIENT_CATEGORIES
    assert FailureCategory.TOOL_AUTH in TERMINAL_CATEGORIES
    assert FailureCategory.TOOL_AUTH not in TRANSIENT_CATEGORIES


# -- propose / validate: nothing executes ----------------------------------

def test_propose_never_touches_the_gateway():
    g = gw()
    reg = ToolRegistry.from_gateway(g)
    propose(call("jira.search", jql="assignee=me"), reg)
    assert g.calls == []


def test_validate_rejects_bad_payloads_before_any_network_call():
    reg = ToolRegistry.from_gateway(gw())
    for payload, category in [
        (call("jira.invented"), FailureCategory.TOOL_NOT_FOUND),
        (call("jira.search"), FailureCategory.TOOL_ARG_INVALID),          # missing jql
        (call("jira.search", jql="x", max_results="ten"), FailureCategory.TOOL_ARG_INVALID),
        (call("jira.search", jql="x", bogus=1), FailureCategory.TOOL_ARG_INVALID),
    ]:
        ok, cat, _ = validate(propose(payload, reg))
        assert not ok and cat is category


def test_propose_rejects_non_json():
    with pytest.raises(ParseError):
        propose("{broken", ToolRegistry.from_gateway(gw()))


# -- commit gate -----------------------------------------------------------

def test_transient_tool_fault_retries_the_tool_and_never_escalates():
    state = {"n": 0}

    def flaky(args, dry):
        state["n"] += 1
        if state["n"] == 1:
            raise ToolTransientError("jira 503", status=503)
        return [{"key": "ABC-1"}]

    g = gw(**{"jira.search": flaky})
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    out = gate.commit(propose(call("jira.search", jql="x"), gate.registry, uuid4()))
    assert out.committed and out.attempts == 2


def test_persistent_transient_fault_does_not_escalate():
    g = gw(**{"jira.search": http_status_handler(503)})
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    out = gate.commit(propose(call("jira.search", jql="x"), gate.registry, uuid4()))
    assert not out.committed
    assert out.category is FailureCategory.TOOL_TRANSIENT
    assert not out.should_escalate      # paying a frontier model will not fix a 503


def test_semantic_tool_error_escalates():
    g = gw(**{"jira.search": http_status_handler(404, "Project 'PROJ' not found")})
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    out = gate.commit(propose(call("jira.search", jql="project=PROJ"), gate.registry, uuid4()))
    assert out.should_escalate and "PROJ" in out.detail


def test_auth_error_is_terminal_neither_retried_nor_escalated():
    g = gw(**{"jira.search": http_status_handler(401)})
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    out = gate.commit(propose(call("jira.search", jql="x"), gate.registry, uuid4()))
    assert out.should_surface and not out.should_escalate
    assert out.attempts == 1            # not retried


def test_irreversible_call_is_dry_run_first():
    g = gw()
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    out = gate.commit(propose(call("webex.post", room="r", text="t"), gate.registry, uuid4()))
    assert out.committed
    assert [c[2] for c in g.calls] == [True, False]   # dry-run, then the real call


def test_read_only_call_is_not_dry_run():
    g = gw()
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    gate.commit(propose(call("jira.search", jql="x"), gate.registry, uuid4()))
    assert [c[2] for c in g.calls] == [False]


def test_idempotency_key_prevents_double_execution_on_retry():
    g = gw()
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    p = propose(call("webex.post", room="r", text="t"), gate.registry, uuid4())
    gate.commit(p)
    gate.commit(p)                                    # same proposal, replayed
    real = [c for c in g.calls if not c[2]]
    assert len(real) == 1


def test_different_arguments_get_a_different_idempotency_key():
    reg = ToolRegistry.from_gateway(gw())
    rid = uuid4()
    a = propose(call("webex.post", room="r", text="one"), reg, rid)
    b = propose(call("webex.post", room="r", text="two"), reg, rid)
    assert a.idempotency_key() != b.idempotency_key()


def test_irreversible_can_require_explicit_approval():
    g = gw()
    gate = CommitGate(g, ToolRegistry.from_gateway(g), require_approval_for_irreversible=True)
    p = propose(call("webex.post", room="r", text="t"), gate.registry, uuid4())
    assert not gate.commit(p).committed
    assert gate.commit(p, approved=True).committed


def test_proposal_risk_class_follows_irreversibility():
    reg = ToolRegistry.from_gateway(gw())
    assert propose(call("webex.post", room="r", text="t"), reg).risk_class is RiskClass.IRREVERSIBLE_WRITE
    assert propose(call("jira.search", jql="x"), reg).risk_class is RiskClass.READ_ONLY


# -- result verification ---------------------------------------------------

def test_empty_read_result_is_flagged_as_suspect():
    g = gw(**{"jira.search": lambda args, dry: []})
    gate = CommitGate(g, ToolRegistry.from_gateway(g), result_verifiers=[EmptyResultVerifier()])
    out = gate.commit(propose(call("jira.search", jql="project=WRONG"), gate.registry, uuid4()))
    assert not out.committed and out.category is FailureCategory.TOOL_RESULT_SUSPECT


def test_argument_echo_catches_an_ignored_filter():
    g = gw(**{"jira.search": lambda args, dry: [{"project": "OTHER"}]})
    v = ArgumentEchoVerifier(echo_fields={"jira.search": ("jql", "project")})
    gate = CommitGate(g, ToolRegistry.from_gateway(g), result_verifiers=[v])
    out = gate.commit(propose(call("jira.search", jql="PROJ"), gate.registry, uuid4()))
    assert out.category is FailureCategory.TOOL_RESULT_SUSPECT


# -- router integration ----------------------------------------------------

def build(small_output, **handlers):
    g = gw(**handlers)
    reg = ToolRegistry.from_gateway(g)
    return SmartRouter(
        small=MockClient("small/default", handler=always(small_output)),
        frontier=MockClient("frontier/default",
                            handler=always(call("jira.search", jql="assignee=me"))),
        ladder=VerifierLadder(free=[JSONSchemaVerifier(), ToolCallVerifier(reg)]),
        embedder=HashingEmbedder(dim=64), cache=ExactCache(),
        commit_gate=CommitGate(g, reg),
    ), g


def test_router_executes_a_verified_call():
    router, g = build(call("jira.search", jql="assignee=me"))
    trace = router.route("get my jira tickets")
    assert trace.tool_committed and trace.tool_name == "jira.search"
    assert len(g.calls) == 1


def test_router_never_executes_the_small_models_bad_call():
    """The SLM's invented tool never reaches the gateway; the frontier's correction does."""
    router, g = build(call("jira.invented", jql="x"))
    trace = router.route("get my jira tickets")
    assert trace.escalated
    assert "jira.invented" not in [c[0] for c in g.calls]
    # The corrected call from the frontier goes through the same gate and commits.
    assert [c[0] for c in g.calls] == ["jira.search"]
    assert trace.tool_committed


def test_router_escalates_a_semantic_tool_error_with_the_gateway_message():
    router, _ = build(
        call("jira.search", jql="project=PROJ"),
        **{"jira.search": http_status_handler(404, "Project 'PROJ' not found")},
    )
    trace = router.route("get my jira tickets")
    assert trace.escalated
    sent = router.frontier.calls[-1][-1]["content"]
    assert "Project 'PROJ' not found" in sent and "TOOL_SEMANTIC" in sent


def test_router_surfaces_an_auth_error_without_paying_the_frontier():
    router, _ = build(call("jira.search", jql="x"), **{"jira.search": http_status_handler(403)})
    trace = router.route("get my jira tickets")
    assert trace.surfaced_to_human and not trace.escalated
    assert router.frontier.calls == []


def test_router_derives_its_prompt_tool_block_from_the_registry():
    router, _ = build(call("jira.search", jql="x"))
    assert {t["name"] for t in router.tools} == {"jira.search", "webex.post"}


# -- nested schema validation (complex connectors) -------------------------

SP_SCHEMA = {
    "type": "object",
    "properties": {
        "site_id": {"type": "string", "pattern": r"^[\w.-]+,"},
        "filter": {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "operator": {"enum": ["eq", "ne", "gt", "lt", "contains"]},
                "value": {"type": ["string", "number", "boolean"]},
            },
            "required": ["field", "operator", "value"],
            "additionalProperties": False,
        },
        "select": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "top": {"type": "integer", "minimum": 1, "maximum": 5000},
    },
    "required": ["site_id", "filter"],
    "additionalProperties": False,
}
SHAREPOINT = ToolSpec("sharepoint.search_items", {}, required=("site_id", "filter"),
                      server="sharepoint", argument_schema=SP_SCHEMA)


def _sp(**args):
    reg = ToolRegistry({SHAREPOINT.name: SHAREPOINT})
    return validate(propose(json.dumps(
        {"tool": "sharepoint.search_items", "arguments": args}), reg))


VALID_FILTER = {"field": "Modified", "operator": "gt", "value": "2026-01-01"}
SITE = "contoso.sharepoint.com,abc"


def test_nested_object_is_validated_not_just_typechecked():
    """`{"filter": "object"} + isinstance(dict)` accepted a hallucinated interior."""
    ok, cat, detail = _sp(site_id=SITE, filter={"fieldd": "Modified", "operator": "gt", "value": 1})
    assert not ok and cat is FailureCategory.TOOL_ARG_INVALID
    assert "arguments.filter" in detail


@pytest.mark.parametrize("args,fragment", [
    ({"site_id": SITE, "filter": {**VALID_FILTER, "operator": "greaterthan"}}, "operator"),
    ({"site_id": SITE, "filter": {**VALID_FILTER, "value": {"$now": 1}}}, "filter.value"),
    ({"site_id": SITE, "filter": VALID_FILTER, "top": 99999}, "maximum"),
    ({"site_id": SITE, "filter": VALID_FILTER, "select": ["Title", 7]}, "select[1]"),
    ({"site_id": "contoso", "filter": VALID_FILTER}, "pattern"),
    ({"site_id": SITE, "filter": VALID_FILTER, "bogus": 1}, "unknown field"),
])
def test_nested_violations_are_caught_with_a_precise_path(args, fragment):
    ok, _, detail = _sp(**args)
    assert not ok and fragment in detail


def test_a_valid_complex_payload_passes():
    ok, _, _ = _sp(site_id=SITE, filter=VALID_FILTER, select=["Title"], top=50)
    assert ok


def test_booleans_are_not_accepted_where_a_number_is_declared():
    from smart_router.tools.schema import validate_schema
    ok, _ = validate_schema(True, {"type": "integer"})
    assert not ok


def test_constrained_decoding_schema_matches_the_validated_schema():
    """The model is handed the same contract the verifier enforces."""
    reg = ToolRegistry({SHAREPOINT.name: SHAREPOINT})
    branch = reg.envelope_schema()["oneOf"][0]
    assert branch["properties"]["arguments"] == SP_SCHEMA
