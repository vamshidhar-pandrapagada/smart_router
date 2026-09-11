"""Proof that the router is not wired to any particular tool.

Registers a deliberately heterogeneous fleet -- git, Outlook, web search, and a custom
tool with a nested schema -- through one gateway, and exercises every failure path. No
router code knows any of these exist.
"""

import json
from uuid import uuid4

import pytest

from smart_router.tools.commit import CommitGate, propose, validate
from smart_router.tools.errors import ToolTransientError
from smart_router.tools.gateway import MockMCPGateway, http_status_handler
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.tools.reversibility import DEFAULT_POLICY, IrreversibilityPolicy
from smart_router.schemas.verification import FailureCategory

FLEET = [
    # Declared irreversible, supports dry-run.
    ToolSpec("git.commit", {"message": "string", "paths": "array"},
             required=("message",), irreversible=True, supports_dry_run=True, server="git"),
    # UNDECLARED — inference must catch this one.
    ToolSpec("git.push", {"remote": "string", "branch": "string"},
             required=("remote", "branch"), server="git"),
    ToolSpec("git.log", {"max_count": "integer"}, server="git"),
    # Nested array-of-objects schema.
    ToolSpec("outlook.send_mail", {}, required=("to", "subject"), server="outlook",
             argument_schema={
                 "type": "object",
                 "properties": {
                     "to": {"type": "array", "minItems": 1,
                            "items": {"type": "object",
                                      "properties": {"address": {"type": "string",
                                                                 "pattern": r"^[^@]+@[^@]+$"},
                                                     "name": {"type": "string"}},
                                      "required": ["address"],
                                      "additionalProperties": False}},
                     "subject": {"type": "string"},
                     "importance": {"enum": ["low", "normal", "high"]},
                 },
                 "required": ["to", "subject"], "additionalProperties": False}),
    ToolSpec("web.search", {"query": "string", "top_k": "integer"},
             required=("query",), server="websearch"),
    # A custom tool whose name matches no known verb.
    ToolSpec("custom.recalculate_risk_scores", {"portfolio_id": "string", "dry": "boolean"},
             required=("portfolio_id",), server="custom"),
]


def registry():
    return ToolRegistry.from_gateway(MockMCPGateway(specs=FLEET))


def call(tool, **args):
    return json.dumps({"tool": tool, "arguments": args})


def test_one_gateway_registers_every_server():
    reg = registry()
    assert {s.server for s in FLEET} == {"git", "outlook", "websearch", "custom"}
    assert len(reg) == len(FLEET)


def test_undeclared_mutating_tool_is_not_treated_as_safe():
    """The fail-open hole: git.push was silently read-only before inference."""
    assert registry().get("git.push").is_irreversible() is True
    assert registry().get("git.log").is_irreversible() is False


def test_unknown_verb_defaults_conservatively():
    assert registry().get("custom.recalculate_risk_scores").is_irreversible() is True


def test_read_only_verb_wins_over_a_contained_mutating_verb():
    """`get_deployment_status` contains 'deploy' but only observes."""
    assert DEFAULT_POLICY.infer("ops.get_deployment_status")[0] is False


def test_audit_lists_every_guessed_tool_for_an_operator_to_annotate():
    audited = {a["tool"] for a in registry().audit()}
    assert "git.push" in audited                 # undeclared
    assert "git.commit" not in audited           # declared explicitly


def test_an_override_beats_inference():
    pol = IrreversibilityPolicy(overrides={"custom.recalculate_risk_scores": False})
    assert pol.infer("custom.recalculate_risk_scores") == (False, "override")


@pytest.mark.parametrize("tool,args,fragment", [
    ("git.commit", {"paths": ["a.py"]}, "missing required field 'message'"),
    ("git.commit", {"message": "x", "paths": "a.py"}, "expected array"),
    ("outlook.send_mail", {"to": [], "subject": "s"}, "at least 1 items"),
    ("outlook.send_mail", {"to": [{"address": "not-an-email"}], "subject": "s"}, "pattern"),
    ("outlook.send_mail", {"to": [{"addr": "a@b.com"}], "subject": "s"}, "required field 'address'"),
    ("outlook.send_mail", {"to": [{"address": "a@b.com"}], "subject": "s",
                           "importance": "urgent"}, "must be one of"),
    ("web.search", {"query": "x", "top_k": "many"}, "expected integer"),
    ("custom.recalculate_risk_scores", {"portfolio_id": "p", "dry": "yes"}, "expected boolean"),
])
def test_malformed_payloads_are_caught_across_every_connector(tool, args, fragment):
    ok, cat, detail = validate(propose(call(tool, **args), registry()))
    assert not ok and cat is FailureCategory.TOOL_ARG_INVALID
    assert fragment in detail


def test_valid_payloads_pass_across_every_connector():
    for tool, args in [
        ("git.commit", {"message": "fix", "paths": ["a.py"]}),
        ("git.push", {"remote": "origin", "branch": "main"}),
        ("git.log", {"max_count": 10}),
        ("outlook.send_mail", {"to": [{"address": "a@b.com", "name": "A"}],
                               "subject": "hi", "importance": "high"}),
        ("web.search", {"query": "vllm prefix caching", "top_k": 5}),
        ("custom.recalculate_risk_scores", {"portfolio_id": "P-1", "dry": True}),
    ]:
        ok, _, detail = validate(propose(call(tool, **args), registry()))
        assert ok, f"{tool}: {detail}"


@pytest.mark.parametrize("status,category,escalates", [
    (503, FailureCategory.TOOL_TRANSIENT, False),
    (404, FailureCategory.TOOL_SEMANTIC, True),
    (403, FailureCategory.TOOL_AUTH, False),
])
def test_error_classes_behave_identically_for_every_connector(status, category, escalates):
    for tool, args in [("git.log", {}), ("web.search", {"query": "x"}),
                       ("outlook.send_mail", {"to": [{"address": "a@b.com"}], "subject": "s"})]:
        g = MockMCPGateway(specs=FLEET, handlers={tool: http_status_handler(status)})
        gate = CommitGate(g, ToolRegistry.from_gateway(g), max_tool_retries=0)
        out = gate.commit(propose(call(tool, **args), gate.registry, uuid4()))
        assert out.category is category, tool
        assert out.should_escalate is escalates, tool


def test_undeclared_irreversible_tool_now_requires_approval():
    """Before inference, git.push slipped past the approval gate entirely."""
    g = MockMCPGateway(specs=FLEET)
    gate = CommitGate(g, ToolRegistry.from_gateway(g), require_approval_for_irreversible=True)
    p = propose(call("git.push", remote="origin", branch="main"), gate.registry, uuid4())
    assert not gate.commit(p).committed
    assert gate.commit(p, approved=True).committed


def test_read_only_tool_is_unaffected_by_the_approval_gate():
    g = MockMCPGateway(specs=FLEET)
    gate = CommitGate(g, ToolRegistry.from_gateway(g), require_approval_for_irreversible=True)
    out = gate.commit(propose(call("web.search", query="x"), gate.registry, uuid4()))
    assert out.committed


def test_transient_retry_works_for_a_custom_tool():
    state = {"n": 0}

    def flaky(args, dry):
        state["n"] += 1
        if state["n"] == 1:
            raise ToolTransientError("upstream 503", status=503)
        return {"score": 0.4}

    g = MockMCPGateway(specs=FLEET, handlers={"custom.recalculate_risk_scores": flaky})
    gate = CommitGate(g, ToolRegistry.from_gateway(g))
    out = gate.commit(propose(call("custom.recalculate_risk_scores", portfolio_id="P-1"),
                              gate.registry, uuid4()))
    assert out.committed and out.attempts == 2
