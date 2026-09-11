"""What the router does with a tool outcome.

The commit gate's own verdicts (`should_escalate`, `should_surface`) were tested; what the
router did with them was not. It escalated every uncommitted outcome except auth -- paying
a frontier model to hit a dead endpoint, or to re-propose a write nobody approved -- and
its frontier-direct path skipped the gate entirely.
"""

import json

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.mock import MockClient, always
from smart_router.routing.router import SmartRouter
from smart_router.schemas.routing import RouteReason
from smart_router.tools.commit import CommitGate
from smart_router.tools.gateway import MockMCPGateway, http_status_handler
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.verify.free import JSONSchemaVerifier
from smart_router.verify.ladder import VerifierLadder

SEARCH = ToolSpec("jira_search", {"jql": "string"}, required=("jql",), irreversible=False)
POST = ToolSpec("webex_post", {"room": "string", "text": "string"},
                required=("room", "text"), irreversible=True)


def call(tool, **args):
    return json.dumps({"tool": tool, "arguments": args})


def build(handlers, small_out, frontier_out, **gate_kw):
    gw = MockMCPGateway(specs=[SEARCH, POST], handlers=handlers)
    router = SmartRouter(
        small=MockClient("small/default", handler=always(small_out)),
        frontier=MockClient("frontier/default", handler=always(frontier_out)),
        ladder=VerifierLadder(free=[JSONSchemaVerifier()]),
        embedder=HashingEmbedder(dim=64), cache=ExactCache(),
        commit_gate=CommitGate(gw, ToolRegistry.from_gateway(gw), **gate_kw),
    )
    return router, gw


def test_endpoint_still_down_after_the_retry_surfaces_without_paying_the_frontier():
    router, gw = build({"jira_search": http_status_handler(503)},
                       call("jira_search", jql="x"), call("jira_search", jql="x"))
    trace = router.route("list my tickets")
    assert not trace.escalated and trace.surfaced_to_human
    assert router.frontier.calls == []
    assert len(gw.calls) == 2                     # one attempt plus the gate's single retry
    assert trace.final_output.startswith("TOOL_TRANSIENT")


def test_unapproved_irreversible_call_surfaces_as_approval_required():
    router, gw = build({}, call("webex_post", room="r", text="t"),
                       call("webex_post", room="r", text="t"),
                       require_approval_for_irreversible=True)
    trace = router.route("send the standup summary")
    assert not trace.escalated and router.frontier.calls == [] and gw.calls == []
    assert trace.final_output.startswith("APPROVAL_REQUIRED")


def test_policy_forced_tool_calls_go_through_the_commit_gate():
    router, gw = build({"jira_search": lambda a, d: [{"key": "ATL-1"}]},
                       "unused", call("jira_search", jql="project=ATL"))
    trace = router.route("list the CONFIDENTIAL tickets")
    assert trace.decision.reason is RouteReason.POLICY_FORCED
    assert trace.tool_committed and [c[0] for c in gw.calls] == ["jira_search"]


def test_a_malformed_policy_forced_call_is_surfaced_not_executed():
    router, gw = build({}, "unused", call("jira_search", jql=123))
    trace = router.route("list the CONFIDENTIAL tickets")
    assert gw.calls == []
    assert trace.surfaced_to_human and trace.final_output.startswith("TOOL_ARG_INVALID")


def test_policy_forced_prose_is_returned_unchanged():
    router, _ = build({}, "unused", "Here is the summary.")
    trace = router.route("summarise the CONFIDENTIAL memo")
    assert trace.final_output == "Here is the summary." and not trace.surfaced_to_human


def test_infrastructure_failures_do_not_count_against_injected_rules(monkeypatch):
    router, _ = build({"jira_search": http_status_handler(503)},
                      call("jira_search", jql="x"), call("jira_search", jql="x"))
    credited = []
    monkeypatch.setattr(router, "_credit", lambda retrieved, succeeded: credited.append(succeeded))
    router.route("list my tickets")
    assert credited == []


def test_semantic_tool_errors_still_escalate_once():
    router, _ = build({"jira_search": http_status_handler(404, "Project 'PROJ' not found")},
                      call("jira_search", jql="project=PROJ"),
                      call("jira_search", jql="assignee=currentUser()"))
    trace = router.route("list my tickets")
    assert trace.escalated and len(router.frontier.calls) == 1
    # The frontier's call fails the same way; there is no tier above it, so it surfaces.
    assert trace.surfaced_to_human
