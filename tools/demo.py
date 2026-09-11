"""Tool execution demo: the Jira MCP path end to end. No API key, no live server.

    python -m smart_router.tools.demo

Four scenarios, each taking a different branch:
  1. clean read            -> verified, executed, committed
  2. bad payload           -> caught before the gateway; Jira never contacted
  3. Jira 404              -> escalated with the gateway's own error attached
  4. Jira 403              -> surfaced to a human; no frontier call is paid for
"""

from __future__ import annotations

import json

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.mock import MockClient, always
from smart_router.routing.router import SmartRouter
from smart_router.tools.commit import CommitGate
from smart_router.tools.errors import ToolSemanticError, ToolTransientError
from smart_router.tools.gateway import MockMCPGateway, http_status_handler
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.verify.free import JSONSchemaVerifier, ToolCallVerifier
from smart_router.verify.ladder import VerifierLadder

TOOLS = [
    ToolSpec("jira.search", {"jql": "string", "max_results": "integer"},
             required=("jql",), description="Search Jira issues by JQL", server="jira"),
    ToolSpec("webex.post", {"room": "string", "text": "string"},
             required=("room", "text"), irreversible=True, supports_dry_run=True,
             description="Post a message to a Webex room", server="webex"),
]

TICKETS = [{"key": "ATL-14", "status": "Open"}, {"key": "ATL-22", "status": "Blocked"}]


def scenario(name, small_output, handlers, expect):
    g = MockMCPGateway(specs=TOOLS, handlers=handlers)
    reg = ToolRegistry.from_gateway(g)
    router = SmartRouter(
        small=MockClient("small/default", handler=always(small_output)),
        frontier=MockClient(
            "frontier/default",
            handler=always(json.dumps(
                {"tool": "jira.search", "arguments": {"jql": "assignee=currentUser()"}})),
        ),
        ladder=VerifierLadder(free=[JSONSchemaVerifier(), ToolCallVerifier(reg)]),
        embedder=HashingEmbedder(dim=64), cache=ExactCache(),
        commit_gate=CommitGate(g, reg),
    )
    trace = router.route("get my jira tickets")

    print(f"\n{name}")
    print(f"  SLM proposed     {small_output[:58]}")
    if g.calls:
        for tool, args, dry in g.calls:
            mode = "dry-run" if dry else "commit "
            print(f"  gateway {mode}  {tool}({args})")
    else:
        print("  gateway calls    none — Jira was never contacted")
    print(f"  escalated        {trace.escalated}")
    print(f"  surfaced         {trace.surfaced_to_human}")
    print(f"  tool committed   {trace.tool_committed}")
    if trace.tool_failure:
        print(f"  tool failure     {trace.tool_failure}")
    print(f"  frontier calls   {len(router.frontier.calls)}")
    print(f"  -> {expect}")


def main() -> None:
    print("=" * 78)
    print("MCP gateway — all servers reach the router through one endpoint")
    print("=" * 78)
    for spec in TOOLS:
        flags = " irreversible" if spec.irreversible else ""
        print(f"  {spec.server:8} {spec.name:16} required={list(spec.required)}{flags}")

    scenario(
        "1. Clean read",
        json.dumps({"tool": "jira.search", "arguments": {"jql": "assignee=currentUser()"}}),
        {"jira.search": lambda a, d: TICKETS},
        "executed once, committed",
    )
    scenario(
        "2. Bad payload — max_results as a string",
        json.dumps({"tool": "jira.search",
                    "arguments": {"jql": "assignee=currentUser()", "max_results": "ten"}}),
        {"jira.search": lambda a, d: TICKETS},
        "bad payload never reached Jira; the frontier's corrected call committed",
    )
    scenario(
        "3. Jira 404 — project does not exist",
        json.dumps({"tool": "jira.search", "arguments": {"jql": "project=PROJ"}}),
        # Args-aware, so the frontier's corrected query actually succeeds -- otherwise
        # the demo cannot show the recovery, only the failure.
        {"jira.search": lambda a, d: (_ for _ in ()).throw(
            ToolSemanticError("Project 'PROJ' not found", status=404))
            if "PROJ" in a.get("jql", "") else TICKETS},
        "escalated with Jira's error; the frontier's corrected query then committed",
    )
    scenario(
        "4. Jira 403 — token lacks scope",
        json.dumps({"tool": "jira.search", "arguments": {"jql": "assignee=currentUser()"}}),
        {"jira.search": http_status_handler(403, "insufficient scope")},
        "surfaced to a human; no frontier call paid for",
    )

    # Transient faults retry the tool rather than escalating.
    state = {"n": 0}

    def flaky(args, dry):
        state["n"] += 1
        if state["n"] == 1:
            raise ToolTransientError("jira 503", status=503)
        return TICKETS

    scenario(
        "5. Jira 503 then recovery",
        json.dumps({"tool": "jira.search", "arguments": {"jql": "assignee=currentUser()"}}),
        {"jira.search": flaky},
        "tool retried once; frontier never involved",
    )


if __name__ == "__main__":
    main()
