"""End-to-end: read a git repo through MCP, summarise it, draft a Webex message.

    export GEMINI_API_KEY=...
    python -m smart_router.tools.yantra_demo --repo /path/to/clone \
        --small gemini/gemma-4-31b-it --frontier gemini/gemini-3.1-pro-preview

Two honest limitations this demo does NOT hide:

1. `mcp-server-git` reads a LOCAL repository. It cannot fetch a GitHub URL, so the repo
   must be cloned first. The router is not doing the fetching.
2. The router routes ONE request at a time. It is not an agent loop -- it does not call a
   tool, read the result, decide a next step and call again. So this is three scripted
   turns through the router, not autonomous planning. Multi-step agentic planning is a
   layer above the router and is not built.

What it does demonstrate: a real MCP server behind the gateway, schema validation before
any call, live model routing with verification, and the commit gate refusing to send an
irreversible message without approval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from uuid import uuid4

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.client import LiteLLMClient
from smart_router.gateway.model_config import register
from smart_router.prompt import ConversationHistory
from smart_router.routing.policy import PolicyGate
from smart_router.routing.router import SmartRouter
from smart_router.telemetry import format_report, report
from smart_router.tools.commit import CommitGate, propose
from smart_router.tools.gateway import CompositeGateway, MockMCPGateway
from smart_router.tools.registry import ToolRegistry, ToolSpec
from smart_router.tools.stdio_gateway import StdioMCPGateway
from smart_router.verify.free import JSONSchemaVerifier, RefusalVerifier, TruncationVerifier
from smart_router.verify.ladder import VerifierLadder

#: Stands in for a Webex MCP server. Declared irreversible and drafted only -- the point
#: is to watch the commit gate refuse it, not to send anything.
WEBEX = ToolSpec(
    name="webex_post_message",
    arg_types={},
    required=("room", "markdown"),
    description="Post a markdown message to a Webex room",
    irreversible=True,
    server="webex",
    argument_schema={
        "type": "object",
        "properties": {
            "room": {"type": "string"},
            "markdown": {"type": "string"},
        },
        "required": ["room", "markdown"],
        "additionalProperties": False,
    },
)

SYSTEM = (
    "You are an engineering assistant with tools. When a tool is appropriate, reply with "
    "ONLY a JSON object: {\"tool\": <name>, \"arguments\": {...}}. When answering in "
    "prose, reply with prose and no JSON. Never invent tool names or arguments."
)


def build_router(repo: str, small: str, frontier: str, args) -> tuple[SmartRouter, CommitGate]:
    git_gw = StdioMCPGateway(
        command="uvx",
        args=["--from", "mcp-server-git", "mcp-server-git", "--repository", repo],
        fixed_arguments={"repo_path": repo},
    )
    git_gw.start()
    webex_gw = MockMCPGateway(specs=[WEBEX], handlers={
        "webex_post_message": lambda a, dry: {"sent": False, "note": "draft only"},
    })
    gateway = CompositeGateway([git_gw, webex_gw])
    registry = ToolRegistry.from_gateway(gateway)

    register(small, input_usd_per_mtok=args.small_in, output_usd_per_mtok=args.small_out)
    register(frontier, input_usd_per_mtok=args.frontier_in, output_usd_per_mtok=args.frontier_out)

    gate = CommitGate(
        gateway, registry,
        dry_run_irreversible=False,          # MCP has no dry-run
        require_approval_for_irreversible=True,   # nothing sends without a human
    )
    router = SmartRouter(
        small=LiteLLMClient(small), frontier=LiteLLMClient(frontier),
        ladder=VerifierLadder(free=[TruncationVerifier(), RefusalVerifier()]),
        embedder=HashingEmbedder(dim=256), policy=PolicyGate(),
        cache=ExactCache(), commit_gate=gate, system_prompt=SYSTEM,
    )
    return router, gate


def show(n: int, prompt: str, trace) -> None:
    print(f"\n{'=' * 78}\nTurn {n}: {prompt[:66]}\n{'=' * 78}")
    print(f"  tier            {trace.decision.tier.value} ({trace.decision.reason.value})")
    print(f"  risk class      {trace.decision.risk_class.value}")
    if trace.tool_name:
        print(f"  tool            {trace.tool_name}")
        print(f"  tool committed  {trace.tool_committed}")
    if trace.tool_failure:
        print(f"  tool outcome    {trace.tool_failure}")
    print(f"  escalated       {trace.escalated}")
    print(f"  cost            ${trace.total_cost_usd:.5f}   latency {trace.total_latency_ms:.0f} ms")
    body = (trace.final_output or "").strip()
    print(f"  output          {body[:400]}{'...' if len(body) > 400 else ''}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="path to a LOCAL clone")
    p.add_argument("--small", default="gemini/gemma-4-31b-it")
    p.add_argument("--frontier", default="gemini/gemini-3.1-pro-preview")
    p.add_argument("--small-in", type=float, default=0.10)
    p.add_argument("--small-out", type=float, default=0.30)
    p.add_argument("--frontier-in", type=float, default=2.00)
    p.add_argument("--frontier-out", type=float, default=12.00)
    p.add_argument("--room", default="engineering-standup")
    args = p.parse_args(argv)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) first.", file=sys.stderr)
        return 1
    if not os.path.isdir(os.path.join(args.repo, ".git")):
        print(f"{args.repo} is not a git repo. Clone the target first.", file=sys.stderr)
        return 1

    router, gate = build_router(args.repo, args.small, args.frontier, args)
    print(f"{len(gate.registry)} tools behind one gateway "
          f"(git via stdio MCP + webex mock)\n")
    for row in gate.registry.audit():
        flag = "IRREVERSIBLE" if row["inferred_irreversible"] else "read-only"
        print(f"  {flag:13} {row['tool']:22} {row['why']}")

    history = ConversationHistory()
    traces = []

    # Turn 1 — a read through the live git MCP server.
    q1 = "Show me the 10 most recent commits in this repository."
    t1 = router.route(q1, history=history)
    traces.append(t1)
    show(1, q1, t1)
    history.append("user", q1)
    history.append("assistant", (t1.final_output or "")[:2000])

    # Turn 2 — prose summary over what the tool returned. No tool call expected.
    q2 = ("Based on those commits, summarise in 3 short bullets what this project is and "
          "what changed recently. Reply in prose, not JSON.")
    t2 = router.route(q2, history=history)
    traces.append(t2)
    show(2, q2, t2)
    history.append("user", q2)
    history.append("assistant", (t2.final_output or "")[:2000])

    # Turn 3 — an irreversible tool. The gate must refuse it without approval.
    q3 = (f"Draft a Webex message to the room '{args.room}' containing that summary. "
          f"Use the webex_post_message tool.")
    t3 = router.route(q3, history=history)
    traces.append(t3)
    show(3, q3, t3)

    print(f"\n{'=' * 78}\nWhat the commit gate did with the Webex call\n{'=' * 78}")
    if t3.tool_name == WEBEX.name and not t3.tool_committed:
        print("  REFUSED — irreversible tool, no human approval. Nothing was sent.")
        print("  The draft is above; approving it would be a separate, explicit step.")
    elif t3.tool_name != WEBEX.name:
        print(f"  The model did not propose the Webex tool (proposed: {t3.tool_name}).")
        print("  That is a model capability result, not a router failure — worth logging.")

    print("\n" + format_report(report(traces)))
    print("\nLive numbers, not a measurement: no ground truth, no control arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
