"""Runnable end-to-end demo. No API key required.

Reproduces the trace the plan was designed around: a scheduling request where every
free and cheap verifier passes and only the judge catches the error -- then reports
measured cost and latency against a frontier-only control arm.

    python -m smart_router.demo
"""

from __future__ import annotations

import json

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.base import ModelResponse
from smart_router.gateway.mock import MockClient, always
from smart_router.prompt import ConversationHistory
from smart_router.routing.policy import PolicyGate
from smart_router.routing.router import SmartRouter
from smart_router.schemas.routing import RouteTier
from smart_router.telemetry import format_report, report
from smart_router.verify.cheap import LogprobVerifier, SelfConsistencyVerifier
from smart_router.verify.expensive import JudgeSampler, LLMJudge
from smart_router.verify.free import (
    JSONSchemaVerifier, RefusalVerifier, ToolCallVerifier, ToolRegistry, ToolSpec,
    TruncationVerifier,
)
from smart_router.verify.ladder import VerifierLadder

TOOLS = ToolRegistry([
    ToolSpec(
        name="scheduler.create_job",
        arg_types={"name": "string", "schedule": "string", "action": "string",
                   "send_if_empty": "boolean"},
        required=("name", "schedule", "action"),
    ),
    ToolSpec(name="outlook.search", arg_types={"query": "string"}, required=("query",)),
])

# Schema-perfect, confidently produced, and wrong in two places: "0 8 * * *" fires seven
# days a week where the user said weekdays, and send_if_empty contradicts "skip it if
# there is nothing new". No free verifier can see either.
BAD_JOB = json.dumps({
    "tool": "scheduler.create_job",
    "arguments": {"name": "Leadership DL digest", "schedule": "0 8 * * *",
                  "action": "outlook.digest", "send_if_empty": True},
}, indent=2)

GOOD_SEARCH = json.dumps({"tool": "outlook.search", "arguments": {"query": "is:unread"}})

# A realistic system prompt. Length matters here: it is the bulk of the cacheable prefix,
# and with a toy prompt the cache ratio is meaningless.
SYSTEM = (
    "You are an enterprise assistant with access to Outlook, Jira and Webex via MCP "
    "connectors. Resolve relative dates against the user's timezone. Never widen the "
    "scope of a request. When creating standing configuration, mirror the user's stated "
    "cadence exactly. Prefer tool calls over prose when an action is requested. "
) * 24

# What a correct answer to the scheduling turn looks like, for the control arm.
FRONTIER_ANSWER = (
    "Created 'Leadership DL digest': weekdays at 08:00, skipped when there is nothing new."
)


def judge_handler(messages):
    """Stands in for a real judge. Flags the two mismatches a human would."""
    proposed = messages[-1]["content"]
    findings = []
    if '"0 8 * * *"' in proposed and "weekday" in proposed.lower():
        findings.append('schedule "0 8 * * *" fires 7 days/week; request said weekday mornings')
    if '"send_if_empty": true' in proposed and "nothing new" in proposed.lower():
        findings.append('send_if_empty=true contradicts "skip it if there is nothing new"')
    verdict = "FAIL" if findings else "PASS"
    return json.dumps({"verdict": verdict, "findings": findings})


def build() -> SmartRouter:
    judge_client = MockClient("judge/default", handler=judge_handler, latency_ms=2000.0)
    return SmartRouter(
        small=MockClient("small/default", handler=always(BAD_JOB), latency_ms=800.0,
                         logprobs=[-0.21] * 40),
        frontier=MockClient("frontier/default", handler=always(FRONTIER_ANSWER),
                            latency_ms=4000.0),
        ladder=VerifierLadder(
            free=[TruncationVerifier(), RefusalVerifier(), JSONSchemaVerifier(),
                  ToolCallVerifier(TOOLS)],
            cheap=[LogprobVerifier(),
                   SelfConsistencyVerifier(sampler=lambda c: [c.response.text] * 3)],
            judge=LLMJudge(lambda m: judge_client.complete(m)),
            # Random sampling measures the system; risk routing protects it.
            sampler=JudgeSampler(random_rate=0.02),
        ),
        embedder=HashingEmbedder(dim=384),
        policy=PolicyGate(),
        cache=ExactCache(),
        system_prompt=SYSTEM,
    )


def main() -> None:
    router = build()
    history = ConversationHistory()
    traces = []

    print("=" * 78)
    print("Session: prep-the-day agent")
    print("=" * 78)

    turns = [
        "Summarize my unread Outlook mail",
        "Set me up a daily digest of unread mail from the leadership DL, weekday "
        "mornings at 8, and skip it if there's nothing new.",
        "Summarize my unread Outlook mail",  # exact-cache hit
    ]

    for i, turn in enumerate(turns, 1):
        # Turn 1 and 3 are read-only; the scheduling turn is the interesting one.
        router.small._handler = always(GOOD_SEARCH if "Summarize" in turn else BAD_JOB)
        trace = router.route(turn, tenant_id="acme", session_id="s1", history=history)
        traces.append(trace)

        print(f"\nTurn {i}: {turn[:64]}{'...' if len(turn) > 64 else ''}")
        print(f"  tier          {trace.decision.tier.value}  ({trace.decision.reason.value})")
        print(f"  risk class    {trace.decision.risk_class.value}")
        if trace.verification:
            for r in trace.verification.results:
                mark = "PASS" if r.passed else "FAIL"
                detail = f" -- {r.detail}" if r.detail else ""
                print(f"    [{mark}] {r.verifier_id}{detail}")
        print(f"  escalated     {trace.escalated}")
        print(f"  cost          ${trace.total_cost_usd:.4f}")
        print(f"  latency       {trace.total_latency_ms:.0f} ms")

        if trace.decision.tier is not RouteTier.CACHE:
            history.append("user", turn)
            history.append("assistant", trace.final_output or "")

    # Control arm: the same traffic routed 100% to frontier. Measured, not estimated.
    control = SmartRouter(
        small=MockClient("frontier/default", handler=always(FRONTIER_ANSWER), latency_ms=4000.0),
        frontier=MockClient("frontier/default", handler=always(FRONTIER_ANSWER), latency_ms=4000.0),
        ladder=VerifierLadder(),
        embedder=HashingEmbedder(dim=384),
        policy=PolicyGate(frontier_only_tenants=frozenset({"acme"})),
        cache=ExactCache(),
        system_prompt=SYSTEM,
    )
    ctrl_hist = ConversationHistory()
    ctrl_traces = []
    for turn in turns:
        t = control.route(turn, tenant_id="acme", history=ctrl_hist)
        ctrl_traces.append(t)
        ctrl_hist.append("user", turn)
        ctrl_hist.append("assistant", t.final_output or "")

    r = report(traces)
    ctrl = report(ctrl_traces)
    print("\n" + "=" * 78)
    print("Router")
    print("=" * 78)
    print(format_report(r, control_cost_per_1k_usd=ctrl.cost_per_1k_usd))
    print("\nControl arm (100% frontier)")
    print(format_report(ctrl))


if __name__ == "__main__":
    main()
