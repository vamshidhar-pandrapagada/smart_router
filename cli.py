"""Command line entry point.

    python -m smart_router models            [--filter gemma]
    python -m smart_router probe             --small <id> --frontier <id>
    python -m smart_router route             --small <id> --frontier <id> --query "..." [--reflexion]
    python -m smart_router learn             --small <id> --frontier <id> [--critic <id>]
    python -m smart_router reflexion-status  [--reflexion-dir DIR]

Reflexion is OFF unless enabled with --reflexion or SMART_ROUTER_REFLEXION=1. Its state
lives in --reflexion-dir (default .smart_router/reflexion, or SMART_ROUTER_REFLEXION_DIR),
so successive `route` runs accumulate failures and corrections and `learn` turns them into
rules offline. `route` never learns and `learn` never serves -- that separation is the
point.

Run `probe` first. Until you know whether your endpoints expose logprobs and constrained
decoding, no measurement from `route` means much.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REFLEXION_DIR = Path(".smart_router/reflexion")


def _warn_if_no_key() -> None:
    if not any(k.endswith("_API_KEY") or k.endswith("_CREDENTIALS") for k in os.environ):
        print("warning: no *_API_KEY in the environment\n", file=sys.stderr)


def _add_model_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--small", required=True, help="litellm model id for the small model")
    sp.add_argument("--frontier", required=True, help="litellm model id for the frontier model")
    sp.add_argument("--api-key", default=None, help="overrides the provider env var")
    sp.add_argument("--api-base", default=None)


def _pricing_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--small-in", type=float, default=0.20, help="small $/M input tokens")
    sp.add_argument("--small-out", type=float, default=0.60, help="small $/M output tokens")
    sp.add_argument("--frontier-in", type=float, default=15.0)
    sp.add_argument("--frontier-out", type=float, default=75.0)


def _ladder_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--expect-json", action="store_true",
                    help="enable the JSON schema verifier")
    sp.add_argument("--small-logprobs", action="store_true",
                    help="set if the probe reported logprobs: yes")
    sp.add_argument("--small-constrained", action="store_true",
                    help="set if the probe reported constrained decoding: yes")
    sp.add_argument("--judge", default=None,
                    help="litellm model id for the LLM judge -- mid-tier, not frontier")
    sp.add_argument("--judge-in", type=float, default=3.0)
    sp.add_argument("--judge-out", type=float, default=15.0)
    sp.add_argument("--judge-rate", type=float, default=0.02,
                    help="random judge sample rate (irreversible routes always judge)")
    sp.add_argument("--tenant", default="default")


def _embedder_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--embedder", choices=["hashing", "st"], default="hashing",
        help="'hashing' is lexical and dependency-free; 'st' is a local sentence-transformer, "
             "which is what reflexion needs to cluster and retrieve by meaning",
    )


def _reflexion_args(sp: argparse.ArgumentParser, *, with_toggle: bool = True) -> None:
    if with_toggle:
        sp.add_argument("--reflexion", dest="reflexion", action="store_true",
                        help="enable reflexion (or SMART_ROUTER_REFLEXION=1)")
        sp.add_argument("--no-reflexion", dest="reflexion", action="store_false",
                        help="disable reflexion, overriding the environment")
        sp.set_defaults(reflexion=None)
    sp.add_argument("--reflexion-mode", choices=["rules", "exemplars"], default=None)
    sp.add_argument("--reflexion-dir", type=Path, default=None)


def _reflexion_config(args, *, enabled: bool | None = None):
    from smart_router.memory.reflexion import ReflexionConfig

    overrides: dict = {}
    toggle = getattr(args, "reflexion", None)
    if toggle is not None:
        overrides["enabled"] = toggle
    if enabled is not None:
        overrides["enabled"] = enabled
    if getattr(args, "reflexion_mode", None):
        overrides["mode"] = args.reflexion_mode
    config = ReflexionConfig.from_env(**overrides)
    if getattr(args, "reflexion_dir", None) is not None:
        config.state_dir = Path(args.reflexion_dir).expanduser()
    elif config.state_dir is None:
        config.state_dir = DEFAULT_REFLEXION_DIR
    return config


def _embedder(args):
    from smart_router.embed import HashingEmbedder, SentenceTransformerEmbedder

    if getattr(args, "embedder", "hashing") == "st":
        return SentenceTransformerEmbedder()
    return HashingEmbedder(dim=256)


def _build(args, *, reflexion_enabled: bool | None = None):
    """One construction path for `route` and `learn`, so replay verifies with exactly the
    ladder that live routing uses."""
    from smart_router.cache.exact import ExactCache
    from smart_router.gateway.client import LiteLLMClient
    from smart_router.gateway.model_config import register
    from smart_router.memory.reflexion import Reflexion
    from smart_router.routing.policy import PolicyGate
    from smart_router.routing.router import SmartRouter
    from smart_router.verify.cheap import LogprobVerifier
    from smart_router.verify.expensive import JudgeSampler, LLMJudge
    from smart_router.verify.free import JSONSchemaVerifier, RefusalVerifier, TruncationVerifier
    from smart_router.verify.ladder import VerifierLadder

    register(args.small, input_usd_per_mtok=args.small_in, output_usd_per_mtok=args.small_out,
             supports_logprobs=args.small_logprobs,
             supports_constrained_decoding=args.small_constrained)
    register(args.frontier, input_usd_per_mtok=args.frontier_in,
             output_usd_per_mtok=args.frontier_out)

    judge = None
    if args.judge:
        register(args.judge, input_usd_per_mtok=args.judge_in, output_usd_per_mtok=args.judge_out)
        judge_client = LiteLLMClient(args.judge, api_key=args.api_key, api_base=args.api_base)
        judge = LLMJudge(lambda messages: judge_client.complete(messages))

    embedder = _embedder(args)
    reflexion = Reflexion(_reflexion_config(args, enabled=reflexion_enabled), embedder)
    router = SmartRouter(
        small=LiteLLMClient(args.small, api_key=args.api_key, api_base=args.api_base),
        frontier=LiteLLMClient(args.frontier, api_key=args.api_key, api_base=args.api_base),
        ladder=VerifierLadder(
            free=[TruncationVerifier(), RefusalVerifier()]
            + ([JSONSchemaVerifier()] if args.expect_json else []),
            cheap=[LogprobVerifier()],
            judge=judge,
            sampler=JudgeSampler(random_rate=args.judge_rate),
        ),
        embedder=embedder,
        policy=PolicyGate(),
        cache=ExactCache(),
        reflexion=reflexion,
    )
    return router, reflexion


def _print_reflexion(reflexion) -> None:
    status = reflexion.status()
    if not status["enabled"]:
        print("\nreflexion          off  (enable with --reflexion or SMART_ROUTER_REFLEXION=1)")
        return
    print(f"\nreflexion          on   mode={status['mode']}  dir={status['state_dir']}")
    dropped = status["failures_dropped_on_load"]
    print(f"  failures logged  {status['failures']}"
          + (f"  ({dropped} dropped on load: embedder mismatch or corrupt line)" if dropped else ""))
    print(f"  rules            {status['rules'] or 'none'}")
    print(f"  exemplars        {status['exemplars']}")
    print(f"  golden cases     {status['golden']}")
    for tenant, version in sorted(status["snapshots"].items()):
        print(f"  snapshot         {tenant}: {version}")
    if status["write_errors"]:
        print(f"  write errors     {status['write_errors']} (state kept in memory only)")
    if status["embedder"].startswith("hashing"):
        print("  note             the hashing embedder is lexical: clustering and retrieval "
              "will miss paraphrases. Use --embedder st for real runs.")


def cmd_models(args) -> int:
    try:
        import litellm
    except ImportError:
        print("litellm not installed: uv pip install 'litellm>=1.92.0'", file=sys.stderr)
        return 1
    needle = (args.filter or "").lower()
    names = sorted(n for n in litellm.model_list if needle in n.lower())
    if not names:
        print(f"no known model ids match {args.filter!r}")
        print("litellm's list lags new releases; an id missing here may still work.")
        return 0
    for n in names:
        print(f"  {n}")
    print(f"\n{len(names)} match(es). This list is litellm's and lags provider releases.")
    return 0


def cmd_probe(args) -> int:
    from smart_router.probe import probe

    _warn_if_no_key()
    print("Capability probe — plan §A.4. Two real calls per capability.\n")
    ok = True
    for model in (args.small, args.frontier):
        result = probe(model, api_key=args.api_key, api_base=args.api_base)
        print(result.render())
        print()
        ok = ok and result.reachable
    if not ok:
        print("At least one endpoint is unreachable. Check the model id and the API key env var.")
        return 1
    print("If constrained decoding is NO on the small model, Phase 2 cannot eliminate")
    print("schema errors at source and the ladder must detect them instead.")
    print("If cached-token reporting is NO, cost numbers cannot be verified — fix that first.")
    return 0


def cmd_route(args) -> int:
    from smart_router.telemetry import format_report, report

    _warn_if_no_key()
    router, reflexion = _build(args)
    queries = args.query or [
        "List three bullet points summarising what a smart router does.",
        "What is 2 + 2?",
    ]
    traces = []
    for q in queries:
        trace = router.route(q, tenant_id=args.tenant)
        traces.append(trace)
        print(f"\nQ: {q[:70]}")
        print(f"  tier       {trace.decision.tier.value} ({trace.decision.reason.value})")
        print(f"  escalated  {trace.escalated}")
        if trace.decision.injected_rule_ids:
            print(f"  injected   {', '.join(trace.decision.injected_rule_ids)}"
                  f"  (snapshot {trace.decision.snapshot_version})")
        if trace.verification:
            for r in trace.verification.results:
                print(f"    [{'PASS' if r.passed else 'FAIL'}] {r.verifier_id}"
                      f"{' — ' + r.detail if r.detail else ''}")
        print(f"  cost       ${trace.total_cost_usd:.5f}")
        print(f"  latency    {trace.total_latency_ms:.0f} ms")
        print(f"  cached     {trace.cache_hit_ratio:.0%} of prompt tokens")
        print(f"  output     {(trace.final_output or '')[:100]}")

    print("\n" + format_report(report(traces)))
    _print_reflexion(reflexion)
    print("\nThese are live numbers but not a measurement: no ground truth, no control arm.")
    print("Phase 1 needs both before any of this can be called a result.")
    return 0


def cmd_learn(args) -> int:
    from smart_router.gateway.client import LiteLLMClient
    from smart_router.gateway.model_config import REGISTRY, register

    _warn_if_no_key()
    router, reflexion = _build(args, reflexion_enabled=True)
    critic_complete = None
    if reflexion.uses_rules:
        critic_model = args.critic or args.frontier
        if critic_model not in REGISTRY:
            register(critic_model, input_usd_per_mtok=args.frontier_in,
                     output_usd_per_mtok=args.frontier_out)
        critic = LiteLLMClient(critic_model, api_key=args.api_key, api_base=args.api_base)
        critic_complete = lambda messages: critic.complete(messages)  # noqa: E731

    judged = " and judge calls" if router.ladder.judge is not None else ""
    print(f"Learning offline — mode={reflexion.mode.value}. Replay issues real small-model "
          f"calls{judged}; this costs money.\n")
    summary = reflexion.learn(
        router, critic_complete=critic_complete,
        live_model=args.small, tenant_id=args.learn_tenant,
    )
    print(summary.render())
    _print_reflexion(reflexion)
    return 0


def cmd_reflexion_status(args) -> int:
    from smart_router.memory.reflexion import Reflexion

    config = _reflexion_config(args, enabled=True)
    if not config.state_dir.exists():
        print(f"no reflexion state at {config.state_dir}")
        return 0
    _print_reflexion(Reflexion(config, _embedder(args)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smart_router")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_models = sub.add_parser("models", help="list model ids litellm recognises")
    p_models.add_argument("--filter", default="gemini", help="substring filter")
    p_models.set_defaults(func=cmd_models)

    p_probe = sub.add_parser("probe", help="check what the endpoints actually expose")
    _add_model_args(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    p_route = sub.add_parser("route", help="run the cascade against live models")
    _add_model_args(p_route)
    _pricing_args(p_route)
    _ladder_args(p_route)
    _embedder_args(p_route)
    _reflexion_args(p_route)
    p_route.add_argument("--query", action="append", help="repeatable")
    p_route.set_defaults(func=cmd_route)

    p_learn = sub.add_parser("learn", help="run the offline reflexion loop over logged failures")
    _add_model_args(p_learn)
    _pricing_args(p_learn)
    _ladder_args(p_learn)
    _embedder_args(p_learn)
    _reflexion_args(p_learn, with_toggle=False)
    p_learn.add_argument("--critic", default=None,
                         help="model that distils failures into rules (default: --frontier)")
    p_learn.add_argument("--learn-tenant", default=None,
                         help="restrict learning to one tenant (default: all)")
    p_learn.set_defaults(func=cmd_learn)

    p_status = sub.add_parser("reflexion-status", help="show what reflexion has stored")
    _embedder_args(p_status)
    _reflexion_args(p_status, with_toggle=False)
    p_status.set_defaults(func=cmd_reflexion_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
