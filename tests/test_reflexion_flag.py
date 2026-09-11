"""Reflexion behind one flag (memory/reflexion.py) and its router wiring."""

import json
from pathlib import Path

import pytest

from smart_router.cache.exact import ExactCache
from smart_router.embed import HashingEmbedder
from smart_router.gateway.base import ModelResponse
from smart_router.gateway.mock import MockClient, always
from smart_router.memory.exemplars import Exemplar
from smart_router.memory.reflexion import (
    ENV_DIR, ENV_ENABLED, ENV_MODE,
    ExemplarStore, Reflexion, ReflexionConfig, ReflexionMode, make_replay,
)
from smart_router.routing.router import SmartRouter
from smart_router.schemas.rule import GLOBAL_SCOPE
from smart_router.schemas.verification import FailureCategory, VerifierClass, VerifierResult
from smart_router.verify.free import JSONSchemaVerifier
from smart_router.verify.ladder import VerifierLadder

QUERIES = [f"find open blockers in project atlas{i} assigned to me" for i in range(10)]


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dim=128):
        super().__init__(dim)
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return super().encode(texts)


class NeedsRuleVerifier:
    """A semantic failure a rule can fix: passes only when the model answers GOOD."""

    verifier_id = "test.needs_rule"
    verifier_class = VerifierClass.FREE

    def check(self, ctx):
        ok = ctx.response.text.strip() == "GOOD"
        return VerifierResult(
            verifier_id=self.verifier_id, verifier_class=VerifierClass.FREE, passed=ok,
            category=None if ok else FailureCategory.SILENT_SEMANTIC,
            detail="" if ok else "output ignored the jql-as-string invariant",
        )


def rule_sensitive(messages):
    """GOOD only when a learned rule is present in the prompt."""
    return "GOOD" if any("<<injected-rules>>" in m["content"] for m in messages) else "BAD"


def critic(_messages):
    return ModelResponse(model="critic/mock", text=json.dumps({
        "trigger_condition": "constructing a jql argument for the jira search tool",
        "causal_invariant": "the connector types jql as a string; a nested object is rejected",
        "deduction_guideline": "serialize conditions into jql string syntax",
    }))


def build(tmp_path, *, mode="rules", enabled=True, small=rule_sensitive, embedder=None, **cfg):
    emb = embedder or HashingEmbedder(dim=128)
    config = ReflexionConfig(enabled=enabled, mode=mode, state_dir=tmp_path / "state", **cfg)
    reflexion = Reflexion(config, emb)
    router = SmartRouter(
        small=MockClient("small/default", handler=small),
        frontier=MockClient("frontier/default", handler=always("FRONTIER ANSWER")),
        ladder=VerifierLadder(free=[NeedsRuleVerifier()]),
        embedder=emb, cache=ExactCache(), reflexion=reflexion,
    )
    return router, reflexion


# -- the flag --------------------------------------------------------------

def test_disabled_reflexion_is_inert(tmp_path):
    emb = CountingEmbedder()
    router, rx = build(tmp_path, enabled=False, small=always("GOOD"), embedder=emb)
    trace = router.route("find open blockers")
    assert not rx.enabled and router.reflexion is None
    assert emb.calls == 0                        # no embedding forced
    assert trace.decision.injected_rule_ids == []
    assert not (tmp_path / "state").exists()     # nothing written


def test_config_reads_the_environment_and_flags_win():
    cfg = ReflexionConfig.from_env({ENV_ENABLED: "1", ENV_MODE: "exemplars", ENV_DIR: "/tmp/rx"})
    assert cfg.enabled and cfg.mode is ReflexionMode.EXEMPLARS
    assert cfg.state_dir == Path("/tmp/rx")
    assert not ReflexionConfig.from_env({}).enabled
    assert not ReflexionConfig.from_env({ENV_ENABLED: "0"}).enabled
    assert not ReflexionConfig.from_env({ENV_ENABLED: "1"}, enabled=False).enabled


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        ReflexionConfig(mode="banana")


def test_learning_requires_the_flag(tmp_path):
    router, rx = build(tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="disabled"):
        rx.learn(router, live_model="small/default")


# -- capture on the request path -------------------------------------------

def test_escalations_are_logged_and_corrections_captured_in_both_modes(tmp_path):
    router, rx = build(tmp_path, small=always("BAD"))
    trace = router.route(QUERIES[0])
    assert trace.escalated
    assert len(rx.failure_log) == 1
    assert len(rx.exemplars) == 1                # captured even in rules mode


def test_verified_successes_become_regression_cases(tmp_path):
    router, rx = build(tmp_path, small=always("GOOD"))
    router.route("list my open tickets")
    assert [c.query for c in rx.golden.cases()] == ["list my open tickets"]


def test_state_survives_a_restart(tmp_path):
    router, _ = build(tmp_path, small=always("BAD"))
    router.route(QUERIES[0])
    _, rx2 = build(tmp_path, small=always("BAD"))
    assert len(rx2.failure_log) == 1 and len(rx2.exemplars) == 1


def test_failures_from_a_different_embedder_are_dropped_not_mixed(tmp_path):
    router, _ = build(tmp_path, small=always("BAD"), embedder=HashingEmbedder(dim=64))
    router.route(QUERIES[0])
    _, rx2 = build(tmp_path, small=always("BAD"), embedder=HashingEmbedder(dim=128))
    assert len(rx2.failure_log) == 0
    assert rx2.failure_log.dropped_on_load == 1


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path):
    router, _ = build(tmp_path, small=always("BAD"))
    router.route(QUERIES[0])
    with (tmp_path / "state" / "failures.jsonl").open("a") as f:
        f.write('{"request_id": "trunc')          # crash mid-append
    _, rx2 = build(tmp_path, small=always("BAD"))
    assert len(rx2.failure_log) == 1


# -- rules mode, end to end ------------------------------------------------

def test_rules_mode_learns_promotes_and_injects(tmp_path):
    router, rx = build(tmp_path, min_similarity=-1.0)
    for q in QUERIES:
        assert router.route(q).escalated
    assert len(rx.failure_log) == 10

    summary = rx.learn(router, critic_complete=critic, live_model="small/default")
    assert len(summary.loop.promoted) == 1, summary.render()
    assert summary.consumed_failures == 10
    assert len(rx.failure_log) == 0              # consumed: will not be re-synthesized

    trace = router.route("find open blockers in project zeta assigned to me")
    assert not trace.escalated                   # the learned rule fixed it
    assert trace.decision.injected_rule_ids[0].startswith("RULE-")
    assert trace.decision.snapshot_version.startswith("rules:default:")
    assert rx.rule_store.all()[0].application_count == 11   # 10 replay trials + 1 live


def test_clusters_too_small_to_learn_from_keep_accumulating(tmp_path):
    router, rx = build(tmp_path)
    for q in QUERIES[:3]:
        router.route(q)
    summary = rx.learn(router, critic_complete=critic, live_model="small/default")
    assert summary.loop.candidates_synthesized == 0
    assert len(rx.failure_log) == 3              # not consumed; waits for more evidence


def test_rules_mode_needs_a_critic(tmp_path):
    router, rx = build(tmp_path)
    with pytest.raises(ValueError, match="critic"):
        rx.learn(router, live_model="small/default")


# -- exemplars mode --------------------------------------------------------

def test_exemplar_mode_few_shots_a_past_correction(tmp_path):
    router, rx = build(tmp_path, mode="exemplars", small=always("BAD"), min_similarity=-1.0)
    q = "summarise the open blockers in project atlas for me"
    assert router.route(q).escalated
    rx.learn(router, live_model="small/default")   # exemplars: refresh, no critic
    second = router.route(q)                       # escalations are never cached
    assert second.decision.injected_rule_ids[0].startswith("EX-")
    assert any("FRONTIER ANSWER" in m["content"] for m in router.small.calls[-1])


def test_one_tenants_memory_never_reaches_another_tenants_prompt(tmp_path):
    router, rx = build(tmp_path, mode="exemplars", small=always("BAD"), min_similarity=-1.0)
    q = "summarise the open blockers in project atlas for me"
    router.route(q, tenant_id="acme")
    rx.refresh()

    other = router.route(q, tenant_id="globex")
    assert other.decision.injected_rule_ids == []
    assert not any("FRONTIER ANSWER" in m["content"] for m in router.small.calls[-1])

    owner = router.route(q, tenant_id="acme")
    assert owner.decision.injected_rule_ids       # the owning tenant does get it


def test_exemplars_are_never_global(tmp_path):
    _, rx = build(tmp_path, mode="exemplars")
    rx.record_exemplar("q", "o", GLOBAL_SCOPE)    # request path: skipped, never raises
    assert len(rx.exemplars) == 0
    with pytest.raises(ValueError, match="GLOBAL"):
        ExemplarStore().add("q", "o", tenant_id=GLOBAL_SCOPE)


def test_exemplars_are_scrubbed_on_entry(tmp_path):
    store = ExemplarStore(tmp_path / "ex.jsonl")
    store.add("rotate key sk-abcd1234efgh please", "done", tenant_id="acme")
    assert "sk-" not in store.all()[0].query


def test_exemplar_ids_are_stable_and_accept_request_ids():
    """rule_id used to apply :06d to a string (crash) and salted hash() (unstable)."""
    assert Exemplar(query="q", verified_output="o",
                    source_request_id="3f2a9c1e-aaaa").rule_id == "EX-3f2a9c1e"
    a = Exemplar(query="same query", verified_output="o")
    assert a.rule_id == Exemplar(query="same query", verified_output="x").rule_id


# -- replay ----------------------------------------------------------------

def test_replay_verifies_but_never_executes_tools():
    from smart_router.tools.commit import CommitGate
    from smart_router.tools.gateway import MockMCPGateway
    from smart_router.tools.registry import ToolRegistry, ToolSpec

    spec = ToolSpec("webex_post_message", {"room": "string", "text": "string"},
                    required=("room", "text"), irreversible=True)
    gw = MockMCPGateway(specs=[spec])
    payload = json.dumps({"tool": "webex_post_message", "arguments": {"room": "r", "text": "t"}})
    router = SmartRouter(
        small=MockClient("small/default", handler=always(payload)),
        frontier=MockClient("frontier/default", handler=always(payload)),
        ladder=VerifierLadder(free=[JSONSchemaVerifier()]),
        embedder=HashingEmbedder(dim=64),
        commit_gate=CommitGate(gw, ToolRegistry.from_gateway(gw)),
    )
    assert make_replay(router, with_judge=False)("post the summary", []) is True
    assert gw.calls == []                          # verified, never executed


def test_replay_forces_the_judge_so_judge_caught_failures_can_be_resolved():
    from smart_router.verify.expensive import JudgeSampler, LLMJudge

    judged = []

    def judge(_messages):
        judged.append(1)
        return ModelResponse(model="j", text=json.dumps({"verdict": "FAIL", "findings": ["x"]}))

    router = SmartRouter(
        small=MockClient("small/default", handler=always("ok")),
        frontier=MockClient("frontier/default", handler=always("ok")),
        ladder=VerifierLadder(judge=LLMJudge(judge), sampler=JudgeSampler(random_rate=0.0)),
        embedder=HashingEmbedder(dim=64),
    )
    # Live routing would sample this read-only query at 0%; replay must not.
    assert make_replay(router, with_judge=True)("what is on my calendar", []) is False
    assert judged == [1]
    assert make_replay(router, with_judge=False)("what is on my calendar", []) is True


# -- status and CLI --------------------------------------------------------

def test_status_reports_what_is_stored(tmp_path):
    router, rx = build(tmp_path, small=always("BAD"))
    router.route(QUERIES[0])
    s = rx.status()
    assert s["enabled"] and s["mode"] == "rules"
    assert s["failures"] == 1 and s["exemplars"] == 1 and s["write_errors"] == 0


def test_cli_status_on_missing_state(tmp_path, capsys):
    from smart_router.cli import main

    assert main(["reflexion-status", "--reflexion-dir", str(tmp_path / "none")]) == 0
    assert "no reflexion state" in capsys.readouterr().out
