import json
from uuid import uuid4

import pytest

from smart_router.gateway.base import ModelResponse
from smart_router.schemas.routing import RiskClass
from smart_router.schemas.verification import FailureCategory, VerifierClass
from smart_router.verify.base import VerificationContext
from smart_router.verify.cheap import LogprobVerifier, SelfConsistencyVerifier
from smart_router.verify.expensive import JudgeSampler, LLMJudge
from smart_router.verify.free import (
    GroundingVerifier,
    JSONSchemaVerifier,
    RefusalVerifier,
    ToolCallVerifier,
    ToolRegistry,
    ToolSpec,
    TruncationVerifier,
)
from smart_router.verify.ladder import VerifierLadder, decomposition

REGISTRY = ToolRegistry([
    ToolSpec(
        name="scheduler.create_job",
        arg_types={"name": "string", "schedule": "string", "send_if_empty": "boolean"},
        required=("name", "schedule"),
    )
])


def ctx(text, **kw):
    return VerificationContext(query=kw.pop("query", "q"), response=ModelResponse(text=text, model="m"), **kw)


def test_invalid_json_is_flagged_preventable():
    r = JSONSchemaVerifier().check(ctx("{not json"))
    assert not r.passed
    assert r.category is FailureCategory.SCHEMA_VIOLATION
    # Constrained decoding would have removed this at source, so it belongs to P not R.
    assert r.preventable_by_construction


def test_unknown_tool_is_flagged():
    payload = json.dumps({"tool": "scheduler.nope", "arguments": {}})
    r = ToolCallVerifier(REGISTRY).check(ctx(payload))
    assert not r.passed and r.category is FailureCategory.TOOL_NOT_FOUND


def test_wrong_argument_type_is_flagged():
    payload = json.dumps({"tool": "scheduler.create_job",
                          "arguments": {"name": "d", "schedule": {"cron": "0 8 * * *"}}})
    r = ToolCallVerifier(REGISTRY).check(ctx(payload))
    assert not r.passed and r.category is FailureCategory.TOOL_ARG_INVALID


def test_wellformed_call_with_wrong_values_passes_every_free_verifier():
    """The residue. This is exactly what the judge exists for."""
    payload = json.dumps({
        "tool": "scheduler.create_job",
        "arguments": {"name": "digest", "schedule": "0 8 * * *", "send_if_empty": True},
    })
    c = ctx(payload)
    for v in (TruncationVerifier(), RefusalVerifier(), JSONSchemaVerifier(), ToolCallVerifier(REGISTRY)):
        assert v.check(c).passed


def test_truncation_and_refusal():
    assert not TruncationVerifier().check(
        VerificationContext(query="q", response=ModelResponse(text="x", model="m", finish_reason="length"))
    ).passed
    assert not RefusalVerifier().check(ctx("I cannot help with that")).passed


def test_grounding_catches_fabricated_quotes_but_not_paraphrase():
    src = ("The Q3 budget review is pending approval.",)
    assert not GroundingVerifier().check(
        ctx('The email says "the Q3 budget review was approved".', grounding=src)
    ).passed
    # Paraphrased fabrication passes -- a known and stated gap.
    assert GroundingVerifier().check(
        ctx("The Q3 budget review has been approved.", grounding=src)
    ).passed


def test_logprob_verifier_skips_when_unavailable():
    v = LogprobVerifier()
    assert not v.available(ctx("{}"))
    assert v.check(ctx("{}")).passed


def test_self_consistency_passes_a_confidently_wrong_model():
    """3/3 agreement measures stability, not correctness."""
    v = SelfConsistencyVerifier(sampler=lambda c: ["WRONG", "WRONG", "WRONG"])
    assert v.check(ctx("WRONG")).passed


def test_ladder_short_circuits_before_the_judge():
    judged = []
    judge = LLMJudge(lambda m: (judged.append(1), ModelResponse(text='{"verdict":"FAIL"}', model="j"))[1])
    ladder = VerifierLadder(
        free=[JSONSchemaVerifier()], judge=judge,
        sampler=JudgeSampler(random_rate=1.0),
    )
    out = ladder.run(uuid4(), ctx("{broken"))
    assert out.escalate and not judged


def test_transient_fault_retries_rather_than_escalating():
    class Boom:
        verifier_id, verifier_class = "t", VerifierClass.FREE

        def check(self, c):
            from smart_router.schemas.verification import VerifierResult
            return VerifierResult(verifier_id="t", verifier_class=VerifierClass.FREE,
                                  passed=False, category=FailureCategory.TIMEOUT)

    out = VerifierLadder(free=[Boom()]).run(uuid4(), ctx("{}"))
    assert out.retry_small and not out.escalate


@pytest.mark.parametrize("risk,expected", [
    (RiskClass.IRREVERSIBLE_WRITE, True),
    (RiskClass.READ_ONLY, False),
])
def test_judge_gate_always_fires_on_irreversible_routes(risk, expected):
    sampler = JudgeSampler(random_rate=0.0)
    should, _ = sampler.should_judge(risk)
    assert should is expected


def test_judge_gate_fires_above_a_cost_of_being_wrong_threshold():
    sampler = JudgeSampler(random_rate=0.0, cost_of_being_wrong_threshold_usd=10.0)
    assert sampler.should_judge(RiskClass.REVERSIBLE_WRITE, 25.0)[0]
    assert not sampler.should_judge(RiskClass.REVERSIBLE_WRITE, 1.0)[0]


def test_decomposition_splits_preventable_from_residue():
    ladder = VerifierLadder(free=[JSONSchemaVerifier(), ToolCallVerifier(REGISTRY)])
    outcomes = [
        ladder.run(uuid4(), ctx("{broken")),                                    # P
        ladder.run(uuid4(), ctx(json.dumps({"tool": "nope", "arguments": {}}))),  # P
        ladder.run(uuid4(), ctx(json.dumps({"tool": "scheduler.create_job",
                                            "arguments": {"name": "d", "schedule": "0 8 * * *"}}))),
    ]
    d = decomposition(outcomes)
    assert d == {"detected": 2, "preventable_by_construction": 2, "residue": 0, "judged": 0}
