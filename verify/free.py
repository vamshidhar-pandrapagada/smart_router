"""Free verifiers -- deterministic, ~5ms, $0 (plan SSB).

These catch the structural failure class. Most of what they catch is *preventable by
construction* with grammar-constrained decoding, which is strictly better than detecting
it after the fact; each result carries `preventable_by_construction` so the Phase 1
harness can split P from R rather than crediting the cascade for failures a decoding
constraint would have removed for free.

What they cannot catch: a well-formed call carrying wrong values. That is the residue,
and it is why the judge exists.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from smart_router.schemas.verification import FailureCategory, VerifierClass, VerifierResult
from smart_router.verify.base import VerificationContext

_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm unable", "i am unable",
    "as an ai", "i won't", "i will not",
)

# Single source of truth: the same registry the prompt's tool block and the commit
# gate derive from. Re-exported here so existing imports keep working.
from smart_router.tools.registry import JSON_TYPES as _TYPES  # noqa: E402
from smart_router.tools.registry import ToolRegistry, ToolSpec  # noqa: E402,F401


def _fail(vid: str, cat: FailureCategory, detail: str, latency_ms: float, preventable: bool) -> VerifierResult:
    return VerifierResult(
        verifier_id=vid,
        verifier_class=VerifierClass.FREE,
        passed=False,
        category=cat,
        detail=detail,
        latency_ms=latency_ms,
        preventable_by_construction=preventable,
    )


def _ok(vid: str, latency_ms: float) -> VerifierResult:
    return VerifierResult(
        verifier_id=vid, verifier_class=VerifierClass.FREE, passed=True, latency_ms=latency_ms
    )


class TruncationVerifier:
    verifier_id = "free.truncation"
    verifier_class = VerifierClass.FREE

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000
        if ctx.response.truncated:
            return _fail(self.verifier_id, FailureCategory.TRUNCATION, "finish_reason=length", ms, False)
        if not ctx.response.text.strip():
            return _fail(self.verifier_id, FailureCategory.TRUNCATION, "empty completion", ms, False)
        return _ok(self.verifier_id, ms)


class RefusalVerifier:
    verifier_id = "free.refusal"
    verifier_class = VerifierClass.FREE

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        lowered = ctx.response.text.lower()
        ms = (time.perf_counter() - t0) * 1000
        for marker in _REFUSAL_MARKERS:
            if marker in lowered:
                return _fail(self.verifier_id, FailureCategory.REFUSAL, f"matched {marker!r}", ms, False)
        return _ok(self.verifier_id, ms)


class JSONSchemaVerifier:
    """Parses the completion and validates it against the tool-call envelope."""

    verifier_id = "free.json_schema"
    verifier_class = VerifierClass.FREE

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        try:
            json.loads(ctx.response.text)
        except json.JSONDecodeError as exc:
            ms = (time.perf_counter() - t0) * 1000
            return _fail(
                self.verifier_id, FailureCategory.SCHEMA_VIOLATION, f"invalid JSON: {exc.msg}", ms,
                preventable=True,
            )
        return _ok(self.verifier_id, (time.perf_counter() - t0) * 1000)


class ToolCallVerifier:
    """Validates a proposed call against the connector's declared contract.

    Delegates to the same recursive validator the commit gate uses. It previously walked
    the flat `arg_types` map, which disagreed with the gate whenever a tool declared a
    full `argument_schema`: the ladder rejected valid calls as "unknown argument" while
    the gate accepted them. Two validators, one registry, opposite verdicts.

    Every failure here is preventable by construction: a decoder constrained to the same
    schema cannot emit an unknown tool, omit a required argument, or mistype one.
    """

    verifier_id = "free.tool_call"
    verifier_class = VerifierClass.FREE

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        ms = lambda: (time.perf_counter() - t0) * 1000  # noqa: E731

        try:
            payload: Any = json.loads(ctx.response.text)
        except json.JSONDecodeError:
            # JSONSchemaVerifier owns this failure; do not double-report it.
            return _ok(self.verifier_id, ms())

        if not isinstance(payload, dict) or "tool" not in payload:
            return _ok(self.verifier_id, ms())

        from smart_router.tools.commit import ToolProposal
        from smart_router.tools.commit import validate as validate_proposal

        name = payload.get("tool")
        arguments = payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _fail(
                self.verifier_id, FailureCategory.TOOL_ARG_INVALID,
                "arguments is not an object", ms(), preventable=True,
            )

        proposal = ToolProposal(
            tool=name if isinstance(name, str) else str(name),
            arguments=arguments,
            spec=self.registry.get(name) if isinstance(name, str) else None,
            raw=ctx.response.text,
        )
        ok, category, detail = validate_proposal(proposal)
        if ok:
            return _ok(self.verifier_id, ms())
        return _fail(self.verifier_id, category, detail, ms(), preventable=True)


class GroundingVerifier:
    """Every quoted span in the answer must appear in the supplied source material.

    Catches a narrow slice of UNGROUNDED_CLAIM -- fabricated verbatim quotes. Paraphrased
    fabrication passes straight through, which is exactly the gap the judge covers.
    """

    verifier_id = "free.grounding"
    verifier_class = VerifierClass.FREE

    def __init__(self, min_quote_len: int = 12) -> None:
        self.min_quote_len = min_quote_len

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        if not ctx.grounding:
            return _ok(self.verifier_id, (time.perf_counter() - t0) * 1000)
        haystack = "\n".join(ctx.grounding).lower()
        text = ctx.response.text
        quotes = [
            seg for seg in text.split('"')[1::2] if len(seg.strip()) >= self.min_quote_len
        ]
        for q in quotes:
            if q.strip().lower() not in haystack:
                return _fail(
                    self.verifier_id, FailureCategory.UNGROUNDED_CLAIM,
                    f"quoted span not present in source: {q[:60]!r}",
                    (time.perf_counter() - t0) * 1000, preventable=False,
                )
        return _ok(self.verifier_id, (time.perf_counter() - t0) * 1000)
