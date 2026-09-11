"""Capability probe -- answers the plan's SSA.4 question empirically (Phase 0 gate).

Before any measurement is trustworthy you have to know what the endpoint actually
exposes, because two verifier tiers depend on it:

* **logprobs** -- without them `LogprobVerifier` is dead and the cheap tier is just
  self-consistency, which cannot see confident error.
* **constrained/structured decoding** -- without it the schema-error class must be
  *detected* after the fact instead of *prevented* at generation, which changes both the
  cost model and how much residue the learning loop has to work on.
* **cached prompt tokens** -- if the provider does not report them you cannot verify that
  the SS2.2 prompt ordering is doing anything, and every cost number is a guess.

Providers advertise these inconsistently and change them without notice, so this asks the
endpoint rather than trusting documentation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

PROBE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass
class ProbeResult:
    model: str
    reachable: bool = False
    error: str | None = None
    supports_logprobs: bool = False
    supports_constrained_decoding: bool = False
    reports_cached_tokens: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.reachable:
            return f"  {self.model}\n    UNREACHABLE — {self.error}"
        def mark(b: bool) -> str:
            return "yes" if b else "NO "
        lines = [
            f"  {self.model}",
            f"    reachable              yes  ({self.latency_ms:.0f} ms)",
            f"    logprobs               {mark(self.supports_logprobs)}"
            f"  {'cheap verifier tier available' if self.supports_logprobs else 'LogprobVerifier is dead'}",
            f"    constrained decoding   {mark(self.supports_constrained_decoding)}"
            f"  {'schema errors preventable' if self.supports_constrained_decoding else 'schema errors only detectable'}",
            f"    cached-token reporting {mark(self.reports_cached_tokens)}"
            f"  {'cache math verifiable' if self.reports_cached_tokens else 'cost numbers unverifiable'}",
            f"    tokens                 {self.prompt_tokens} in / {self.completion_tokens} out"
            f" ({self.cached_prompt_tokens} cached)",
        ]
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def _usage(resp) -> tuple[int, int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0, 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        cached,
    )


def probe(model: str, *, api_key: str | None = None, api_base: str | None = None) -> ProbeResult:
    """One real call per capability. Failures are recorded, never raised."""
    try:
        import litellm
    except ImportError:
        return ProbeResult(model, error="litellm not installed — pip install 'smart-router[providers]'")

    import logging

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    litellm.suppress_debug_info = True

    result = ProbeResult(model)
    base: dict[str, Any] = {"model": model, "temperature": 0.0, "timeout": 60}
    if api_key:
        base["api_key"] = api_key
    if api_base:
        base["api_base"] = api_base

    # 1. Reachability, token accounting, latency.
    started = time.perf_counter()
    try:
        resp = litellm.completion(
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
            **base,
        )
    except Exception as exc:
        # Provider errors often carry an embedded traceback; keep the first line only.
        message = str(exc).split("Traceback")[0].strip().splitlines()
        result.error = f"{type(exc).__name__}: {(message[0] if message else str(exc))[:200]}"
        return result
    result.reachable = True
    result.latency_ms = (time.perf_counter() - started) * 1000
    p, c, cached = _usage(resp)
    result.prompt_tokens, result.completion_tokens, result.cached_prompt_tokens = p, c, cached
    if p == 0:
        result.notes.append("provider reported no token usage — cost accounting will be blind")

    # 2. Logprobs.
    try:
        lp_resp = litellm.completion(
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16, logprobs=True, **base,
        )
        choice = lp_resp.choices[0]
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        result.supports_logprobs = bool(content)
        if lp is not None and not content:
            result.notes.append("logprobs field present but empty")
    except Exception as exc:
        result.notes.append(f"logprobs rejected: {str(exc)[:110]}")

    # 3. Constrained / structured decoding.
    try:
        sd_resp = litellm.completion(
            messages=[{"role": "user", "content": "Answer: what is 2+2? Use the schema."}],
            max_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "probe", "schema": PROBE_SCHEMA, "strict": True},
            },
            **base,
        )
        text = sd_resp.choices[0].message.content or ""
        payload = json.loads(text)
        result.supports_constrained_decoding = isinstance(payload, dict) and "answer" in payload
        if not result.supports_constrained_decoding:
            result.notes.append("structured output accepted but did not honour the schema")
    except Exception as exc:
        result.notes.append(f"constrained decoding rejected: {str(exc)[:110]}")

    # 4. Cached-token reporting: send a long, repeated prefix twice.
    try:
        prefix = "You are a meticulous enterprise assistant. " * 220
        msgs = [{"role": "system", "content": prefix}, {"role": "user", "content": "Say ok."}]
        litellm.completion(messages=msgs, max_tokens=8, **base)
        second = litellm.completion(messages=msgs, max_tokens=8, **base)
        _, _, cached2 = _usage(second)
        result.reports_cached_tokens = cached2 > 0
        result.cached_prompt_tokens = max(result.cached_prompt_tokens, cached2)
        if not result.reports_cached_tokens:
            result.notes.append(
                "no cached tokens on an identical repeated prefix — either caching is off "
                "or the provider does not report it; SS2.2 ordering cannot be verified"
            )
    except Exception as exc:
        result.notes.append(f"cache probe failed: {str(exc)[:110]}")

    return result
