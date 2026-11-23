"""Scriptable in-process client.

Exists so the whole cascade runs, and is tested, with no provider key. It simulates
provider prefix caching faithfully enough to prove that the SS2.2 block order actually
works: it records every message-prefix it has served and, on the next call, counts the
longest previously-seen prefix as cached. Reorder the blocks and the hit rate collapses.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Sequence

from smart_router.gateway.base import ModelResponse, ProviderFault, ProviderTimeout
from smart_router.gateway.model_config import settings_for
from smart_router.prompt import approx_tokens

#: A handler returns the completion text, or raises ProviderTimeout / ProviderFault.
Handler = Callable[[Sequence[dict[str, str]]], str]


class MockClient:
    def __init__(
        self,
        model_name: str = "small/default",
        *,
        handler: Handler | None = None,
        latency_ms: float = 0.0,
        logprobs: list[float] | None = None,
        finish_reason: str = "stop",
        simulate_prefix_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.settings = settings_for(model_name)
        self.supports_constrained_decoding = self.settings.supports_constrained_decoding
        self.supports_logprobs = self.settings.supports_logprobs
        self._handler = handler or (lambda _messages: "{}")
        self._latency_ms = latency_ms
        self._logprobs = logprobs
        self._finish_reason = finish_reason
        self._simulate_prefix_cache = simulate_prefix_cache
        self._seen_prefixes: set[str] = set()
        self.calls: list[list[dict[str, str]]] = []

    @staticmethod
    def _key(messages: Sequence[dict[str, str]]) -> str:
        return json.dumps(list(messages), sort_keys=True, separators=(",", ":"))

    def _cached_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        """Longest previously-served message-prefix, in tokens."""
        if not self._simulate_prefix_cache:
            return 0
        cached = 0
        for i in range(len(messages), 0, -1):
            if self._key(messages[:i]) in self._seen_prefixes:
                cached = sum(approx_tokens(m["content"]) for m in messages[:i])
                break
        for i in range(1, len(messages) + 1):
            self._seen_prefixes.add(self._key(messages[:i]))
        return cached

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ModelResponse:
        started = time.perf_counter()
        self.calls.append([dict(m) for m in messages])
        cached = self._cached_tokens(messages)
        text = self._handler(messages)  # may raise ProviderTimeout / ProviderFault
        prompt_tokens = sum(approx_tokens(m["content"]) for m in messages)
        completion_tokens = approx_tokens(text)
        latency = self._latency_ms or (time.perf_counter() - started) * 1000.0
        return ModelResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            completion_tokens=completion_tokens,
            latency_ms=latency,
            finish_reason=self._finish_reason,
            logprobs=self._logprobs,
        )


def always(text: str) -> Handler:
    return lambda _messages: text


def raises(exc: Exception) -> Handler:
    def _handler(_messages: Sequence[dict[str, str]]) -> str:
        raise exc

    return _handler


def sequence(*texts: str) -> Handler:
    """Return each text in turn; repeat the last one thereafter."""
    box = {"i": 0}

    def _handler(_messages: Sequence[dict[str, str]]) -> str:
        i = min(box["i"], len(texts) - 1)
        box["i"] += 1
        return texts[i]

    return _handler


__all__ = ["MockClient", "always", "raises", "sequence", "ProviderFault", "ProviderTimeout"]
