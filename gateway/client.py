"""LiteLLM-backed client.

Adapted from pramana-ai's `pramana/gateway/client.py`, which is the real shared
dependency between the two projects (`litellm>=1.92.0`). Import of litellm is deferred
so the core package -- and its tests -- run without the `providers` extra installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from smart_router.gateway.base import ModelResponse, ProviderFault, ProviderTimeout
from smart_router.gateway.model_config import settings_for

logger = logging.getLogger(__name__)


class LiteLLMClient:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.settings = settings_for(model_name)
        self.model_name = model_name
        self.supports_constrained_decoding = self.settings.supports_constrained_decoding
        self.supports_logprobs = self.settings.supports_logprobs
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._litellm = None

    def _client(self):
        if self._litellm is None:
            try:
                import litellm
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "LiteLLMClient needs the 'providers' extra: "
                    "pip install 'smart-router[providers]'"
                ) from exc
            logging.getLogger("LiteLLM").setLevel(logging.ERROR)
            self._litellm = litellm
        return self._litellm

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ModelResponse:  # pragma: no cover - requires a live provider
        litellm = self._client()
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": self.settings.temperature if temperature is None else temperature,
            "timeout": timeout or self.settings.timeout_seconds,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if schema is not None and self.supports_constrained_decoding:
            # Constrained decoding removes the SCHEMA_VIOLATION class at source rather
            # than detecting it after the fact (plan SSB). Only sent when the endpoint
            # actually supports it -- the Phase 0 capability probe settles that.
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema}}
        if self.supports_logprobs:
            kwargs["logprobs"] = True

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                resp = litellm.completion(**kwargs)
            except Exception as exc:
                last_exc = exc
                if "timeout" in str(exc).lower():
                    raise ProviderTimeout(str(exc)) from exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_base_delay * (2**attempt))
                continue

            usage = getattr(resp, "usage", None)
            cached = 0
            if usage is not None:
                details = getattr(usage, "prompt_tokens_details", None)
                cached = int(getattr(details, "cached_tokens", 0) or 0)
            choice = resp.choices[0]
            return ModelResponse(
                text=choice.message.content or "",
                model=self.model_name,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                cached_prompt_tokens=cached,
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            )

        raise ProviderFault(str(last_exc))
