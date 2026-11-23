"""Provider abstraction. Nothing above this package may import a provider SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class ProviderTimeout(Exception):
    """Request exceeded the configured timeout."""


class ProviderFault(Exception):
    """Provider returned an error. Transient; retried once, never escalated."""


@dataclass
class ModelResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    #: Per-token logprobs when the endpoint exposes them. Absent on many managed
    #: endpoints -- the cheap verifier tier degrades accordingly (plan SS A.4).
    logprobs: list[float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class ModelClient(Protocol):
    model_name: str
    supports_constrained_decoding: bool
    supports_logprobs: bool

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ModelResponse: ...
