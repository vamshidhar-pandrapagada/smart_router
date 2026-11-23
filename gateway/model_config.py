"""Per-model settings: pricing, timeout, capability flags."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    name: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    #: Providers commonly discount cached input heavily. Getting this wrong makes every
    #: cost number in the harness wrong, which is why SS2.2 ordering is a Phase 0 gate.
    cached_input_discount: float = 0.10
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    supports_constrained_decoding: bool = False
    supports_logprobs: bool = False

    def cost(self, prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int) -> float:
        uncached = max(0, prompt_tokens - cached_prompt_tokens)
        rate = self.input_usd_per_mtok / 1_000_000
        cost = uncached * rate
        cost += cached_prompt_tokens * rate * self.cached_input_discount
        cost += completion_tokens * self.output_usd_per_mtok / 1_000_000
        return cost


#: Illustrative. Verify against current provider pricing before trusting the harness.
REGISTRY: dict[str, ModelSettings] = {
    "small/default": ModelSettings(
        name="small/default",
        input_usd_per_mtok=0.20,
        output_usd_per_mtok=0.60,
        timeout_seconds=15.0,
        supports_constrained_decoding=False,
        supports_logprobs=False,
    ),
    "frontier/default": ModelSettings(
        name="frontier/default",
        input_usd_per_mtok=15.0,
        output_usd_per_mtok=75.0,
        timeout_seconds=120.0,
    ),
    "judge/default": ModelSettings(
        name="judge/default",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        timeout_seconds=60.0,
    ),
}


def register(
    name: str,
    *,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    cached_input_discount: float = 0.10,
    timeout_seconds: float = 60.0,
    supports_constrained_decoding: bool = False,
    supports_logprobs: bool = False,
) -> ModelSettings:
    """Register a real provider model.

    Pricing is a caller input on purpose: published rates change, differ by region and
    tier, and a stale constant hard-coded here would silently corrupt every cost number
    the harness produces. Set it from the provider's current price list.
    """
    settings = ModelSettings(
        name=name,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        cached_input_discount=cached_input_discount,
        timeout_seconds=timeout_seconds,
        supports_constrained_decoding=supports_constrained_decoding,
        supports_logprobs=supports_logprobs,
    )
    REGISTRY[name] = settings
    return settings


def settings_for(model: str) -> ModelSettings:
    if model not in REGISTRY:
        raise KeyError(f"unknown model {model!r}; register it in gateway.model_config.REGISTRY")
    return REGISTRY[model]
