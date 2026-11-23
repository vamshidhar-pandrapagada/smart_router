"""Prompt assembly contract (plan SS2.2) -- cache-critical.

Provider prefix caching matches on the longest common prefix from the *start* of the
prompt, so any volatile block placed early invalidates everything after it. The block
order below is mandatory, not stylistic:

    1. system            stable across all users and turns
    2. tools             stable until the tool registry changes
    3. tenant context    stable within a session
    4. history           grows monotonically; cached from turn 2 onward
    ------------------------------------------- cache boundary
    5. rules             volatile per query
    6. current turn      volatile

Two failure modes this module exists to prevent:

* Rules placed in the system prefix. Top-k retrieval differs per query, so a rules block
  at position 2 re-pays full price on the whole conversation every turn -- roughly a 10x
  input-cost regression on a five-turn session, to save ~300 tokens.
* The injected rule block persisted into history. That turns the stable region volatile
  and silently destroys the cache; the symptom is a cache hit rate that *falls* as
  sessions lengthen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

Message = dict[str, str]

#: Roles the conversation log is allowed to contain. A rules block is not one of them.
_CLEAN_ROLES = frozenset({"user", "assistant", "tool"})

_RULE_SENTINEL = "<<injected-rules>>"


def approx_tokens(text: str) -> int:
    """Rough token count. Replace with the provider tokenizer before trusting cost math."""
    return max(1, len(text) // 4)


class HistoryContaminationError(ValueError):
    """Raised when an injected rule block is about to be persisted into history."""


@dataclass
class ConversationHistory:
    """Clean user/assistant/tool messages only.

    The rule block is re-inserted fresh at the tail on every turn and never stored here.
    """

    messages: list[Message] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        if role not in _CLEAN_ROLES:
            raise HistoryContaminationError(
                f"role {role!r} is not a clean conversation role; "
                "only user/assistant/tool may be persisted into history"
            )
        if _RULE_SENTINEL in content:
            raise HistoryContaminationError(
                "refusing to persist an injected rule block into conversation history "
                "(plan SS2.2): this makes the stable prefix volatile and destroys the cache"
            )
        self.messages.append({"role": role, "content": content})

    def __len__(self) -> int:
        return len(self.messages)


@dataclass(frozen=True)
class AssembledPrompt:
    """A prompt split at the cache boundary."""

    prefix: tuple[Message, ...]
    volatile: tuple[Message, ...]
    injected_rule_ids: tuple[str, ...]
    injected_token_count: int

    @property
    def messages(self) -> list[Message]:
        return [dict(m) for m in (*self.prefix, *self.volatile)]

    @property
    def prefix_tokens(self) -> int:
        return sum(approx_tokens(m["content"]) for m in self.prefix)

    @property
    def volatile_tokens(self) -> int:
        return sum(approx_tokens(m["content"]) for m in self.volatile)

    def prefix_fingerprint(self) -> str:
        """Identity of the cacheable region.

        Two requests sharing this fingerprint can share a provider prefix cache entry.
        The evaluation harness asserts that it is stable across queries within a session
        and *extends* (rather than changes) as turns accumulate.
        """
        payload = json.dumps(list(self.prefix), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def prefix_chain(self) -> list[str]:
        """Per-block fingerprints, in order.

        Turn N's chain must start with turn N-1's chain for the cache to survive. This is
        what makes 'the prefix extended' distinguishable from 'the prefix changed'.
        """
        out: list[str] = []
        running = hashlib.sha256()
        for m in self.prefix:
            running.update(json.dumps(m, sort_keys=True, separators=(",", ":")).encode())
            out.append(running.hexdigest()[:16])
        return out


def _render_rules(rules: Sequence[Any]) -> str:
    """Rules and exemplars both render themselves, so the two memory arms share this path
    -- which is what makes a matched-token-budget comparison possible at all."""
    lines = [_RULE_SENTINEL]
    for r in rules:
        render = getattr(r, "render", None)
        if callable(render):
            lines.append(render())
            continue
        lines.append(
            f"- {getattr(r, 'rule_id', 'RULE-???')}\n"
            f"  TRIGGER: {getattr(r, 'trigger_condition', '')}\n"
            f"  INVARIANT: {getattr(r, 'causal_invariant', '')}\n"
            f"  DEDUCTION: {getattr(r, 'deduction_guideline', '')}"
        )
    return "\n".join(lines)


def assemble(
    *,
    system: str,
    tools: Sequence[dict[str, Any]] | None = None,
    tenant_context: str | None = None,
    history: ConversationHistory | Iterable[Message] | None = None,
    rules: Sequence[Any] = (),
    current_turn: str,
) -> AssembledPrompt:
    """Build a prompt in cache-safe order.

    `rules` land *after* history and *before* the current turn -- never in the system
    prefix. They are returned as part of the volatile region so callers can see exactly
    what they are re-paying for on every request.
    """
    prefix: list[Message] = [{"role": "system", "content": system}]

    if tools:
        prefix.append(
            {
                "role": "system",
                "content": "TOOLS:\n" + json.dumps(list(tools), sort_keys=True, indent=2),
            }
        )
    if tenant_context:
        prefix.append({"role": "system", "content": f"CONTEXT:\n{tenant_context}"})

    if history is not None:
        hist_messages = history.messages if isinstance(history, ConversationHistory) else list(history)
        prefix.extend(dict(m) for m in hist_messages)

    volatile: list[Message] = []
    rule_ids: tuple[str, ...] = ()
    injected_tokens = 0
    if rules:
        rendered = _render_rules(rules)
        volatile.append({"role": "system", "content": rendered})
        rule_ids = tuple(getattr(r, "rule_id", "RULE-???") for r in rules)
        injected_tokens = approx_tokens(rendered)

    volatile.append({"role": "user", "content": current_turn})

    return AssembledPrompt(
        prefix=tuple(prefix),
        volatile=tuple(volatile),
        injected_rule_ids=rule_ids,
        injected_token_count=injected_tokens,
    )
