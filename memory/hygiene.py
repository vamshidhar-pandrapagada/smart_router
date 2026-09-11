"""Anti-memorization filters (plan SSC.5).

Two different controls share the name `leak_ground_truth` in loose usage. They are not
the same and both are needed:

* **Input-side blinding** -- the critic is not shown the frontier's answer while
  synthesizing. Enforced in `memory/critic.py` by what is put in the prompt.
* **Output-side filtering** -- this module. A synthesized rule must state a generalizable
  constraint, not the answer to one query. "Enforce anti-memorization" is a goal; these
  are the mechanisms.

A rule that memorizes one answer is not learning; it is a cache with extra steps, and a
worse cache than `cache/exact.py` because it costs tokens on every unrelated request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[A-Za-z0-9_@./#-]+")
#: Literal shapes a generalizable invariant should never need to name.
_LITERALS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"           # dates
    r"|\b[A-Z]{2,}-\d+\b"              # ticket ids: ABC-123
    r"|\b[\w.+-]+@[\w-]+\.[\w.]+\b"    # emails
    r"|\bhttps?://\S+"                 # urls
    r"|\b[0-9a-f]{16,}\b"              # hashes / ids
)


@dataclass
class HygieneVerdict:
    ok: bool
    reason: str = ""


@dataclass
class AntiMemorizationFilter:
    #: Max share of rule tokens that also appear in the source query or answer.
    max_ngram_overlap: float = 0.35
    ngram_n: int = 3
    #: A rule embedded too close to its single source instance has memorized it.
    max_source_similarity: float = 0.93

    @staticmethod
    def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        tokens = [t.lower() for t in _TOKEN.findall(text)]
        return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}

    def check(
        self,
        rule_text: str,
        *,
        source_query: str,
        source_answer: str,
        rule_vector: np.ndarray | None = None,
        source_vector: np.ndarray | None = None,
    ) -> HygieneVerdict:
        literal = _LITERALS.search(rule_text)
        if literal:
            return HygieneVerdict(
                False, f"rule names an instance literal: {literal.group(0)!r}"
            )

        rule_grams = self._ngrams(rule_text, self.ngram_n)
        if rule_grams:
            source_grams = self._ngrams(
                f"{source_query}\n{source_answer}", self.ngram_n
            )
            overlap = len(rule_grams & source_grams) / len(rule_grams)
            if overlap > self.max_ngram_overlap:
                return HygieneVerdict(
                    False,
                    f"{overlap:.0%} {self.ngram_n}-gram overlap with source "
                    f"(max {self.max_ngram_overlap:.0%})",
                )

        if rule_vector is not None and source_vector is not None:
            sim = float(rule_vector.reshape(-1) @ source_vector.reshape(-1))
            if sim > self.max_source_similarity:
                return HygieneVerdict(
                    False, f"rule embeds at {sim:.2f} to its single source instance"
                )

        return HygieneVerdict(True, "no memorization signal")


_SECRET = re.compile(
    r"\b(sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,})\b"
    r"|\b[\w.+-]+@[\w-]+\.[\w.]+\b"
    r"|\b\d{3}-\d{2}-\d{4}\b"
)


def scrub(text: str) -> str:
    """Redact secrets and direct identifiers before a rule is persisted (SSF)."""
    return _SECRET.sub("[REDACTED]", text)
