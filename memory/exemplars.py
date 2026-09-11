"""Exemplar retrieval -- the control arm the rule ledger must beat (plan SSC.0).

Same failure archive, different use: instead of a critic distilling an abstract rule,
store the verified-correct output and few-shot the k most similar past successes. No
critic, no lifecycle, no replay harness, no poisoning surface -- four modules collapse
into one.

Concrete examples frequently outperform distilled invariants, because there is no
distillation step in which a critic can invent a wrong generalization. Rules earn their
place in three specific places: compression (many failures into one rule), generalization
across surface-dissimilar queries, and prohibitions -- which cannot be demonstrated well,
since a model shown a counterexample may imitate the wrong half.

Compare the two arms at MATCHED TOKEN BUDGET or the comparison is uninformative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smart_router.vectors import VectorSnapshot, build_snapshot


@dataclass
class Exemplar:
    query: str
    verified_output: str
    tenant_id: str = "default"
    source_request_id: str = ""

    @property
    def rule_id(self) -> str:
        """Lets an exemplar stand in wherever a rule is expected.

        Previously `f"EX-{request_id or hash(query) % 10**6:06d}"`, which applied an
        integer format to the request id -- a string -- and raised the first time an
        exemplar carried one. It also used `hash()`, which Python salts per process, so
        the same exemplar got a different id on every run and logged ids could not be
        matched back. Both paths are now stable strings.
        """
        if self.source_request_id:
            return f"EX-{self.source_request_id.replace('-', '')[:8]}"
        from smart_router.memory.failures import fingerprint

        return f"EX-{fingerprint(self.query)[:8]}"

    @property
    def trigger_text(self) -> str:
        return self.query

    @property
    def token_cost(self) -> int:
        return max(1, (len(self.query) + len(self.verified_output)) // 4)

    def render(self) -> str:
        return f"- EXAMPLE\n  REQUEST: {self.query}\n  CORRECT OUTPUT: {self.verified_output}"


@dataclass
class ExemplarRetriever:
    snapshot: VectorSnapshot[Exemplar] | None
    embedder_name: str
    k: int = 3
    max_tokens: int = 400
    min_similarity: float = 0.35

    def retrieve(self, query_vector: np.ndarray):
        from smart_router.memory.retrieval import Retrieved

        if self.snapshot is None or len(self.snapshot) == 0:
            return Retrieved([], 0)
        kept, tokens, below, dropped = [], 0, 0, 0
        for s in self.snapshot.search(query_vector, self.k, self.embedder_name):
            if s.score < self.min_similarity:
                below += 1
                continue
            if tokens + s.item.token_cost > self.max_tokens:
                dropped += 1
                continue
            kept.append(s.item)
            tokens += s.item.token_cost
        return Retrieved(kept, tokens, dropped, below)


def build_exemplar_snapshot(version: str, embedder, exemplars: list[Exemplar]):
    return build_snapshot(version, embedder, exemplars, lambda e: e.trigger_text)
