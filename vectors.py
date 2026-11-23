"""In-memory vector store (plan SS2.4). No vector database.

At the working-set size this design implies -- rules in the hundreds to low thousands,
capped by the SSD token budget and utility eviction -- a brute-force numpy matmul is
sub-millisecond and *exact*. Past ~50k vectors, move to in-process ANN (faiss/hnswlib).
A hosted vector store is not on this path: its network round trip alone (20-50ms) exceeds
the entire pre-call budget.

Snapshot pinning and in-memory vectors are the same design: cut a snapshot, load it into
each replica, serve until the next cut. No hot-path write, no consistency problem, and
replay is exact because `version` names the vector set that was live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

import numpy as np

T = TypeVar("T")

#: Beyond this, brute force stops being comfortably sub-millisecond.
ANN_THRESHOLD = 50_000


@dataclass(frozen=True)
class Scored(Generic[T]):
    item: T
    score: float


class VectorSnapshot(Generic[T]):
    """Immutable set of vectors plus their payloads."""

    def __init__(
        self,
        version: str,
        embedder_name: str,
        vectors: np.ndarray,
        items: Sequence[T],
    ) -> None:
        if len(items) != vectors.shape[0]:
            raise ValueError(f"{vectors.shape[0]} vectors for {len(items)} items")
        if vectors.size and vectors.shape[0] > ANN_THRESHOLD:
            raise ValueError(
                f"{vectors.shape[0]} vectors exceeds the brute-force threshold "
                f"({ANN_THRESHOLD}); move to in-process ANN rather than raising this limit"
            )
        self.version = version
        #: Swapping the embedder invalidates every stored vector, so its identity is
        #: pinned to the snapshot and checked at query time.
        self.embedder_name = embedder_name
        self._vectors = vectors.astype(np.float32, copy=False)
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query: np.ndarray, k: int, embedder_name: str) -> list[Scored[T]]:
        if embedder_name != self.embedder_name:
            raise ValueError(
                f"snapshot {self.version} was built with {self.embedder_name!r} but the "
                f"live embedder is {embedder_name!r}; re-embed before serving"
            )
        if not self._items or k <= 0:
            return []
        sims = self._vectors @ query.reshape(-1)
        top = np.argsort(-sims)[: min(k, len(self._items))]
        return [Scored(self._items[int(i)], float(sims[int(i)])) for i in top]

    def search_mmr(
        self,
        query: np.ndarray,
        k: int,
        embedder_name: str,
        *,
        lambda_: float = 0.7,
        candidate_pool: int = 40,
    ) -> list[Scored[T]]:
        """Maximal Marginal Relevance selection.

        Plain top-k returns near-duplicate rules, which wastes the token budget on
        redundancy. MMR trades a little relevance for coverage.
        """
        if embedder_name != self.embedder_name:
            raise ValueError(
                f"snapshot {self.version} was built with {self.embedder_name!r} but the "
                f"live embedder is {embedder_name!r}; re-embed before serving"
            )
        if not self._items or k <= 0:
            return []

        q = query.reshape(-1)
        sims = self._vectors @ q
        pool = list(np.argsort(-sims)[: min(candidate_pool, len(self._items))])

        selected: list[int] = []
        while pool and len(selected) < k:
            if not selected:
                best = pool.pop(0)
                selected.append(int(best))
                continue
            chosen = self._vectors[selected]
            best_idx, best_score = None, -np.inf
            for cand in pool:
                redundancy = float(np.max(chosen @ self._vectors[cand]))
                score = lambda_ * float(sims[cand]) - (1.0 - lambda_) * redundancy
                if score > best_score:
                    best_idx, best_score = cand, score
            pool.remove(best_idx)
            selected.append(int(best_idx))

        return [Scored(self._items[i], float(sims[i])) for i in selected]


def build_snapshot(
    version: str,
    embedder,
    items: Sequence[T],
    text_of,
) -> VectorSnapshot[T]:
    """Embed `items` and freeze them into a snapshot."""
    if not items:
        return VectorSnapshot(version, embedder.name, np.zeros((0, embedder.dim), np.float32), [])
    vectors = embedder.encode([text_of(i) for i in items])
    return VectorSnapshot(version, embedder.name, vectors, items)
