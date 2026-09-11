"""Local, in-process embedding (plan SS2.4).

The embedding MUST run locally. An embedding API call is 20-50ms of network, which
breaks the <20ms pre-call budget and destroys the property that makes the pre-router
nearly free: one vector, computed once, reused by the cache, the difficulty classifier
and rule retrieval.

`HashingEmbedder` is the default so the package runs and tests with no model download.
It is deterministic and fast but semantically blind -- swap in `SentenceTransformerEmbedder`
before drawing any conclusion about semantic cache hit rate or retrieval quality.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    #: Model identity. Versioned alongside each snapshot: swapping the embedder
    #: invalidates every stored vector (see the risk register).
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return mat / norms


class HashingEmbedder:
    """Deterministic hashed bag-of-tokens. No dependencies, no semantics."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[i, idx] += sign
        return _l2_normalize(out)


class SentenceTransformerEmbedder:
    """Real local model. Requires the `embed` extra."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "SentenceTransformerEmbedder needs the 'embed' extra: "
                "pip install 'smart-router[embed]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        vecs = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)
