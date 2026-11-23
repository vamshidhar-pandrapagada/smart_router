"""Difficulty classifier -- layer 3, ~2ms (plan SSA.1, SSA.2).

A logistic head over the query embedding. Explicitly *not* a small LLM judging
difficulty: anything that calls a model to decide which model to call has already lost.
It consumes the vector the semantic cache and rule retrieval already computed, so its
marginal cost is a dot product.

Labels come from the cascade itself (SSA.2): every escalation is a negative example,
every verifier-passed response a positive. No separate labelling effort. Until enough
labels exist the classifier abstains and the router runs pure cascade -- that is the
Phase 0/1 cold-start behaviour, and it is what generates the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingExample:
    vector: np.ndarray
    succeeded: bool


@dataclass
class DifficultyClassifier:
    """Logistic regression trained by batch gradient descent."""

    dim: int
    #: Below this many labels the classifier abstains rather than guessing.
    min_labels: int = 200
    l2: float = 1e-3
    epochs: int = 300
    lr: float = 0.5
    _w: np.ndarray | None = field(default=None, repr=False)
    _b: float = field(default=0.0, repr=False)
    _n_labels: int = field(default=0, repr=False)
    #: Pinned so a snapshot swap cannot silently feed vectors from a different space.
    embedder_name: str | None = field(default=None, repr=False)

    @property
    def trained(self) -> bool:
        return self._w is not None and self._n_labels >= self.min_labels

    def fit(self, examples: list[TrainingExample], embedder_name: str) -> None:
        self._n_labels = len(examples)
        self.embedder_name = embedder_name
        if len(examples) < self.min_labels:
            self._w = None
            return
        X = np.stack([e.vector.reshape(-1) for e in examples]).astype(np.float64)
        y = np.array([1.0 if e.succeeded else 0.0 for e in examples])
        w = np.zeros(X.shape[1])
        b = 0.0
        n = len(y)
        for _ in range(self.epochs):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = p - y
            w -= self.lr * ((X.T @ err) / n + self.l2 * w)
            b -= self.lr * float(err.mean())
        self._w, self._b = w, b

    def predict(self, vector: np.ndarray, embedder_name: str | None = None) -> float | None:
        """P(small model succeeds), or None while abstaining."""
        if not self.trained:
            return None
        if (
            embedder_name is not None
            and self.embedder_name is not None
            and embedder_name != self.embedder_name
        ):
            raise ValueError(
                f"classifier was fit on {self.embedder_name!r} vectors but was handed "
                f"{embedder_name!r}; re-fit before serving"
            )
        z = float(vector.reshape(-1) @ self._w + self._b)
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))
