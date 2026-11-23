"""Failure log and clustering -- the suppression gate's input (plan SSC.2).

Most production failures are transient or one-off and yield no generalizable rule.
Minting a candidate from every failure spends the replay harness on noise, so a candidate
is only proposed once a *cluster* of similar failures exists.

This module is also useful on its own, before any critic is built: it answers whether
recurring non-structural failures exist in the traffic at all. If every cluster is a
schema error, constrained decoding is the answer and the critic never earns its place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from smart_router.schemas.verification import FailureCategory


def fingerprint(query: str) -> str:
    """Stable id for a query without storing the query (SSF)."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


@dataclass
class FailureRecord:
    request_id: str
    tenant_id: str
    query: str
    query_fingerprint: str
    category: FailureCategory
    verifier_id: str
    detail: str
    rejected_output: str
    vector: np.ndarray
    preventable_by_construction: bool
    model: str
    frontier_output: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FailureCluster:
    category: FailureCategory
    tenant_id: str
    records: list[FailureRecord]
    centroid: np.ndarray

    def __len__(self) -> int:
        return len(self.records)

    @property
    def preventable_share(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.preventable_by_construction for r in self.records) / len(self.records)

    @property
    def distinct_queries(self) -> int:
        """Distinct source queries.

        A cluster of one query retried twenty times is not a recurring failure mode; it
        is one failure seen repeatedly. The gate counts this, not `len(records)`.
        """
        return len({r.query_fingerprint for r in self.records})


def record_to_json(rec: FailureRecord, embedder_name: str | None) -> dict:
    return {
        "request_id": rec.request_id,
        "tenant_id": rec.tenant_id,
        "query": rec.query,
        "query_fingerprint": rec.query_fingerprint,
        "category": rec.category.value,
        "verifier_id": rec.verifier_id,
        "detail": rec.detail,
        "rejected_output": rec.rejected_output,
        "vector": [round(float(x), 6) for x in rec.vector.reshape(-1)],
        "preventable_by_construction": rec.preventable_by_construction,
        "model": rec.model,
        "frontier_output": rec.frontier_output,
        "created_at": rec.created_at.isoformat(),
        "embedder": embedder_name,
    }


def record_from_json(d: dict) -> FailureRecord:
    return FailureRecord(
        request_id=d["request_id"],
        tenant_id=d["tenant_id"],
        query=d["query"],
        query_fingerprint=d["query_fingerprint"],
        category=FailureCategory(d["category"]),
        verifier_id=d["verifier_id"],
        detail=d.get("detail", ""),
        rejected_output=d.get("rejected_output", ""),
        vector=np.asarray(d["vector"], dtype=np.float32),
        preventable_by_construction=bool(d.get("preventable_by_construction", False)),
        model=d.get("model", ""),
        frontier_output=d.get("frontier_output"),
        created_at=datetime.fromisoformat(d["created_at"]),
    )


class FailureLog:
    """Failure records, optionally persisted as append-only JSONL.

    Append rather than rewrite, because recording happens on the request path. Three
    properties the persistence has to guarantee:

    * **A write failure never fails the request.** A full disk degrades the log to
      in-memory and is counted in `write_errors`; the user still gets an answer.
    * **A torn final line is skipped, not fatal.** A crash mid-append leaves half a JSON
      object; loading drops it and keeps the rest.
    * **Vectors from a different embedder are dropped, not mixed.** Clustering cosine
      similarities across two embedding spaces produces confident nonsense, so records
      carry the embedder that produced them and a mismatch is excluded on load and
      counted in `dropped_on_load`.
    """

    def __init__(
        self,
        max_records: int = 50_000,
        path: Path | str | None = None,
        embedder_name: str | None = None,
    ) -> None:
        self.max_records = max_records
        self.path = Path(path) if path else None
        self.embedder_name = embedder_name
        self._records: list[FailureRecord] = []
        self.dropped_on_load = 0
        self.write_errors = 0
        if self.path is not None and self.path.exists():
            self.load()

    def record(self, rec: FailureRecord) -> None:
        self._records.append(rec)
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record_to_json(rec, self.embedder_name)) + "\n")
            except OSError:
                self.write_errors += 1
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
            self._rewrite()

    def load(self) -> None:
        self._records = []
        self.dropped_on_load = 0
        if self.path is None or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                self.dropped_on_load += 1
                continue
            stored = d.get("embedder")
            if self.embedder_name and stored not in (None, self.embedder_name):
                self.dropped_on_load += 1
                continue
            try:
                self._records.append(record_from_json(d))
            except (KeyError, ValueError, TypeError):
                self.dropped_on_load += 1
        self._records = self._records[-self.max_records :]

    def remove(self, request_ids) -> int:
        """Drop consumed records so the next learning pass does not re-pay for them."""
        ids = set(request_ids)
        if not ids:
            return 0
        before = len(self._records)
        self._records = [r for r in self._records if r.request_id not in ids]
        removed = before - len(self._records)
        if removed:
            self._rewrite()
        return removed

    def _rewrite(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in self._records:
                    f.write(json.dumps(record_to_json(r, self.embedder_name)) + "\n")
            tmp.replace(self.path)
        except OSError:
            self.write_errors += 1

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[FailureRecord]:
        return list(self._records)

    def cluster(
        self, *, similarity_threshold: float = 0.72, tenant_id: str | None = None
    ) -> list[FailureCluster]:
        """Greedy agglomerative clustering within (tenant, category).

        Failures are never clustered across categories -- a timeout and a wrong tool
        argument share nothing generalizable even when the queries look alike.
        """
        buckets: dict[tuple[str, FailureCategory], list[FailureRecord]] = {}
        for r in self._records:
            if tenant_id is not None and r.tenant_id != tenant_id:
                continue
            buckets.setdefault((r.tenant_id, r.category), []).append(r)

        clusters: list[FailureCluster] = []
        for (tid, category), records in buckets.items():
            groups: list[list[FailureRecord]] = []
            centroids: list[np.ndarray] = []
            for rec in records:
                v = rec.vector.reshape(-1)
                placed = False
                for i, c in enumerate(centroids):
                    if float(v @ c) >= similarity_threshold:
                        groups[i].append(rec)
                        stacked = np.stack([g.vector.reshape(-1) for g in groups[i]])
                        mean = stacked.mean(axis=0)
                        centroids[i] = mean / max(float(np.linalg.norm(mean)), 1e-12)
                        placed = True
                        break
                if not placed:
                    groups.append([rec])
                    centroids.append(v / max(float(np.linalg.norm(v)), 1e-12))
            for group, centroid in zip(groups, centroids):
                clusters.append(FailureCluster(category, tid, group, centroid))

        clusters.sort(key=len, reverse=True)
        return clusters
