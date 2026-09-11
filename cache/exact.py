"""Exact response cache (plan SS2.4).

Phase 0 builds this and nothing else. Normalized-query hash into a key-value store: no
vectors, no similarity threshold to tune, no false hits. Its standalone hit rate is a
Phase 0 deliverable because it decides whether semantic caching is worth building at all.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

_WS = re.compile(r"\s+")


def normalize(query: str) -> str:
    return _WS.sub(" ", query.strip().lower())


def cache_key(query: str, *, tenant_id: str, model: str) -> str:
    """Tenant and model are part of the key.

    Sharing a cache entry across tenants would leak one tenant's answer to another;
    sharing across models would attribute one model's output to another.
    """
    payload = f"{tenant_id}\x00{model}\x00{normalize(query)}"
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class _Entry:
    value: str
    expires_at: float


class ExactCache:
    """Bounded LRU with TTL. Per-replica; there is no shared-cache story in Phase 0."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: float = 3600.0) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def get(self, query: str, *, tenant_id: str, model: str) -> str | None:
        key = cache_key(query, tenant_id=tenant_id, model=model)
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        if entry.expires_at <= time.monotonic():
            del self._entries[key]
            self.stats.expirations += 1
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry.value

    def put(self, query: str, value: str, *, tenant_id: str, model: str) -> None:
        key = cache_key(query, tenant_id=tenant_id, model=model)
        self._entries[key] = _Entry(value, time.monotonic() + self.ttl_seconds)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def __len__(self) -> int:
        return len(self._entries)
