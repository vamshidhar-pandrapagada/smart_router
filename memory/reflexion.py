"""Reflexion behind one flag.

The memory modules -- failure log, suppression gate, critic, replay, lifecycle, rule
store, exemplars -- were built as separate pieces. Turning reflexion on meant hand-wiring
three objects into the router, nothing survived a restart, and the offline loop had no
entry point. This module is that wiring:

    reflexion = Reflexion(ReflexionConfig.from_env(), embedder)
    router = SmartRouter(..., reflexion=reflexion)

**Off by default, and inert when off.** A disabled Reflexion forces no embedding, retrieves
nothing and writes no file; the router behaves exactly as if it were absent.

**Two modes, one capture path**, so the plan's SSC.0 bake-off runs on the same traffic:

* ``rules``      failures are clustered, gated, distilled by a critic, replayed and
                 promoted through the lifecycle before any rule is served.
* ``exemplars``  the frontier's corrected answers to past escalations are retrieved and
                 few-shot directly. No critic, no lifecycle, no poisoning surface.

Both modes capture failures, corrections and verified successes, so switching modes later
starts from history rather than from zero.

**Where things run:**

    request path   retrieve per tenant; log failures; record successes and corrections.
                   Cheap appends. Never waits on learning, never fails a request.
    learn()        cluster -> gate -> critic -> replay -> lifecycle -> refresh.
                   Offline, and it costs money: replay makes real model calls.

**Tenant isolation is structural.** Each tenant gets its own snapshot and retriever, built
from that tenant's rules (plus GLOBAL rules, which require human review before promotion)
or that tenant's exemplars (never GLOBAL -- they are raw user data). One tenant's learned
text cannot reach another tenant's prompt, because it is never in the vector set searched.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from smart_router.memory.critic import ReflexionCritic
from smart_router.memory.exemplars import Exemplar, ExemplarRetriever
from smart_router.memory.failures import FailureLog, fingerprint
from smart_router.memory.gate import SuppressionGate
from smart_router.memory.hygiene import scrub
from smart_router.memory.loop import LearningLoop, LoopReport
from smart_router.memory.replay import GoldenCase, ReplayHarness
from smart_router.memory.retrieval import Retrieved, RuleRetriever
from smart_router.memory.store import RuleStore
from smart_router.schemas.rule import GLOBAL_SCOPE
from smart_router.vectors import build_snapshot

ENV_ENABLED = "SMART_ROUTER_REFLEXION"
ENV_MODE = "SMART_ROUTER_REFLEXION_MODE"
ENV_DIR = "SMART_ROUTER_REFLEXION_DIR"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ReflexionMode(str, Enum):
    RULES = "rules"
    EXEMPLARS = "exemplars"


@dataclass
class ReflexionConfig:
    enabled: bool = False
    mode: ReflexionMode = ReflexionMode.RULES
    #: Where failures, rules, exemplars and golden cases persist. None keeps everything
    #: in memory, which is right for tests and wrong for anything that should learn.
    state_dir: Path | None = None
    k: int = 4
    max_tokens: int = 400
    min_similarity: float = 0.35
    min_distinct_queries: int = 5
    golden_capacity: int = 200
    exemplar_capacity: int = 2000
    #: Golden cases replayed per candidate. Each costs two small-model calls (with and
    #: without the rule), plus a judge call when replay_with_judge is set.
    replay_golden_sample: int = 40
    #: Force the judge during replay. Without it, a failure only the judge can see -- the
    #: cron case -- passes replay with or without the rule, and the resolution test
    #: certifies rules that fix nothing.
    replay_with_judge: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReflexionMode):
            try:
                self.mode = ReflexionMode(str(self.mode).lower())
            except ValueError:
                allowed = [m.value for m in ReflexionMode]
                raise ValueError(
                    f"reflexion mode must be one of {allowed}, got {self.mode!r}"
                ) from None
        if self.state_dir is not None:
            self.state_dir = Path(self.state_dir).expanduser()

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, **overrides: Any
    ) -> "ReflexionConfig":
        """Environment first, explicit overrides (CLI flags) on top."""
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        raw = env.get(ENV_ENABLED)
        if raw is not None:
            values["enabled"] = raw.strip().lower() in _TRUTHY
        if env.get(ENV_MODE):
            values["mode"] = env[ENV_MODE]
        if env.get(ENV_DIR):
            values["state_dir"] = Path(env[ENV_DIR])
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


class _Jsonl:
    """Append-only JSONL with periodic compaction.

    Appends because writes happen on the request path. A torn final line from a crash
    mid-append is skipped on read. A write failure is counted, never raised.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        self.lines = 0
        self.write_errors = 0

    def read(self) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.lines = len(rows)
        return rows

    def append(self, row: dict) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.lines += 1
        except OSError:
            self.write_errors += 1

    def rewrite(self, rows: Sequence[dict]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            tmp.replace(self.path)
            self.lines = len(rows)
        except OSError:
            self.write_errors += 1


class ExemplarStore:
    """Frontier corrections per tenant, deduplicated by query fingerprint.

    Exemplars are raw user requests and model outputs, so they are scrubbed on entry and
    never GLOBAL: unlike a distilled rule there is no abstraction step between one
    tenant's data and another tenant's prompt.

    Caveat carried explicitly: the frontier's answer is treated as correct. It passed the
    commit gate when a tool was involved, but nothing adjudicated it. That is the
    label-bias assumption this arm makes, and why the offline judge audit exists.
    """

    def __init__(self, path: Path | None = None, capacity: int = 2000) -> None:
        self.capacity = capacity
        self._file = _Jsonl(path)
        self._items: OrderedDict[tuple[str, str], Exemplar] = OrderedDict()
        for row in self._file.read():
            try:
                self._put(Exemplar(
                    query=row["query"], verified_output=row["output"],
                    tenant_id=row["tenant_id"], source_request_id=row.get("request_id", ""),
                ))
            except KeyError:
                continue
        self._trim()

    def _put(self, ex: Exemplar) -> None:
        key = (ex.tenant_id, fingerprint(ex.query))
        self._items.pop(key, None)  # newest wins and moves to the back
        self._items[key] = ex

    def _trim(self) -> None:
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    @staticmethod
    def _row(ex: Exemplar) -> dict:
        return {"query": ex.query, "output": ex.verified_output,
                "tenant_id": ex.tenant_id, "request_id": ex.source_request_id}

    def add(self, query: str, output: str, *, tenant_id: str, request_id: str = "") -> Exemplar:
        if tenant_id == GLOBAL_SCOPE:
            raise ValueError("exemplars are raw tenant data and cannot be GLOBAL")
        ex = Exemplar(query=scrub(query), verified_output=scrub(output),
                      tenant_id=tenant_id, source_request_id=request_id)
        self._put(ex)
        self._trim()
        # Compact occasionally rather than on every add: rewriting on each request once
        # the store is full would put a full-file write on the request path.
        if self._file.lines >= 4 * self.capacity:
            self._file.rewrite([self._row(e) for e in self._items.values()])
        else:
            self._file.append(self._row(ex))
        return ex

    def all(self) -> list[Exemplar]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


class GoldenSet:
    """Recent distinct verified successes -- the regression battery for replay.

    'Verified' means the small model's output passed the ladder and committed. That is
    the proxy label, not adjudicated correctness, so the regression check inherits the
    same label bias as the Wilson bound: noted here, not solved here.

    Most-recent rather than a uniform reservoir, because regression should reflect the
    traffic a rule will meet next, not the traffic of six months ago.
    """

    def __init__(self, path: Path | None = None, capacity: int = 200) -> None:
        self.capacity = capacity
        self._file = _Jsonl(path)
        self._items: OrderedDict[tuple[str, str], GoldenCase] = OrderedDict()
        for row in self._file.read():
            if "query" in row:
                self._put(row["query"], row.get("tenant_id", "default"))
        self._trim()

    def _put(self, query: str, tenant_id: str) -> None:
        key = (tenant_id, fingerprint(query))
        self._items.pop(key, None)
        self._items[key] = GoldenCase(query=query, tenant_id=tenant_id)

    def _trim(self) -> None:
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def add(self, query: str, tenant_id: str) -> None:
        q = scrub(query)
        self._put(q, tenant_id)
        self._trim()
        if self._file.lines >= 4 * self.capacity:
            self._file.rewrite(
                [{"query": c.query, "tenant_id": c.tenant_id} for c in self._items.values()]
            )
        else:
            self._file.append({"query": q, "tenant_id": tenant_id})

    def cases(self, tenant_id: str | None = None, limit: int | None = None) -> list[GoldenCase]:
        items = [c for c in self._items.values() if tenant_id is None or c.tenant_id == tenant_id]
        items.reverse()  # most recent first
        return items[:limit] if limit else items

    def __len__(self) -> int:
        return len(self._items)


def make_replay(router, *, with_judge: bool = True) -> Callable[[str, Sequence[Any]], bool]:
    """Counterfactual replay through the router's own small model and verifiers.

    Two properties matter more than anything else here:

    * **It never executes tools.** Verification only; the commit gate is not consulted.
      Replaying a failing "create a job" request must not create jobs.
    * **The judge is forced on when `with_judge` is set.** For judge-caught failures --
      the cron case, where every free and cheap verifier passes -- a replay without the
      judge passes with or without the rule, so the resolution test would certify rules
      that fix nothing. Live routing samples the judge; replay must not.

    Uses the router's own decoding path (`_call`), so constrained decoding and the prompt
    ordering are exactly what live traffic gets.
    """
    from smart_router.prompt import assemble
    from smart_router.schemas.routing import RouteTier
    from smart_router.verify.base import VerificationContext
    from smart_router.verify.expensive import JudgeSampler
    from smart_router.verify.ladder import VerifierLadder

    source = router.ladder
    ladder = VerifierLadder(
        free=list(source.free),
        cheap=list(source.cheap),
        judge=source.judge if with_judge else None,
        sampler=JudgeSampler(random_rate=1.0),
    )

    def replay(query: str, rules: Sequence[Any]) -> bool:
        prompt = assemble(
            system=router.system_prompt, tools=router.tools,
            rules=list(rules), current_turn=query,
        )
        try:
            _, _, resp = router._call(router.small, prompt.messages, RouteTier.SMALL)
        except Exception:  # noqa: BLE001 - a replay that cannot run is a failed replay
            return False
        policy = router.policy.evaluate(query, tenant_id="replay")
        ctx = VerificationContext(query=query, response=resp, risk_class=policy.risk_class)
        outcome = ladder.run(
            uuid4(), ctx,
            risk_class=policy.risk_class,
            cost_of_being_wrong_usd=policy.cost_of_being_wrong_usd,
        )
        return outcome.passed

    return replay


@dataclass
class LearnSummary:
    mode: str
    loop: LoopReport | None
    consumed_failures: int
    exemplars: int
    snapshots: dict[str, str]

    def render(self) -> str:
        lines = [f"mode              {self.mode}"]
        if self.loop is not None:
            lines.append(f"loop              {self.loop.summary()}")
            for label, items in (
                ("gated out", self.loop.clusters_gated_out),
                ("quarantined", self.loop.quarantined),
                ("replay rejected", self.loop.replay_rejected),
                ("held in shadow", self.loop.held_in_shadow),
                ("promoted", self.loop.promoted),
            ):
                for item in items:
                    lines.append(f"  {label:15} {item}")
            lines.append(
                f"failures consumed {self.consumed_failures} "
                "(clusters that reached the critic; the rest keep accumulating)"
            )
        else:
            lines.append(f"exemplars         {self.exemplars} (no distillation in this mode)")
        for tenant, version in sorted(self.snapshots.items()):
            lines.append(f"snapshot          {tenant}: {version}")
        return "\n".join(lines)


def _version(mode: str, tenant: str, items: Sequence[Any]) -> str:
    """Content-addressed: identical memory yields an identical version, for replay."""
    digest = hashlib.sha256(
        "\n".join(sorted(i.render() for i in items)).encode()
    ).hexdigest()[:8]
    return f"{mode}:{tenant}:{digest}"


class Reflexion:
    def __init__(self, config: ReflexionConfig, embedder) -> None:
        self.config = config
        self.embedder = embedder
        self.failure_log: FailureLog | None = None
        self.rule_store: RuleStore | None = None
        self.exemplars: ExemplarStore | None = None
        self.golden: GoldenSet | None = None
        self._retrievers: dict[str, Any] = {}
        self._versions: dict[str, str] = {}
        if not config.enabled:
            return

        root = config.state_dir
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

        def at(name: str) -> Path | None:
            return root / name if root is not None else None

        self.failure_log = FailureLog(path=at("failures.jsonl"), embedder_name=embedder.name)
        self.rule_store = RuleStore(at("rules.json"))
        self.exemplars = ExemplarStore(at("exemplars.jsonl"), capacity=config.exemplar_capacity)
        self.golden = GoldenSet(at("golden.jsonl"), capacity=config.golden_capacity)
        self.refresh()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def mode(self) -> ReflexionMode:
        return self.config.mode

    @property
    def uses_rules(self) -> bool:
        return self.config.mode is ReflexionMode.RULES

    # -- request path ----------------------------------------------------------

    def retrieve(self, vector, tenant_id: str) -> Retrieved | None:
        retriever = self._retrievers.get(tenant_id) or self._retrievers.get(GLOBAL_SCOPE)
        return retriever.retrieve(vector) if retriever is not None else None

    def snapshot_version_for(self, tenant_id: str) -> str | None:
        return self._versions.get(tenant_id) or self._versions.get(GLOBAL_SCOPE)

    def record_success(self, query: str, tenant_id: str) -> None:
        if self.golden is not None:
            self.golden.add(query, tenant_id)

    def record_exemplar(self, query: str, output: str, tenant_id: str, request_id: str = "") -> None:
        # Never raises on the request path: a GLOBAL tenant or an empty answer is skipped.
        if self.exemplars is None or tenant_id == GLOBAL_SCOPE or not (output or "").strip():
            return
        self.exemplars.add(query, output, tenant_id=tenant_id, request_id=request_id)

    # -- offline ---------------------------------------------------------------

    def refresh(self) -> dict[str, str]:
        """Cut fresh per-tenant snapshots and swap the retrievers in one assignment.

        Called after learning, never on the request path. A snapshot is immutable while
        served, which keeps the prompt tail stable for the provider prefix cache and
        makes every request replayable from its logged version.
        """
        if not self.enabled:
            return {}
        c = self.config
        retrievers: dict[str, Any] = {}
        versions: dict[str, str] = {}
        if self.uses_rules:
            tenants = {GLOBAL_SCOPE} | {r.tenant_id for r in self.rule_store.promoted()}
            for tenant in tenants:
                # promoted(tenant) = this tenant's rules plus GLOBAL ones, which already
                # required human review before promotion.
                rules = self.rule_store.promoted(tenant)
                snap = build_snapshot(
                    _version("rules", tenant, rules), self.embedder, rules,
                    lambda r: r.trigger_text,
                )
                versions[tenant] = snap.version
                retrievers[tenant] = RuleRetriever(
                    snap, self.embedder.name, k=c.k,
                    max_tokens=c.max_tokens, min_similarity=c.min_similarity,
                )
        else:
            by_tenant: dict[str, list[Exemplar]] = {}
            for ex in self.exemplars.all():
                by_tenant.setdefault(ex.tenant_id, []).append(ex)
            for tenant, items in by_tenant.items():
                snap = build_snapshot(
                    _version("exemplars", tenant, items), self.embedder, items,
                    lambda e: e.trigger_text,
                )
                versions[tenant] = snap.version
                retrievers[tenant] = ExemplarRetriever(
                    snap, self.embedder.name, k=c.k,
                    max_tokens=c.max_tokens, min_similarity=c.min_similarity,
                )
        self._retrievers, self._versions = retrievers, versions
        return dict(versions)

    def learn(
        self,
        router,
        *,
        critic_complete=None,
        live_model: str,
        tenant_id: str | None = None,
    ) -> LearnSummary:
        """Run the offline loop, then refresh what the request path serves."""
        if not self.enabled:
            raise RuntimeError("reflexion is disabled; enable it before learning")

        if not self.uses_rules:
            versions = self.refresh()
            return LearnSummary("exemplars", None, 0, len(self.exemplars), versions)

        if critic_complete is None:
            raise ValueError("rules mode needs a critic model to distil failures into rules")

        harness = ReplayHarness(
            replay=make_replay(router, with_judge=self.config.replay_with_judge),
            golden=self.golden.cases(tenant_id, limit=self.config.replay_golden_sample),
        )
        loop = LearningLoop(
            failure_log=self.failure_log,
            store=self.rule_store,
            critic=ReflexionCritic(critic_complete),
            harness=harness,
            gate=SuppressionGate(min_distinct_queries=self.config.min_distinct_queries),
            embedder=self.embedder,
        )
        report = loop.run(live_model=live_model, tenant_id=tenant_id)
        consumed = self.failure_log.remove(report.consumed_request_ids)
        versions = self.refresh()
        return LearnSummary("rules", report, consumed, len(self.exemplars), versions)

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "mode": self.mode.value,
            "state_dir": str(self.config.state_dir) if self.config.state_dir else None,
            "embedder": self.embedder.name,
            "failures": len(self.failure_log),
            "failures_dropped_on_load": self.failure_log.dropped_on_load,
            "rules": dict(Counter(r.state.value for r in self.rule_store.all())),
            "exemplars": len(self.exemplars),
            "golden": len(self.golden),
            "snapshots": dict(self._versions),
            "write_errors": (
                self.failure_log.write_errors
                + self.exemplars._file.write_errors
                + self.golden._file.write_errors
            ),
        }
