"""Router orchestrator -- the request lifecycle of plan SS2.1.

Order is load-bearing and matches the decision ladder in SSA.1:

    policy gate (0ms) -> exact cache -> embed once -> difficulty classifier
      -> prompt assembly (SS2.2 order) -> small model -> verifier ladder
      -> commit, or escalate to frontier WITH the failure context attached

Two behaviours worth stating plainly because they are easy to get wrong:

* A transient fault (timeout, provider error) retries the small model once. It does not
  escalate: sending a 503 to a frontier model spends real money on a problem the small
  model never had.
* Escalation carries the failed attempt and the verifier's finding. Re-running the query
  cold at the frontier throws away information that improves its answer and shortens it.
"""

from __future__ import annotations

import time
from typing import Sequence
from uuid import UUID, uuid4

import numpy as np

from smart_router.cache.exact import ExactCache
from smart_router.gateway.base import ModelClient, ProviderFault, ProviderTimeout
from smart_router.prompt import ConversationHistory, assemble
from smart_router.routing.classifier import DifficultyClassifier
from smart_router.routing.policy import PolicyGate
from smart_router.routing.threshold import DEFAULT_COST_MODEL, CostModel
from smart_router.schemas.routing import RouteReason, RouteTier, RoutingDecision
from smart_router.schemas.trace import ModelCall, RequestTrace
from smart_router.schemas.verification import FailureCategory, VerificationOutcome
from smart_router.verify.base import VerificationContext
from smart_router.verify.ladder import VerifierLadder

TOOL_ESCALATION_TEMPLATE = (
    "A smaller model proposed a tool call. It was well-formed and passed verification, "
    "but the tool itself rejected it.\n"
    "PROPOSED CALL:\n{output}\n\n"
    "GATEWAY ERROR ({category}): {detail}\n\n"
    "Diagnose why the call was wrong and issue a corrected one. If the error indicates a "
    "missing or unknown identifier, discover the valid value first rather than guessing."
)

ESCALATION_TEMPLATE = (
    "A smaller model attempted this request and its output was rejected by verification.\n"
    "REJECTED OUTPUT:\n{output}\n\n"
    "VERIFIER FINDING ({category}): {detail}\n\n"
    "Produce a correct response. Do not repeat the rejected output's error."
)


class SmartRouter:
    def __init__(
        self,
        *,
        small: ModelClient,
        frontier: ModelClient,
        ladder: VerifierLadder,
        embedder,
        policy: PolicyGate | None = None,
        classifier: DifficultyClassifier | None = None,
        cache: ExactCache | None = None,
        cost_model: CostModel = DEFAULT_COST_MODEL,
        retriever=None,
        failure_log=None,
        rule_store=None,
        commit_gate=None,
        reflexion=None,
        system_prompt: str = "You are a helpful enterprise assistant.",
        tools: list[dict] | None = None,
        snapshot_version: str | None = None,
    ) -> None:
        self.small = small
        self.frontier = frontier
        self.ladder = ladder
        self.embedder = embedder
        self.policy = policy or PolicyGate()
        self.classifier = classifier
        self.cache = cache if cache is not None else ExactCache()
        self.cost_model = cost_model
        #: Rule or exemplar retriever. Both expose retrieve(vector) -> Retrieved, so the
        #: two memory arms are swappable for the SSC.0 bake-off.
        self.retriever = retriever
        self.failure_log = failure_log
        self.rule_store = rule_store
        #: Two-phase commit. Without it the router returns a *proposed* call and never
        #: executes anything -- safe against benchmarks, unsafe against live connectors,
        #: because nothing then stops the caller running the payload itself.
        self.commit_gate = commit_gate
        self.system_prompt = system_prompt
        # Prompt tool block and verifier registry derive from the same registry, so they
        # cannot drift.
        if tools is None and commit_gate is not None:
            tools = commit_gate.registry.prompt_block()
        self.tools = tools or []
        self.snapshot_version = snapshot_version
        #: Reflexion behind one flag. Disabled or absent, the router behaves exactly as
        #: if the memory did not exist: no embedding forced, no retrieval, no file
        #: written. Enabled, it supplies the failure log, the rule store and per-tenant
        #: retrieval.
        self.reflexion = reflexion if (reflexion is not None and reflexion.enabled) else None
        if self.reflexion is not None:
            self.failure_log = self.reflexion.failure_log
            self.rule_store = self.reflexion.rule_store if self.reflexion.uses_rules else None

    # -- helpers -------------------------------------------------------------

    def _call(
        self, client: ModelClient, messages, tier: RouteTier
    ) -> tuple[str, ModelCall, object]:
        from smart_router.gateway.model_config import settings_for

        # Constrained decoding removes the schema-error class at source rather than
        # detecting it afterwards -- but only if the schema is actually sent.
        schema = None
        if getattr(client, "supports_constrained_decoding", False) and self.commit_gate:
            schema = self.commit_gate.registry.envelope_schema()
        resp = client.complete(messages, schema=schema)
        settings = settings_for(client.model_name)
        return resp.text, ModelCall(
            model=client.model_name,
            tier=tier,
            prompt_tokens=resp.prompt_tokens,
            cached_prompt_tokens=resp.cached_prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=settings.cost(
                resp.prompt_tokens, resp.cached_prompt_tokens, resp.completion_tokens
            ),
            latency_ms=resp.latency_ms,
        ), resp

    # -- main ----------------------------------------------------------------

    def route(
        self,
        query: str,
        *,
        tenant_id: str = "default",
        session_id: str | None = None,
        history: ConversationHistory | None = None,
        grounding: tuple[str, ...] = (),
        rules: Sequence = (),
        override_frontier: bool = False,
        request_id: UUID | None = None,
    ) -> RequestTrace:
        rid = request_id or uuid4()
        t0 = time.perf_counter()

        # 1. Deterministic policy gate.
        policy = self.policy.evaluate(
            query, tenant_id=tenant_id, override_frontier=override_frontier
        )

        # 2. Exact cache. Semantic caching is deferred until this hit rate is measured.
        if not policy.force_frontier:
            cached = self.cache.get(query, tenant_id=tenant_id, model=self.small.model_name)
            if cached is not None:
                decision = RoutingDecision(
                    request_id=rid, tenant_id=tenant_id, tier=RouteTier.CACHE,
                    reason=RouteReason.CACHE_HIT, selected_model="cache",
                    risk_class=policy.risk_class,
                    cost_of_being_wrong_usd=policy.cost_of_being_wrong_usd,
                    snapshot_version=self.snapshot_version,
                    decision_latency_ms=(time.perf_counter() - t0) * 1000,
                )
                return RequestTrace(
                    request_id=rid, tenant_id=tenant_id, session_id=session_id, query=query,
                    decision=decision, final_output=cached, committed=True,
                )

        # 3. One embedding, reused by cache, classifier and (Phase 5) rule retrieval.
        vector: np.ndarray | None = None
        p_success: float | None = None
        threshold: float | None = None
        needs_vector = (
            self.classifier is not None
            or self.retriever is not None
            or self.reflexion is not None
        )
        if not policy.force_frontier and needs_vector:
            # One vector, reused by the classifier and by retrieval.
            vector = self.embedder.encode([query])[0]
            if self.classifier is not None:
                p_success = self.classifier.predict(vector, self.embedder.name)
                threshold = self.cost_model.min_success_prob(policy.cost_of_being_wrong_usd)

        # 4. Tier selection.
        if policy.force_frontier:
            tier, reason = RouteTier.FRONTIER, RouteReason.POLICY_FORCED
        elif p_success is not None and threshold is not None and p_success < threshold:
            tier, reason = RouteTier.FRONTIER, RouteReason.PREDICTED_HARD
        else:
            tier, reason = RouteTier.SMALL, RouteReason.OPTIMISTIC

        retrieved = None
        snapshot_version = self.snapshot_version
        if tier is RouteTier.SMALL and vector is not None:
            if self.reflexion is not None:
                # Per tenant: one tenant's rules or exemplars never reach another's prompt.
                retrieved = self.reflexion.retrieve(vector, tenant_id)
                snapshot_version = self.reflexion.snapshot_version_for(tenant_id)
            elif self.retriever is not None:
                retrieved = self.retriever.retrieve(vector)
            if retrieved is not None:
                rules = retrieved.rules or rules

        prompt = assemble(
            system=self.system_prompt, tools=self.tools, history=history,
            rules=rules, current_turn=query,
        )
        decision = RoutingDecision(
            request_id=rid, tenant_id=tenant_id, tier=tier, reason=reason,
            selected_model=(self.small if tier is RouteTier.SMALL else self.frontier).model_name,
            risk_class=policy.risk_class,
            p_small_succeeds=p_success, threshold_used=threshold,
            cost_of_being_wrong_usd=policy.cost_of_being_wrong_usd,
            snapshot_version=snapshot_version,
            injected_rule_ids=list(prompt.injected_rule_ids),
            injected_token_count=prompt.injected_token_count,
            decision_latency_ms=(time.perf_counter() - t0) * 1000,
        )
        trace = RequestTrace(
            request_id=rid, tenant_id=tenant_id, session_id=session_id, query=query,
            decision=decision,
        )

        if tier is RouteTier.FRONTIER:
            text, call, _ = self._call(self.frontier, prompt.messages, RouteTier.FRONTIER)
            trace.calls.append(call)
            # Policy sends the riskiest traffic straight here, so its tool calls need the
            # gate more than anyone's. This path used to return the raw proposal
            # unexecuted: nothing ran, and a caller that executed it got no gate at all.
            return self._finish_frontier(trace, text)

        # 5. Optimistic small-model attempt, with one retry for transient faults only.
        for attempt in (0, 1):
            try:
                text, call, resp = self._call(self.small, prompt.messages, RouteTier.SMALL)
            except (ProviderTimeout, ProviderFault) as exc:
                category = (
                    FailureCategory.TIMEOUT
                    if isinstance(exc, ProviderTimeout)
                    else FailureCategory.PROVIDER_FAULT
                )
                if attempt == 0:
                    trace.small_retried = True
                    continue
                return self._escalate(trace, prompt.messages, "", category, str(exc))
            trace.calls.append(call)
            break

        ctx = VerificationContext(
            query=query, response=resp, risk_class=policy.risk_class, grounding=grounding
        )
        outcome = self.ladder.run(
            rid, ctx,
            risk_class=policy.risk_class,
            cost_of_being_wrong_usd=policy.cost_of_being_wrong_usd,
        )
        trace.verification = outcome

        if outcome.retry_small and not trace.small_retried:
            trace.small_retried = True
            text, call, resp = self._call(self.small, prompt.messages, RouteTier.SMALL)
            trace.calls.append(call)
            ctx = VerificationContext(
                query=query, response=resp, risk_class=policy.risk_class, grounding=grounding
            )
            outcome = self.ladder.run(
                rid, ctx,
                risk_class=policy.risk_class,
                cost_of_being_wrong_usd=policy.cost_of_being_wrong_usd,
            )
            trace.verification = outcome

        if outcome.escalate or outcome.retry_small:
            failing = next((r for r in outcome.results if not r.passed), None)
            self._credit(retrieved, succeeded=False)
            self._log_failure(trace, query, text, failing, vector, tenant_id)
            return self._escalate(
                trace, prompt.messages, text,
                outcome.first_failure or FailureCategory.SILENT_SEMANTIC,
                failing.detail if failing else "",
            )

        # Verification passed. Only now may a side effect happen.
        if self.commit_gate is not None:
            tool_outcome = self._commit(trace, text, rid)
            if tool_outcome is not None and not tool_outcome.committed:
                if not tool_outcome.should_escalate:
                    # Not a capability failure, so a frontier model cannot fix it: an
                    # endpoint still down after the gate's retry, a missing credential, or
                    # an irreversible call nobody approved. Escalating would pay frontier
                    # prices to hit the same wall. Surface it -- and do not count it
                    # against any injected rule: the rule did not take the endpoint down.
                    return self._surface(trace, tool_outcome)
                self._credit(retrieved, succeeded=False)
                self._log_failure(trace, query, text, None, vector, tenant_id,
                                  category=tool_outcome.category, detail=tool_outcome.detail)
                return self._escalate(
                    trace, prompt.messages, text, tool_outcome.category,
                    tool_outcome.detail, template=TOOL_ESCALATION_TEMPLATE,
                )

        trace.final_output = text
        trace.committed = True
        self._credit(retrieved, succeeded=True)
        if self.reflexion is not None:
            # A verified small-model success becomes a regression case for replay.
            self.reflexion.record_success(query, tenant_id)
        # Only cache responses that produced no side effect; replaying a cached write is
        # not the same thing as replaying a cached read.
        if not trace.tool_committed or not self._is_irreversible(text):
            self.cache.put(query, text, tenant_id=tenant_id, model=self.small.model_name)
        return trace

    def _is_irreversible(self, text: str) -> bool:
        if self.commit_gate is None:
            return False
        from smart_router.tools.commit import ParseError, propose

        try:
            return propose(text, self.commit_gate.registry).irreversible
        except ParseError:
            return False

    def _commit(self, trace, text, rid):
        """Execute the proposed call. Everything expensive already happened."""
        from smart_router.tools.commit import ParseError, propose

        try:
            proposal = propose(text, self.commit_gate.registry, rid)
        except ParseError:
            return None  # not a tool call; nothing to execute
        outcome = self.commit_gate.commit(proposal)
        trace.tool_name = proposal.tool
        trace.tool_committed = outcome.committed
        trace.tool_attempts = outcome.attempts
        trace.tool_dry_run = outcome.dry_run_result is not None
        if not outcome.committed:
            trace.tool_failure = f"{outcome.category.value}: {outcome.detail}"
        return outcome

    def _credit(self, retrieved, *, succeeded: bool) -> None:
        """Credit assignment (plan SSC.1).

        Only rules that were actually retrieved above the similarity floor are credited.
        Incrementing every co-injected rule makes inert ones look good and computes the
        Wilson bound over contaminated counts.
        """
        if retrieved is None or self.rule_store is None:
            return
        for rule in retrieved.rules:
            self.rule_store.record_outcome(rule.rule_id, succeeded=succeeded, triggered=True)

    def _log_failure(
        self, trace, query, output, failing, vector, tenant_id,
        category=None, detail=None,
    ) -> None:
        """Feed the offline loop. Nothing on the request path waits for this."""
        if self.failure_log is None or vector is None:
            return
        if failing is None and category is None:
            return
        from smart_router.memory.failures import FailureRecord, fingerprint

        self.failure_log.record(
            FailureRecord(
                request_id=str(trace.request_id),
                tenant_id=tenant_id,
                query=query,
                query_fingerprint=fingerprint(query),
                category=(failing.category if failing else category)
                or FailureCategory.SILENT_SEMANTIC,
                verifier_id=failing.verifier_id if failing else "tools.commit",
                detail=(failing.detail if failing else detail) or "",
                rejected_output=output,
                vector=vector,
                preventable_by_construction=(
                    failing.preventable_by_construction if failing else False
                ),
                model=self.small.model_name,
            )
        )

    def _escalate(
        self, trace: RequestTrace, messages: list[dict], rejected: str,
        category: FailureCategory, detail: str, template: str = ESCALATION_TEMPLATE,
    ) -> RequestTrace:
        trace.escalated = True
        trace.decision = trace.decision.model_copy(
            update={"reason": RouteReason.VERIFIER_ESCALATION,
                    "tier": RouteTier.FRONTIER,
                    "selected_model": self.frontier.model_name}
        )
        escalation = list(messages) + [
            {
                "role": "user",
                "content": template.format(
                    output=rejected or "(no output produced)",
                    category=category.value,
                    detail=detail or "unspecified",
                ),
            }
        ]
        text, call, _ = self._call(self.frontier, escalation, RouteTier.FRONTIER)
        trace.calls.append(call)
        self._finish_frontier(trace, text)
        if self.reflexion is not None and not trace.surfaced_to_human:
            # The frontier's correction to a request the small model got wrong: raw
            # material for the exemplar arm. Captured in both modes, so switching arms
            # later starts with data. Policy-forced traffic never reaches this path --
            # sensitive requests bypass the small model -- so it never becomes an exemplar.
            self.reflexion.record_exemplar(
                trace.query, text, trace.tenant_id, str(trace.request_id)
            )
        return trace

    def _finish_frontier(self, trace: RequestTrace, text: str) -> RequestTrace:
        """Frontier output -- direct or escalated -- goes through the same commit gate.

        Bounded at one attempt: there is no tier above the frontier, so a tool failure on
        its call is surfaced rather than retried or escalated again.
        """
        if self.commit_gate is not None:
            outcome = self._commit(trace, text, trace.request_id)
            if outcome is not None and not outcome.committed:
                return self._surface(trace, outcome)
        trace.final_output = text
        trace.committed = True
        return trace

    @staticmethod
    def _surface(trace: RequestTrace, outcome) -> RequestTrace:
        """Return a tool outcome to the caller without escalating it."""
        trace.surfaced_to_human = True
        trace.final_output = f"{outcome.category.value}: {outcome.detail}"
        trace.committed = True
        return trace
