"""Rule contracts (plan SS2.5, SSC).

`RouterRule` is pramana's `EpistemicRule` grammar -- trigger / causal invariant /
deduction guideline -- plus the production lifecycle it needs to be safe: provenance,
tenant scope, a confidence interval rather than a raw ratio, and the model checkpoint the
evidence was gathered against.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from smart_router.schemas.verification import FailureCategory


class RuleState(str, Enum):
    PROPOSED = "PROPOSED"
    SHADOW = "SHADOW"
    PROMOTED = "PROMOTED"
    DEMOTED = "DEMOTED"
    RETIRED = "RETIRED"
    #: Failed a hygiene filter. Never served, kept for audit.
    QUARANTINED = "QUARANTINED"


#: Rules scoped here apply to every tenant and therefore require human review before
#: promotion: a GLOBAL rule is text one tenant's traffic wrote into everyone's prompt.
GLOBAL_SCOPE = "GLOBAL"


class RuleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Hashes, never verbatim user text (SSF).
    query_fingerprints: list[str] = Field(default_factory=list)
    failure_category: FailureCategory
    frontier_trace_ids: list[str] = Field(default_factory=list)
    verifier_trace_ids: list[str] = Field(default_factory=list)
    authored_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cluster_size: int = 1


class RouterRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(pattern=r"^RULE-[0-9]{3,}$")
    tenant_id: str
    trigger_condition: str
    causal_invariant: str
    deduction_guideline: str
    provenance: RuleProvenance
    state: RuleState = RuleState.PROPOSED

    #: Telemetry. `application_count` increments only when this rule's trigger actually
    #: matched -- see the credit-assignment note in memory/lifecycle.py.
    application_count: int = 0
    success_count: int = 0
    false_trigger_count: int = 0
    wilson_lower_bound: float = 0.0

    #: The small-model checkpoint the replay evidence was gathered against. A model
    #: upgrade invalidates it and forces re-validation (SSC.3 staleness).
    validated_against_model: str | None = None
    token_cost: int = 0
    human_reviewed: bool = False
    quarantine_reason: str | None = None

    def render(self) -> str:
        return (
            f"- {self.rule_id}\n"
            f"  TRIGGER: {self.trigger_condition}\n"
            f"  INVARIANT: {self.causal_invariant}\n"
            f"  DEDUCTION: {self.deduction_guideline}"
        )

    @property
    def trigger_text(self) -> str:
        """What the retrieval index embeds."""
        return f"{self.trigger_condition} {self.causal_invariant}"

    def is_servable(self) -> bool:
        return self.state is RuleState.PROMOTED
