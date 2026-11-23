"""Entity grounding: does this call name something the conversation ever mentioned?

Raised in review of the architecture article, and correctly: schema validation catches
malformed structure, not wrong intent. A payload can carry the right types and every
required field and still target the wrong account, order or ticket, because the model
misread which entity the user meant. "Cancel order 4471" becoming `order_id=4417` passes
every structural check and then fires a destructive write.

This closes part of that gap deterministically and for free, before the judge: pull the
identifier-shaped arguments out of a proposed call and confirm each one actually appears
in the conversation -- in what the user said, or in what an earlier tool returned. An
identifier the model produced from nowhere is the signature of a misread entity.

What it cannot do, stated plainly:

* **It cannot tell a right identifier from a wrong one that was also mentioned.** If two
  order numbers are in play and the model picks the other one, both are grounded. That
  needs the judge, or confirmation.
* **It flags legitimate indirection.** "Cancel my latest order" gives the model an id from
  a tool result -- grounded, fine -- but an id composed from a user's own records is not.
  Hence a flag that escalates rather than a hard block.

So it is a narrowing, not a solution: it catches invented and transposed identifiers,
which is the cheap half of the problem.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from smart_router.schemas.verification import FailureCategory, VerifierClass, VerifierResult
from smart_router.verify.base import VerificationContext

#: Argument names that denote an entity rather than free text.
ENTITY_FIELD = re.compile(
    r"(^|_)(id|ids|key|ref|reference|number|no|code)$"
    r"|(account|order|ticket|issue|invoice|customer|client|user|member|room|channel"
    r"|project|repo|repository|branch|file|path|record|entity|target|recipient)",
    re.IGNORECASE,
)

#: Values that look like identifiers rather than prose.
IDENTIFIER_SHAPES = (
    re.compile(r"^[0-9]{3,}$"),                                  # 4471
    re.compile(r"^[A-Za-z]{2,}[-_][0-9]{1,}$"),                  # ATL-14, ORD_99
    re.compile(r"^[0-9a-fA-F]{8,}$"),                            # hex ids
    re.compile(r"^[0-9a-fA-F-]{32,}$"),                          # uuids
    re.compile(r"^[A-Z]{2,}[0-9]{2,}$"),                         # AC4471
)

_WORD = re.compile(r"[A-Za-z0-9_-]+")


def looks_like_identifier(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return abs(value) >= 100
    if not isinstance(value, str):
        return False
    token = value.strip()
    if not token or " " in token:          # prose is not an identifier
        return False
    return any(p.match(token) for p in IDENTIFIER_SHAPES)


def extract_entities(arguments: dict[str, Any], _path: str = "") -> list[tuple[str, str]]:
    """(path, value) for every identifier-shaped argument, nested fields included."""
    found: list[tuple[str, str]] = []
    for key, value in arguments.items():
        path = f"{_path}.{key}" if _path else key
        if isinstance(value, dict):
            found.extend(extract_entities(value, path))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    found.extend(extract_entities(item, f"{path}[{i}]"))
                elif looks_like_identifier(item):
                    found.append((f"{path}[{i}]", str(item)))
        elif looks_like_identifier(value):
            # A name that denotes an entity, or a value too distinctive to be anything else.
            if ENTITY_FIELD.search(key) or not isinstance(value, int):
                found.append((path, str(value)))
    return found


@dataclass
class EntityGroundingVerifier:
    """Every identifier in the call must appear in the conversation."""

    verifier_id: str = "free.entity_grounding"
    verifier_class: VerifierClass = VerifierClass.FREE
    #: Identifiers below this length are too collision-prone to check usefully.
    min_length: int = 3
    #: Argument paths never checked -- free text that may legitimately contain numbers.
    ignore_paths: tuple[str, ...] = field(default=("text", "body", "message", "markdown",
                                                   "content", "comment", "description"))

    def _grounding_tokens(self, ctx: VerificationContext) -> set[str]:
        sources = [ctx.query, *ctx.grounding]
        return {t.lower() for s in sources for t in _WORD.findall(str(s))}

    def check(self, ctx: VerificationContext) -> VerifierResult:
        t0 = time.perf_counter()
        ms = lambda: (time.perf_counter() - t0) * 1000  # noqa: E731

        try:
            payload = json.loads(ctx.response.text)
        except json.JSONDecodeError:
            return VerifierResult(verifier_id=self.verifier_id,
                                  verifier_class=VerifierClass.FREE, passed=True,
                                  latency_ms=ms())
        if not isinstance(payload, dict) or not isinstance(payload.get("arguments"), dict):
            return VerifierResult(verifier_id=self.verifier_id,
                                  verifier_class=VerifierClass.FREE, passed=True,
                                  latency_ms=ms())

        grounded = self._grounding_tokens(ctx)
        for path, value in extract_entities(payload["arguments"]):
            if any(path.split(".")[-1].split("[")[0] == p for p in self.ignore_paths):
                continue
            if len(value) < self.min_length:
                continue
            if value.lower() in grounded:
                continue
            return VerifierResult(
                verifier_id=self.verifier_id, verifier_class=VerifierClass.FREE,
                passed=False, category=FailureCategory.UNGROUNDED_ENTITY,
                detail=(f"{path}={value!r} appears nowhere in the request or in earlier "
                        "tool results; the model may have misread which entity was meant"),
                latency_ms=ms(),
                # No decoder constraint prevents a well-formed reference to the wrong
                # entity, so this belongs to the residue, not to P.
                preventable_by_construction=False,
            )

        return VerifierResult(verifier_id=self.verifier_id,
                              verifier_class=VerifierClass.FREE, passed=True, latency_ms=ms())
