"""Tool registry -- the single source of truth for tool definitions.

Previously the prompt's tool block and the verifier's registry were two unconnected
lists. They could drift: a tool the model was told about but the verifier did not know,
or the reverse, each failing silently. Both now derive from this registry, which in turn
is populated from the MCP gateway's own advertised tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: JSON type name -> acceptable Python types.
JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arg_types: dict[str, str]
    required: tuple[str, ...] = ()
    description: str = ""
    #: Tri-state. True/False when the gateway declares it; None means UNDECLARED, in
    #: which case `is_irreversible()` infers conservatively. A plain False default would
    #: be fail-open: an unannotated `git.push` would skip the dry-run and the judge.
    irreversible: bool | None = None
    #: The gateway executes this without side effects when asked.
    supports_dry_run: bool = False
    #: MCP server behind the gateway. Informational; the gateway does the routing.
    server: str = ""
    #: Full JSON Schema for `arguments`, when the connector declares one. Required for
    #: any tool with nested structure -- `arg_types` can only check top-level types, so
    #: a hallucinated interior (wrong field, invented operator) would pass unnoticed.
    argument_schema: dict[str, Any] | None = None

    def schema_for_arguments(self) -> dict[str, Any]:
        """The declared schema, or one synthesized from the flat arg_types map."""
        if self.argument_schema:
            return self.argument_schema
        return {
            "type": "object",
            "properties": {k: {"type": v} for k, v in self.arg_types.items()},
            "required": list(self.required),
            "additionalProperties": False,
        }

    def is_irreversible(self, policy=None) -> bool:
        if self.irreversible is not None:
            return self.irreversible
        from smart_router.tools.reversibility import DEFAULT_POLICY

        return (policy or DEFAULT_POLICY).infer(self.name)[0]

    def json_schema(self) -> dict[str, Any]:
        """Schema for constrained decoding and for the prompt's tool block."""
        return {
            "type": "object",
            "properties": {
                "tool": {"const": self.name},
                "arguments": self.schema_for_arguments(),
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        }


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.tools, list):  # tolerate ToolRegistry([spec, ...])
            self.tools = {t.name: t for t in self.tools}

    def add(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def prompt_block(self) -> list[dict[str, Any]]:
        """What goes into the prompt's stable tool-definition block.

        Derived from the same registry the verifier checks against, so the two cannot
        drift. This belongs in the cacheable prefix: it is stable until the gateway's
        tool inventory changes.
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "arguments": s.schema_for_arguments(),
                "required": list(s.required),
                "irreversible": s.is_irreversible(),
            }
            for s in sorted(self.tools.values(), key=lambda t: t.name)
        ]

    def envelope_schema(self) -> dict[str, Any]:
        """Union schema over every tool, for constrained decoding."""
        if not self.tools:
            return {"type": "object"}
        return {
            "oneOf": [s.json_schema() for s in sorted(self.tools.values(), key=lambda t: t.name)]
        }

    def audit(self, policy=None) -> list[dict[str, Any]]:
        """Every tool whose reversibility was guessed rather than declared.

        Inference is a safety net. This is the list an operator should annotate.
        """
        from smart_router.tools.reversibility import DEFAULT_POLICY

        pol = policy or DEFAULT_POLICY
        out = []
        for spec in sorted(self.tools.values(), key=lambda t: t.name):
            if spec.irreversible is None:
                inferred, why = pol.infer(spec.name)
                out.append({"tool": spec.name, "inferred_irreversible": inferred, "why": why})
        return out

    @classmethod
    def from_gateway(cls, gateway) -> "ToolRegistry":
        """Populate from the MCP gateway's advertised inventory."""
        return cls({s.name: s for s in gateway.list_tools()})
