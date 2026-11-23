"""Recursive JSON Schema validation for tool arguments.

The flat `arg_types` map was adequate for simple tools -- `{"jql": "string"}` -- but
useless for anything with structure. A SharePoint search takes a nested filter object, and
`{"filter": "object"} + isinstance(dict)` accepts a hallucinated interior: wrong field
name, invented operator, wrong value type. For complex connectors that is most of the
payload surface left unchecked.

This validates the shape the connector actually declares, so a nested error is caught
before the gateway is contacted -- which is also exactly the schema handed to the model
for constrained decoding, so the same contract prevents and detects.

Supports the subset MCP servers use in practice: type, properties, required,
additionalProperties, enum, const, items, minItems/maxItems, minimum/maximum, pattern,
oneOf/anyOf. Unknown keywords are ignored rather than rejected.
"""

from __future__ import annotations

import re
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_ok(value: Any, declared: str) -> bool:
    expected = _TYPES.get(declared)
    if expected is None:
        return True
    # bool is a subclass of int in Python; a schema asking for a number does not mean
    # True is acceptable.
    if declared in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> tuple[bool, str]:
    """Returns (ok, detail). `detail` names the exact path that failed."""
    if not schema:
        return True, ""

    for key in ("oneOf", "anyOf"):
        if key in schema:
            for branch in schema[key]:
                ok, _ = validate_schema(value, branch, path)
                if ok:
                    break
            else:
                return False, f"{path} matches no {key} branch"

    if "const" in schema and value != schema["const"]:
        return False, f"{path} must be {schema['const']!r}, got {value!r}"

    if "enum" in schema and value not in schema["enum"]:
        return False, f"{path} must be one of {schema['enum']}, got {value!r}"

    declared = schema.get("type")
    if isinstance(declared, str) and not _type_ok(value, declared):
        return False, f"{path} expected {declared}, got {type(value).__name__}"
    if isinstance(declared, list) and not any(_type_ok(value, d) for d in declared):
        return False, f"{path} expected one of {declared}, got {type(value).__name__}"

    if isinstance(value, str):
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            return False, f"{path} does not match pattern {pattern!r}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False, f"{path} below minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return False, f"{path} above maximum {schema['maximum']}"

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False, f"{path} needs at least {schema['minItems']} items"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False, f"{path} allows at most {schema['maxItems']} items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                ok, detail = validate_schema(item, item_schema, f"{path}[{i}]")
                if not ok:
                    return False, detail

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for req in schema.get("required", []):
            if req not in value:
                return False, f"{path} missing required field {req!r}"
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    known = ", ".join(sorted(properties)) or "none"
                    return False, f"{path} has unknown field {key!r} (known: {known})"
        for key, sub in properties.items():
            if key in value and isinstance(sub, dict):
                ok, detail = validate_schema(value[key], sub, f"{path}.{key}")
                if not ok:
                    return False, detail

    return True, ""
