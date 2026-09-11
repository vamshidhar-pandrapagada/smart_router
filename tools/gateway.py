"""MCP gateway abstraction.

Every MCP server reaches the router through one gateway, so the router talks to a single
endpoint and the gateway routes to Jira, Outlook, Webex and the rest. That keeps server
topology out of the router entirely: nothing above this module knows which server backs
a tool, and adding a server changes no router code.

Two things the gateway owns that the router must not reimplement:

* **Inventory.** `list_tools()` is the source of truth for what exists. The registry --
  and therefore both the prompt block and the verifier -- derive from it, so they cannot
  drift from what the gateway will actually accept.
* **Error classification.** The gateway knows what a given server's 429 means. It raises
  the taxonomy in `tools/errors.py`; the router only decides retry vs escalate vs
  surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from smart_router.tools.errors import (
    ToolAuthError,
    ToolError,
    ToolNotFoundError,
    ToolTransientError,
    classify_status,
)
from smart_router.tools.registry import ToolSpec


@dataclass
class ToolResult:
    tool: str
    content: Any
    latency_ms: float = 0.0
    dry_run: bool = False
    idempotency_key: str | None = None
    #: Gateway-reported, when available. Lets cost be attributed per tool.
    cost_usd: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Empty is not an error, but it is worth verifying (see verify/result.py)."""
        if self.content is None:
            return True
        if isinstance(self.content, (list, dict, str)):
            return len(self.content) == 0
        return False


class MCPGateway(Protocol):
    def list_tools(self) -> Sequence[ToolSpec]: ...

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> ToolResult: ...


@dataclass
class MockMCPGateway:
    """In-process gateway for tests and the demo.

    Records every call, honours idempotency keys, and lets a handler raise the real
    error types so retry/escalate behaviour can be exercised without a live server.
    """

    specs: list[ToolSpec] = field(default_factory=list)
    #: tool name -> handler(arguments, dry_run) -> content, or raising a ToolError.
    handlers: dict[str, Callable[[dict[str, Any], bool], Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any], bool]] = field(default_factory=list)
    _committed: dict[str, ToolResult] = field(default_factory=dict)

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(self.specs)

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> ToolResult:
        spec = next((s for s in self.specs if s.name == tool), None)
        if spec is None:
            raise ToolNotFoundError(f"gateway does not expose {tool!r}", tool=tool)

        # An idempotency key that has already committed returns the original result
        # rather than executing again. This is what makes a retry safe on an
        # irreversible tool.
        if idempotency_key and not dry_run and idempotency_key in self._committed:
            return self._committed[idempotency_key]

        if dry_run and not spec.supports_dry_run:
            raise ToolError(f"{tool!r} does not support dry-run", tool=tool)

        started = time.perf_counter()
        self.calls.append((tool, dict(arguments), dry_run))
        handler = self.handlers.get(tool, lambda args, dr: {"ok": True})
        content = handler(arguments, dry_run)  # may raise a ToolError subclass

        result = ToolResult(
            tool=tool,
            content=content,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )
        if idempotency_key and not dry_run:
            self._committed[idempotency_key] = result
        return result


@dataclass
class CompositeGateway:
    """One gateway fronting several MCP servers.

    This is what a production MCP gateway looks like from the router's side: Jira,
    SharePoint, git and Webex all reachable through a single `call()`, with the gateway
    owning which backend serves a given tool. The router stays ignorant of topology.

    Tool names must be unique across backends; a collision is a configuration error and
    is raised at construction rather than silently resolved by ordering.
    """

    gateways: Sequence[MCPGateway]

    def __post_init__(self) -> None:
        self._owner: dict[str, MCPGateway] = {}
        for gw in self.gateways:
            for spec in gw.list_tools():
                if spec.name in self._owner:
                    raise ToolError(
                        f"tool {spec.name!r} is exposed by two backends; names must be unique"
                    )
                self._owner[spec.name] = gw

    def list_tools(self) -> Sequence[ToolSpec]:
        return [s for gw in self.gateways for s in gw.list_tools()]

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> ToolResult:
        gw = self._owner.get(tool)
        if gw is None:
            raise ToolNotFoundError(f"no backend exposes {tool!r}", tool=tool)
        return gw.call(tool, arguments, idempotency_key=idempotency_key,
                       dry_run=dry_run, timeout=timeout)


def http_status_handler(status: int, message: str = ""):
    """Handler that raises the taxonomy entry for an HTTP-ish status."""

    def _handler(_args: dict[str, Any], _dry_run: bool):
        raise classify_status(status)(message or f"status {status}", status=status)

    return _handler


__all__ = [
    "MCPGateway", "MockMCPGateway", "CompositeGateway", "ToolResult", "http_status_handler",
    "ToolError", "ToolAuthError", "ToolTransientError", "ToolNotFoundError",
]
