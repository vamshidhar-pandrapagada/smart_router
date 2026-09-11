"""Live MCP gateway over stdio.

Implements the `MCPGateway` protocol against a real MCP server process, so the router
drives `mcp-server-git`, a SharePoint connector or anything else through exactly the same
interface it uses for `MockMCPGateway`.

Two things this had to solve that the mock did not:

**Sync over async.** The MCP client is asyncio; the router is synchronous, because a
verifier ladder that has to be awaited infects every caller. A single background thread
owns the event loop and the session for the gateway's lifetime, and `call()` hands work to
it. One process, one session -- not one per call.

**Errors are not HTTP.** MCP servers return failures as `isError` content, and git reports
"not a valid ref", "nothing to commit", "non-fast-forward" -- not 404s. Mapping that text
into the taxonomy is the gateway's job, which is why `classify` is pluggable: every server
words its failures differently and the router must not learn any of them.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from smart_router.tools.errors import (
    ToolAuthError,
    ToolError,
    ToolNotFoundError,
    ToolSemanticError,
    ToolTransientError,
)
from smart_router.tools.gateway import ToolResult
from smart_router.tools.registry import ToolSpec

#: (tool, error text) -> a ToolError subclass.
Classifier = Callable[[str, str], type[ToolError]]

_AUTH = re.compile(
    r"permission denied|authentication failed|not authorized|forbidden|"
    r"could not read Username|access denied|401|403",
    re.IGNORECASE,
)
_TRANSIENT = re.compile(
    r"could not resolve host|connection (refused|reset|timed out)|timeout|"
    r"temporarily unavailable|network is unreachable|remote end hung up|503|502|429",
    re.IGNORECASE,
)


def default_classifier(_tool: str, message: str) -> type[ToolError]:
    """Text-based classification for servers that do not speak HTTP.

    Order matters: auth first, because "permission denied" often also mentions a
    connection. Anything unmatched is treated as semantic -- a capability failure the
    small model will reproduce -- rather than transient, so it escalates once instead of
    being retried into the same wall.
    """
    if _AUTH.search(message):
        return ToolAuthError
    if _TRANSIENT.search(message):
        return ToolTransientError
    return ToolSemanticError


@dataclass
class StdioMCPGateway:
    command: str
    args: Sequence[str] = ()
    env: dict[str, str] | None = None
    classify: Classifier = default_classifier
    #: Arguments injected into every call (e.g. repo_path for mcp-server-git), so the
    #: model never has to produce infrastructure detail it has no business knowing.
    fixed_arguments: dict[str, Any] = field(default_factory=dict)
    startup_timeout: float = 60.0

    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _session: Any = field(default=None, init=False, repr=False)
    _ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _stop: Any = field(default=None, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _specs: list[ToolSpec] = field(default_factory=list, init=False, repr=False)

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "StdioMCPGateway":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mcp-stdio", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.startup_timeout):
            raise ToolError(f"MCP server {self.command!r} did not start in time")
        if self._error is not None:
            raise ToolError(f"MCP server {self.command!r} failed to start: {self._error}")

    def _run(self) -> None:
        async def main() -> None:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self.command, args=list(self.args), env=self.env
            )
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        self._specs = [
                            _to_spec(t, hidden=set(self.fixed_arguments))
                            for t in (await session.list_tools()).tools
                        ]
                        self._stop = asyncio.Event()
                        self._ready.set()
                        await self._stop.wait()
            except BaseException as exc:  # noqa: BLE001 - reported to start()
                self._error = exc
                self._ready.set()

        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()

    def close(self) -> None:
        if self._loop and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=10)
        self._thread = None

    def _submit(self, coro) -> Any:
        if self._loop is None:
            raise ToolError("gateway not started")
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # -- MCPGateway protocol -------------------------------------------------

    def list_tools(self) -> Sequence[ToolSpec]:
        if self._thread is None:
            self.start()
        return list(self._specs)

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        dry_run: bool = False,
        timeout: float | None = None,
    ) -> ToolResult:
        if self._thread is None:
            self.start()
        if not any(s.name == tool for s in self._specs):
            raise ToolNotFoundError(f"server does not expose {tool!r}", tool=tool)
        if dry_run:
            # MCP has no dry-run concept. Refusing is the honest answer: silently
            # executing a call the gate believed was a rehearsal would be far worse.
            raise ToolError(f"{tool!r}: MCP has no dry-run; gate on approval instead", tool=tool)

        payload = {**self.fixed_arguments, **arguments}
        started = time.perf_counter()
        result = self._submit(self._session.call_tool(tool, payload))
        latency = (time.perf_counter() - started) * 1000.0

        text = _content_text(result)
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise self.classify(tool, text)(text.strip()[:400] or "tool reported an error", tool=tool)

        return ToolResult(
            tool=tool, content=text, latency_ms=latency,
            dry_run=False, idempotency_key=idempotency_key,
        )


def _content_text(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _to_spec(tool: Any, hidden: set[str] | None = None) -> ToolSpec:
    """MCP Tool -> ToolSpec. The server's own schema becomes the validated contract.

    Gateway-supplied arguments are stripped from the exposed schema. `fixed_arguments`
    are injected at call time, so leaving them in would make the validator demand
    infrastructure detail (`repo_path`) from the model -- rejecting every well-formed
    call. Hiding them here keeps one contract for the prompt, the decoder and the
    validator alike.
    """
    hidden = hidden or set()
    schema = dict(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {})
    properties = {k: v for k, v in (schema.get("properties") or {}).items() if k not in hidden}
    if schema:
        schema = {
            **schema,
            "properties": properties,
            "required": [r for r in (schema.get("required") or ()) if r not in hidden],
        }
    arg_types = {
        k: (v.get("type") if isinstance(v, dict) else "string") or "string"
        for k, v in properties.items()
    }
    return ToolSpec(
        name=tool.name,
        arg_types=arg_types,
        required=tuple(schema.get("required") or ()),
        description=(getattr(tool, "description", "") or "").strip()[:200],
        argument_schema=schema or None,
        # Reversibility is left undeclared: MCP has no field for it, so the registry's
        # conservative inference decides and `audit()` surfaces every guess.
        irreversible=None,
    )
