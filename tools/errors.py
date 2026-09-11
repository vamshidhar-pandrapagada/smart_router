"""Tool-execution error taxonomy.

Distinct from provider errors on purpose. `ProviderTimeout` means the *LLM endpoint*
timed out; `ToolTransientError` means the *MCP server behind the gateway* did. Same
words, different failure domain, different correct response -- conflating them sends
Jira outages to a frontier model.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base for anything the MCP gateway reports."""

    def __init__(self, message: str, *, tool: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.tool = tool
        self.status = status


class ToolTransientError(ToolError):
    """503, 429, socket drop, gateway timeout.

    Retry the *tool*. Do not escalate: a frontier model calling the same dead endpoint
    produces the same error at ~90x the cost.
    """


class ToolSemanticError(ToolError):
    """400, 404, 422 -- the call was well-formed but wrong.

    'Project PROJ not found', malformed JQL, an id that does not exist. The small model
    lacked the context or reasoning to get it right and will reproduce the error on
    retry, so escalate with the gateway's own error text attached.
    """


class ToolAuthError(ToolError):
    """401, 403.

    Neither retried nor escalated. A frontier model cannot fix a missing token or an
    unauthorized scope, and spending a frontier call to rediscover that wastes money and
    delays the only useful outcome: surfacing the credential problem to a human.
    """


class ToolNotFoundError(ToolError):
    """The gateway does not expose this tool. A registry/prompt drift bug, not a model bug."""


#: HTTP-ish status to exception. Gateways vary; override via MCPGateway.classify.
def classify_status(status: int) -> type[ToolError]:
    if status in (401, 403):
        return ToolAuthError
    if status in (408, 425, 429, 500, 502, 503, 504):
        return ToolTransientError
    if status == 404:
        return ToolSemanticError
    if 400 <= status < 500:
        return ToolSemanticError
    return ToolTransientError
