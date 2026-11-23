"""Integration against a real MCP server (`mcp-server-git`).

Everything else in the suite runs against `MockMCPGateway`. These tests exist because a
mock cannot surface the two things that actually broke when this met a live server:

* tool names using the `git_commit` single-underscore convention, which the reversibility
  tokenizer did not split;
* gateway-supplied arguments (`repo_path`) being demanded from the model by the validator,
  because injection happened after validation.

Skipped when `uvx` or the `mcp` package is unavailable.
"""

import json
import shutil
import subprocess
from uuid import uuid4

import pytest

pytest.importorskip("mcp")
pytestmark = pytest.mark.live_mcp

if shutil.which("uvx") is None:
    pytest.skip("uvx not available", allow_module_level=True)

from smart_router.schemas.verification import FailureCategory
from smart_router.tools.commit import CommitGate, propose
from smart_router.tools.registry import ToolRegistry
from smart_router.tools.stdio_gateway import StdioMCPGateway, default_classifier
from smart_router.tools.errors import ToolAuthError, ToolSemanticError, ToolTransientError


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    path = tmp_path_factory.mktemp("repo")
    run = lambda *a: subprocess.run(a, cwd=path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "Test")
    (path / "a.txt").write_text("hello\n")
    run("git", "add", "a.txt")
    run("git", "commit", "-q", "-m", "initial")
    return str(path)


@pytest.fixture(scope="module")
def gateway(repo):
    gw = StdioMCPGateway(
        command="uvx",
        args=["--from", "mcp-server-git", "mcp-server-git", "--repository", repo],
        fixed_arguments={"repo_path": repo},
    )
    gw.start()
    yield gw
    gw.close()


@pytest.fixture(scope="module")
def gate(gateway):
    return CommitGate(gateway, ToolRegistry.from_gateway(gateway), dry_run_irreversible=False)


def _call(gate, tool, **args):
    return gate.commit(propose(json.dumps({"tool": tool, "arguments": args}),
                               gate.registry, uuid4()))


def test_tools_are_discovered_from_the_live_server(gateway):
    names = {s.name for s in gateway.list_tools()}
    assert {"git_log", "git_status", "git_commit", "git_diff"} <= names


def test_gateway_supplied_arguments_are_hidden_from_the_contract(gate):
    """repo_path is injected at call time, so the model must not be asked for it."""
    spec = gate.registry.get("git_log")
    assert "repo_path" not in spec.required
    assert "repo_path" not in (spec.schema_for_arguments().get("properties") or {})


def test_reversibility_is_inferred_correctly_for_real_git_tool_names(gate):
    reg = gate.registry
    assert reg.get("git_log").is_irreversible() is False
    assert reg.get("git_status").is_irreversible() is False
    assert reg.get("git_diff").is_irreversible() is False
    assert reg.get("git_commit").is_irreversible() is True
    assert reg.get("git_add").is_irreversible() is True
    assert reg.get("git_create_branch").is_irreversible() is True


def test_read_tools_execute_against_the_live_server(gate):
    out = _call(gate, "git_log", max_count=2)
    assert out.committed and "Commit" in str(out.result.content)


def test_bad_argument_type_is_caught_before_the_server_is_contacted(gate):
    out = _call(gate, "git_log", max_count="two")
    assert out.category is FailureCategory.TOOL_ARG_INVALID
    assert "expected integer" in out.detail


def test_a_git_error_maps_to_TOOL_SEMANTIC_and_escalates(gate):
    """git reports exit codes and prose, not HTTP statuses."""
    out = _call(gate, "git_show", revision="does-not-exist")
    assert out.category is FailureCategory.TOOL_SEMANTIC
    assert out.should_escalate
    assert "did not resolve" in out.detail


def test_a_tool_the_server_does_not_expose_is_rejected(gate):
    out = _call(gate, "git_push", remote="origin")
    assert out.category is FailureCategory.TOOL_NOT_FOUND


def test_mcp_has_no_dry_run_and_the_gateway_says_so_rather_than_executing(gateway):
    from smart_router.tools.errors import ToolError

    with pytest.raises(ToolError, match="no dry-run"):
        gateway.call("git_log", {"max_count": 1}, dry_run=True)


@pytest.mark.parametrize("message,expected", [
    ("Permission denied (publickey)", ToolAuthError),
    ("could not read Username for 'https://github.com'", ToolAuthError),
    ("fatal: could not resolve host: github.com", ToolTransientError),
    ("Connection timed out", ToolTransientError),
    ("Ref 'nope' did not resolve to an object", ToolSemanticError),
    ("nothing to commit, working tree clean", ToolSemanticError),
])
def test_git_prose_errors_classify_without_http_statuses(message, expected):
    assert default_classifier("git_x", message) is expected
