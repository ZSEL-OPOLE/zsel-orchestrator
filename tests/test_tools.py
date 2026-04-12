"""Tests for orchestrator tools — kubectl, git, shell, file read.

All tests focus on the security allowlist / blocklist logic without actually
executing system commands (mocked via AsyncMock where needed).
"""

import os
from unittest.mock import AsyncMock, patch

import pytest


# ── kubectl ───────────────────────────────────────────────────────────────


class TestToolKubectl:
    """Kubectl tool security and routing tests."""

    @pytest.mark.asyncio
    async def test_empty_command_returns_error(self) -> None:
        from src.tools import tool_kubectl

        result = await tool_kubectl({"command": ""})
        assert result.success is False
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_blocked_exec_pattern(self) -> None:
        from src.tools import tool_kubectl

        result = await tool_kubectl({"command": "exec -it mypod -- bash"})
        assert result.success is False
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_blocked_port_forward(self) -> None:
        from src.tools import tool_kubectl

        result = await tool_kubectl({"command": "port-forward svc/myapp 8080:8080"})
        assert result.success is False
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_unknown_verb_rejected(self) -> None:
        from src.tools import tool_kubectl

        result = await tool_kubectl({"command": "frobnicate pods"})
        assert result.success is False
        assert "Unknown verb" in result.error

    @pytest.mark.asyncio
    async def test_safe_get_invokes_kubectl(self) -> None:
        from src.tools import tool_kubectl

        with patch("src.tools._run_command", new=AsyncMock(return_value=("NAME\nfoo\n", "", 0))):
            result = await tool_kubectl({"command": "get pods -n default"})
        assert result.success is True
        assert "NAME" in result.output
        assert result.metadata["verb"] == "get"
        assert result.metadata["is_write"] is False

    @pytest.mark.asyncio
    async def test_write_verb_flagged_in_metadata(self) -> None:
        from src.tools import tool_kubectl

        with patch("src.tools._run_command", new=AsyncMock(return_value=("applied\n", "", 0))):
            result = await tool_kubectl({"command": "apply -f deployment.yaml"})
        assert result.metadata["is_write"] is True

    @pytest.mark.asyncio
    async def test_duration_recorded(self) -> None:
        from src.tools import tool_kubectl

        with patch("src.tools._run_command", new=AsyncMock(return_value=("ok\n", "", 0))):
            result = await tool_kubectl({"command": "get namespaces"})
        assert result.duration_ms >= 0


# ── git ───────────────────────────────────────────────────────────────────


class TestToolGit:
    """Git tool security tests."""

    @pytest.mark.asyncio
    async def test_missing_repo_dir_fails(self) -> None:
        from src.tools import tool_git

        result = await tool_git({"command": "status"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_force_push_blocked(self) -> None:
        from src.tools import tool_git

        result = await tool_git({"command": "push --force origin main", "repo_dir": "/workspace/repo"})
        assert result.success is False
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_reset_hard_blocked(self) -> None:
        from src.tools import tool_git

        result = await tool_git({"command": "reset --hard HEAD~5", "repo_dir": "/workspace/repo"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_safe_status_runs(self) -> None:
        from src.tools import tool_git

        with patch("src.tools._run_command", new=AsyncMock(return_value=("On branch main\n", "", 0))):
            result = await tool_git({"command": "status", "repo_dir": "/workspace/repo"})
        assert result.success is True
        assert "branch" in result.output


# ── shell ─────────────────────────────────────────────────────────────────


class TestToolShell:
    """Shell tool — blocklist and safe execution tests."""

    @pytest.mark.asyncio
    async def test_empty_command_fails(self) -> None:
        from src.tools import tool_shell

        result = await tool_shell({"command": ""})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_rm_rf_root_blocked(self) -> None:
        from src.tools import tool_shell

        result = await tool_shell({"command": "rm -rf /"})
        assert result.success is False
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fork_bomb_blocked(self) -> None:
        from src.tools import tool_shell

        result = await tool_shell({"command": ":(){ :|:& };:"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_safe_echo_runs(self) -> None:
        from src.tools import tool_shell

        with patch("src.tools._run_command", new=AsyncMock(return_value=("hello\n", "", 0))):
            result = await tool_shell({"command": "echo hello"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_invalid_quoting_fails_gracefully(self) -> None:
        from src.tools import tool_shell

        result = await tool_shell({"command": "echo 'unterminated"})
        assert result.success is False
        assert "Parse error" in result.error


# ── read_file ─────────────────────────────────────────────────────────────


class TestToolReadFile:
    """File read tool — path security tests."""

    @pytest.mark.asyncio
    async def test_no_path_fails(self) -> None:
        from src.tools import tool_read_file

        result = await tool_read_file({})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_path_outside_workspace_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORCH_WORKSPACE_ROOT", "/workspace")
        from src.tools import tool_read_file

        result = await tool_read_file({"path": "/etc/passwd"})
        assert result.success is False
        assert "outside workspace" in result.error

    @pytest.mark.asyncio
    async def test_valid_workspace_file_reads(self, tmp_path: os.PathLike, monkeypatch: pytest.MonkeyPatch) -> None:
        tmp = str(tmp_path)
        monkeypatch.setenv("ORCH_WORKSPACE_ROOT", tmp)
        test_file = os.path.join(tmp, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello world")
        from src.tools import tool_read_file

        result = await tool_read_file({"path": test_file})
        assert result.success is True
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_missing_file_fails_gracefully(self, tmp_path: os.PathLike, monkeypatch: pytest.MonkeyPatch) -> None:
        tmp = str(tmp_path)
        monkeypatch.setenv("ORCH_WORKSPACE_ROOT", tmp)
        from src.tools import tool_read_file

        result = await tool_read_file({"path": os.path.join(tmp, "nonexistent.txt")})
        assert result.success is False
