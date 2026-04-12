"""Tests for orchestrator agents — data models, registry, message bus, and tool extraction."""

import time
import uuid

import pytest

from src.agents import (
    AgentDefinition,
    AgentMessage,
    AgentRegistry,
    AgentRole,
    MessageType,
    Task,
    TaskStep,
)


# ── Data model tests ──────────────────────────────────────────────────────


class TestAgentRole:
    def test_all_roles_are_str_enum(self) -> None:
        for role in AgentRole:
            assert isinstance(role.value, str)

    def test_required_roles_exist(self) -> None:
        roles = {r.value for r in AgentRole}
        assert "sre" in roles
        assert "backend_dev" in roles
        assert "security" in roles
        assert "tester" in roles

    def test_roles_are_unique(self) -> None:
        values = [r.value for r in AgentRole]
        assert len(values) == len(set(values))


class TestMessageType:
    def test_all_types_are_str_enum(self) -> None:
        for mt in MessageType:
            assert isinstance(mt.value, str)

    def test_required_types_exist(self) -> None:
        types = {t.value for t in MessageType}
        assert "task" in types
        assert "answer" in types
        assert "handoff" in types


class TestAgentMessage:
    def test_auto_id_generated(self) -> None:
        msg = AgentMessage(from_agent="alpha", to_agent="beta", content="hi")
        assert msg.id != ""
        # Should be a valid UUID
        uuid.UUID(msg.id)

    def test_auto_timestamp_set(self) -> None:
        before = time.time()
        msg = AgentMessage()
        assert msg.timestamp >= before

    def test_explicit_id_preserved(self) -> None:
        msg = AgentMessage(id="my-id-123")
        assert msg.id == "my-id-123"

    def test_default_type_is_task(self) -> None:
        msg = AgentMessage()
        assert msg.msg_type == MessageType.TASK


class TestTaskStep:
    def test_auto_id_generated(self) -> None:
        step = TaskStep(description="Deploy app")
        uuid.UUID(step.id)

    def test_default_status_pending(self) -> None:
        step = TaskStep()
        assert step.status == "pending"

    def test_dependencies_empty_by_default(self) -> None:
        step = TaskStep()
        assert step.dependencies == []


class TestTask:
    def test_auto_id_generated(self) -> None:
        task = Task(description="Fix the cluster")
        uuid.UUID(task.id)

    def test_default_requester_is_user(self) -> None:
        task = Task()
        assert task.requester == "user"

    def test_steps_empty_by_default(self) -> None:
        task = Task()
        assert task.steps == []

    def test_default_status_pending(self) -> None:
        task = Task()
        assert task.status == "pending"


# ── AgentDefinition ───────────────────────────────────────────────────────


class TestAgentDefinition:
    def test_create_minimal(self) -> None:
        defn = AgentDefinition(name="test-sre", role=AgentRole.SRE, system_prompt="You are an SRE.")
        assert defn.name == "test-sre"
        assert defn.role == AgentRole.SRE
        assert defn.preferred_model == "qwen3.5:9b"  # default

    def test_custom_model(self) -> None:
        defn = AgentDefinition(
            name="gemma-agent", role=AgentRole.BACKEND_DEV, system_prompt="Dev.", preferred_model="gemma4:e4b"
        )
        assert defn.preferred_model == "gemma4:e4b"

    def test_capabilities_empty_by_default(self) -> None:
        defn = AgentDefinition(name="x", role=AgentRole.TESTER, system_prompt=".")
        assert defn.capabilities == []
        assert defn.tools == []


# ── AgentRegistry ─────────────────────────────────────────────────────────


class TestAgentRegistry:
    def _make_defn(self, name: str = "agent-1", role: AgentRole = AgentRole.SRE) -> AgentDefinition:
        return AgentDefinition(name=name, role=role, system_prompt=f"You are {name}.", capabilities=["kubectl"])

    def test_register_and_get(self) -> None:
        reg = AgentRegistry()
        defn = self._make_defn()
        agent = reg.register(defn)
        assert reg.get("agent-1") is agent

    def test_get_nonexistent_returns_none(self) -> None:
        reg = AgentRegistry()
        assert reg.get("nobody") is None

    def test_get_by_role(self) -> None:
        reg = AgentRegistry()
        reg.register(self._make_defn("sre-1", AgentRole.SRE))
        reg.register(self._make_defn("sre-2", AgentRole.SRE))
        reg.register(self._make_defn("dev-1", AgentRole.BACKEND_DEV))
        sres = reg.get_by_role(AgentRole.SRE)
        assert len(sres) == 2
        devs = reg.get_by_role(AgentRole.BACKEND_DEV)
        assert len(devs) == 1

    def test_get_by_capability(self) -> None:
        reg = AgentRegistry()
        defn1 = AgentDefinition(name="cap-a", role=AgentRole.SRE, system_prompt=".", capabilities=["kubectl", "git"])
        defn2 = AgentDefinition(name="cap-b", role=AgentRole.DBA, system_prompt=".", capabilities=["psql"])
        reg.register(defn1)
        reg.register(defn2)
        assert len(reg.get_by_capability("kubectl")) == 1
        assert len(reg.get_by_capability("psql")) == 1
        assert len(reg.get_by_capability("nonexistent")) == 0

    def test_count_property(self) -> None:
        reg = AgentRegistry()
        assert reg.count == 0
        reg.register(self._make_defn("a"))
        assert reg.count == 1
        reg.register(self._make_defn("b"))
        assert reg.count == 2

    def test_list_agents_returns_all(self) -> None:
        reg = AgentRegistry()
        reg.register(self._make_defn("x"))
        reg.register(self._make_defn("y"))
        agents = reg.list_agents()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "x" in names
        assert "y" in names

    def test_list_agents_includes_capabilities(self) -> None:
        reg = AgentRegistry()
        reg.register(self._make_defn("z"))
        agents = reg.list_agents()
        assert "capabilities" in agents[0]
        assert "kubectl" in agents[0]["capabilities"]


# ── Agent._extract_tool_calls ──────────────────────────────────────────────


class TestAgentToolExtraction:
    """Test the regex-based tool call extraction — no LLM needed."""

    def _make_agent(self) -> object:
        from src.agents import Agent

        defn = AgentDefinition(name="tester", role=AgentRole.TESTER, system_prompt=".")
        return Agent(defn)

    def test_no_tool_calls_returns_empty(self) -> None:
        agent = self._make_agent()
        result = agent._extract_tool_calls("Just a normal response with no tools.")
        assert result == []

    def test_single_tool_call_extracted(self) -> None:
        agent = self._make_agent()
        text = '```tool:kubectl\n{"command": "get pods"}\n```'
        calls = agent._extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "kubectl"
        assert calls[0]["args"]["command"] == "get pods"

    def test_multiple_tool_calls_extracted(self) -> None:
        agent = self._make_agent()
        text = (
            'First call:\n```tool:kubectl\n{"command": "get pods"}\n```\n'
            'Second call:\n```tool:shell\n{"command": "echo hi"}\n```'
        )
        calls = agent._extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["tool"] == "kubectl"
        assert calls[1]["tool"] == "shell"

    def test_invalid_json_falls_back_to_command_key(self) -> None:
        agent = self._make_agent()
        text = "```tool:shell\nget pods --all-namespaces\n```"
        calls = agent._extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["args"]["command"] == "get pods --all-namespaces"
