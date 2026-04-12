"""
ZSEL Orchestrator — główny mozg systemu.

Przepływ:
  1. Użytkownik przesyła wysokopoziomowe zadanie
  2. Planner dekomponuje je na kroki z przypisanymi agentami
  3. Executor uruchamia kroki (z uwzględnieniem zależności)
  4. Self-improvement loop zapisuje wyniki do Qdrant
  5. Kolejne podobne zadania korzystają z historii

Wzorce:
  - Plan-Execute (OpenAI Swarm style)
  - Agent handoff (CrewAI style)
  - Knowledge grounding (RAG na każdym etapie)
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

from .agents import (
    AgentFactory,
    AgentMessage,
    AgentRegistry,
    AgentRole,
    MessageBus,
    MessageType,
    Task,
    TaskStep,
    register_builtin_agents,
)
from .config import get_settings
from .knowledge import ErrorEntry, KnowledgeEntry, get_knowledge_base
from .llm import get_llm

logger = logging.getLogger(__name__)


# ── Task Planner ──────────────────────────────────────────────────────────


class TaskPlanner:
    """
    Dekomponuje zadania na kroki.

    Używa LLM do stworzenia planu (lista kroków z agentami).
    Wzbogaca prompt o podobne zadania z historii (RAG).
    """

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    async def plan(self, task: Task) -> list[TaskStep]:
        """Create execution plan for a task."""
        llm = get_llm()
        kb = get_knowledge_base()

        # RAG: find similar past tasks
        similar = await kb.find_similar_tasks(task.description, limit=3)
        similar_context = ""
        if similar:
            similar_context = "\n## Podobne poprzednie zadania:\n"
            for s in similar:
                status = "✅" if s["success"] else "❌"
                similar_context += f"{status} {s['description'][:100]}\n   Agenty: {', '.join(s['agents_used'])}\n"
                if s.get("learnings"):
                    similar_context += f"   Learnings: {s['learnings'][:150]}\n"

        # RAG: find relevant infra knowledge
        knowledge = await kb.search_knowledge(task.description, limit=5)
        infra_context = ""
        if knowledge:
            infra_context = "\n## Relevantna wiedza z bazy:\n"
            for k in knowledge:
                infra_context += f"- [{k.category}] {k.content[:200]}\n"

        available_agents = "\n".join(f"- {a['name']}: {a['description']} (capabilities: {', '.join(a['capabilities'])})" for a in self._registry.list_agents())

        prompt = f"""Jesteś Planner w autonomicznej firmie AI. Stwórz PRECYZYJNY plan wykonania poniższego zadania.

## Zadanie:
{task.description}

## Dostępni agenci:
{available_agents}
{similar_context}
{infra_context}

## INSTRUKCJE:
1. Rozbij zadanie na konkretne, atomowe kroki
2. Każdy krok musi być przypisany do JEDNEGO agenta
3. Kroki mogą mieć zależności (depends_on: lista ID kroków które muszą być skończone)
4. Pierwsze kroki = diagnostyka/research, środkowe = implementacja, ostatnie = weryfikacja
5. Zawsze dodaj krok weryfikacyjny na końcu
6. Max 8 kroków (unikaj over-engineeringu)

## OUTPUT — TYLKO VALID JSON (bez markdown):
{{
  "steps": [
    {{
      "id": "step-1",
      "description": "Konkretny opis co zrobić",
      "agent": "nazwa_agenta",
      "tools_needed": ["kubectl", "git"],
      "depends_on": []
    }},
    {{
      "id": "step-2",
      "description": "...",
      "agent": "nazwa_agenta",
      "tools_needed": [],
      "depends_on": ["step-1"]
    }}
  ],
  "reasoning": "Krótkie uzasadnienie planu"
}}"""

        response = await llm.generate(prompt, model=llm._think_model, think=True, temperature=0.4, max_tokens=4096)

        # Parse JSON from response
        steps = self._parse_plan(response)
        logger.info("Planner created %d steps for task: %s", len(steps), task.description[:60])
        return steps

    def _parse_plan(self, response: str) -> list[TaskStep]:
        """Extract steps from LLM response."""
        # Try to extract JSON from response
        import re

        json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', response)
        if not json_match:
            # Fallback: single step
            return [
                TaskStep(
                    description="Wykonaj zadanie bezpośrednio",
                    agent="sre",
                    tools_needed=["kubectl"],
                )
            ]
        try:
            data = json.loads(json_match.group())
            steps = []
            for s in data.get("steps", []):
                steps.append(
                    TaskStep(
                        id=s.get("id", str(uuid.uuid4())),
                        description=s.get("description", ""),
                        agent=s.get("agent", "sre"),
                        tools_needed=s.get("tools_needed", []),
                        dependencies=s.get("depends_on", []),
                    )
                )
            return steps
        except json.JSONDecodeError as e:
            logger.error("Failed to parse plan JSON: %s", e)
            return [TaskStep(description=response[:500], agent="sre")]


# ── Task Executor ─────────────────────────────────────────────────────────


class TaskExecutor:
    """
    Wykonuje plan zadania.

    - Respektuje zależności między krokami
    - Uruchamia niezależne kroki równolegle
    - Przekazuje wyniki poprzednich kroków jako kontekst
    - Obsługuje błędy i retry
    """

    def __init__(self, registry: AgentRegistry, bus: MessageBus):
        self._registry = registry
        self._bus = bus

    async def execute_plan(self, task: Task, *, on_update: Any = None) -> str:
        """Execute all steps, respecting dependencies. Returns final result."""
        completed: dict[str, str] = {}  # step_id -> result
        failed: dict[str, str] = {}  # step_id -> error

        task.status = "executing"
        all_results = []

        # Execute steps respecting dependencies (topological-ish)
        max_rounds = 10
        pending = list(task.steps)

        for _ in range(max_rounds):
            if not pending:
                break

            # Find steps that can run now (all deps completed)
            ready = [s for s in pending if all(dep in completed for dep in s.dependencies) and s.id not in failed]

            if not ready:
                # All remaining steps have failed deps
                for s in pending:
                    if not all(dep in completed for dep in s.dependencies):
                        s.status = "failed"
                        s.error = "Dependency failed"
                        failed[s.id] = "Dependency failed"
                break

            # Execute ready steps in parallel
            results = await asyncio.gather(
                *[self._execute_step(s, task, completed) for s in ready],
                return_exceptions=True,
            )

            for step, result in zip(ready, results):
                pending.remove(step)
                if isinstance(result, Exception):
                    step.status = "failed"
                    step.error = str(result)
                    failed[step.id] = str(result)
                    logger.error("Step %s failed: %s", step.id, result)
                elif result.startswith("ERROR:"):
                    step.status = "failed"
                    step.error = result[6:]
                    failed[step.id] = result[6:]
                else:
                    step.status = "completed"
                    step.result = result
                    completed[step.id] = result

                all_results.append(f"## Krok: {step.description}\n**Agent**: {step.agent}\n**Status**: {step.status}\n{step.result or step.error}")

                if on_update:
                    await on_update(step)

        # Aggregate final result
        success_count = sum(1 for s in task.steps if s.status == "completed")
        fail_count = sum(1 for s in task.steps if s.status == "failed")

        if fail_count == 0:
            task.status = "completed"
        elif success_count > 0:
            task.status = "partial"
        else:
            task.status = "failed"

        return "\n\n---\n\n".join(all_results)

    async def _execute_step(self, step: TaskStep, task: Task, completed: dict[str, str]) -> str:
        """Execute a single step with the assigned agent."""
        step.status = "running"
        step.started_at = time.time()

        agent = self._registry.get(step.agent)
        if not agent:
            # Try to get by role
            agents = self._registry.get_by_role(AgentRole(step.agent))
            agent = agents[0] if agents else None

        if not agent:
            return f"ERROR: Agent '{step.agent}' not found"

        # Build context from completed steps
        context_parts = []
        if completed:
            context_parts.append("## Wyniki poprzednich kroków:")
            for dep_id in step.dependencies:
                if dep_id in completed:
                    context_parts.append(f"### {dep_id}:\n{completed[dep_id][:500]}")
        context = "\n".join(context_parts)

        # Send message to bus
        await self._bus.send(
            AgentMessage(
                from_agent="orchestrator",
                to_agent=step.agent,
                msg_type=MessageType.TASK,
                content=step.description,
                context={"task_id": task.id, "step_id": step.id},
                task_id=task.id,
            )
        )

        # Execute
        try:
            result = await asyncio.wait_for(
                agent.execute(step.description, context=context),
                timeout=300.0,  # 5 min per step
            )
            step.completed_at = time.time()

            # Send result back on bus
            await self._bus.send(
                AgentMessage(
                    from_agent=step.agent,
                    to_agent="orchestrator",
                    msg_type=MessageType.REPORT,
                    content=result[:200],
                    task_id=task.id,
                    reply_to=step.id,
                )
            )
            return result
        except TimeoutError:
            return "ERROR: Step execution timed out (5min)"
        except Exception as e:
            return f"ERROR: {e}"


# ── Self-Improvement Loop ─────────────────────────────────────────────────


class SelfImprovementLoop:
    """
    Pętla samoudoskonalania.

    Po każdym zadaniu:
    1. Analizuje co poszło dobrze / źle
    2. Wyciąga learnings
    3. Zapisuje do Qdrant (task_history + knowledge)
    4. Aktualizuje pamieci agentów
    5. (Opcjonalnie) Proponuje nowe agenty jeśli brakuje kompetencji
    """

    async def process_completed_task(
        self,
        task: Task,
        factory: AgentFactory,
        registry: AgentRegistry,
    ) -> str:
        """Analyze completed task and extract learnings."""
        kb = get_knowledge_base()
        llm = get_llm()

        # Gather task data
        steps_summary = []
        for s in task.steps:
            steps_summary.append(f"- [{s.status}] {s.agent}: {s.description[:100]}" + (f"\n  Result: {s.result[:200]}" if s.result else "") + (f"\n  Error: {s.error}" if s.error else ""))

        analysis_prompt = f"""Analizuj zakończone zadanie i wyciągnij wnioski.

## Zadanie:
{task.description}

## Status: {task.status}

## Wykonane kroki:
{chr(10).join(steps_summary)}

## WNIOSKI (odpowiedz w JSON):
{{
  "success": true/false,
  "learnings": "Kluczowe obserwacje i lekcje (max 3 zdania)",
  "what_worked": "Co zadziałało dobrze",
  "what_failed": "Co nie zadziałało i dlaczego",
  "knowledge_to_store": [
    {{"category": "infra/app/error/network", "content": "Konkretna wiedza do zapamiętania", "tags": ["tag1"]}}
  ],
  "missing_capabilities": "Czy brakło jakiegoś agenta/kompetencji? Co powinna mieć firma AI?"
}}"""

        response = await llm.generate(analysis_prompt, model=llm._fast_model, temperature=0.3, max_tokens=2048)

        # Parse analysis
        import re

        json_match = re.search(r'\{[\s\S]*"success"[\s\S]*\}', response)
        learnings = ""
        knowledge_items = []

        if json_match:
            try:
                data = json.loads(json_match.group())
                learnings = data.get("learnings", "")
                knowledge_items = data.get("knowledge_to_store", [])

                # Log errors from failed steps
                if not data.get("success", True) and data.get("what_failed"):
                    for step in task.steps:
                        if step.status == "failed" and step.error:
                            await kb.log_error(
                                ErrorEntry(
                                    error_message=step.error[:200],
                                    context=f"Task: {task.description[:100]}\nStep: {step.description[:100]}",
                                    agent=step.agent,
                                    task_id=task.id,
                                    severity="medium",
                                    resolved=False,
                                )
                            )
            except json.JSONDecodeError:
                learnings = response[:300]

        # Store knowledge items
        for item in knowledge_items:
            if item.get("content"):
                await kb.add_knowledge(
                    KnowledgeEntry(
                        content=item["content"],
                        source=f"task:{task.id}",
                        category=item.get("category", "task"),
                        tags=item.get("tags", []) + ["auto-learned"],
                        metadata={"task_id": task.id, "task_desc": task.description[:100]},
                    )
                )

        # Record task in history for future pattern matching
        agents_used = list({s.agent for s in task.steps})
        success = task.status == "completed"
        duration_ms = (task.completed_at - task.started_at) * 1000 if task.completed_at and task.started_at else 0

        await kb.record_task(
            task_id=task.id,
            description=task.description,
            agents_used=agents_used,
            result="\n".join(s.result[:100] for s in task.steps if s.result),
            success=success,
            duration_ms=duration_ms,
            learnings=learnings,
        )

        logger.info(
            "Self-improvement: task %s processed. Knowledge items: %d, Learnings: %s",
            task.id,
            len(knowledge_items),
            learnings[:80],
        )
        return learnings


# ── Main Orchestrator ─────────────────────────────────────────────────────


class Orchestrator:
    """
    Centralny punkt koordynacji — wirtualna firma AI.

    Przyjmuje zadania, planuje, wykonuje, uczy się.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self.bus = MessageBus()
        self.factory = AgentFactory(self.registry)
        self.planner = TaskPlanner(self.registry)
        self.executor = TaskExecutor(self.registry, self.bus)
        self.improvement = SelfImprovementLoop()
        self._tasks: dict[str, Task] = {}
        self._running = False

    async def initialize(self):
        """Initialize orchestrator — register agents, connect KB."""
        kb = get_knowledge_base()
        await kb.initialize()
        register_builtin_agents(self.registry)

        logger.info(
            "Orchestrator initialized: %d agents, %d KB collections",
            self.registry.count,
            4,
        )
        self._running = True

    async def close(self):
        """Clean shutdown."""
        llm = get_llm()
        await llm.close()
        kb = get_knowledge_base()
        await kb.close()
        self._running = False

    async def submit_task(
        self,
        description: str,
        *,
        requester: str = "user",
        priority: str = "normal",
        on_update: Any = None,
    ) -> Task:
        """
        Submit a task for the AI company to execute.

        High-level idea → plan → multi-agent execution → learnings.
        """
        task = Task(
            description=description,
            requester=requester,
        )
        task.started_at = time.time()
        self._tasks[task.id] = task

        logger.info("Task submitted [%s]: %s", task.id, description[:80])

        try:
            # 1. Plan
            task.status = "planning"
            task.steps = await self.planner.plan(task)
            logger.info("Plan created: %d steps", len(task.steps))

            if on_update:
                await on_update({"type": "plan", "steps": [asdict(s) for s in task.steps]})

            # 2. Execute
            final_result = await self.executor.execute_plan(task, on_update=on_update)
            task.result = final_result
            task.completed_at = time.time()

            # 3. Self-improve
            if get_settings().learning_enabled:
                task.learnings = await self.improvement.process_completed_task(task, self.factory, self.registry)

        except Exception as e:
            logger.error("Task %s failed with exception: %s", task.id, e)
            task.status = "failed"
            task.result = f"Orchestrator error: {e}"
            kb = get_knowledge_base()
            await kb.log_error(
                ErrorEntry(
                    error_message=str(e),
                    context=f"Task: {description[:200]}",
                    agent="orchestrator",
                    task_id=task.id,
                    severity="high",
                )
            )

        return task

    async def stream_task(self, description: str, *, requester: str = "user") -> AsyncIterator[dict]:
        """
        Submit and stream task progress as SSE events.
        Yields dicts with 'type', 'data'.
        """
        updates: asyncio.Queue = asyncio.Queue()
        task_container: list[Task] = []

        async def on_update(data):
            await updates.put(data)

        async def run():
            task = await self.submit_task(description, requester=requester, on_update=on_update)
            task_container.append(task)
            await updates.put({"type": "completed", "task": asdict(task)})

        asyncio.create_task(run())

        while True:
            try:
                update = await asyncio.wait_for(updates.get(), timeout=300.0)
                yield update
                if update.get("type") == "completed":
                    break
            except TimeoutError:
                yield {"type": "error", "data": "Task timeout"}
                break

    async def chat_with_agent(
        self,
        agent_name: str,
        message: str,
        *,
        task_id: str = "",
    ) -> str:
        """Direct chat with a specific agent (for debugging/exploration)."""
        agent = self.registry.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found. Available: {[a['name'] for a in self.registry.list_agents()]}"
        return await agent.execute(message, context=f"Task context: {task_id}" if task_id else "")

    async def spawn_agent(
        self,
        description: str,
        role: str = "sre",
        capabilities: list[str] | None = None,
    ) -> str:
        """Spawn a new specialized agent dynamically."""
        from .agents import AgentRole as AR

        try:
            role_enum = AR(role)
        except ValueError:
            role_enum = AR.SRE

        name = description.lower()[:20].replace(" ", "-").replace("_", "-")
        name = f"dynamic-{name}-{str(uuid.uuid4())[:4]}"

        agent = await self.factory.create_agent(
            name=name,
            role=role_enum,
            description=description,
            capabilities=capabilities or ["general"],
        )
        return agent.name

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, *, limit: int = 20) -> list[dict]:
        tasks = sorted(self._tasks.values(), key=lambda t: t.started_at or 0, reverse=True)
        return [
            {
                "id": t.id,
                "description": t.description[:100],
                "status": t.status,
                "steps": len(t.steps),
                "requester": t.requester,
                "started_at": t.started_at,
                "completed_at": t.completed_at,
            }
            for t in tasks[:limit]
        ]

    async def get_status(self) -> dict:
        """Health and status overview of the AI company."""
        kb = get_knowledge_base()
        llm = get_llm()
        kb_stats = await kb.get_stats()
        llm_health = await llm.health_check()
        return {
            "status": "running" if self._running else "stopped",
            "agents": self.registry.list_agents(),
            "tasks": {
                "total": len(self._tasks),
                "running": sum(1 for t in self._tasks.values() if t.status in ("planning", "executing")),
                "completed": sum(1 for t in self._tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self._tasks.values() if t.status == "failed"),
            },
            "knowledge": kb_stats,
            "llm": llm_health,
            "messages_exchanged": self.bus.total_messages,
        }


# Singleton
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
