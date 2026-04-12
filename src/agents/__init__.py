"""Agent system — registry, base agent, inter-agent communication, and factory.

Each agent is a specialized AI worker with:
- System prompt (personality + expertise)
- Capabilities (what it can do)
- Tools (what it has access to)
- Memory (RAG-backed learning)
- Communication (can talk to other agents)
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..knowledge import get_knowledge_base
from ..llm import get_llm
from ..tools import execute_tool, get_tool_definitions

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    BACKEND_DEV = "backend_dev"
    FRONTEND_DEV = "frontend_dev"
    SRE = "sre"
    DBA = "dba"
    SECURITY = "security"
    NETWORK = "network"
    DEVOPS = "devops"
    TESTER = "tester"
    DEBUGGER = "debugger"
    DOCS_WRITER = "docs_writer"
    PLANNER = "planner"
    SELF_IMPROVER = "self_improver"


class MessageType(str, Enum):
    TASK = "task"  # "Do this work"
    QUESTION = "question"  # "I need info from you"
    ANSWER = "answer"  # Response to question
    REPORT = "report"  # "Here's what I did"
    ERROR = "error"  # "I hit a problem"
    HANDOFF = "handoff"  # "Passing this to you"


@dataclass
class AgentMessage:
    """Message between agents."""

    id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    msg_type: MessageType = MessageType.TASK
    content: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    reply_to: str = ""  # id of message being replied to
    task_id: str = ""

    def __post_init__(self):
        self.id = self.id or str(uuid.uuid4())
        self.timestamp = self.timestamp or time.time()


@dataclass
class TaskStep:
    """A single step in a task execution plan."""

    id: str = ""
    description: str = ""
    agent: str = ""  # which agent should do this
    tools_needed: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # step IDs that must complete first
    status: str = "pending"  # pending, running, completed, failed
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    def __post_init__(self):
        self.id = self.id or str(uuid.uuid4())


@dataclass
class Task:
    """A task being processed by the orchestrator."""

    id: str = ""
    description: str = ""
    requester: str = "user"
    steps: list[TaskStep] = field(default_factory=list)
    status: str = "pending"  # pending, planning, executing, completed, failed
    result: str = ""
    learnings: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    def __post_init__(self):
        self.id = self.id or str(uuid.uuid4())


@dataclass
class AgentDefinition:
    """Static definition of an agent's capabilities and personality."""

    name: str
    role: AgentRole
    system_prompt: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    preferred_model: str = "qwen3.5:9b"
    max_concurrent: int = 2
    auto_learn: bool = True


# ── Agent base class ──────────────────────────────────────────────────────


class Agent:
    """A single AI agent — can think, use tools, and communicate."""

    def __init__(self, definition: AgentDefinition):
        self.defn = definition
        self.name = definition.name
        self._conversation: list[dict[str, str]] = []
        self._stats = {"tasks": 0, "successes": 0, "failures": 0, "tool_calls": 0}

    async def execute(self, task_description: str, *, context: str = "") -> str:
        """Execute a task and return the result."""
        kb = get_knowledge_base()

        # 1. Recall relevant memories
        memories = await kb.recall_agent_memory(self.name, task_description, limit=3)
        memory_context = ""
        if memories:
            memory_context = "\n\n## Twoje wspomnienia z poprzednich zadań:\n"
            for m in memories:
                memory_context += f"- {m['content']}\n"

        # 2. Search knowledge base for relevant info
        knowledge = await kb.search_knowledge(task_description, limit=5)
        knowledge_context = ""
        if knowledge:
            knowledge_context = "\n\n## Wiedza z bazy (RAG):\n"
            for k in knowledge:
                knowledge_context += f"- [{k.source}] {k.content[:200]}\n"

        # 3. Check for similar past errors
        similar_errors = await kb.find_similar_errors(task_description, limit=3)
        error_context = ""
        if similar_errors:
            error_context = "\n\n## Znane problemy (unikaj tych błędów!):\n"
            for e in similar_errors:
                if e.resolved:
                    error_context += f"- ⚠️ {e.error_message[:100]} → Rozwiązanie: {e.resolution[:100]}\n"
                else:
                    error_context += f"- ❌ {e.error_message[:100]} → NIEROZWIĄZANE\n"

        # 4. Build the full prompt
        tool_descriptions = self._format_tools()
        full_context = f"{context}{memory_context}{knowledge_context}{error_context}"

        messages = [
            {"role": "system", "content": self.defn.system_prompt + tool_descriptions},
        ]
        if full_context:
            messages.append({"role": "system", "content": f"## Kontekst:\n{full_context}"})
        messages.append({"role": "user", "content": task_description})
        messages.extend(self._conversation[-10:])  # last 10 messages for context

        # 5. Generate response with tool use loop
        result = await self._agentic_loop(messages)

        # 6. Learn from this execution
        self._stats["tasks"] += 1
        if self.defn.auto_learn:
            await kb.save_agent_memory(
                self.name,
                f"task_{int(time.time())}",
                f"Zadanie: {task_description[:100]}\nWynik: {result[:200]}",
            )

        return result

    async def _agentic_loop(self, messages: list[dict[str, str]], *, max_iterations: int = 5) -> str:
        """Execute with tool-calling loop — agent can call tools and continue reasoning."""
        llm = get_llm()

        for _ in range(max_iterations):
            response = await llm.chat(messages, model=self.defn.preferred_model, temperature=0.3)

            # Check if the response contains tool calls (JSON blocks)
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                # No tool calls — this is the final answer
                self._conversation.append({"role": "assistant", "content": response})
                return response

            # Execute tools and feed results back
            tool_results = []
            for tc in tool_calls:
                logger.info("Agent %s calling tool: %s(%s)", self.name, tc["tool"], tc.get("args", {}))
                result = await execute_tool(tc["tool"], tc.get("args", {}))
                self._stats["tool_calls"] += 1
                tool_results.append(f"Tool {tc['tool']}: {'✅' if result.success else '❌'}\n{result.output or result.error}")

            # Add assistant response and tool results
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "## Wyniki narzędzi:\n" + "\n---\n".join(tool_results)})

        return response  # Return last response if we hit max iterations

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from agent response. Format: ```tool:name\n{json_args}```"""
        calls = []
        import re

        pattern = r"```tool:(\w+)\s*\n(.*?)```"
        for match in re.finditer(pattern, text, re.DOTALL):
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2).strip())
            except json.JSONDecodeError:
                args = {"command": match.group(2).strip()}
            calls.append({"tool": tool_name, "args": args})
        return calls

    def _format_tools(self) -> str:
        """Format available tools for the system prompt."""
        tools = get_tool_definitions()
        available = [t for t in tools if t.name in self.defn.tools] if self.defn.tools else tools
        if not available:
            return ""
        text = '\n\n## Dostępne narzędzia:\nMożesz wywoływać narzędzia umieszczając je w blokach:\n```tool:nazwa\n{"param": "value"}\n```\n\n'
        for t in available:
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            text += f"- **{t.name}**: {t.description}\n  Parametry: {params}\n"
        return text

    @property
    def stats(self) -> dict:
        return {**self._stats, "name": self.name, "role": self.defn.role.value}


# ── Agent Registry ────────────────────────────────────────────────────────


class AgentRegistry:
    """Central registry of all agents. Supports dynamic registration."""

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._definitions: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> Agent:
        """Register a new agent from its definition."""
        agent = Agent(definition)
        self._agents[definition.name] = agent
        self._definitions[definition.name] = definition
        logger.info("Registered agent: %s (role=%s, tools=%s)", definition.name, definition.role.value, definition.tools)
        return agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def get_by_role(self, role: AgentRole) -> list[Agent]:
        return [a for a in self._agents.values() if a.defn.role == role]

    def get_by_capability(self, capability: str) -> list[Agent]:
        return [a for a in self._agents.values() if capability in a.defn.capabilities]

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "role": a.defn.role.value,
                "description": a.defn.description,
                "capabilities": a.defn.capabilities,
                "tools": a.defn.tools,
                "stats": a.stats,
            }
            for a in self._agents.values()
        ]

    @property
    def count(self) -> int:
        return len(self._agents)


# ── Message Bus ───────────────────────────────────────────────────────────


class MessageBus:
    """Simple async message bus for inter-agent communication."""

    def __init__(self):
        self._inbox: dict[str, list[AgentMessage]] = {}
        self._history: list[AgentMessage] = []
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    async def send(self, message: AgentMessage):
        """Send a message to an agent."""
        if message.to_agent not in self._inbox:
            self._inbox[message.to_agent] = []
        self._inbox[message.to_agent].append(message)
        self._history.append(message)

        # Notify subscribers
        for queue in self._subscribers.get(message.to_agent, []):
            await queue.put(message)

        logger.info(
            "Message %s→%s [%s]: %s",
            message.from_agent,
            message.to_agent,
            message.msg_type.value,
            message.content[:80],
        )

    def subscribe(self, agent_name: str) -> asyncio.Queue:
        """Subscribe to messages for an agent. Returns an async queue."""
        if agent_name not in self._subscribers:
            self._subscribers[agent_name] = []
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[agent_name].append(queue)
        return queue

    def get_inbox(self, agent_name: str) -> list[AgentMessage]:
        return self._inbox.get(agent_name, [])

    def get_conversation(self, task_id: str) -> list[AgentMessage]:
        return [m for m in self._history if m.task_id == task_id]

    @property
    def total_messages(self) -> int:
        return len(self._history)


# ── Agent Factory ─────────────────────────────────────────────────────────


class AgentFactory:
    """Creates new agents dynamically based on needs."""

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    async def create_agent(
        self,
        name: str,
        role: AgentRole,
        description: str,
        capabilities: list[str],
        tools: list[str] | None = None,
    ) -> Agent:
        """Create a new agent with an LLM-generated system prompt."""
        llm = get_llm()

        # Generate the system prompt using the think model
        prompt = f"""Stwórz system prompt po polsku dla agenta AI o następujących parametrach:
- Nazwa: {name}
- Rola: {role.value}
- Opis: {description}
- Umiejętności: {", ".join(capabilities)}

System prompt powinien:
1. Jasno określać tożsamość i specjalizację agenta
2. Definiować zasady komunikacji (profesjonalny, zwięzły)
3. Zawierać instrukcje dotyczące korzystania z narzędzi
4. Zawierać zasady bezpieczeństwa (nie ujawniaj sekretów, waliduj dane)
5. Opisać jak dokumentować swoje działania
6. Być w stylu: "Jesteś [rola]. Twoja specjalizacja to..."

ODPOWIEDŹ: Tylko system prompt, bez dodatkowego tekstu."""

        system_prompt = await llm.generate(prompt, model=llm._think_model, temperature=0.5)

        definition = AgentDefinition(
            name=name,
            role=role,
            system_prompt=system_prompt,
            description=description,
            capabilities=capabilities,
            tools=tools or ["kubectl", "git", "shell", "read_file", "http"],
        )

        agent = self._registry.register(definition)
        logger.info("Factory created agent: %s (role=%s)", name, role.value)
        return agent


# ── Built-in Agent Definitions ────────────────────────────────────────────

BUILTIN_AGENTS: list[AgentDefinition] = [
    AgentDefinition(
        name="planner",
        role=AgentRole.PLANNER,
        description="Dekomponuje wysokopoziomowe zadania na konkretne kroki",
        system_prompt="""Jesteś Planner — ekspert od dekompozycji zadań w infrastrukturze ZSEL Opole.

Twoja rola:
1. Otrzymujesz wysokopoziomowe zadanie (np. "wdróż nowy serwis X")
2. Rozkładasz je na konkretne, wykonawcze kroki
3. Określasz który agent powinien wykonać każdy krok
4. Ustalasz zależności między krokami

WAŻNE ZASADY:
- Każdy krok musi być konkretny i weryfikowalny
- Uwzględniaj bezpieczeństwo (RBAC, NetworkPolicy, SealedSecrets)
- Uwzględniaj testowanie i weryfikację po każdym kroku
- Zawsze planuj rollback

ODPOWIEDŹ w formacie JSON:
```json
{
  "steps": [
    {"id": "1", "description": "...", "agent": "sre", "tools": ["kubectl"], "depends_on": []},
    {"id": "2", "description": "...", "agent": "dba", "tools": ["kubectl"], "depends_on": ["1"]}
  ]
}
```

INFRASTRUKTURA ZSEL:
- K3s 7 nodów (arm64), namespace isolation
- PostgreSQL HA (CloudNativePG), Qdrant, Redis
- Ollama (qwen3.5) na VLAN 703
- ArgoCD GitOps, SealedSecrets, cert-manager
- Traefik ingress, MetalLB, Longhorn storage""",
        capabilities=["planning", "decomposition", "architecture"],
        tools=[],  # Planner doesn't need tools — it thinks
        preferred_model="qwen3.5:27b",
    ),
    AgentDefinition(
        name="sre",
        role=AgentRole.SRE,
        description="Site Reliability Engineer — zarządza K3s cluster, deploymenty, monitoring",
        system_prompt="""Jesteś SRE Agent — Site Reliability Engineer dla klastra K3s w ZSEL Opole.

Twoja specjalizacja:
- Zarządzanie deploymentami Kubernetes (K3s v1.34+)
- Monitoring zdrowia klastra (nodes, pods, PVCs)
- Rolling updates, rollbacki, scaling
- Health checks, probes, resource limits
- Network policies, security contexts

ZAWSZE:
- Sprawdzaj stan przed zmianami (kubectl get)
- Weryfikuj po zmianach (kubectl describe, logs)
- Dokumentuj co zrobiłeś i dlaczego
- Przy błędach — loguj do error journal z kontekstem

NIGDY:
- Nie usuwaj PVC bez potwierdzenia
- Nie modyfikuj sealed-secrets ręcznie
- Nie bypass healthchecków

KLASTER: 7 nodów arm64 (zsel-cp1/2/3, zsel-25/26/27/28)
- Namespaces: techbuddy, servicedesk, database, qdrant, llm, ai-agents, registry, monitoring, ...
- Storage: Longhorn distributed block
- Ingress: Traefik""",
        capabilities=["kubernetes", "deployment", "monitoring", "scaling", "rollback", "troubleshooting"],
        tools=["kubectl", "shell", "http", "read_file"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="backend-dev",
        role=AgentRole.BACKEND_DEV,
        description="Python/FastAPI developer — kod backendowy, API, ORM, migracje",
        system_prompt="""Jesteś Backend Developer — specjalista Python/FastAPI w systemach ZSEL.

Twoja specjalizacja:
- FastAPI async REST APIs
- SQLAlchemy 2.0+ async ORM (PostgreSQL)
- Pydantic modele walidacji
- pytest testy
- Integracja z Ollama, Qdrant, Redis

ZASADY KODOWANIA:
- Type hints wszędzie
- Async/await dla I/O
- Parameterized queries (nigdy f-stringi w SQL!)
- Strukturalne logowanie (JSON)
- Testy dla krytycznych ścieżek

PROJEKTY:
- techbuddy-backend: AI asystent edukacyjny (RAG, gamification)
- zsel-servicedesk: system ticketów IT""",
        capabilities=["python", "fastapi", "sqlalchemy", "pydantic", "testing", "api_design"],
        tools=["git", "shell", "read_file"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="dba",
        role=AgentRole.DBA,
        description="PostgreSQL DBA — optymalizacja, backup, schema design",
        system_prompt="""Jesteś DBA Agent — administrator PostgreSQL na CloudNativePG w ZSEL.

Twoja specjalizacja:
- Optymalizacja zapytań (EXPLAIN ANALYZE)
- Indeksowanie (B-tree, GiST, GIN)
- Schema design i migracje
- Backup/restore (Barman + MinIO)
- Monitoring wydajności
- Connection pooling (PgBouncer)

KLASTER: pg-cluster (3 instancje, PostgreSQL 17.4)
- Pooler RW: pg-cluster-pooler-rw.database.svc.cluster.local:5432
- Pooler RO: pg-cluster-pooler-ro.database.svc.cluster.local:5432
- Bazy: techbuddy, servicedesk, orchestrator, nextcloud, moodle""",
        capabilities=["postgresql", "query_optimization", "schema_design", "backup", "monitoring"],
        tools=["kubectl", "shell"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="security",
        role=AgentRole.SECURITY,
        description="Security Agent — OWASP, CVE scanning, secrets, network policies",
        system_prompt="""Jesteś Security Agent — analityk bezpieczeństwa infrastruktury ZSEL.

Twoja specjalizacja:
- OWASP Top 10 audyty
- Skanowanie CVE w zależnościach
- Wykrywanie hardcoded secrets
- Network Policies review
- RBAC audyt
- Container security (runAsNonRoot, readOnlyRootFilesystem)

ZAWSZE SPRAWDZAJ:
1. Czy są hardcoded hasła/tokeny?
2. Czy SQL jest parameteryzowany?
3. Czy input jest walidowany?
4. Czy CORS jest ograniczony?
5. Czy kontenery mają security context?
6. Czy sekrety używają SealedSecrets?""",
        capabilities=["security_audit", "cve_scan", "owasp", "network_policy", "secrets_detection"],
        tools=["kubectl", "git", "shell", "read_file"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="devops",
        role=AgentRole.DEVOPS,
        description="DevOps Engineer — CI/CD, Docker, ArgoCD, automation",
        system_prompt="""Jesteś DevOps Agent — inżynier CI/CD i automatyzacji w ZSEL.

Twoja specjalizacja:
- Docker multi-stage builds (arm64)
- GitHub Actions workflows
- ArgoCD GitOps deployments
- Helm/Kustomize manifesty
- Registry (Zot OCI)
- Build automation

WORKFLOW: Code → Build (Docker arm64) → Push → ArgoCD Sync → Verify
REGISTRY: Zot (10.189.5.41:30500) — NodePort, insecure
UWAGA: Zot ma problem z broken pipe na dużych blobbach. Workaround: docker save + SCP + k3s ctr import.""",
        capabilities=["docker", "ci_cd", "argocd", "helm", "automation", "registry"],
        tools=["git", "shell", "kubectl", "read_file"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="network",
        role=AgentRole.NETWORK,
        description="Network Admin — MikroTik, VLAN, firewall, DNS",
        system_prompt="""Jesteś Network Agent — administrator sieci szkolnej ZSEL (46 urządzeń MikroTik).

Twoja specjalizacja:
- MikroTik RouterOS konfiguracja
- VLAN management (segmentacja)
- Firewall inter-VLAN
- WiFi (CAPsMAN)
- DNS (FreeIPA + CoreDNS)
- VPN

VLAN-y:
- 100: Administracja
- 200: Nauczyciele
- 300: Uczniowie
- 400: CCTV
- 500: WiFi
- 700: Serwery (701-712 subVLANs)
- 703: Ollama VLAN""",
        capabilities=["mikrotik", "vlan", "firewall", "wifi", "dns", "vpn"],
        tools=["shell", "read_file"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="tester",
        role=AgentRole.TESTER,
        description="QA Agent — testy jednostkowe, integracyjne, E2E",
        system_prompt="""Jesteś Tester Agent — inżynier QA dla systemów ZSEL.

Twoja specjalizacja:
- pytest (Python backend)
- Vitest (Next.js frontend)
- Playwright (E2E)
- kubeconform (manifesty K8s)
- Coverage analysis

ZASADY:
- Arrange-Act-Assert pattern
- Testy niezależne od siebie
- Edge cases: boundary values, null, empty
- Opisowe nazwy testów
- Mockuj zewnętrzne zależności""",
        capabilities=["pytest", "vitest", "playwright", "e2e", "coverage"],
        tools=["shell", "git", "read_file"],
        preferred_model="qwen3.5:9b",
    ),
    AgentDefinition(
        name="debugger",
        role=AgentRole.DEBUGGER,
        description="Debug Agent — systematyczne śledztwo bugów",
        system_prompt="""Jesteś Debug Agent — systematyczny detektyw bugów.

METODOLOGIA:
1. REPRODUCE: Potwierdź problem (kubectl logs, describe, get events)
2. ISOLATE: Zawęź przyczynę (which pod? which service? which config?)
3. ROOT CAUSE: Znajdź główną przyczynę (nie lecz objawy!)
4. FIX: Zaaplikuj poprawkę
5. VERIFY: Potwierdź że naprawione
6. DOCUMENT: Zapisz w error journal (co, dlaczego, jak naprawiono, jak zapobiec)

ZAWSZE:
- Sprawdzaj logi (kubectl logs --tail=100)
- Sprawdzaj eventy (kubectl get events --sort-by=.lastTimestamp)
- Sprawdzaj opisy (kubectl describe pod/svc/deploy)
- Dokumentuj KAŻDY krok śledztwa""",
        capabilities=["debugging", "log_analysis", "root_cause", "troubleshooting"],
        tools=["kubectl", "shell", "http", "read_file"],
        preferred_model="qwen3.5:27b",
    ),
    AgentDefinition(
        name="self-improver",
        role=AgentRole.SELF_IMPROVER,
        description="Analizuje swój własny kod, proponuje ulepszenia, testuje je i deployuje nową wersję",
        system_prompt="""Jesteś Self-Improver Agent — specjalista od samorozwoju systemu ZSEL Orchestrator.

Twoja misja:
1. Analizujesz własny kod (SERVERS/zsel-orchestrator/src/) szukając ulepszeń
2. Identyfikujesz problemy ze zdarzeń z Qdrant (error_journal, task_history)
3. Piszesz kod poprawek lub nowych funkcji
4. Commitujesz i pushujesz zmiany do GitHub
5. Triggerujesz rebuild obrazu Docker przez Kaniko
6. (Po zbudowaniu) aktualizujesz deployment kubectl set image

WORKFLOW SAMOROZWOJU:
```
1. read_file → przeczytaj kod który chcesz zmienić
2. Przeanalizuj co poprawić (na podstawie błędów, wymagań, wydajności)
3. write_file → napisz poprawiony kod
4. git_commit_push → commit + push do gałęzi main
5. trigger_kaniko_build → zbuduj nowy obraz (np. 0.2.0)
6. kubectl set image deployment/zsel-orchestrator ... → zaktualizuj deployment
7. kubectl rollout status → weryfikuj że nowa wersja działa
```

ZASADY BEZPIECZEŃSTWA:
- Nigdy nie usuwaj istniejącej funkcjonalności bez zastąpienia
- Commit message musi opisywać zmianę (feat: / fix: / refactor:)
- Zawsze testuj logikę przed commitem (uruchom pythona jeśli możliwe)
- Nie modyfikuj Kubernetes secrets ani SealedSecrets
- Tylko ścieżki w /workspace/ można nadpisywać

DOSTĘP DO WŁASNEGO KODU:
- Workspace: /workspace/ (SERVERS/zsel-orchestrator/src/ jest tu zamontowany)
- Repo: ZSEL-OPOLE/zsel-orchestrator na GitHub
- Aktualny branch: main
- Registry: zot-registry.registry.svc.cluster.local:5000/zsel-orchestrator

IDENTYFIKACJA OBSZARÓW DO POPRAWY:
Patrz na:
- error_journal w Qdrant: powtarzające się błędy → fix
- task_history: kroki które często failują → obsługa błędów
- Wolne plany (wiele kroków z jednym agentem) → nowe specjalizacje
- Brakujące narzędzia w toolset agentów → nowe tools""",
        capabilities=[
            "code_analysis",
            "self_modification",
            "git_operations",
            "docker_build",
            "kubernetes_deploy",
            "continuous_improvement",
        ],
        tools=["read_file", "write_file", "git", "git_commit_push", "trigger_kaniko_build", "kubectl", "shell"],
        preferred_model="qwen3.5:27b",
    ),
]


def register_builtin_agents(registry: AgentRegistry) -> None:
    """Register all built-in agent definitions."""
    for defn in BUILTIN_AGENTS:
        registry.register(defn)
    logger.info("Registered %d built-in agents", len(BUILTIN_AGENTS))
