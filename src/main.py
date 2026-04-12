"""ZSEL Orchestrator — FastAPI HTTP API + WebSocket streaming."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .health import get_health_aggregator
from .knowledge import ErrorEntry, KnowledgeEntry, get_knowledge_base
from .orchestrator import get_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = get_orchestrator()
    await orch.initialize()
    health = get_health_aggregator()
    await health.start()
    logger.info("Orchestrator ready: %d agents", orch.registry.count)
    yield
    await health.stop()
    await orch.close()


app = FastAPI(
    title="ZSEL Orchestrator",
    description="Wirtualna firma AI — autonomiczny multi-agent system z RAG i samoudoskonalaniem.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────


class TaskRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=4000)
    requester: str = "user"
    priority: str = "normal"


class ChatRequest(BaseModel):
    agent: str
    message: str
    task_id: str = ""


class SpawnAgentRequest(BaseModel):
    description: str
    role: str = "sre"
    capabilities: list[str] = []


class AddKnowledgeRequest(BaseModel):
    content: str
    source: str
    category: str = "infra"
    tags: list[str] = []


class SearchKnowledgeRequest(BaseModel):
    query: str
    limit: int = 10
    category: str | None = None


class LogErrorRequest(BaseModel):
    error_message: str
    context: str = ""
    agent: str = "user"
    severity: str = "medium"


# ── Health ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "zsel-orchestrator"}


@app.get("/ready")
async def ready():
    orch = get_orchestrator()
    return {
        "ready": orch._running,
        "agents": orch.registry.count,
    }


@app.get("/health/services")
async def health_services():
    """Consolidated health of all ZSEL services."""
    ha = get_health_aggregator()
    return ha.get_all()


@app.get("/health/services/unhealthy")
async def health_unhealthy():
    """Only unhealthy/degraded services."""
    ha = get_health_aggregator()
    return {"unhealthy": ha.get_unhealthy()}


@app.get("/health/services/{service_name}")
async def health_service(service_name: str):
    """Health of a specific service."""
    ha = get_health_aggregator()
    result = ha.get_service(service_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    return result


@app.get("/status")
async def status():
    orch = get_orchestrator()
    return await orch.get_status()


# ── Tasks ─────────────────────────────────────────────────────────────────


@app.post("/tasks")
async def submit_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """Submit a task. Returns task ID immediately, execution runs in background."""
    orch = get_orchestrator()
    import time
    import uuid

    from .agents import Task

    task_id = str(uuid.uuid4())
    # Create placeholder task
    task = Task(
        id=task_id,
        description=req.description,
        requester=req.requester,
        status="queued",
        started_at=time.time(),
    )
    orch._tasks[task_id] = task

    # Execute in background
    background_tasks.add_task(
        orch.submit_task,
        req.description,
        requester=req.requester,
    )

    return {"task_id": task_id, "status": "queued"}


@app.post("/tasks/sync")
async def submit_task_sync(req: TaskRequest):
    """Submit a task and wait for completion. Max 5min."""
    orch = get_orchestrator()
    try:
        task = await asyncio.wait_for(
            orch.submit_task(req.description, requester=req.requester),
            timeout=300.0,
        )
        return asdict(task)
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Task execution timed out")


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    orch = get_orchestrator()
    task = orch.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return asdict(task)


@app.get("/tasks")
async def list_tasks(limit: int = 20):
    orch = get_orchestrator()
    return orch.list_tasks(limit=limit)


@app.get("/tasks/{task_id}/stream")
async def stream_task_sse(task_id: str):
    """Server-Sent Events stream for task progress."""
    orch = get_orchestrator()
    task = orch.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        async for update in orch.stream_task(task.description, requester=task.requester):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── WebSocket ─────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_orchestrator(websocket: WebSocket):
    """
    WebSocket interface for real-time interaction.

    Send: {"action": "task", "description": "..."}
         {"action": "chat", "agent": "sre", "message": "..."}
         {"action": "status"}
    Receive: progress events and final result
    """
    await websocket.accept()
    orch = get_orchestrator()
    logger.info("WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            action = msg.get("action")

            if action == "task":
                description = msg.get("description", "").strip()
                if not description:
                    await websocket.send_json({"error": "Empty description"})
                    continue

                await websocket.send_json({"type": "start", "description": description})

                async for update in orch.stream_task(description):
                    try:
                        await websocket.send_json(update)
                    except WebSocketDisconnect:
                        return

            elif action == "chat":
                agent_name = msg.get("agent", "sre")
                message = msg.get("message", "")
                result = await orch.chat_with_agent(agent_name, message, task_id=msg.get("task_id", ""))
                await websocket.send_json({"type": "chat_response", "agent": agent_name, "result": result})

            elif action == "status":
                status_data = await orch.get_status()
                await websocket.send_json({"type": "status", **status_data})

            elif action == "agents":
                await websocket.send_json({"type": "agents", "agents": orch.registry.list_agents()})

            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# ── Agents ────────────────────────────────────────────────────────────────


@app.get("/agents")
async def list_agents():
    orch = get_orchestrator()
    return orch.registry.list_agents()


@app.post("/agents/{agent_name}/chat")
async def chat_with_agent(agent_name: str, req: ChatRequest):
    orch = get_orchestrator()
    result = await orch.chat_with_agent(agent_name, req.message, task_id=req.task_id)
    return {"agent": agent_name, "result": result}


@app.post("/agents/spawn")
async def spawn_agent(req: SpawnAgentRequest):
    orch = get_orchestrator()
    name = await orch.spawn_agent(req.description, role=req.role, capabilities=req.capabilities)
    return {"name": name, "status": "spawned"}


@app.get("/agents/{agent_name}/memory")
async def get_agent_memory(agent_name: str, query: str = "recent tasks"):
    kb = get_knowledge_base()
    memories = await kb.recall_agent_memory(agent_name, query, limit=10)
    return {"agent": agent_name, "memories": memories}


# ── Knowledge ─────────────────────────────────────────────────────────────


@app.post("/knowledge")
async def add_knowledge(req: AddKnowledgeRequest):
    kb = get_knowledge_base()
    entry = KnowledgeEntry(
        content=req.content,
        source=req.source,
        category=req.category,
        tags=req.tags,
    )
    point_id = await kb.add_knowledge(entry)
    return {"id": point_id, "indexed": True}


@app.post("/knowledge/search")
async def search_knowledge(req: SearchKnowledgeRequest):
    kb = get_knowledge_base()
    entries = await kb.search_knowledge(
        req.query,
        limit=req.limit,
        category=req.category,
    )
    return [
        {
            "id": e.id,
            "content": e.content,
            "source": e.source,
            "category": e.category,
            "tags": e.tags,
            "score": e.score,
        }
        for e in entries
    ]


@app.get("/knowledge/stats")
async def knowledge_stats():
    kb = get_knowledge_base()
    return await kb.get_stats()


# ── Error Journal ─────────────────────────────────────────────────────────


@app.post("/errors")
async def log_error(req: LogErrorRequest):
    kb = get_knowledge_base()
    entry = ErrorEntry(
        error_message=req.error_message,
        context=req.context,
        agent=req.agent,
        severity=req.severity,
    )
    point_id = await kb.log_error(entry)
    return {"id": point_id, "logged": True}


@app.post("/errors/search")
async def search_errors(req: SearchKnowledgeRequest):
    kb = get_knowledge_base()
    entries = await kb.find_similar_errors(req.query, limit=req.limit)
    return [
        {
            "id": e.id,
            "error_message": e.error_message,
            "context": e.context,
            "resolution": e.resolution,
            "root_cause": e.root_cause,
            "severity": e.severity,
            "resolved": e.resolved,
            "agent": e.agent,
            "score": e.score,
        }
        for e in entries
    ]


@app.post("/errors/{error_id}/resolve")
async def resolve_error(error_id: str, resolution: str, root_cause: str = "", prevention: str = ""):
    kb = get_knowledge_base()
    await kb.resolve_error(error_id, resolution, root_cause=root_cause, prevention=prevention)
    return {"resolved": True}


# ── Bus / Messages ────────────────────────────────────────────────────────


@app.get("/messages")
async def get_messages(task_id: str):
    orch = get_orchestrator()
    msgs = orch.bus.get_conversation(task_id)
    return [
        {
            "id": m.id,
            "from": m.from_agent,
            "to": m.to_agent,
            "type": m.msg_type.value,
            "content": m.content[:300],
            "timestamp": m.timestamp,
        }
        for m in msgs
    ]


# ── Self-Development ───────────────────────────────────────────────────────


class SelfAnalyzeRequest(BaseModel):
    focus: str = Field(
        "Przeanalizuj logi błędów z ostatnich zadań i zaproponuj co można poprawić.",
        description="Obszar analizy lub konkretne pytanie do self-improver agenta",
    )


class SelfRebuildRequest(BaseModel):
    new_version: str = Field(..., description="Nowa wersja obrazu, np. '0.2.0'")
    git_revision: str = Field("main", description="Branch/tag do zbudowania")
    reason: str = Field("", description="Opis co zmieniło się w tej wersji")


@app.post("/self/analyze")
async def self_analyze(req: SelfAnalyzeRequest):
    """
    Poproś self-improver agenta o analizę kodu i zaproponowanie ulepszeń.
    Nie modyfikuje kodu — tylko analiza i propozycje.
    """
    orch = get_orchestrator()
    agent = orch.registry.get("self-improver")
    if not agent:
        raise HTTPException(status_code=503, detail="self-improver agent not registered")

    prompt = f"""Wykonaj analizę stanu orchestratora i zaproponuj konkretne ulepszenia.

Twoje zadanie:
{req.focus}

Kroki:
1. Przejrzyj najnowsze błędy z error_journal (kubectl lub HTTP do /errors/search)
2. Przejrzyj ostatnie zadania (GET /tasks) — które failowały?
3. Zaproponuj konkretne zmiany kodu z przykładami
4. NIE modyfikuj kodu teraz — tylko analiza i raport

Format raportu:
- Lista problemów (z priorytetem: KRYTYCZNY/WAŻNY/DROBNY)
- Propozycje kodu (diff-style lub cały fragment)
- Rekomendacja kolejnej wersji (0.x.y)"""

    result = await asyncio.wait_for(
        agent.execute(prompt),
        timeout=300.0,
    )
    return {"analysis": result, "agent": "self-improver"}


@app.post("/self/rebuild")
async def self_rebuild(req: SelfRebuildRequest):
    """
    Triggeruje Kaniko Job który buduje nowy obraz orchestratora z aktualnego kodu w repo.
    UWAGA: Wymaga że zmiany były już wcześniej spushowane do GitHub.
    """
    from .tools import execute_tool

    result = await execute_tool(
        "trigger_kaniko_build",
        {
            "image_tag": req.new_version,
            "git_revision": req.git_revision,
            "context_subdir": "SERVERS/zsel-orchestrator",
        },
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=f"Kaniko Job failed: {result.error}")

    # Log this as knowledge
    kb = get_knowledge_base()
    from .knowledge import KnowledgeEntry

    await kb.add_knowledge(
        KnowledgeEntry(
            content=f"Rebuild triggered for v{req.new_version} (branch: {req.git_revision}). Reason: {req.reason or 'manual'}",
            source="self:rebuild",
            category="deployment",
            tags=["rebuild", "self-improvement", f"v{req.new_version}"],
        )
    )

    return {
        "triggered": True,
        "job": result.metadata.get("job_name"),
        "image": result.metadata.get("image"),
        "output": result.output,
    }


@app.post("/self/evolve")
async def self_evolve(background_tasks: BackgroundTasks):
    """
    Pełna pętla samorozwoju: analiza → plan ulepszeń → implementacja → commit → rebuild.
    Uruchamia się w tle — wynik możesz śledzić przez /tasks/{id}.
    """
    orch = get_orchestrator()
    import time as _time

    task_description = (
        "Samorozwój orchestratora: "
        "1) Przejrzyj error_journal i task_history w Qdrant. "
        "2) Zidentyfikuj TOP-3 problemy do naprawienia. "
        "3) Zaproponuj i zaimplementuj konkretne poprawki kodu. "
        "4) Commituj i pushuj zmiany do main. "
        "5) Zaraportuj co zostało zmienione i dlaczego."
    )

    import uuid

    from .agents import Task

    task_id = str(uuid.uuid4())
    task = Task(
        id=task_id,
        description=task_description,
        requester="self-evolve-api",
        status="queued",
        started_at=_time.time(),
    )
    orch._tasks[task_id] = task

    background_tasks.add_task(
        orch.submit_task,
        task_description,
        requester="self-evolve-api",
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "description": "Pętla samorozwoju uruchomiona w tle. Śledź przez GET /tasks/{task_id}",
    }
