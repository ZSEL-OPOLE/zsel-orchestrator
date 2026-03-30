"""ZSEL Orchestrator — FastAPI HTTP API + WebSocket streaming."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .orchestrator import get_orchestrator
from .knowledge import KnowledgeEntry, ErrorEntry, get_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = get_orchestrator()
    await orch.initialize()
    logger.info("Orchestrator ready: %d agents", orch.registry.count)
    yield
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


@app.get("/status")
async def status():
    orch = get_orchestrator()
    return await orch.get_status()


# ── Tasks ─────────────────────────────────────────────────────────────────


@app.post("/tasks")
async def submit_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """Submit a task. Returns task ID immediately, execution runs in background."""
    orch = get_orchestrator()
    from .agents import Task
    import uuid
    import time

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
    except asyncio.TimeoutError:
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
