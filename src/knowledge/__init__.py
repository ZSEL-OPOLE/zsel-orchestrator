"""RAG Knowledge Base — Qdrant-backed vector store for infrastructure knowledge."""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ..config import get_settings
from ..llm import get_llm

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A piece of knowledge stored in the vector DB."""

    id: str = ""
    content: str = ""
    source: str = ""  # e.g. "git:GITOPS/06-apps/...", "error:kubectl-timeout", "task:deploy-xyz"
    category: str = ""  # infra, app, network, security, error, task, agent
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    score: float = 0.0  # populated on search

    def to_payload(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp or time.time(),
        }


@dataclass
class ErrorEntry:
    """An error with context, resolution, and learnings."""

    id: str = ""
    error_message: str = ""
    context: str = ""  # what was happening when error occurred
    stack_trace: str = ""
    resolution: str = ""  # how it was fixed
    root_cause: str = ""
    prevention: str = ""  # how to prevent in future
    agent: str = ""  # which agent encountered it
    task_id: str = ""
    severity: str = "medium"  # low, medium, high, critical
    resolved: bool = False
    timestamp: float = 0.0
    score: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_message": self.error_message,
            "context": self.context,
            "stack_trace": self.stack_trace[:2000],
            "resolution": self.resolution,
            "root_cause": self.root_cause,
            "prevention": self.prevention,
            "agent": self.agent,
            "task_id": self.task_id,
            "severity": self.severity,
            "resolved": self.resolved,
            "timestamp": self.timestamp or time.time(),
        }

    @property
    def search_text(self) -> str:
        """Text used for embedding — combines error + context + resolution."""
        parts = [self.error_message, self.context]
        if self.resolution:
            parts.append(f"Resolution: {self.resolution}")
        if self.root_cause:
            parts.append(f"Root cause: {self.root_cause}")
        return "\n".join(parts)


class KnowledgeBase:
    """
    RAG-based knowledge system using Qdrant vectors + Ollama embeddings.

    Collections:
    - orch_knowledge:     Infrastructure docs, manifests, configs, learnings
    - orch_error_journal: Errors with resolutions — learn from past mistakes
    - orch_task_history:  Completed tasks with results — pattern matching
    - orch_agent_memory:  Per-agent memories and specialization data
    """

    def __init__(self):
        s = get_settings()
        self._client: AsyncQdrantClient | None = None
        self._host = s.qdrant_host
        self._port = s.qdrant_port
        self._dim = s.ollama_embed_dim
        self._collections = {
            "knowledge": s.qdrant_collection_knowledge,
            "errors": s.qdrant_collection_errors,
            "tasks": s.qdrant_collection_tasks,
            "agents": s.qdrant_collection_agents,
        }
        self._stats = {
            "indexed": 0,
            "searches": 0,
            "errors_logged": 0,
        }

    async def initialize(self):
        """Create Qdrant client and ensure all collections exist."""
        self._client = AsyncQdrantClient(host=self._host, port=self._port)
        for name, collection in self._collections.items():
            await self._ensure_collection(collection)
            count = await self._count(collection)
            logger.info("Collection %s (%s): %d points", name, collection, count)

    async def _ensure_collection(self, name: str):
        """Create collection if it doesn't exist."""
        try:
            await self._client.get_collection(name)
        except Exception:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self._dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created collection: %s (dim=%d)", name, self._dim)

    async def _count(self, collection: str) -> int:
        info = await self._client.get_collection(collection)
        return info.points_count or 0

    async def close(self):
        if self._client:
            await self._client.close()

    # ── Knowledge CRUD ────────────────────────────────────────────────────

    async def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Index a knowledge entry. Returns the point ID."""
        llm = get_llm()
        entry.id = entry.id or str(uuid.uuid4())
        entry.timestamp = entry.timestamp or time.time()
        vector = await llm.embed(entry.content[:2000])
        if not vector:
            logger.warning("Empty embedding for knowledge entry: %s", entry.source)
            return ""
        await self._client.upsert(
            collection_name=self._collections["knowledge"],
            points=[
                models.PointStruct(
                    id=entry.id,
                    vector=vector,
                    payload=entry.to_payload(),
                )
            ],
        )
        self._stats["indexed"] += 1
        return entry.id

    async def search_knowledge(
        self,
        query: str,
        *,
        limit: int = 10,
        category: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[KnowledgeEntry]:
        """Search knowledge base by semantic similarity."""
        llm = get_llm()
        vector = await llm.embed(query[:2000])
        if not vector:
            return []

        query_filter = None
        if category:
            query_filter = models.Filter(must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))])

        results = await self._client.query_points(
            collection_name=self._collections["knowledge"],
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )
        self._stats["searches"] += 1
        entries = []
        for point in results.points:
            e = KnowledgeEntry(
                id=str(point.id),
                content=point.payload.get("content", ""),
                source=point.payload.get("source", ""),
                category=point.payload.get("category", ""),
                tags=point.payload.get("tags", []),
                metadata=point.payload.get("metadata", {}),
                timestamp=point.payload.get("timestamp", 0),
                score=point.score,
            )
            entries.append(e)
        return entries

    # ── Error Journal ─────────────────────────────────────────────────────

    async def log_error(self, entry: ErrorEntry) -> str:
        """Log an error to the journal. Returns point ID."""
        llm = get_llm()
        entry.id = entry.id or str(uuid.uuid4())
        entry.timestamp = entry.timestamp or time.time()
        vector = await llm.embed(entry.search_text[:2000])
        if not vector:
            return ""
        await self._client.upsert(
            collection_name=self._collections["errors"],
            points=[models.PointStruct(id=entry.id, vector=vector, payload=entry.to_payload())],
        )
        self._stats["errors_logged"] += 1
        logger.info("Error logged: %s (agent=%s, resolved=%s)", entry.error_message[:80], entry.agent, entry.resolved)
        return entry.id

    async def find_similar_errors(self, error_text: str, *, limit: int = 5) -> list[ErrorEntry]:
        """Find previously seen errors similar to this one."""
        llm = get_llm()
        vector = await llm.embed(error_text[:2000])
        if not vector:
            return []
        results = await self._client.query_points(
            collection_name=self._collections["errors"],
            query=vector,
            limit=limit,
            score_threshold=0.5,
        )
        entries = []
        for p in results.points:
            e = ErrorEntry(
                id=str(p.id),
                error_message=p.payload.get("error_message", ""),
                context=p.payload.get("context", ""),
                resolution=p.payload.get("resolution", ""),
                root_cause=p.payload.get("root_cause", ""),
                prevention=p.payload.get("prevention", ""),
                agent=p.payload.get("agent", ""),
                severity=p.payload.get("severity", "medium"),
                resolved=p.payload.get("resolved", False),
                timestamp=p.payload.get("timestamp", 0),
                score=p.score,
            )
            entries.append(e)
        return entries

    async def resolve_error(self, error_id: str, resolution: str, root_cause: str = "", prevention: str = ""):
        """Update an error entry with resolution info."""
        llm = get_llm()
        # Get existing
        points = await self._client.retrieve(self._collections["errors"], ids=[error_id])
        if not points:
            return
        payload = points[0].payload
        payload["resolution"] = resolution
        payload["root_cause"] = root_cause
        payload["prevention"] = prevention
        payload["resolved"] = True
        # Re-embed with resolution context for better future matching
        search_text = f"{payload['error_message']}\n{payload['context']}\nResolution: {resolution}\nRoot cause: {root_cause}"
        vector = await llm.embed(search_text[:2000])
        if vector:
            await self._client.upsert(
                collection_name=self._collections["errors"],
                points=[models.PointStruct(id=error_id, vector=vector, payload=payload)],
            )

    # ── Task History ──────────────────────────────────────────────────────

    async def record_task(
        self,
        task_id: str,
        description: str,
        agents_used: list[str],
        result: str,
        success: bool,
        duration_ms: float,
        learnings: str = "",
    ) -> str:
        """Record a completed task for pattern matching."""
        llm = get_llm()
        text = f"Task: {description}\nResult: {result[:500]}"
        if learnings:
            text += f"\nLearnings: {learnings}"
        vector = await llm.embed(text[:2000])
        if not vector:
            return ""
        point_id = task_id or str(uuid.uuid4())
        await self._client.upsert(
            collection_name=self._collections["tasks"],
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "description": description,
                        "agents_used": agents_used,
                        "result": result[:2000],
                        "success": success,
                        "duration_ms": duration_ms,
                        "learnings": learnings,
                        "timestamp": time.time(),
                    },
                )
            ],
        )
        return point_id

    async def find_similar_tasks(self, description: str, *, limit: int = 5) -> list[dict]:
        """Find previously completed tasks similar to this one."""
        llm = get_llm()
        vector = await llm.embed(description[:2000])
        if not vector:
            return []
        results = await self._client.query_points(
            collection_name=self._collections["tasks"],
            query=vector,
            limit=limit,
            score_threshold=0.4,
        )
        return [{**p.payload, "id": str(p.id), "similarity": p.score} for p in results.points]

    # ── Agent Memory ──────────────────────────────────────────────────────

    async def save_agent_memory(self, agent_name: str, memory_key: str, content: str):
        """Save a memory for a specific agent."""
        llm = get_llm()
        point_id = hashlib.sha256(f"{agent_name}:{memory_key}".encode()).hexdigest()[:32]
        vector = await llm.embed(f"{agent_name} {memory_key}: {content}"[:2000])
        if not vector:
            return
        await self._client.upsert(
            collection_name=self._collections["agents"],
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "agent": agent_name,
                        "key": memory_key,
                        "content": content,
                        "timestamp": time.time(),
                    },
                )
            ],
        )

    async def recall_agent_memory(self, agent_name: str, query: str, *, limit: int = 5) -> list[dict]:
        """Recall relevant memories for an agent."""
        llm = get_llm()
        vector = await llm.embed(f"{agent_name}: {query}"[:2000])
        if not vector:
            return []
        results = await self._client.query_points(
            collection_name=self._collections["agents"],
            query=vector,
            query_filter=models.Filter(must=[models.FieldCondition(key="agent", match=models.MatchValue(value=agent_name))]),
            limit=limit,
            score_threshold=0.3,
        )
        return [{**p.payload, "similarity": p.score} for p in results.points]

    # ── Stats ─────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        counts = {}
        for name, collection in self._collections.items():
            counts[name] = await self._count(collection)
        return {**self._stats, "collections": counts}


# Singleton
_kb: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
