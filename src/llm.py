"""LLM client — Ollama integration for generation and embeddings."""

import logging

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for Ollama API — generation, chat, and embeddings."""

    def __init__(self):
        s = get_settings()
        self._base_url = s.ollama_base_url
        self._think_model = s.ollama_think_model
        self._fast_model = s.ollama_fast_model
        self._embed_model = s.ollama_embed_model
        self._timeout = s.ollama_timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Generation ────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        think: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Single-turn generation. Returns the response text."""
        client = await self._get_client()
        body: dict = {
            "model": model or self._fast_model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            body["system"] = system

        resp = await client.post("/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        think: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Multi-turn chat. Returns the assistant's response."""
        client = await self._get_client()
        body: dict = {
            "model": model or self._fast_model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = await client.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    async def think(self, prompt: str, *, system: str = "") -> str:
        """Deep thinking with large model and thinking enabled."""
        return await self.generate(
            prompt,
            system=system,
            model=self._think_model,
            think=True,
            temperature=0.6,
            max_tokens=8192,
        )

    # ── Embeddings ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        client = await self._get_client()
        resp = await client.post(
            "/api/embed",
            json={"model": self._embed_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        client = await self._get_client()
        resp = await client.post(
            "/api/embed",
            json={"model": self._embed_model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])

    # ── Health ────────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Check Ollama connectivity and available models."""
        try:
            client = await self._get_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"status": "ok", "models": models}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton
_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
