"""Tests for ZSEL Orchestrator configuration."""

import os

import pytest
from pydantic import ValidationError


class TestSettings:
    """Tests for the Settings class."""

    def test_default_models(self) -> None:
        from src.config import Settings

        s = Settings()
        assert s.ollama_think_model == "qwen3.5:27b"
        assert s.ollama_fast_model == "qwen3.5:9b"
        assert s.ollama_embed_model == "nomic-embed-text"

    def test_default_embed_dim(self) -> None:
        from src.config import Settings

        s = Settings()
        assert s.ollama_embed_dim == 768

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables with ORCH_ prefix should override defaults."""
        monkeypatch.setenv("ORCH_OLLAMA_FAST_MODEL", "gemma4:e4b")
        from src.config import Settings

        s = Settings()
        assert s.ollama_fast_model == "gemma4:e4b"

    def test_max_task_depth_default(self) -> None:
        from src.config import Settings

        s = Settings()
        assert s.max_task_depth == 5
        assert s.max_parallel_agents == 3

    def test_learning_enabled_default(self) -> None:
        from src.config import Settings

        s = Settings()
        assert s.learning_enabled is True

    def test_qdrant_collection_names(self) -> None:
        from src.config import Settings

        s = Settings()
        assert "orch_" in s.qdrant_collection_knowledge
        assert "orch_" in s.qdrant_collection_errors
        assert "orch_" in s.qdrant_collection_tasks

    def test_get_settings_is_cached(self) -> None:
        from src.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # lru_cache should return same instance
