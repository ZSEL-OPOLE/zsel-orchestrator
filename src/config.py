"""Configuration — all settings from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for ZSEL Orchestrator."""

    model_config = {"env_prefix": "ORCH_"}

    # --- LLM (Ollama) ---
    ollama_base_url: str = "http://ollama.llm.svc.cluster.local:11434"
    ollama_think_model: str = "qwen3.5:27b"
    ollama_fast_model: str = "qwen3.5:9b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_dim: int = 768
    ollama_timeout: float = 120.0

    # --- Vector DB (Qdrant) ---
    qdrant_host: str = "qdrant.qdrant.svc.cluster.local"
    qdrant_port: int = 6333
    qdrant_collection_knowledge: str = "orch_knowledge"
    qdrant_collection_errors: str = "orch_error_journal"
    qdrant_collection_tasks: str = "orch_task_history"
    qdrant_collection_agents: str = "orch_agent_memory"

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://orchestrator:orchestrator@pg-cluster-pooler-rw.database.svc.cluster.local:5432/orchestrator"

    # --- Redis ---
    redis_url: str = "redis://redis.cache.svc.cluster.local:6379/2"

    # --- Service ---
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    # --- Git / self-development ---
    git_token: str = ""  # ORCH_GIT_TOKEN — GitHub token z repo scope
    git_repo_url: str = "https://github.com/ZSEL-OPOLE/zsel-orchestrator.git"
    git_repo_name: str = "ZSEL-OPOLE/zsel-orchestrator"
    workspace_root: str = "/workspace"

    # --- Container build (Kaniko) ---
    zot_registry_internal: str = "zot-registry.registry.svc.cluster.local:5000"
    orchestrator_image_name: str = "zsel-orchestrator"
    kaniko_service_account: str = "kaniko-builder"
    kaniko_namespace: str = "orchestrator"

    # --- Health Aggregator (internal K8s service URLs) ---
    health_url_techbuddy: str = "http://techbuddy-backend.techbuddy.svc.cluster.local:8000/api/v1/health"
    health_url_servicedesk: str = "http://servicedesk.servicedesk.svc.cluster.local:8000/api/v1/health"
    health_url_keycloak: str = "http://keycloak-http.keycloak.svc.cluster.local:8080/health/ready"
    health_url_moodle: str = "http://moodle.moodle.svc.cluster.local:8080/admin/tool/health/"
    health_url_nextcloud: str = "http://nextcloud.nextcloud.svc.cluster.local:8080/status.php"
    health_url_stalwart: str = "http://stalwart.stalwart.svc.cluster.local:8080/healthz"
    health_url_argocd: str = "http://argocd-server.argocd.svc.cluster.local:8080/healthz"
    health_url_grafana: str = "http://grafana.monitoring.svc.cluster.local:3000/api/health"

    # --- Orchestrator ---
    max_task_depth: int = 5
    max_parallel_agents: int = 3
    learning_enabled: bool = True
    auto_index_interval: int = 3600  # seconds between auto-indexing


@lru_cache
def get_settings() -> Settings:
    return Settings()
