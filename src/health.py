"""Health Aggregator — polls all ZSEL services and exposes consolidated health status.

Runs as a background task in the orchestrator, polling every 30 seconds.
Each service check uses httpx with a short timeout to avoid blocking.
Results are cached in-memory and optionally in Redis.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
CHECK_TIMEOUT_SECONDS = 5.0


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health check result for a single service."""

    name: str
    url: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    response_time_ms: float = 0.0
    status_code: int = 0
    detail: str = ""
    last_check: float = 0.0
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 1),
            "status_code": self.status_code,
            "detail": self.detail,
            "last_check": self.last_check,
            "last_check_ago_s": round(time.time() - self.last_check, 1) if self.last_check else 0,
            "consecutive_failures": self.consecutive_failures,
        }


# ── Service definitions ───────────────────────────────────────────────────

def _build_service_checks() -> list[dict[str, str]]:
    """Build the list of services to check from config / known internal URLs."""
    settings = get_settings()
    return [
        {
            "name": "techbuddy-backend",
            "url": settings.health_url_techbuddy,
            "namespace": "techbuddy",
        },
        {
            "name": "servicedesk-backend",
            "url": settings.health_url_servicedesk,
            "namespace": "servicedesk",
        },
        {
            "name": "keycloak",
            "url": settings.health_url_keycloak,
            "namespace": "keycloak",
        },
        {
            "name": "moodle",
            "url": settings.health_url_moodle,
            "namespace": "moodle",
        },
        {
            "name": "nextcloud",
            "url": settings.health_url_nextcloud,
            "namespace": "nextcloud",
        },
        {
            "name": "stalwart-mail",
            "url": settings.health_url_stalwart,
            "namespace": "stalwart",
        },
        {
            "name": "ollama",
            "url": f"{settings.ollama_base_url}/api/tags",
            "namespace": "llm",
        },
        {
            "name": "qdrant",
            "url": f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz",
            "namespace": "qdrant",
        },
        {
            "name": "argocd",
            "url": settings.health_url_argocd,
            "namespace": "argocd",
        },
        {
            "name": "grafana",
            "url": settings.health_url_grafana,
            "namespace": "monitoring",
        },
    ]


# ── HealthAggregator ──────────────────────────────────────────────────────


class HealthAggregator:
    """Background service that polls all ZSEL services health endpoints."""

    def __init__(self) -> None:
        self._results: dict[str, ServiceHealth] = {}
        self._services = _build_service_checks()
        self._task: asyncio.Task | None = None
        self._running = False
        self._client: httpx.AsyncClient | None = None

        # Pre-populate results
        for svc in self._services:
            self._results[svc["name"]] = ServiceHealth(
                name=svc["name"],
                url=svc["url"],
            )

    async def start(self) -> None:
        """Start the background polling loop."""
        self._running = True
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(CHECK_TIMEOUT_SECONDS, connect=3.0),
            verify=False,
            follow_redirects=True,
        )
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("HealthAggregator started — polling %d services every %ds", len(self._services), POLL_INTERVAL_SECONDS)

    async def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        logger.info("HealthAggregator stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop — checks all services concurrently."""
        # Initial poll immediately
        await self._check_all()

        while self._running:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            if not self._running:
                break
            await self._check_all()

    async def _check_all(self) -> None:
        """Check all services concurrently."""
        tasks = [self._check_service(svc) for svc in self._services]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_service(self, svc: dict[str, str]) -> None:
        """Check a single service health endpoint."""
        name = svc["name"]
        url = svc["url"]
        result = self._results[name]

        start = time.monotonic()
        try:
            resp = await self._client.get(url)
            elapsed_ms = (time.monotonic() - start) * 1000

            result.status_code = resp.status_code
            result.response_time_ms = elapsed_ms
            result.last_check = time.time()

            if resp.status_code < 400:
                result.status = ServiceStatus.HEALTHY
                result.detail = ""
                result.consecutive_failures = 0
            elif resp.status_code < 500:
                result.status = ServiceStatus.DEGRADED
                result.detail = f"HTTP {resp.status_code}"
                result.consecutive_failures = 0
            else:
                result.consecutive_failures += 1
                result.status = ServiceStatus.UNHEALTHY
                result.detail = f"HTTP {resp.status_code}"

        except httpx.ConnectError:
            result.consecutive_failures += 1
            result.status = ServiceStatus.UNHEALTHY
            result.detail = "Connection refused"
            result.response_time_ms = (time.monotonic() - start) * 1000
            result.last_check = time.time()
        except httpx.TimeoutException:
            result.consecutive_failures += 1
            result.status = ServiceStatus.UNHEALTHY
            result.detail = f"Timeout ({CHECK_TIMEOUT_SECONDS}s)"
            result.response_time_ms = CHECK_TIMEOUT_SECONDS * 1000
            result.last_check = time.time()
        except Exception as exc:
            result.consecutive_failures += 1
            result.status = ServiceStatus.UNHEALTHY
            result.detail = str(exc)[:200]
            result.response_time_ms = (time.monotonic() - start) * 1000
            result.last_check = time.time()
            logger.warning("Health check failed for %s: %s", name, exc)

        # Log status changes
        if result.consecutive_failures == 1:
            logger.warning("Service %s became UNHEALTHY: %s", name, result.detail)
        elif result.consecutive_failures == 0 and result.status == ServiceStatus.HEALTHY:
            pass  # Normal — don't spam logs

    # ── Public API ────────────────────────────────────────────────────────

    def get_all(self) -> dict[str, Any]:
        """Get consolidated health status for all services."""
        services = [r.to_dict() for r in self._results.values()]
        healthy_count = sum(1 for r in self._results.values() if r.status == ServiceStatus.HEALTHY)
        total = len(self._results)

        if healthy_count == total:
            overall = "healthy"
        elif healthy_count >= total * 0.7:
            overall = "degraded"
        elif healthy_count > 0:
            overall = "critical"
        else:
            overall = "down"

        return {
            "status": overall,
            "healthy": healthy_count,
            "total": total,
            "services": services,
            "poll_interval_s": POLL_INTERVAL_SECONDS,
        }

    def get_service(self, name: str) -> dict[str, Any] | None:
        """Get health for a specific service."""
        result = self._results.get(name)
        return result.to_dict() if result else None

    def get_unhealthy(self) -> list[dict[str, Any]]:
        """Get only unhealthy services."""
        return [
            r.to_dict()
            for r in self._results.values()
            if r.status in (ServiceStatus.UNHEALTHY, ServiceStatus.DEGRADED, ServiceStatus.UNKNOWN)
        ]


# ── Singleton ─────────────────────────────────────────────────────────────

_health_aggregator: HealthAggregator | None = None


def get_health_aggregator() -> HealthAggregator:
    """Get or create the singleton HealthAggregator."""
    global _health_aggregator
    if _health_aggregator is None:
        _health_aggregator = HealthAggregator()
    return _health_aggregator
