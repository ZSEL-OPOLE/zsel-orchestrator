"""Tests for HealthAggregator data models and non-network logic."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.health import (
    POLL_INTERVAL_SECONDS,
    ServiceHealth,
    ServiceStatus,
    _build_service_checks,
)


class TestServiceStatus:
    def test_values_are_strings(self) -> None:
        for status in ServiceStatus:
            assert isinstance(status.value, str)

    def test_required_statuses_exist(self) -> None:
        values = {s.value for s in ServiceStatus}
        assert "healthy" in values
        assert "unhealthy" in values
        assert "degraded" in values
        assert "unknown" in values


class TestServiceHealth:
    def test_default_status_unknown(self) -> None:
        sh = ServiceHealth(name="test-svc", url="http://example")
        assert sh.status == ServiceStatus.UNKNOWN

    def test_to_dict_keys(self) -> None:
        sh = ServiceHealth(name="svc", url="http://x", status=ServiceStatus.HEALTHY)
        d = sh.to_dict()
        assert "name" in d
        assert "status" in d
        assert "response_time_ms" in d
        assert "consecutive_failures" in d
        assert "last_check_ago_s" in d

    def test_to_dict_status_is_string(self) -> None:
        sh = ServiceHealth(name="svc", url="http://x", status=ServiceStatus.DEGRADED)
        d = sh.to_dict()
        assert d["status"] == "degraded"  # string, not enum

    def test_last_check_ago_zero_when_never_checked(self) -> None:
        sh = ServiceHealth(name="svc", url="http://x")
        d = sh.to_dict()
        # last_check == 0.0 → last_check_ago_s should be 0
        assert d["last_check_ago_s"] == 0

    def test_consecutive_failures_starts_at_zero(self) -> None:
        sh = ServiceHealth(name="svc", url="http://x")
        assert sh.consecutive_failures == 0


class TestBuildServiceChecks:
    def test_returns_list(self) -> None:
        checks = _build_service_checks()
        assert isinstance(checks, list)
        assert len(checks) > 0

    def test_each_entry_has_name_and_url(self) -> None:
        for svc in _build_service_checks():
            assert "name" in svc
            assert "url" in svc
            assert svc["name"] != ""
            assert svc["url"].startswith("http")

    def test_includes_key_services(self) -> None:
        names = {s["name"] for s in _build_service_checks()}
        assert "techbuddy-backend" in names
        assert "keycloak" in names
        assert "ollama" in names
        assert "qdrant" in names

    def test_no_duplicate_names(self) -> None:
        names = [s["name"] for s in _build_service_checks()]
        assert len(names) == len(set(names))


class TestHealthAggregatorInit:
    def test_pre_populates_results_as_unknown(self) -> None:
        from src.health import HealthAggregator

        agg = HealthAggregator()
        result = agg.get_all()
        # All services start as unknown — healthy count = 0
        assert result["healthy"] == 0
        assert result["total"] > 0
        assert result["status"] == "down"  # no healthy services yet

    def test_get_all_structure(self) -> None:
        from src.health import HealthAggregator

        agg = HealthAggregator()
        result = agg.get_all()
        assert "status" in result
        assert "healthy" in result
        assert "total" in result
        assert "services" in result
        assert "poll_interval_s" in result
        assert result["total"] > 0

    def test_services_list_has_name_and_status(self) -> None:
        from src.health import HealthAggregator

        agg = HealthAggregator()
        result = agg.get_all()
        for svc in result["services"]:
            assert "name" in svc
            assert "status" in svc
            assert svc["status"] == "unknown"  # not polled yet

    def test_poll_interval_is_reasonable(self) -> None:
        # Should be between 10 and 300 seconds
        assert 10 <= POLL_INTERVAL_SECONDS <= 300
