# ═══════════════════════════════════════════════════════════════════════════════
# ZSEL Orchestrator — multi-stage arm64 build
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ── Final image ───────────────────────────────────────────────────────────────
FROM python:3.12-slim

RUN groupadd -r orch && useradd -r -g orch -d /app -s /sbin/nologin orch

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/

# Healthcheck using internal /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

EXPOSE 8080

USER orch

ENV PYTHONUNBUFFERED=1
ENV ORCH_LOCAL_MODE=false

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
