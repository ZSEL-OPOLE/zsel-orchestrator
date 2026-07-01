# ═══════════════════════════════════════════════════════════════════════════════
# axe-core-runner — WCAG accessibility auditor (ARM64)
# Node 20 slim + Chromium (system) + axe-core + playwright-core
# Brak GUI — headless only.
# ═══════════════════════════════════════════════════════════════════════════════

FROM node:20-bookworm-slim AS deps

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

# ── Final image ───────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-liberation \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r axe && useradd -r -g axe -d /app -s /sbin/nologin axe

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY runner.mjs ./

ENV CHROMIUM_PATH=/usr/bin/chromium
ENV NODE_ENV=production

# Tymczasowe katalogi wymagane przez Chromium (readOnlyRootFilesystem=true w K8s → tmpfs mounts)
VOLUME ["/tmp"]

USER axe

ENTRYPOINT ["node", "runner.mjs"]
