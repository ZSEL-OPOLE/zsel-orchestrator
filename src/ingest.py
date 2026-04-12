"""
Knowledge Ingestion — zasilanie Qdrant wiedzą o całej infrastrukturze.

Indeksuje:
  - GITOPS/ — manifesty K8s, ArgoCD, Helm, dokumenty
  - SERVERS/ — kod aplikacji (Python, TypeScript), README
  - NETWORK/ — konfiguracje MikroTik, VLAN
  - scripts/ — skrypty deployu, seedy

Każdy plik jest chunkowany (okno 1000 tokenów, overlap 200).
Duplikaty wykrywane przez SHA256 hash.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

# Max file size to index (skip big binary/generated files)
MAX_FILE_SIZE = 200 * 1024  # 200KB
CHUNK_SIZE = 1000  # ~chars per chunk
CHUNK_OVERLAP = 200

# Files/dirs to skip
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".next",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "alembic/versions",
    "migrations",
    ".idea",
    ".vscode",
}

SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".lock",  # package-lock, poetry.lock — noisy
}

PRIORITY_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".json",
    ".toml",  # Infra configs
    ".py",
    ".ts",
    ".tsx",
    ".js",  # Code
    ".md",
    ".txt",  # Docs
    ".sh",
    ".bash",  # Scripts
    ".conf",
    ".cfg",
    ".ini",  # Config files
    ".rsc",  # MikroTik RouterOS scripts
    ".env.example",  # Env templates
}

# Category detection by path pattern
CATEGORY_MAP = [
    ("GITOPS", "infra"),
    ("NETWORK", "network"),
    ("scripts", "scripts"),
    ("techbuddy", "app"),
    ("servicedesk", "app"),
    ("zsel-ai-agents", "app"),
    ("zsel-orchestrator", "app"),
    ("00-docs", "docs"),
    ("k3s", "infra"),
    ("platform", "infra"),
    ("manifests", "infra"),
]


def _detect_category(path: str) -> str:
    for pattern, category in CATEGORY_MAP:
        if pattern.lower() in path.lower():
            return category
    return "general"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_file_metadata(path: Path, workspace_root: str) -> dict:
    """Extract metadata from file."""
    rel = str(path.relative_to(workspace_root))
    return {
        "file_path": rel,
        "extension": path.suffix,
        "size": path.stat().st_size,
        "repo": rel.split("/")[0] if "/" in rel else "root",
    }


def _should_skip(path: Path) -> bool:
    """Check if file/dir should be skipped."""
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if path.stat().st_size > MAX_FILE_SIZE:
        return True
    return False


async def run_ingestion(workspace: str, *, force: bool = False, console: Any = None) -> dict:
    """
    Main ingestion entry point.

    Walks workspace, indexes all relevant files into Qdrant.
    Returns stats dict.
    """
    from .knowledge import KnowledgeEntry, get_knowledge_base
    from .orchestrator import get_orchestrator

    orch = get_orchestrator()
    await orch.initialize()
    kb = get_knowledge_base()

    workspace_path = Path(workspace)
    if not workspace_path.exists():
        raise ValueError(f"Workspace not found: {workspace}")

    stats = {
        "files_scanned": 0,
        "files_indexed": 0,
        "chunks_created": 0,
        "files_skipped": 0,
        "errors": 0,
        "start_time": time.time(),
    }

    def log(msg: str):
        if console:
            console.print(msg)
        else:
            logger.info(msg)

    log(f"[cyan]Scanning workspace:[/cyan] {workspace}")

    # Walk all files
    for path in workspace_path.rglob("*"):
        if not path.is_file():
            continue

        # Skip hidden dirs and configured SKIP_DIRS
        parts = set(path.parts)
        if any(skip in parts for skip in SKIP_DIRS):
            continue
        if any(p.startswith(".") for p in path.parts[-3:] if not p.startswith(".git")):
            continue

        if _should_skip(path):
            stats["files_skipped"] += 1
            continue

        if path.suffix not in PRIORITY_EXTENSIONS and path.suffix != "":
            stats["files_skipped"] += 1
            continue

        stats["files_scanned"] += 1

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Cannot read %s: %s", path, e)
            stats["errors"] += 1
            continue

        if not text.strip():
            continue

        rel_path = str(path.relative_to(workspace_path))
        category = _detect_category(rel_path)
        metadata = _get_file_metadata(path, workspace)

        # Chunk file
        chunks = _chunk_text(text)
        file_indexed = False

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            chunk_id = f"{_file_hash(rel_path)}-{i}"

            # Add source context to chunk
            source_header = f"# File: {rel_path}\n# Chunk: {i + 1}/{len(chunks)}\n\n"
            content = source_header + chunk

            try:
                entry = KnowledgeEntry(
                    id=chunk_id,
                    content=content,
                    source=f"ws:{rel_path}",
                    category=category,
                    tags=[path.suffix.lstrip("."), category, metadata["repo"]],
                    metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                )
                await kb.add_knowledge(entry)
                stats["chunks_created"] += 1
                file_indexed = True
            except Exception as e:
                logger.error("Index error for %s chunk %d: %s", rel_path, i, e)
                stats["errors"] += 1

        if file_indexed:
            stats["files_indexed"] += 1
            if stats["files_indexed"] % 50 == 0:
                log(f"  [dim]Indexed {stats['files_indexed']} files, {stats['chunks_created']} chunks...[/dim]")

    # Also ingest built-in knowledge (infra facts, known patterns)
    await _ingest_builtin_knowledge(kb)

    stats["duration_s"] = time.time() - stats["start_time"]
    await orch.close()

    log("[bold green]✅ Ingestion complete:[/bold green]")
    log(f"  Files scanned: {stats['files_scanned']}")
    log(f"  Files indexed: {stats['files_indexed']}")
    log(f"  Chunks created: {stats['chunks_created']}")
    log(f"  Skipped: {stats['files_skipped']}")
    log(f"  Errors: {stats['errors']}")
    log(f"  Duration: {stats['duration_s']:.1f}s")

    return stats


async def _ingest_builtin_knowledge(kb: Any):
    """Ingest hardcoded infra knowledge that's not in files."""
    from .knowledge import KnowledgeEntry

    BUILTIN = [
        # Cluster topology
        KnowledgeEntry(
            content="""K3s cluster ZSEL Opole (v1.34+, arm64):
Nodes:
- zsel-cp1 (10.189.5.21 → 10.43.11.x) — control plane, limactl /opt/homebrew/bin
- zsel-cp2 (10.189.5.22 → 10.43.11.x) — control plane, limactl /usr/local/bin, CORDON during deploys
- zsel-cp3 (10.189.5.23 → 10.43.11.x) — control plane, limactl /usr/local/bin
- zsel-25  (10.189.5.25 → 10.43.11.x) — worker w3, limactl /usr/local/bin, qwen3.5:122b
- zsel-26  (10.189.5.26 → 10.43.11.x) — worker w1, limactl /opt/homebrew/bin
- zsel-27  (10.189.5.27 → 10.43.11.x) — worker w4, limactl /usr/local/bin, qwen3.5:122b
- zsel-28  (10.189.5.28 → 10.43.11.x) — worker w2, limactl /opt/homebrew/bin
SSH: ssh admin@10.189.5.{21-28}
Lima VMs: cp1→.21, cp2→.22, cp3→.23, w3→.25, w1→.26, w4→.27, w2→.28""",
            source="builtin:cluster-topology",
            category="infra",
            tags=["k3s", "cluster", "nodes", "topology"],
        ),
        # Ollama bridge
        KnowledgeEntry(
            content="""Ollama — lokalne LLM w ZSEL:
- Uruchomiony natively na macOS hostach w VLAN 703 (10.189.3.11-15)
- Dostęp z K3s via bridge: ClusterIP Service + Manual Endpoints
- Adres z K8s: http://ollama.llm.svc.cluster.local:11434
- Namespace: llm
- Modele: qwen3.5:122b (.25/.27), qwen3.5:35b/27b/9b/4b (reszta), nomic-embed-text (wszystkie)
- Qwen 3.5 gotcha: think=false wymagane! Inaczej odpowiedź w polu 'thinking', 'response' puste.
- LaunchAgent: ~/Library/LaunchAgents/com.ollama.serve.plist (auto-start)""",
            source="builtin:ollama-bridge",
            category="infra",
            tags=["ollama", "llm", "qwen3.5", "bridge", "vlan703"],
        ),
        # Qdrant
        KnowledgeEntry(
            content="""Qdrant — wektorowa baza danych:
- Namespace: qdrant
- ClusterIP: qdrant.qdrant.svc.cluster.local:6333
- Collections:
  * techbuddy_documents: 62,286 punktów (dane szkoły, FAQ, akty prawne ISAP)
  * techbuddy_knowledge: 37 punktów (stary seed)
  * techbuddy_reasoning_bank: wzorce routing (ReasoningBank)
  * orch_knowledge: wiedza infrastrukturalna (orchestrator)
  * orch_error_journal: dziennik błędów
  * orch_task_history: historia zadań
  * orch_agent_memory: pamięci agentów
- Embedding model: nomic-embed-text (768 dim, Ollama)
- Polish embedding alternative: sdadas/mmlw-roberta-large (1024 dim)""",
            source="builtin:qdrant",
            category="infra",
            tags=["qdrant", "vector-db", "embeddings"],
        ),
        # PostgreSQL
        KnowledgeEntry(
            content="""PostgreSQL — baza danych ZSEL (CloudNativePG):
- Namespace: database
- Cluster: pg-cluster (3 instancje, PostgreSQL 17.4)
- Pooler RW: pg-cluster-pooler-rw.database.svc.cluster.local:5432
- Pooler RO: pg-cluster-pooler-ro.database.svc.cluster.local:5432
- Bazy: techbuddy, servicedesk, orchestrator, nextcloud, moodle
- PROBLEM ZNANY: pooler pods nie docierają do K8s API (10.43.0.1:443) — fix: delete pooler pods → auto-recreate
- Backup: Barman + MinIO (s3://barman/)
- Storage: Longhorn 100Gi per instance""",
            source="builtin:postgresql",
            category="infra",
            tags=["postgresql", "cnpg", "database", "pgbouncer"],
        ),
        # Registry fix
        KnowledgeEntry(
            content="""Zot Registry — PROBLEM ZNANY (broken pipe na dużych blobbach):
- Zot NodePort: 10.189.5.41:30500 (mapa do zsel-cp1)
- PROBLEM: docker push przez NodePort → broken pipe na blobbach >12MB
- OBJAWY: "unexpected EOF", "broken pipe", "connection reset"
- WORKAROUND: docker save → scp → limactl copy → k3s ctr images import
  ```bash
  docker save IMAGE | gzip > /tmp/image.tar.gz
  scp /tmp/image.tar.gz admin@10.189.5.21:/tmp/
  /opt/homebrew/bin/limactl copy /tmp/image.tar.gz zsel-cp1:/tmp/image.tar.gz
  /opt/homebrew/bin/limactl shell zsel-cp1 -- sudo bash -c \
    "gunzip -c /tmp/image.tar.gz | k3s ctr images import -"
  ```
- Powtórz dla pozostałych nodów (różne ścieżki limactl!)""",
            source="builtin:zot-registry-workaround",
            category="infra",
            tags=["zot", "registry", "bug", "workaround", "broken-pipe"],
        ),
        # TechBuddy patterns
        KnowledgeEntry(
            content="""TechBuddy Backend — znane problemy i rozwiązania:
PROBLEM 1: AI hallucination — szukał tylko w techbuddy_knowledge (37 pt), pomijał techbuddy_documents (62k pt)
FIX: unified_agent.py szuka OBIE kolekcje, merge po score, top 8, próg 0.25

PROBLEM 2: CrashLoopBackOff → PgBouncer pooler nie osiągał K8s API
FIX: kubectl delete pods -l cnpg.io/cluster=pg-cluster -n database -l role=pooler
     Poolery się odradzą 1/1 → restart TechBuddy pods

PROBLEM 3: Qwen 3.5 nie odpowiada — pusty response
FIX: dodaj "think": false do żądania Ollama API

PROBLEM 4: GENERAL prompt miał hardcoded fake dane szkoły
FIX: usunięto fakty z prompta, dodano anti-hallucination rules, dane tylko z RAG""",
            source="builtin:techbuddy-fixes",
            category="app",
            tags=["techbuddy", "bugs", "rag", "qwen3.5", "pgbouncer"],
        ),
        # Network
        KnowledgeEntry(
            content="""Sieć ZSEL — topologia VLAN:
VLAN 100: Administracja
VLAN 200: Nauczyciele  
VLAN 300: Uczniowie
VLAN 400: CCTV/Monitoring
VLAN 500: WiFi (CAPsMAN)
VLAN 700: Serwery (główna sieć serwerowa)
  VLAN 701: Management
  VLAN 702: Storage (Longhorn, MinIO)
  VLAN 703: Ollama (LLM hosts) — 10.189.3.0/24
  VLAN 704: K3s nodes — 10.189.5.0/24
MikroTik: 46 urządzeń RouterOS, główne switche: CPDZSEL01, CPDZSEL02
Ingress: Traefik, MetalLB (10.189.4.x)
DNS: FreeIPA (LDAP+Kerberos) + CoreDNS (K8s)
VPN: WireGuard "vpn-zsel", scutil""",
            source="builtin:network-topology",
            category="network",
            tags=["vlan", "mikrotik", "network", "traefik", "metallb"],
        ),
        # Security patterns
        KnowledgeEntry(
            content="""Bezpieczeństwo ZSEL — wzorce:
ZAWSZE:
- SealedSecrets dla secretów K8s (kubeseal --cert /tmp/sealed-secrets-cert.pem)
- runAsNonRoot: true, readOnlyRootFilesystem: true, drop ALL capabilities
- seccompProfile: RuntimeDefault lub Localhost
- NetworkPolicy per namespace (CNI: Cilium)
- Parameterized SQL (nigdy f-strings!)
- Validate all user inputs

NIGDY:
- Nie commituj plain kind:Secret do GIT!
- Nie używaj :latest w produkcji
- Nie uruchamiaj jako root
- Nie expose wewnętrznych API bez auth

RBAC: ServiceAccount per deployment, minimal permissions
Cert-manager: Let's Encrypt (produkcja) / self-signed (dev)
Realm: ZSEL.OPOLE.PL, Keycloak OIDC (auth.zsel.opole.pl)""",
            source="builtin:security-patterns",
            category="security",
            tags=["security", "rbac", "sealedsecrets", "networkpolicy", "owasp"],
        ),
    ]

    for entry in BUILTIN:
        try:
            await kb.add_knowledge(entry)
        except Exception as e:
            logger.error("Builtin ingest error for %s: %s", entry.source, e)

    logger.info("Ingested %d built-in knowledge entries", len(BUILTIN))


if __name__ == "__main__":
    import sys

    workspace = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    asyncio.run(run_ingestion(workspace))
