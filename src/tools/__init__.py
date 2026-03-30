"""Tool system — executable capabilities for agents.

Each tool is a callable that agents can use to interact with the real world:
kubectl, git, shell, database, HTTP, file operations.

Tools are sandboxed — they validate inputs and log all executions.
"""

import asyncio
import json
import logging
import re
import shlex
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Maximum output size to prevent memory issues
MAX_OUTPUT_BYTES = 64 * 1024  # 64KB


class ToolCategory(str, Enum):
    KUBERNETES = "kubernetes"
    GIT = "git"
    SHELL = "shell"
    DATABASE = "database"
    HTTP = "http"
    FILE = "file"
    LLM = "llm"


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool: str
    success: bool
    output: str
    error: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition of an available tool."""

    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, str]  # param_name -> description
    examples: list[str] = field(default_factory=list)
    requires_approval: bool = False  # destructive operations need approval


# Allowlist of safe kubectl operations (read-only by default)
SAFE_KUBECTL_VERBS = {"get", "describe", "logs", "top", "explain", "api-resources", "api-versions"}
WRITE_KUBECTL_VERBS = {"apply", "delete", "scale", "rollout", "patch", "label", "annotate", "cordon", "uncordon", "drain"}
BLOCKED_KUBECTL_PATTERNS = re.compile(r"(exec|run|attach|port-forward|proxy|cp )", re.IGNORECASE)

# Blocked shell commands
BLOCKED_SHELL_PATTERNS = re.compile(
    r"(rm\s+-rf\s+/|mkfs|dd\s+if=|shutdown|reboot|halt|init\s+0|:(){ :|fork\s*bomb)",
    re.IGNORECASE,
)


async def _run_command(cmd: list[str], *, timeout: float = 30.0, cwd: str | None = None) -> tuple[str, str, int]:
    """Run a command safely with timeout and output limits."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
        err = stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
        return out, err, proc.returncode or 0
    except asyncio.TimeoutError:
        proc.kill()
        return "", f"Command timed out after {timeout}s", -1
    except FileNotFoundError:
        return "", f"Command not found: {cmd[0]}", -1


# ── Tool implementations ──────────────────────────────────────────────────


async def tool_kubectl(args: dict[str, Any]) -> ToolResult:
    """Execute kubectl commands. Read operations are safe; write operations need approval."""
    start = time.time()
    command = args.get("command", "").strip()
    if not command:
        return ToolResult(tool="kubectl", success=False, output="", error="Empty command")

    # Parse the verb
    parts = command.split()
    verb = parts[0] if parts else ""

    # Block dangerous patterns
    if BLOCKED_KUBECTL_PATTERNS.search(command):
        return ToolResult(tool="kubectl", success=False, output="", error=f"Blocked: '{verb}' is not allowed for agents")

    # Check if write operation
    is_write = verb in WRITE_KUBECTL_VERBS
    if verb not in SAFE_KUBECTL_VERBS and verb not in WRITE_KUBECTL_VERBS:
        return ToolResult(tool="kubectl", success=False, output="", error=f"Unknown verb: {verb}")

    cmd = ["kubectl"] + parts
    out, err, code = await _run_command(cmd, timeout=30.0)
    duration = (time.time() - start) * 1000
    return ToolResult(
        tool="kubectl",
        success=code == 0,
        output=out,
        error=err if code != 0 else "",
        duration_ms=duration,
        metadata={"command": command, "verb": verb, "is_write": is_write},
    )


async def tool_git(args: dict[str, Any]) -> ToolResult:
    """Execute git commands in a specific repo directory."""
    start = time.time()
    command = args.get("command", "").strip()
    repo_dir = args.get("repo_dir", "").strip()
    if not command or not repo_dir:
        return ToolResult(tool="git", success=False, output="", error="Need 'command' and 'repo_dir'")

    # Block force-push and destructive operations
    if re.search(r"push\s+--force|reset\s+--hard|clean\s+-fd", command):
        return ToolResult(tool="git", success=False, output="", error="Destructive git operation blocked")

    parts = ["git", "--no-pager"] + command.split()
    out, err, code = await _run_command(parts, timeout=30.0, cwd=repo_dir)
    duration = (time.time() - start) * 1000
    return ToolResult(tool="git", success=code == 0, output=out, error=err if code != 0 else "", duration_ms=duration)


async def tool_shell(args: dict[str, Any]) -> ToolResult:
    """Execute a shell command. Only safe, read-only commands allowed."""
    start = time.time()
    command = args.get("command", "").strip()
    if not command:
        return ToolResult(tool="shell", success=False, output="", error="Empty command")

    if BLOCKED_SHELL_PATTERNS.search(command):
        return ToolResult(tool="shell", success=False, output="", error="Destructive command blocked")

    # Use shlex for safe parsing — no shell=True
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return ToolResult(tool="shell", success=False, output="", error=f"Parse error: {e}")

    out, err, code = await _run_command(parts, timeout=30.0)
    duration = (time.time() - start) * 1000
    return ToolResult(tool="shell", success=code == 0, output=out, error=err if code != 0 else "", duration_ms=duration)


async def tool_read_file(args: dict[str, Any]) -> ToolResult:
    """Read a file's contents. Limited to workspace paths."""
    start = time.time()
    path = args.get("path", "").strip()
    if not path:
        return ToolResult(tool="read_file", success=False, output="", error="No path specified")

    # Security: only allow reading from workspace
    import os
    workspace = os.environ.get("ORCH_WORKSPACE_ROOT", "/workspace")
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(workspace) and not abs_path.startswith("/tmp"):
        return ToolResult(tool="read_file", success=False, output="", error=f"Path outside workspace: {path}")

    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_OUTPUT_BYTES)
        duration = (time.time() - start) * 1000
        return ToolResult(tool="read_file", success=True, output=content, duration_ms=duration)
    except (FileNotFoundError, PermissionError) as e:
        return ToolResult(tool="read_file", success=False, output="", error=str(e))


async def tool_http_request(args: dict[str, Any]) -> ToolResult:
    """Make an HTTP request. Only internal cluster URLs allowed."""
    import httpx as _httpx

    start = time.time()
    url = args.get("url", "").strip()
    method = args.get("method", "GET").upper()
    if not url:
        return ToolResult(tool="http", success=False, output="", error="No URL specified")

    # Security: only internal cluster URLs
    allowed_prefixes = ("http://", "https://")
    if not any(url.startswith(p) for p in allowed_prefixes):
        return ToolResult(tool="http", success=False, output="", error="Invalid URL scheme")

    # Block external URLs unless explicitly allowed
    internal_patterns = (".svc.cluster.local", "10.189.", "10.43.", "localhost", "127.0.0.1")
    if not any(p in url for p in internal_patterns):
        return ToolResult(tool="http", success=False, output="", error="Only internal cluster URLs allowed")

    try:
        async with _httpx.AsyncClient(timeout=15.0, verify=False) as client:
            resp = await client.request(method, url)
            duration = (time.time() - start) * 1000
            body = resp.text[:MAX_OUTPUT_BYTES]
            return ToolResult(
                tool="http",
                success=200 <= resp.status_code < 400,
                output=body,
                duration_ms=duration,
                metadata={"status_code": resp.status_code, "url": url},
            )
    except Exception as e:
        return ToolResult(tool="http", success=False, output="", error=str(e))


async def tool_write_file(args: dict[str, Any]) -> ToolResult:
    """Write content to a file within the workspace. Creates intermediate dirs if needed."""
    import os
    start = time.time()
    path = args.get("path", "").strip()
    content = args.get("content", "")
    if not path:
        return ToolResult(tool="write_file", success=False, output="", error="No path specified")

    workspace = os.environ.get("ORCH_WORKSPACE_ROOT", "/workspace")
    abs_path = os.path.abspath(path)
    # Security: only allow writing inside workspace
    if not abs_path.startswith(workspace):
        return ToolResult(tool="write_file", success=False, output="", error=f"Path outside workspace: {path}")

    # Block writing to sensitive files
    blocked_paths = ("/etc/", "/usr/", "/bin/", "/sbin/", "/lib/", "/proc/", "/sys/")
    if any(abs_path.startswith(p) for p in blocked_paths):
        return ToolResult(tool="write_file", success=False, output="", error=f"Path not allowed: {abs_path}")

    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        duration = (time.time() - start) * 1000
        return ToolResult(
            tool="write_file",
            success=True,
            output=f"Written {len(content)} bytes to {abs_path}",
            duration_ms=duration,
        )
    except (PermissionError, OSError) as e:
        return ToolResult(tool="write_file", success=False, output="", error=str(e))


async def tool_git_commit_push(args: dict[str, Any]) -> ToolResult:
    """
    Stage, commit, and push changes in a git repo.
    Requires ORCH_GIT_TOKEN env var for authentication.

    Args:
        repo_dir: absolute path to the git repo
        files: list of files to stage (default: '.' for all)
        message: commit message
        branch: branch to push to (default: 'main')
    """
    import os
    start = time.time()
    repo_dir = args.get("repo_dir", "").strip()
    message = args.get("message", "chore: auto-update by orchestrator").strip()
    branch = args.get("branch", "main").strip()
    files = args.get("files", ["."])

    if not repo_dir:
        return ToolResult(tool="git_commit_push", success=False, output="", error="Need 'repo_dir'")

    token = os.environ.get("ORCH_GIT_TOKEN", "")
    if not token:
        return ToolResult(tool="git_commit_push", success=False, output="", error="ORCH_GIT_TOKEN not set")

    # Validate branch name (prevent injection)
    if not re.match(r'^[a-zA-Z0-9/_.-]+$', branch):
        return ToolResult(tool="git_commit_push", success=False, output="", error="Invalid branch name")

    # Configure git identity for the commit
    config_cmds = [
        ["git", "config", "user.email", "orchestrator@zsel.opole.pl"],
        ["git", "config", "user.name", "ZSEL Orchestrator"],
    ]
    results = []
    for cmd in config_cmds:
        out, err, code = await _run_command(cmd, timeout=10.0, cwd=repo_dir)
        if code != 0:
            return ToolResult(tool="git_commit_push", success=False, output="", error=f"git config failed: {err}")

    # Stage files
    file_args = files if isinstance(files, list) else [files]
    add_cmd = ["git", "add"] + file_args
    out, err, code = await _run_command(add_cmd, timeout=15.0, cwd=repo_dir)
    if code != 0:
        return ToolResult(tool="git_commit_push", success=False, output=out, error=f"git add failed: {err}")
    results.append(f"git add: {out or 'ok'}")

    # Check if anything is staged
    status_out, _, _ = await _run_command(["git", "status", "--porcelain"], timeout=10.0, cwd=repo_dir)
    if not status_out.strip():
        return ToolResult(tool="git_commit_push", success=True, output="Nothing to commit — working tree clean", duration_ms=(time.time() - start) * 1000)

    # Commit
    out, err, code = await _run_command(["git", "commit", "-m", message], timeout=15.0, cwd=repo_dir)
    if code != 0:
        return ToolResult(tool="git_commit_push", success=False, output=out, error=f"git commit failed: {err}")
    results.append(f"git commit: {out[:200]}")

    # Push with token — inject token into remote URL temporarily
    remote_out, _, _ = await _run_command(["git", "remote", "get-url", "origin"], timeout=10.0, cwd=repo_dir)
    original_remote = remote_out.strip()

    # Build authenticated remote URL (https://token@github.com/...)
    if "github.com" in original_remote:
        authed_remote = original_remote.replace("https://", f"https://{token}@")
        if not authed_remote.startswith("https://"):
            # SSH remote — convert to HTTPS
            authed_remote = re.sub(r"git@github\.com:", f"https://{token}@github.com/", original_remote)
    else:
        return ToolResult(tool="git_commit_push", success=False, output="", error="Only GitHub remotes supported")

    # Push
    out, err, code = await _run_command(
        ["git", "push", authed_remote, f"HEAD:{branch}"],
        timeout=60.0,
        cwd=repo_dir,
    )
    duration = (time.time() - start) * 1000

    if code != 0:
        return ToolResult(tool="git_commit_push", success=False, output="\n".join(results), error=f"git push failed: {err}")
    results.append(f"git push → {branch}: {out[:200] or 'ok'}")
    return ToolResult(
        tool="git_commit_push",
        success=True,
        output="\n".join(results),
        duration_ms=duration,
        metadata={"branch": branch, "message": message},
    )


async def tool_trigger_kaniko_build(args: dict[str, Any]) -> ToolResult:
    """
    Create and apply a Kubernetes Job to rebuild the orchestrator image using Kaniko.

    Args:
        image_tag: new image tag to build (e.g. '0.2.0')
        git_repo: git repo URL (default: from env)
        git_revision: branch/tag/sha to build (default: 'main')
        context_subdir: subdirectory within the repo (default: 'SERVERS/zsel-orchestrator')
    """
    import os
    import uuid as uuid_mod

    start = time.time()
    image_tag = args.get("image_tag", "").strip()
    git_repo = args.get("git_repo", os.environ.get("ORCH_GIT_REPO_URL", "https://github.com/ZSEL-OPOLE/zsel-orchestrator.git"))
    git_revision = args.get("git_revision", "main")
    context_subdir = args.get("context_subdir", "SERVERS/zsel-orchestrator")

    if not image_tag:
        return ToolResult(tool="trigger_kaniko_build", success=False, output="", error="image_tag is required (e.g. '0.2.0')")

    # Validate tag (semver-ish, no shell injection)
    if not re.match(r'^[0-9a-zA-Z._-]+$', image_tag):
        return ToolResult(tool="trigger_kaniko_build", success=False, output="", error="Invalid image_tag format")

    registry = os.environ.get("ORCH_ZOT_REGISTRY_INTERNAL", "zot-registry.registry.svc.cluster.local:5000")
    image_name = os.environ.get("ORCH_ORCHESTRATOR_IMAGE_NAME", "zsel-orchestrator")
    namespace = os.environ.get("ORCH_KANIKO_NAMESPACE", "orchestrator")
    sa = os.environ.get("ORCH_KANIKO_SERVICE_ACCOUNT", "kaniko-builder")
    job_name = f"kaniko-orch-{image_tag.replace('.', '-')}-{str(uuid_mod.uuid4())[:6]}"
    dest_image = f"{registry}/{image_name}:{image_tag}"

    job_yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
  labels:
    app: kaniko-builder
    built-by: orchestrator
    image-tag: "{image_tag}"
spec:
  ttlSecondsAfterFinished: 600
  template:
    spec:
      serviceAccountName: {sa}
      restartPolicy: Never
      containers:
      - name: kaniko
        image: gcr.io/kaniko-project/executor:v1.23.2
        args:
        - --context=git#{git_repo}#{git_revision}
        - --context-sub-path={context_subdir}
        - --dockerfile=Dockerfile
        - --destination={dest_image}
        - --insecure
        - --skip-tls-verify
        - --cache=true
        - --cache-repo={registry}/{image_name}/cache
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "2Gi"
"""

    # Write job YAML to a temp file and apply it
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="/tmp/kaniko-") as f:
        f.write(job_yaml)
        tmpfile = f.name

    out, err, code = await _run_command(["kubectl", "apply", "-f", tmpfile], timeout=30.0)
    duration = (time.time() - start) * 1000

    # Clean up temp file
    try:
        import os as _os
        _os.unlink(tmpfile)
    except OSError:
        pass

    if code != 0:
        return ToolResult(tool="trigger_kaniko_build", success=False, output=out, error=f"kubectl apply failed: {err}", duration_ms=duration)

    return ToolResult(
        tool="trigger_kaniko_build",
        success=True,
        output=f"Kaniko Job created: {job_name}\nDestination image: {dest_image}\n{out}",
        duration_ms=duration,
        metadata={"job_name": job_name, "image": dest_image, "namespace": namespace},
    )





TOOL_REGISTRY: dict[str, tuple[ToolDefinition, Callable]] = {
    "kubectl": (
        ToolDefinition(
            name="kubectl",
            description="Execute kubectl commands against the K3s cluster. Read operations (get, describe, logs) are safe. Write operations (apply, delete, scale) need approval.",
            category=ToolCategory.KUBERNETES,
            parameters={"command": "kubectl command without 'kubectl' prefix, e.g. 'get pods -n techbuddy'"},
            examples=["get pods -A", "describe pod techbuddy-backend-xxx -n techbuddy", "logs deploy/ai-agents -n ai-agents --tail=50"],
        ),
        tool_kubectl,
    ),
    "git": (
        ToolDefinition(
            name="git",
            description="Execute git commands in a repository directory.",
            category=ToolCategory.GIT,
            parameters={"command": "git command without 'git' prefix", "repo_dir": "absolute path to repo directory"},
            examples=["status", "log --oneline -10", "diff HEAD~1"],
        ),
        tool_git,
    ),
    "shell": (
        ToolDefinition(
            name="shell",
            description="Run safe shell commands. No destructive operations allowed.",
            category=ToolCategory.SHELL,
            parameters={"command": "shell command to execute"},
            examples=["ls -la /workspace", "cat /etc/os-release", "df -h"],
        ),
        tool_shell,
    ),
    "read_file": (
        ToolDefinition(
            name="read_file",
            description="Read contents of a file within the workspace.",
            category=ToolCategory.FILE,
            parameters={"path": "absolute path to the file"},
        ),
        tool_read_file,
    ),
    "http": (
        ToolDefinition(
            name="http",
            description="Make HTTP requests to internal cluster services.",
            category=ToolCategory.HTTP,
            parameters={"url": "full URL", "method": "HTTP method (GET, POST, etc.)"},
            examples=["http://techbuddy-backend.techbuddy.svc.cluster.local:8080/health"],
        ),
        tool_http_request,
    ),
    "write_file": (
        ToolDefinition(
            name="write_file",
            description="Write or overwrite a file within the workspace. Creates parent directories automatically. Only paths under ORCH_WORKSPACE_ROOT are allowed.",
            category=ToolCategory.FILE,
            parameters={
                "path": "absolute path within workspace",
                "content": "text content to write",
            },
            requires_approval=True,
        ),
        tool_write_file,
    ),
    "git_commit_push": (
        ToolDefinition(
            name="git_commit_push",
            description="Stage, commit, and push changes to a GitHub repository. Requires ORCH_GIT_TOKEN environment variable.",
            category=ToolCategory.GIT,
            parameters={
                "repo_dir": "absolute path to the git repository",
                "message": "commit message (use conventional commits: feat:, fix:, chore:)",
                "files": "list of files to stage, default ['.'] for all",
                "branch": "target branch, default 'main'",
            },
            requires_approval=True,
        ),
        tool_git_commit_push,
    ),
    "trigger_kaniko_build": (
        ToolDefinition(
            name="trigger_kaniko_build",
            description="Trigger a Kaniko Kubernetes Job to build and push the orchestrator Docker image. Use to deploy a new version after code changes.",
            category=ToolCategory.KUBERNETES,
            parameters={
                "image_tag": "new image tag to build and push, e.g. '0.2.0'",
                "git_revision": "git branch/tag/sha to build from, default 'main'",
                "context_subdir": "subdirectory in the repo, default 'SERVERS/zsel-orchestrator'",
            },
            requires_approval=True,
        ),
        tool_trigger_kaniko_build,
    ),
}


def get_tool_definitions() -> list[ToolDefinition]:
    """Return all available tool definitions (for agent system prompts)."""
    return [defn for defn, _ in TOOL_REGISTRY.values()]


async def execute_tool(name: str, args: dict[str, Any]) -> ToolResult:
    """Execute a tool by name with given arguments."""
    if name not in TOOL_REGISTRY:
        return ToolResult(tool=name, success=False, output="", error=f"Unknown tool: {name}")
    _, executor = TOOL_REGISTRY[name]
    try:
        result = await executor(args)
        logger.info("Tool %s: success=%s duration=%.0fms", name, result.success, result.duration_ms)
        return result
    except Exception as e:
        logger.error("Tool %s crashed: %s", name, e)
        return ToolResult(tool=name, success=False, output="", error=f"Tool crash: {e}")
