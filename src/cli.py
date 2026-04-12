"""
ZSEL Orchestrator CLI — lokalny interfejs do wirtualnej firmy AI.

Użycie:
  orchestrator task "Zdiagnozuj problem z TechBuddy CrashLoopBackOff"
  orchestrator chat sre "Ile nodów ma klaster?"
  orchestrator agents
  orchestrator knowledge search "PgBouncer timeout"
  orchestrator ingest
  orchestrator status
"""

import asyncio
import json

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="orchestrator",
    help="ZSEL AI Orchestrator — wirtualna firma agentów AI.",
    add_completion=False,
)
console = Console()


def _get_base_url() -> str:
    import os

    return os.environ.get("ORCH_API_URL", "http://localhost:8080")


def _local_mode() -> bool:
    """Check if we're running in local (in-process) mode vs API mode."""
    import os

    return os.environ.get("ORCH_LOCAL_MODE", "true").lower() == "true"


async def _run_task_local(description: str, requester: str = "cli") -> dict:
    """Run task in-process (no HTTP, direct import)."""
    from .orchestrator import get_orchestrator

    orch = get_orchestrator()
    await orch.initialize()

    steps_shown = set()

    async def on_update(data):
        if isinstance(data, dict) and data.get("type") == "plan":
            console.print(Panel("[bold cyan]Plan wykonania:[/bold cyan]"))
            for s in data.get("steps", []):
                console.print(f"  [dim]{s['id']}[/dim] → [green]{s['agent']}[/green]: {s['description']}")
        elif hasattr(data, "id") and data.id not in steps_shown:
            steps_shown.add(data.id)
            icon = "✅" if data.status == "completed" else "❌" if data.status == "failed" else "⏳"
            console.print(f"  {icon} [{data.agent}] {data.description[:80]}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Planowanie...", total=None)
        task = await orch.submit_task(description, requester=requester, on_update=on_update)

    await orch.close()
    return {
        "id": task.id,
        "status": task.status,
        "result": task.result,
        "learnings": task.learnings,
        "steps": len(task.steps),
    }


@app.command()
def task(
    description: str = typer.Argument(..., help="Opis zadania (może być wysokopoziomowy)"),
    requester: str = typer.Option("cli", help="Kto zgłasza zadanie"),
    sync: bool = typer.Option(True, help="Czekaj na wynik (synchronicznie)"),
):
    """Prześlij zadanie do wirtualnej firmy AI."""
    console.print(Panel(f"[bold yellow]🤖 Zadanie:[/bold yellow] {description}", border_style="yellow"))

    if _local_mode():
        result = asyncio.run(_run_task_local(description, requester=requester))
    else:
        import httpx

        url = _get_base_url()
        endpoint = "/tasks/sync" if sync else "/tasks"
        resp = httpx.post(f"{url}{endpoint}", json={"description": description, "requester": requester}, timeout=360)
        resp.raise_for_status()
        result = resp.json()

    status = result.get("status", "unknown")
    icon = "✅" if status == "completed" else "⚠️" if status == "partial" else "❌"

    console.print(
        Panel(
            f"[bold]{icon} Status: {status}[/bold]\n\n{result.get('result', '')[:2000]}",
            border_style="green" if status == "completed" else "red",
            title=f"Wynik (task_id: {result.get('id', '?')})",
        )
    )

    if result.get("learnings"):
        console.print(
            Panel(
                f"[italic]{result['learnings']}[/italic]",
                title="💡 Learnings (zapisano do Qdrant)",
                border_style="blue",
            )
        )


@app.command()
def chat(
    agent_name: str = typer.Argument(..., help="Nazwa agenta (sre, backend-dev, debugger, ...)"),
    message: str = typer.Argument(..., help="Wiadomość do agenta"),
    task_id: str = typer.Option("", help="ID powiązanego zadania (opcjonalne)"),
):
    """Porozmawiaj bezpośrednio z konkretnym agentem."""
    console.print(f"[dim]→ {agent_name}:[/dim] {message}")

    async def run():
        if _local_mode():
            from .orchestrator import get_orchestrator

            orch = get_orchestrator()
            await orch.initialize()
            result = await orch.chat_with_agent(agent_name, message, task_id=task_id)
            await orch.close()
            return result
        else:
            import httpx

            resp = httpx.post(
                f"{_get_base_url()}/agents/{agent_name}/chat",
                json={"agent": agent_name, "message": message, "task_id": task_id},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("result", "")

    result = asyncio.run(run())
    console.print(Panel(Markdown(result), title=f"[bold green]{agent_name}[/bold green]", border_style="green"))


@app.command()
def agents():
    """Wylistuj wszystkich dostępnych agentów."""

    async def run():
        if _local_mode():
            from .orchestrator import get_orchestrator

            orch = get_orchestrator()
            await orch.initialize()
            data = orch.registry.list_agents()
            await orch.close()
            return data
        else:
            import httpx

            resp = httpx.get(f"{_get_base_url()}/agents")
            return resp.json()

    data = asyncio.run(run())

    table = Table(title="🤖 Agenci AI — ZSEL Virtual Company", show_header=True, header_style="bold cyan")
    table.add_column("Nazwa", style="bold yellow", width=20)
    table.add_column("Rola", width=15)
    table.add_column("Opis", width=40)
    table.add_column("Narzędzia", width=25)
    table.add_column("Zadania", justify="right", width=8)

    for a in data:
        table.add_row(
            a["name"],
            a["role"],
            a["description"][:40],
            ", ".join(a.get("tools", [])),
            str(a.get("stats", {}).get("tasks", 0)),
        )

    console.print(table)


@app.command()
def status():
    """Status wirtualnej firmy AI i klastra wiedzy."""

    async def run():
        if _local_mode():
            from .orchestrator import get_orchestrator

            orch = get_orchestrator()
            await orch.initialize()
            data = await orch.get_status()
            await orch.close()
            return data
        else:
            import httpx

            resp = httpx.get(f"{_get_base_url()}/status")
            return resp.json()

    data = asyncio.run(run())

    # LLM status
    llm = data.get("llm", {})
    llm_status = "✅ OK" if llm.get("status") == "ok" else f"❌ {llm.get('error', 'error')}"
    models = ", ".join(llm.get("models", [])[:5])

    # Knowledge
    kb = data.get("knowledge", {})
    collections = kb.get("collections", {})

    # Tasks
    tasks = data.get("tasks", {})

    table = Table(title="📊 ZSEL Orchestrator Status", show_header=False, border_style="cyan")
    table.add_column("Klucz", style="bold", width=25)
    table.add_column("Wartość", width=50)

    table.add_row("Status", "🟢 Running" if data.get("status") == "running" else "🔴 Stopped")
    table.add_row("Agenci", str(len(data.get("agents", []))))
    table.add_row("LLM (Ollama)", f"{llm_status}\n{models}")
    table.add_row("Zadania łącznie", str(tasks.get("total", 0)))
    table.add_row("  - Uruchomione", str(tasks.get("running", 0)))
    table.add_row("  - Zakończone ✅", str(tasks.get("completed", 0)))
    table.add_row("  - Nieudane ❌", str(tasks.get("failed", 0)))
    table.add_row("Wiadomości między agentami", str(data.get("messages_exchanged", 0)))
    table.add_row("Wiedza (orch_knowledge)", str(collections.get("knowledge", 0)))
    table.add_row("Dziennik błędów", str(collections.get("errors", 0)))
    table.add_row("Historia zadań", str(collections.get("tasks", 0)))
    table.add_row("Pamięci agentów", str(collections.get("agents", 0)))

    console.print(table)


@app.command()
def ingest(
    workspace: str = typer.Option(
        "/Users/lkolo-prez/Documents/LKP/01_ZSEL/SERVERS-CLEAN",
        help="Root workspace path",
    ),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already indexed"),
):
    """
    Zaindeksuj całą wiedzę infrastrukturalną do Qdrant.

    Skanuje: GITOPS manifesty, SERVERS kody, dokumentację, skrypty.
    """
    from .ingest import run_ingestion

    console.print(Panel(f"[bold cyan]Indeksowanie wiedzy z:[/bold cyan] {workspace}", border_style="cyan"))
    asyncio.run(run_ingestion(workspace, force=force, console=console))


@app.command()
def knowledge(
    subcommand: str = typer.Argument("search", help="search | add | stats"),
    query: str = typer.Argument("", help="Zapytanie wyszukiwania"),
    category: str = typer.Option(None, help="Filtr kategorii (infra/app/error/network)"),
    limit: int = typer.Option(10, help="Liczba wyników"),
):
    """Przeszukaj lub zarządzaj bazą wiedzy."""

    async def run():
        if _local_mode():
            from .orchestrator import get_orchestrator

            orch = get_orchestrator()
            await orch.initialize()
            from .knowledge import get_knowledge_base as _get_kb

            _kb = _get_kb()

            if subcommand == "search" and query:
                entries = await _kb.search_knowledge(query, limit=limit, category=category)
                return entries
            elif subcommand == "stats":
                return await _kb.get_stats()
            return []

    if subcommand == "search":
        if not query:
            console.print("[red]Podaj zapytanie: orchestrator knowledge search 'PgBouncer error'[/red]")
            raise typer.Exit(1)

        results = asyncio.run(run())

        table = Table(title=f"🔍 Wyniki dla: '{query}'")
        table.add_column("Źródło", width=30)
        table.add_column("Kategoria", width=12)
        table.add_column("Treść", width=55)
        table.add_column("Score", justify="right", width=8)

        for e in results:
            table.add_row(
                e.source[:30],
                e.category,
                e.content[:55],
                f"{e.score:.3f}",
            )
        console.print(table)

    elif subcommand == "stats":
        stats = asyncio.run(run())
        console.print_json(json.dumps(stats))


@app.command()
def errors(
    query: str = typer.Argument("", help="Szukaj podobnych błędów"),
    limit: int = typer.Option(10, help="Liczba wyników"),
):
    """Przeszukaj dziennik błędów."""

    async def run():
        if _local_mode():
            from .knowledge import get_knowledge_base as _get_kb
            from .orchestrator import get_orchestrator

            orch = get_orchestrator()
            await orch.initialize()
            _kb = _get_kb()
            if query:
                return await _kb.find_similar_errors(query, limit=limit)
            return []

    results = asyncio.run(run())

    table = Table(title=f"🚨 Dziennik błędów: '{query}'")
    table.add_column("Błąd", width=35)
    table.add_column("Agent", width=12)
    table.add_column("Rozwiązanie", width=35)
    table.add_column("Resolved", width=8)
    table.add_column("Score", justify="right", width=8)

    for e in results:
        resolved_icon = "✅" if e.resolved else "❌"
        table.add_row(
            e.error_message[:35],
            e.agent,
            e.resolution[:35] if e.resolution else "[dim]brak[/dim]",
            resolved_icon,
            f"{e.score:.3f}",
        )
    console.print(table)


@app.command()
def spawn(
    description: str = typer.Argument(..., help="Opis nowego agenta"),
    role: str = typer.Option("sre", help="Rola agenta"),
    capabilities: str = typer.Option("", help="Umiejętności (CSV)"),
):
    """Stwórz nowego, dynamicznego agenta AI."""

    async def run():
        if _local_mode():
            from .orchestrator import get_orchestrator

            orch = get_orchestrator()
            await orch.initialize()
            caps = [c.strip() for c in capabilities.split(",") if c.strip()] if capabilities else []
            name = await orch.spawn_agent(description, role=role, capabilities=caps)
            await orch.close()
            return name

    name = asyncio.run(run())
    console.print(Panel(f"[bold green]✅ Nowy agent stworzony:[/bold green] {name}", border_style="green"))


if __name__ == "__main__":
    app()
