"""Rich terminal display helpers for A-Evolve scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def print_evaluation_summary(
    summary: dict[str, Any],
    *,
    title: str = "Evaluation Summary",
    console: Console | None = None,
) -> None:
    active_console = console or Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Split", str(summary.get("split", "")))
    table.add_row("Tasks", str(summary.get("total", 0)))
    table.add_row("Success", str(summary.get("success", 0)))
    table.add_row("Accuracy", _format_percent(summary.get("accuracy", 0.0)))
    table.add_row("Average score", _format_float(summary.get("avg_score", 0.0)))
    table.add_row("Results CSV", str(summary.get("results_csv", "")))
    active_console.print(table)


def print_evolve_summary(summary: dict[str, Any], *, console: Console | None = None) -> None:
    active_console = console or Console()
    table = Table(title="Evolution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Run ID", str(summary.get("run_id", "")))
    table.add_row("Cycles", str(summary.get("cycles_completed", 0)))
    table.add_row("Final test tasks", str(summary.get("test_total", 0)))
    table.add_row("Final test success", str(summary.get("test_success", 0)))
    table.add_row("Final test accuracy", _format_percent(summary.get("test_accuracy", 0.0)))
    table.add_row("Run directory", str(summary.get("run_dir", "")))
    table.add_row("Workspace", str(summary.get("workspace", "")))
    active_console.print(table)

    score_history = summary.get("score_history") or []
    if score_history:
        score_text = " -> ".join(_format_float(score) for score in score_history)
        active_console.print(Panel(score_text, title="Train Score History", expand=False))


def print_run_header(
    *,
    title: str,
    run_dir: str | Path | None = None,
    workspace: str | Path | None = None,
    source: str | None = None,
    console: Console | None = None,
) -> None:
    active_console = console or Console()
    lines = []
    if source:
        lines.append(f"[bold]Source[/bold]: {source}")
    if run_dir:
        lines.append(f"[bold]Run directory[/bold]: {run_dir}")
    if workspace:
        lines.append(f"[bold]Workspace[/bold]: {workspace}")
    active_console.print(Panel("\n".join(lines), title=title, expand=False))


def _format_percent(value: Any) -> str:
    return f"{float(value or 0.0) * 100:.2f}%"


def _format_float(value: Any) -> str:
    return f"{float(value or 0.0):.4f}"
