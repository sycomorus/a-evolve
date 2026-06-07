"""Generic benchmark evaluation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .benchmarks.base import BenchmarkAdapter
from .protocol.base_agent import BaseAgent


def run_evaluation(
    agent: BaseAgent,
    benchmark: BenchmarkAdapter,
    *,
    split: str = "test",
    limit: int | None = 10,
    output_dir: str | Path,
    show_progress: bool = False,
    console: Console | None = None,
) -> dict[str, Any]:
    """Run ``agent.solve`` and ``benchmark.evaluate`` on one split.

    Writes ``summary.json`` and ``results.csv`` into ``output_dir``.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    tasks = benchmark.get_tasks(split=split, limit=limit)
    task_iterable = tasks

    if show_progress:
        active_console = console or Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Evaluating"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[current_task]}"),
            TimeElapsedColumn(),
            console=active_console,
        ) as progress:
            task_id = progress.add_task(
                "evaluate",
                total=len(tasks),
                current_task="starting",
            )
            for task in tasks:
                rows.append(_evaluate_one(agent, benchmark, task))
                progress.update(
                    task_id,
                    advance=1,
                    current_task=task.id,
                )
    else:
        for task in task_iterable:
            rows.append(_evaluate_one(agent, benchmark, task))

    total = len(rows)
    success = sum(1 for row in rows if _as_bool(row.get("success")))
    score_sum = sum(float(row.get("score") or 0.0) for row in rows)
    summary = {
        "workspace": str(agent.workspace.root),
        "split": split,
        "limit": limit,
        "total": total,
        "success": success,
        "accuracy": (success / total) if total else 0.0,
        "avg_score": (score_sum / total) if total else 0.0,
        "results_csv": str(destination / "results.csv"),
    }

    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_results_csv(destination / "results.csv", rows)
    return summary


def _evaluate_one(
    agent: BaseAgent,
    benchmark: BenchmarkAdapter,
    task: Any,
) -> dict[str, Any]:
    try:
        trajectory = agent.solve(task)
        feedback = benchmark.evaluate(task, trajectory)
        row = {
            "task_id": task.id,
            "success": feedback.success,
            "score": feedback.score,
            "detail": feedback.detail,
        }
        evaluation = feedback.raw.get("evaluation") if isinstance(feedback.raw, dict) else None
        if isinstance(evaluation, dict):
            row.update(evaluation)
        return row
    except Exception as exc:
        return {
            "task_id": task.id,
            "success": False,
            "score": 0.0,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["task_id", "success", "score", "detail"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
