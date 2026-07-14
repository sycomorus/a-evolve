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
from .algorithms.step_opsd import summarize_interaction_metrics
from .protocol.base_agent import BaseAgent
from .task_runner import TaskEvaluation, resolve_agent_parallelism, run_task_evaluations


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

    tasks = benchmark.get_tasks(split=split, limit=limit)
    workers = min(resolve_agent_parallelism(agent), len(tasks)) if tasks else 0

    if show_progress:
        active_console = console or Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Evaluating"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]workers={task.fields[workers]}"),
            TextColumn("[dim]{task.fields[current_task]}"),
            TimeElapsedColumn(),
            console=active_console,
        ) as progress:
            task_id = progress.add_task(
                "evaluate",
                total=len(tasks),
                current_task="starting",
                workers=workers,
            )

            def update_progress(result: TaskEvaluation) -> None:
                progress.update(
                    task_id,
                    advance=1,
                    current_task=result.task.id,
                )

            evaluations = run_task_evaluations(
                agent,
                benchmark,
                tasks,
                on_complete=update_progress,
            )
    else:
        evaluations = run_task_evaluations(agent, benchmark, tasks)

    rows = [_row_from_evaluation(result) for result in evaluations]

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
    if any("has_grounded" in row for row in rows):
        interaction_metrics = summarize_interaction_metrics(rows)
        interaction_metrics_path = destination / "interaction_metrics.json"
        interaction_metrics_path.write_text(
            json.dumps(interaction_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["interaction_metrics"] = interaction_metrics
        summary["interaction_metrics_path"] = str(interaction_metrics_path)

    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_results_csv(destination / "results.csv", rows)
    return summary


def _row_from_evaluation(result: TaskEvaluation) -> dict[str, Any]:
    if result.feedback is None:
        return {
            "task_id": result.task.id,
            "success": False,
            "score": 0.0,
            "detail": result.error or "missing feedback",
        }
    feedback = result.feedback
    row = {
        "task_id": result.task.id,
        "success": feedback.success,
        "score": feedback.score,
        "detail": feedback.detail,
    }
    evaluation = feedback.raw.get("evaluation") if isinstance(feedback.raw, dict) else None
    if isinstance(evaluation, dict):
        row.update(evaluation)
    if isinstance(feedback.raw, dict):
        row["runtime_dir"] = feedback.raw.get("runtime_dir")
        row["task_dir"] = feedback.raw.get("task_dir")
    return row


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
