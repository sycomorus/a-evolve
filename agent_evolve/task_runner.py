"""Task execution helpers shared by evolution and evaluation."""

from __future__ import annotations

import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .benchmarks.base import BenchmarkAdapter
from .protocol.base_agent import BaseAgent
from .types import Feedback, Task, Trajectory


@dataclass
class TaskEvaluation:
    task: Task
    trajectory: Trajectory | None = None
    feedback: Feedback | None = None
    error: str | None = None


def resolve_agent_parallelism(agent: BaseAgent) -> int:
    config = getattr(agent, "config", None)
    raw = getattr(config, "parallelism", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def run_task_evaluations(
    agent: BaseAgent,
    benchmark: BenchmarkAdapter,
    tasks: list[Task],
    *,
    parallelism: int | None = None,
    on_complete: Callable[[TaskEvaluation], None] | None = None,
) -> list[TaskEvaluation]:
    if not tasks:
        return []

    workers = min(parallelism or resolve_agent_parallelism(agent), len(tasks))
    if workers <= 1:
        return _run_sequential(agent, benchmark, tasks, on_complete=on_complete)

    workspace = agent.workspace.root
    agent_class = _class_path(agent.__class__)
    results: list[TaskEvaluation | None] = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one_in_process, agent_class, workspace, benchmark, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            task = tasks[index]
            try:
                result = future.result()
            except Exception as exc:
                result = TaskEvaluation(task=task, error=f"{type(exc).__name__}: {exc}")
            results[index] = result
            if on_complete is not None:
                on_complete(result)

    return [result for result in results if result is not None]


def _run_sequential(
    agent: BaseAgent,
    benchmark: BenchmarkAdapter,
    tasks: list[Task],
    *,
    on_complete: Callable[[TaskEvaluation], None] | None,
) -> list[TaskEvaluation]:
    results = []
    for task in tasks:
        result = _run_one(agent, benchmark, task)
        results.append(result)
        if on_complete is not None:
            on_complete(result)
    return results


def _run_one_in_process(
    agent_class: str,
    workspace: Path,
    benchmark: BenchmarkAdapter,
    task: Task,
) -> TaskEvaluation:
    cls = _import_class(agent_class)
    agent = cls(workspace)
    return _run_one(agent, benchmark, task)


def _run_one(agent: BaseAgent, benchmark: BenchmarkAdapter, task: Task) -> TaskEvaluation:
    try:
        trajectory = agent.solve(task)
        feedback = benchmark.evaluate(task, trajectory)
        return TaskEvaluation(task=task, trajectory=trajectory, feedback=feedback)
    except Exception as exc:
        return TaskEvaluation(task=task, error=f"{type(exc).__name__}: {exc}")


def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _import_class(dotted_path: str) -> type:
    module_name, qualname = dotted_path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
