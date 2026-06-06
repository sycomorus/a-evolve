"""OR-Interact-Bench adapter for IndustryOR tasks."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..base import BenchmarkAdapter
from ...types import Feedback, Task, Trajectory


ANSWER_FILENAME = "submitted_answer.csv"


@dataclass(frozen=True)
class ObjectiveEvaluation:
    task_id: str
    correct: bool
    expected: float
    predicted: float | None
    absolute_error: float | None
    relative_error: float | None
    relative_tolerance: float
    run_status: str | None
    error: str | None


class ORInteractBenchmark(BenchmarkAdapter):
    """Adapter for the 100-task IndustryOR subset of OR-Interact-Bench."""

    def __init__(
        self,
        benchmark_dir: str | Path | None = None,
        dataset: str = "IndustryOR",
        seed: int = 42,
        train_size: int = 50,
    ) -> None:
        self.benchmark_dir = _default_benchmark_dir() if benchmark_dir is None else Path(benchmark_dir).resolve()
        self.dataset = dataset
        self.seed = seed
        self.train_size = train_size
        self._tasks = self._load_tasks()
        self._train, self._test = self._split_tasks()

    def get_tasks(self, split: str = "train", limit: int = 10) -> list[Task]:
        split_key = split.lower()
        if split_key == "train":
            tasks = self._train
        elif split_key in {"holdout", "test"}:
            tasks = self._test
        else:
            raise ValueError(f"unknown split {split!r}; expected train, holdout, or test")
        return tasks[:limit] if limit is not None else list(tasks)

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        task_dir = Path(task.metadata["task_dir"])
        runtime_dir = _runtime_dir_from_trajectory(trajectory)
        if runtime_dir is None:
            evaluation = self._missing_runtime(task)
        else:
            evaluation = evaluate_task_objective(task.id, task_dir, runtime_dir)

        detail = _feedback_detail(evaluation, trajectory)
        return Feedback(
            success=evaluation.correct,
            score=1.0 if evaluation.correct else 0.0,
            detail=detail,
            raw={
                "evaluation": asdict(evaluation),
                "runtime_dir": str(runtime_dir) if runtime_dir else None,
                "task_dir": str(task_dir),
            },
        )

    def _load_tasks(self) -> list[Task]:
        index_path = self.benchmark_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing OR-Interact-Bench index: {index_path}")

        index = json.loads(index_path.read_text(encoding="utf-8"))
        tasks: list[Task] = []
        for item in index.get("tasks", []):
            relative_path = Path(item["path"])
            if len(relative_path.parts) < 2 or relative_path.parts[0] != self.dataset:
                continue
            task_dir = (self.benchmark_dir / relative_path).resolve()
            if not task_dir.is_dir():
                continue
            tasks.append(self._task_from_index_item(item, task_dir))

        if not tasks:
            raise ValueError(f"no tasks found for dataset {self.dataset!r} under {self.benchmark_dir}")
        return tasks

    def _split_tasks(self) -> tuple[list[Task], list[Task]]:
        tasks = list(self._tasks)
        random.Random(self.seed).shuffle(tasks)
        train = tasks[: self.train_size]
        test = tasks[self.train_size :]
        return train, test

    def _task_from_index_item(self, item: dict[str, Any], task_dir: Path) -> Task:
        task_id = str(item["task_id"])
        visible_files = _visible_files(task_dir)
        metadata = {
            "dataset": self.dataset,
            "task_dir": str(task_dir),
            "source_instance_dir": item.get("source_instance_dir"),
            "visible_roots": ["docs", "data"],
            "visible_files": visible_files,
        }
        task_input = (
            f"{self.dataset} optimization task {task_id}. "
            "Only docs/ and data/ are visible to the agent."
        )
        return Task(id=task_id, input=task_input, metadata=metadata)

    def _missing_runtime(self, task: Task) -> ObjectiveEvaluation:
        oracle = _read_oracle(Path(task.metadata["task_dir"]) / "oracle" / "objective.json")
        return ObjectiveEvaluation(
            task_id=task.id,
            correct=False,
            expected=float(oracle["objective_value"]),
            predicted=None,
            absolute_error=None,
            relative_error=None,
            relative_tolerance=float(oracle.get("relative_tolerance", 0.0)),
            run_status=None,
            error="trajectory missing runtime_dir",
        )


def evaluate_task_objective(task_id: str, task_dir: Path, runtime_dir: Path) -> ObjectiveEvaluation:
    oracle = _read_oracle(task_dir / "oracle" / "objective.json")
    expected = float(oracle["objective_value"])
    tolerance = float(oracle.get("relative_tolerance", 0.0))

    run_status = _read_run_status(runtime_dir / "run_summary.json")
    predicted, error = _read_prediction(runtime_dir / ANSWER_FILENAME)
    if predicted is None:
        return ObjectiveEvaluation(
            task_id=task_id,
            correct=False,
            expected=expected,
            predicted=None,
            absolute_error=None,
            relative_error=None,
            relative_tolerance=tolerance,
            run_status=run_status,
            error=error,
        )

    absolute_error = abs(predicted - expected)
    relative_error = absolute_error / max(1.0, abs(expected))
    correct = math.isfinite(predicted) and relative_error <= tolerance
    return ObjectiveEvaluation(
        task_id=task_id,
        correct=correct,
        expected=expected,
        predicted=predicted,
        absolute_error=absolute_error,
        relative_error=relative_error,
        relative_tolerance=tolerance,
        run_status=run_status,
        error=None if correct else "outside relative tolerance",
    )


def _runtime_dir_from_trajectory(trajectory: Trajectory) -> Path | None:
    for step in trajectory.steps:
        runtime_dir = step.get("runtime_dir")
        if runtime_dir:
            return Path(str(runtime_dir))
    return None


def _feedback_detail(evaluation: ObjectiveEvaluation, trajectory: Trajectory) -> str:
    status = "PASS" if evaluation.correct else "FAIL"
    parts = [
        f"Status: {status}",
        f"Run status: {evaluation.run_status or 'unknown'}",
        f"Expected objective: {evaluation.expected:.10g}",
        f"Predicted objective: {_format_optional(evaluation.predicted)}",
        f"Relative error: {_format_optional(evaluation.relative_error)}",
        f"Tolerance: {evaluation.relative_tolerance:.10g}",
    ]
    if evaluation.error:
        parts.append(f"Failure reason: {evaluation.error}")

    diagnostics = _trajectory_diagnostics(trajectory)
    if diagnostics:
        parts.append(f"Trajectory diagnosis: {diagnostics}")
    return "\n".join(parts)


def _trajectory_diagnostics(trajectory: Trajectory) -> str:
    if not trajectory.steps:
        return "no recorded steps"
    status = None
    error = None
    tools: list[str] = []
    for step in trajectory.steps:
        status = step.get("status", status)
        error = step.get("error", error)
        if "tools" in step and isinstance(step["tools"], list):
            tools = [str(t) for t in step["tools"]]
    fragments = []
    if status:
        fragments.append(f"agent status={status}")
    if tools:
        fragments.append(f"tools={', '.join(tools[:12])}")
    if error:
        fragments.append(f"error={error}")
    return "; ".join(fragments)


def _read_oracle(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_run_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        status = summary.get("status")
        return str(status) if status is not None else None
    except Exception:
        return "invalid_summary"


def _read_prediction(path: Path) -> tuple[float | None, str | None]:
    if not path.is_file():
        return None, "missing submitted_answer.csv"

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return None, f"failed to read answer CSV: {type(exc).__name__}: {exc}"

    if not rows:
        return None, "submitted_answer.csv has no rows"

    value = rows[0].get("objective_value")
    if value is None or not str(value).strip():
        return None, "submitted_answer.csv missing objective_value"

    try:
        return float(value), None
    except ValueError:
        return None, f"objective_value is not numeric: {value!r}"


def _visible_files(task_dir: Path) -> dict[str, list[str]]:
    visible: dict[str, list[str]] = {}
    for root_name in ("docs", "data"):
        root = task_dir / root_name
        if root.is_dir():
            visible[root_name] = sorted(
                str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
            )
        else:
            visible[root_name] = []
    return visible


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.10g}"


def _default_benchmark_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "OR-Interact-Bench"
