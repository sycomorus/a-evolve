"""OR-Interact-Bench adapter."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from user.simulator import (
    USER_RESPONSE_REASONS,
    load_grounded_records,
    safe_user_response_summary,
)

from ..base import BenchmarkAdapter
from ...types import Feedback, Task, Trajectory


ANSWER_FILENAME = "submitted_answer.csv"
REFERENCE_SOLUTION_FILENAME = "reference_solution.py"
REFUSAL_CODES = tuple(code for code in USER_RESPONSE_REASONS if code != "answered")


@dataclass(frozen=True)
class ObjectiveEvaluation:
    task_id: str
    correct: bool
    expected: float | None
    predicted: float | None
    absolute_error: float | None
    relative_error: float | None
    relative_tolerance: float
    run_status: str | None
    error: str | None
    skipped: bool = False


class ORInteractBenchmark(BenchmarkAdapter):
    """Adapter for a dataset directory under OR-Interact-Bench."""

    def __init__(
        self,
        benchmark_dir: str | Path | None = None,
        dataset: str = "IndustryOR",
        seed: int = 42,
        train_size: int = 50,
        val_size: int = 0,
        interaction_enabled: bool = False,
    ) -> None:
        self.benchmark_dir = _default_benchmark_dir() if benchmark_dir is None else Path(benchmark_dir).resolve()
        self.dataset = dataset
        self.seed = seed
        self.train_size = train_size
        self.val_size = val_size
        self.interaction_enabled = interaction_enabled
        self._tasks = self._load_tasks()
        self._train, self._val, self._test = self._split_tasks()

    def get_tasks(self, split: str = "train", limit: int = 10) -> list[Task]:
        split_key = split.lower()
        if split_key == "train":
            tasks = self._train
        elif split_key in {"val", "validation"}:
            tasks = self._val
        elif split_key == "holdout":
            tasks = self._val if self.val_size else self._test
        elif split_key == "test":
            tasks = self._test
        else:
            raise ValueError(
                f"unknown split {split!r}; expected train, val, validation, holdout, or test"
            )
        return tasks[:limit] if limit is not None else list(tasks)

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        task_dir = Path(task.metadata["task_dir"])
        runtime_dir = _runtime_dir_from_trajectory(trajectory)
        if runtime_dir is None:
            evaluation = self._missing_runtime(task)
        else:
            evaluation = evaluate_task_objective(task.id, task_dir, runtime_dir)

        detail = _feedback_detail(evaluation, trajectory, task_dir)
        evaluation_record = asdict(evaluation)
        if self.interaction_enabled:
            evaluation_record.update(_interaction_diagnostics(task_dir, trajectory))
        return Feedback(
            success=evaluation.correct,
            score=1.0 if evaluation.correct else 0.0,
            detail=detail,
            raw={
                "evaluation": evaluation_record,
                "runtime_dir": str(runtime_dir) if runtime_dir else None,
                "task_dir": str(task_dir),
            },
        )

    def _load_tasks(self) -> list[Task]:
        index = self._read_index()
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
            tasks = self._load_tasks_from_dataset_dir()

        if not tasks:
            raise ValueError(f"no tasks found for dataset {self.dataset!r} under {self.benchmark_dir}")
        return tasks

    def _read_index(self) -> dict[str, Any]:
        index_path = self.benchmark_dir / "index.json"
        if not index_path.is_file():
            return {"tasks": []}
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _load_tasks_from_dataset_dir(self) -> list[Task]:
        dataset_dir = self.benchmark_dir / self.dataset
        if not dataset_dir.is_dir():
            return []

        tasks = []
        for task_dir in sorted(dataset_dir.glob("task_*")):
            if not task_dir.is_dir():
                continue
            item = {
                "task_id": task_dir.name,
                "path": f"{self.dataset}/{task_dir.name}",
                "source_instance_dir": None,
            }
            tasks.append(self._task_from_index_item(item, task_dir.resolve()))
        return tasks

    def _split_tasks(self) -> tuple[list[Task], list[Task], list[Task]]:
        if self.train_size < 0 or self.val_size < 0:
            raise ValueError("train_size and val_size must be non-negative")
        tasks = list(self._tasks)
        random.Random(self.seed).shuffle(tasks)
        test_start = self.train_size + self.val_size
        if self.val_size and test_start > len(tasks):
            raise ValueError(
                "requested validation split cannot be satisfied: "
                f"train_size={self.train_size}, val_size={self.val_size}, "
                f"available_tasks={len(tasks)}"
            )
        train = tasks[: self.train_size]
        val = tasks[self.train_size : test_start]
        test = tasks[test_start:]
        return train, val, test

    def _task_from_index_item(self, item: dict[str, Any], task_dir: Path) -> Task:
        task_id = str(item["task_id"])
        visible_files = _visible_files(task_dir)
        task_metadata = _read_task_metadata(task_dir)
        metadata = {
            "dataset": self.dataset,
            "task_dir": str(task_dir),
            "source_instance_dir": item.get("source_instance_dir"),
            "visible_roots": ["docs", "data"],
            "visible_files": visible_files,
        }
        category = _task_category(task_metadata)
        if category:
            metadata["category"] = category
        task_input = (
            f"{self.dataset} optimization task {task_id}. "
            "Only docs/ and data/ are visible to the agent."
        )
        return Task(id=task_id, input=task_input, metadata=metadata)

    def _missing_runtime(self, task: Task) -> ObjectiveEvaluation:
        oracle = _read_oracle(Path(task.metadata["task_dir"]) / "oracle" / "objective.json")
        objective_value = oracle.get("objective_value")
        tolerance = float(oracle.get("relative_tolerance", 0.0))
        if objective_value is None:
            return ObjectiveEvaluation(
                task_id=task.id,
                correct=False,
                expected=None,
                predicted=None,
                absolute_error=None,
                relative_error=None,
                relative_tolerance=tolerance,
                run_status=None,
                error="oracle objective_value is null",
                skipped=True,
            )
        return ObjectiveEvaluation(
            task_id=task.id,
            correct=False,
            expected=float(objective_value),
            predicted=None,
            absolute_error=None,
            relative_error=None,
            relative_tolerance=tolerance,
            run_status=None,
            error="trajectory missing runtime_dir",
        )


def evaluate_task_objective(task_id: str, task_dir: Path, runtime_dir: Path) -> ObjectiveEvaluation:
    oracle = _read_oracle(task_dir / "oracle" / "objective.json")
    tolerance = float(oracle.get("relative_tolerance", 0.0))
    run_status = _read_run_status(runtime_dir / "run_summary.json")
    objective_value = oracle.get("objective_value")
    if objective_value is None:
        return ObjectiveEvaluation(
            task_id=task_id,
            correct=False,
            expected=None,
            predicted=None,
            absolute_error=None,
            relative_error=None,
            relative_tolerance=tolerance,
            run_status=run_status,
            error="oracle objective_value is null",
            skipped=True,
        )

    expected = float(objective_value)
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


def _interaction_diagnostics(
    task_dir: Path,
    trajectory: Trajectory,
) -> dict[str, Any]:
    grounded_dir = task_dir / "grounded"
    has_grounded = grounded_dir.is_dir() and any(
        path.is_file() and path.suffix == ".md" and not path.name.startswith(".")
        for path in grounded_dir.iterdir()
    )
    grounded_record_count = len(load_grounded_records(task_dir))
    ask_count = 0
    answered_count = 0
    refusal_counts = {code: 0 for code in REFUSAL_CODES}
    pending_ids: set[str] = set()
    pending_without_id = 0
    events = trajectory.conversation or trajectory.steps
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "tool_call" and event.get("name") == "ask_user":
            ask_count += 1
            call_id = str(event.get("tool_call_id") or event.get("id") or "")
            if call_id:
                pending_ids.add(call_id)
            else:
                pending_without_id += 1
            continue
        if event_type != "tool_output" or event.get("name") != "ask_user":
            continue
        call_id = str(event.get("tool_call_id") or event.get("id") or "")
        if call_id and call_id in pending_ids:
            pending_ids.remove(call_id)
        elif pending_without_id:
            pending_without_id -= 1
        output = event.get("output", event.get("content", {}))
        response = safe_user_response_summary(output)
        if response["answered"]:
            answered_count += 1
        code = response.get("code")
        if code in refusal_counts:
            refusal_counts[code] += 1
    diagnostics = {
        "has_grounded": has_grounded,
        "grounded_record_count": grounded_record_count,
        "has_answerable_grounded": grounded_record_count > 0,
        "ask_count": ask_count,
        "answered_ask_count": answered_count,
        "refused_ask_count": max(0, ask_count - answered_count),
    }
    diagnostics.update(
        {f"{code}_ask_count": count for code, count in refusal_counts.items()}
    )
    return diagnostics


def _feedback_detail(evaluation: ObjectiveEvaluation, trajectory: Trajectory, task_dir: Path) -> str:
    status = "PASS" if evaluation.correct else "FAIL"
    parts = [
        f"Status: {status}",
        f"Run status: {evaluation.run_status or 'unknown'}",
        f"Expected objective: {_format_optional(evaluation.expected)}",
        f"Predicted objective: {_format_optional(evaluation.predicted)}",
        f"Relative error: {_format_optional(evaluation.relative_error)}",
        f"Tolerance: {evaluation.relative_tolerance:.10g}",
    ]
    if evaluation.error:
        parts.append(f"Failure reason: {evaluation.error}")

    diagnostics = _trajectory_diagnostics(trajectory)
    if diagnostics:
        parts.append(f"Trajectory diagnosis: {diagnostics}")
    reference_solution = _read_reference_solution(task_dir)
    if reference_solution:
        parts.append(reference_solution)
    return "\n".join(parts)


def _read_reference_solution(task_dir: Path) -> str:
    path = task_dir / "oracle" / REFERENCE_SOLUTION_FILENAME
    if not path.is_file():
        return ""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Reference solution unavailable: {type(exc).__name__}: {exc}"
    return (
        f"Reference solution path: {path}\n"
        "Reference solution code:\n"
        "```python\n"
        f"{source.rstrip()}\n"
        "```"
    )


def _read_task_metadata(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "metadata.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _task_category(metadata: dict[str, Any]) -> str | None:
    category = metadata.get("category")
    if category is None:
        source = metadata.get("source")
        if isinstance(source, dict):
            category = source.get("category")
    if category is None:
        return None
    text = str(category).strip()
    return text or None


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
