from __future__ import annotations

import csv
import json
from pathlib import Path

from agent_evolve.evaluation import run_evaluation
from agent_evolve.types import Feedback, Task, Trajectory


def test_run_evaluation_writes_summary_and_results(tmp_path: Path) -> None:
    agent = FakeAgent(tmp_path / "workspace")
    benchmark = FakeBenchmark()

    summary = run_evaluation(
        agent,
        benchmark,
        split="test",
        limit=3,
        output_dir=tmp_path / "evaluation",
    )

    assert benchmark.requested == ("test", 3)
    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["accuracy"] == 2 / 3
    assert summary["avg_score"] == 0.5
    assert summary["workspace"] == str(agent.workspace.root)

    persisted = json.loads((tmp_path / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
    results_path = tmp_path / "evaluation" / "results.csv"
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == ["task_1", "task_2", "task_3"]
    assert "predicted" in rows[0]


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root


class FakeAgent:
    def __init__(self, workspace: Path) -> None:
        self.workspace = FakeWorkspace(workspace)

    def solve(self, task: Task) -> Trajectory:
        return Trajectory(task_id=task.id, output=f"answer {task.id}")


class FakeBenchmark:
    def __init__(self) -> None:
        self.requested: tuple[str, int | None] | None = None

    def get_tasks(self, split: str = "train", limit: int = 10) -> list[Task]:
        self.requested = (split, limit)
        return [Task(id=f"task_{index}", input="") for index in range(1, (limit or 0) + 1)]

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        scores = {"task_1": 1.0, "task_2": 0.0, "task_3": 0.5}
        successes = {"task_1", "task_3"}
        return Feedback(
            success=task.id in successes,
            score=scores[task.id],
            detail=f"checked {trajectory.task_id}",
            raw={"evaluation": {"predicted": scores[task.id]}},
        )
