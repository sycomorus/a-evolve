from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agent_evolve.agents.or_interact.react_agent import ORReactAgent
from agent_evolve.benchmarks.or_interact import ORInteractBenchmark
from agent_evolve.types import Task, Trajectory


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "OR-Interact-Bench"


def test_split_is_reproducible_and_disjoint() -> None:
    first = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)
    second = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)

    first_train = [task.id for task in first.get_tasks("train", limit=50)]
    first_test = [task.id for task in first.get_tasks("test", limit=50)]
    second_train = [task.id for task in second.get_tasks("train", limit=50)]
    second_test = [task.id for task in second.get_tasks("holdout", limit=50)]

    assert len(first_train) == 50
    assert len(first_test) == 50
    assert set(first_train).isdisjoint(first_test)
    assert first_train == second_train
    assert first_test == second_test


def test_task_metadata_does_not_expose_oracle_names() -> None:
    benchmark = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)
    task = benchmark.get_tasks("train", limit=1)[0]

    text = json.dumps(task.metadata, ensure_ascii=False).lower()
    assert "oracle" not in text
    assert "objective" not in text
    assert "reference_solution" not in text


def test_evaluate_correct_objective(tmp_path: Path) -> None:
    benchmark = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)
    task = _task_by_id(benchmark, "task_001")
    expected = _oracle_value(task)
    _write_answer(tmp_path, expected)

    feedback = benchmark.evaluate(task, _trajectory(tmp_path))

    assert feedback.success is True
    assert feedback.score == 1.0
    assert feedback.raw["evaluation"]["predicted"] == expected


def test_evaluate_wrong_objective(tmp_path: Path) -> None:
    benchmark = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)
    task = _task_by_id(benchmark, "task_001")
    _write_answer(tmp_path, -1)

    feedback = benchmark.evaluate(task, _trajectory(tmp_path))

    assert feedback.success is False
    assert feedback.score == 0.0
    assert "outside relative tolerance" in feedback.detail


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (None, "missing submitted_answer.csv"),
        ([], "submitted_answer.csv has no rows"),
        ([{"objective_value": "not-a-number"}], "objective_value is not numeric"),
    ],
)
def test_evaluate_bad_answer_csv(tmp_path: Path, rows: list[dict[str, str]] | None, message: str) -> None:
    benchmark = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)
    task = _task_by_id(benchmark, "task_001")
    if rows is not None:
        with (tmp_path / "submitted_answer.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["objective_value"])
            writer.writeheader()
            writer.writerows(rows)

    feedback = benchmark.evaluate(task, _trajectory(tmp_path))

    assert feedback.success is False
    assert message in feedback.detail


def test_agent_prompt_includes_skill_memory_and_evolved_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(workspace, "modeling", "Prefer explicit variable bounds.")
    (workspace / "memory" / "memories.jsonl").write_text(
        json.dumps({"content": "Check objective direction before finalizing."}) + "\n",
        encoding="utf-8",
    )
    _write_tool(
        workspace,
        "diagnose",
        "from typing import Any\n\ndef diagnose() -> dict[str, Any]:\n    return {'ok': True}\n",
    )
    _write_registry(workspace, [{"name": "diagnose", "file": "diagnose.py", "function": "diagnose", "description": "diagnostic helper"}])

    captured: dict[str, object] = {}

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            captured["system_prompt"] = system_prompt
            captured["tools"] = registry.list_tools()

        def run(self):
            from baseline.react.agent import AgentResult

            return AgentResult(status="success", turns=1, objective_value=1)

    monkeypatch.setattr("agent_evolve.agents.or_interact.react_agent.ReActAgent", FakeReActAgent)
    agent = ORReactAgent(workspace)
    task = Task(id="task_x", input="", metadata={"task_dir": str(_visible_task(tmp_path)), "dataset": "IndustryOR"})

    agent.solve(task)

    prompt = str(captured["system_prompt"])
    assert "Prefer explicit variable bounds." in prompt
    assert "Check objective direction before finalizing." in prompt
    assert "diagnose" in captured["tools"]


def test_workspace_tool_overrides_seed_tool(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_tool(
        workspace,
        "list_context",
        "from typing import Any\n\ndef list_context() -> dict[str, Any]:\n    return {'overridden': True}\n",
    )
    _write_registry(workspace, [{"name": "list_context", "file": "list_context.py", "function": "list_context"}])

    agent = ORReactAgent(workspace)

    assert agent.registry.call("list_context") == {"overridden": True}
    assert agent.registry.get("list_context").kind == "evolved"


def test_forbidden_evolved_tool_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_tool(
        workspace,
        "bad_tool",
        "from typing import Any\n\ndef bad_tool() -> dict[str, Any]:\n    return {'path': 'oracle/objective.json'}\n",
    )
    _write_registry(workspace, [{"name": "bad_tool", "file": "bad_tool.py", "function": "bad_tool"}])

    with pytest.raises(ValueError, match="forbidden string"):
        ORReactAgent(workspace)


def _task_by_id(benchmark: ORInteractBenchmark, task_id: str) -> Task:
    for split in ("train", "test"):
        for task in benchmark.get_tasks(split, limit=50):
            if task.id == task_id:
                return task
    raise AssertionError(f"missing task {task_id}")


def _oracle_value(task: Task) -> float:
    path = Path(task.metadata["task_dir"]) / "oracle" / "objective.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["objective_value"])


def _write_answer(runtime_dir: Path, value: float | int | str) -> None:
    with (runtime_dir / "submitted_answer.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["objective_value"])
        writer.writeheader()
        writer.writerow({"objective_value": value})


def _trajectory(runtime_dir: Path) -> Trajectory:
    return Trajectory(task_id="task_001", output="", steps=[{"runtime_dir": str(runtime_dir), "status": "success"}])


def _workspace(path: Path) -> Path:
    (path / "prompts").mkdir(parents=True)
    (path / "skills").mkdir()
    (path / "memory").mkdir()
    (path / "tools").mkdir()
    (path / "manifest.yaml").write_text(
        "name: test\ncontract_version: '1.0'\nagent:\n  entrypoint: agent_evolve.agents.or_interact.react_agent.ORReactAgent\n",
        encoding="utf-8",
    )
    (path / "prompts" / "system.md").write_text("Base prompt", encoding="utf-8")
    (path / "tools" / "registry.yaml").write_text("tools: []\n", encoding="utf-8")
    return path


def _write_skill(workspace: Path, name: str, body: str) -> None:
    skill_dir = workspace / "skills" / name
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _write_registry(workspace: Path, tools: list[dict[str, str]]) -> None:
    lines = ["tools:"]
    for tool in tools:
        lines.append(f"  - name: {tool['name']}")
        for key in ("file", "function", "description"):
            if key in tool:
                lines.append(f"    {key}: {tool[key]}")
    (workspace / "tools" / "registry.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tool(workspace: Path, name: str, source: str) -> None:
    (workspace / "tools" / f"{name}.py").write_text(source, encoding="utf-8")


def _visible_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "data").mkdir()
    (task_dir / "docs" / "business_requirement.md").write_text("Task", encoding="utf-8")
    return task_dir
