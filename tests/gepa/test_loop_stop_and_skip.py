"""Tests for EvolutionLoop manages_own_evaluation skip and stop signal."""
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from agent_evolve.benchmarks.base import BenchmarkAdapter
from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.engine.loop import EvolutionLoop
from agent_evolve.protocol.base_agent import BaseAgent
from agent_evolve.types import Feedback, Observation, StepResult, Task, Trajectory


def _make_mock_agent(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "prompts").mkdir()
    (workspace_root / "prompts" / "system.md").write_text("test prompt")
    agent = MagicMock()
    agent.workspace = AgentWorkspace(workspace_root)
    agent.solve.return_value = Trajectory(task_id="t1", output="out")
    agent.reload_from_fs.return_value = None
    agent.export_to_fs.return_value = None
    return agent

def _make_mock_benchmark():
    benchmark = MagicMock()
    benchmark.get_tasks.return_value = [Task(id="t1", input="do something")]
    benchmark.evaluate.return_value = Feedback(success=True, score=0.9, detail="good")
    return benchmark

class StoppingEngine:
    @property
    def manages_own_evaluation(self) -> bool:
        return False
    def step(self, workspace, observations, history, trial):
        return StepResult(mutated=True, summary="done", stop=True)
    def on_cycle_end(self, accepted, score):
        pass

class SelfManagingStoppingEngine:
    @property
    def manages_own_evaluation(self) -> bool:
        return True
    def step(self, workspace, observations, history, trial):
        assert observations == [], "Expected empty observations for self-managing engine"
        return StepResult(mutated=True, summary="self-managed", stop=True)
    def on_cycle_end(self, accepted, score):
        pass

def test_loop_stops_when_engine_returns_stop_true(tmp_path):
    agent = _make_mock_agent(tmp_path)
    benchmark = _make_mock_benchmark()
    engine = StoppingEngine()
    config = EvolveConfig(max_cycles=10, batch_size=1)
    loop = EvolutionLoop(agent, benchmark, engine, config)
    loop.versioning = MagicMock()
    result = loop.run()
    assert result.cycles_completed == 1
    assert result.converged is True
    assert agent.solve.called

def test_loop_skips_solve_when_manages_own_evaluation(tmp_path):
    agent = _make_mock_agent(tmp_path)
    benchmark = _make_mock_benchmark()
    engine = SelfManagingStoppingEngine()
    config = EvolveConfig(max_cycles=10, batch_size=1)
    loop = EvolutionLoop(agent, benchmark, engine, config)
    loop.versioning = MagicMock()
    result = loop.run()
    assert result.cycles_completed == 1
    assert result.converged is True
    assert not agent.solve.called


def test_loop_train_limit_runs_epochs_over_mini_batches(tmp_path):
    agent = BatchAgent(tmp_path)
    benchmark = BatchBenchmark(task_count=5)
    engine = RecordingEngine()
    config = EvolveConfig(max_cycles=2, batch_size=2, train_limit=5, egl_window=999)
    loop = EvolutionLoop(agent, benchmark, engine, config)
    loop.versioning = MagicMock()
    progress_events = []

    result = loop.run(progress_callback=progress_events.append)

    assert benchmark.requests == [("train", 5)]
    assert engine.batches == [
        ["t1", "t2"],
        ["t3", "t4"],
        ["t5"],
        ["t1", "t2"],
        ["t3", "t4"],
        ["t5"],
    ]
    assert result.cycles_completed == 6
    assert result.details["epochs_completed"] == 2
    assert result.details["updates_completed"] == 6
    assert result.details["total_updates"] == 6
    assert [event["epoch"] for event in progress_events] == [1, 1, 1, 2, 2, 2]
    assert [event["batch_index"] for event in progress_events] == [1, 2, 3, 1, 2, 3]


class BatchAgent:
    def __init__(self, tmp_path: Path):
        workspace_root = tmp_path / "batch_workspace"
        workspace_root.mkdir()
        (workspace_root / "prompts").mkdir()
        (workspace_root / "prompts" / "system.md").write_text("test prompt")
        self.workspace = AgentWorkspace(workspace_root)
        self.export_to_fs = MagicMock()
        self.reload_from_fs = MagicMock()

    def solve(self, task: Task) -> Trajectory:
        return Trajectory(task_id=task.id, output=f"solved {task.id}")


class BatchBenchmark:
    def __init__(self, task_count: int):
        self.tasks = [Task(id=f"t{index}", input="") for index in range(1, task_count + 1)]
        self.requests = []

    def get_tasks(self, split: str = "train", limit: int = 10):
        self.requests.append((split, limit))
        return self.tasks[:limit]

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        return Feedback(success=True, score=1.0, detail=f"checked {task.id}")


class RecordingEngine:
    def __init__(self):
        self.batches = []

    @property
    def manages_own_evaluation(self) -> bool:
        return False

    def step(self, workspace, observations, history, trial):
        self.batches.append([observation.task.id for observation in observations])
        return StepResult(mutated=False, summary="recorded")

    def on_cycle_end(self, accepted, score):
        pass


class ValidationAgent(BaseAgent):
    def solve(self, task: Task) -> Trajectory:
        return Trajectory(task_id=task.id, output=self.system_prompt)


class ValidationBenchmark(BenchmarkAdapter):
    def __init__(self):
        self.train_tasks = [
            Task(id="train-1", input="", metadata={"split": "train"}),
            Task(id="train-2", input="", metadata={"split": "train"}),
        ]
        self.val_tasks = [Task(id="val-1", input="", metadata={"split": "val"})]
        self.validation_evaluations = 0

    def get_tasks(self, split: str = "train", limit: int = 10):
        tasks = self.val_tasks if split in {"val", "validation"} else self.train_tasks
        return tasks[:limit]

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        if task.metadata["split"] == "val":
            self.validation_evaluations += 1
            success = trajectory.output != "bad"
        else:
            success = True
        return Feedback(success=success, score=float(success), detail="checked")


class ValidationMutatingEngine:
    def __init__(self):
        self.candidates = iter(["good", "bad"])
        self.batches = []
        self.accepted = []

    @property
    def manages_own_evaluation(self) -> bool:
        return False

    def step(self, workspace, observations, history, trial):
        self.batches.append([observation.task.id for observation in observations])
        workspace.write_prompt(next(self.candidates))
        return StepResult(mutated=True, summary="changed prompt")

    def on_cycle_end(self, accepted, score):
        self.accepted.append(accepted)


def test_validation_gate_accepts_equal_and_rolls_back_regression(tmp_path):
    workspace = _validation_workspace(tmp_path / "validation_workspace")
    benchmark = ValidationBenchmark()
    engine = ValidationMutatingEngine()
    config = EvolveConfig(
        batch_size=1,
        train_limit=2,
        validation_limit=1,
        max_cycles=1,
        egl_window=999,
    )

    result = EvolutionLoop(
        ValidationAgent(workspace), benchmark, engine, config
    ).run(cycles=1)

    assert (workspace / "prompts" / "system.md").read_text() == "good"
    assert benchmark.validation_evaluations == 3
    assert engine.batches == [["train-1"], ["train-2"]]
    assert engine.accepted == [True, False]
    assert result.details["initial_validation_accuracy"] == 1.0
    assert result.details["validation_accuracy_history"] == [1.0, 0.0]
    assert result.details["validation_accepted_history"] == [True, False]
    assert result.details["final_validation_accuracy"] == 1.0
    assert result.initial_validation_accuracy == 1.0
    assert result.validation_accuracy_history == [1.0, 0.0]
    assert result.validation_accepted_history == [True, False]
    assert result.final_validation_accuracy == 1.0

    validation_dir = workspace / "evolution" / "validation"
    for name in ("baseline", "cycle_0001", "cycle_0002"):
        assert (validation_dir / name / "results.csv").is_file()
        assert (validation_dir / name / "summary.json").is_file()

    metrics = json.loads((workspace / "evolution" / "metrics.json").read_text())
    assert metrics["validation_accuracy_history"] == [1.0, 0.0]
    rejected_summary = json.loads(
        (validation_dir / "cycle_0002" / "summary.json").read_text()
    )
    assert rejected_summary["candidate_accuracy"] == 0.0
    assert rejected_summary["comparison_baseline"] == 1.0
    assert rejected_summary["accepted"] is False
    assert rejected_summary["rolled_back"] is True
    assert rejected_summary["effective_tag"] == "evo-2"
    history = [
        json.loads(line)
        for line in (workspace / "evolution" / "history.jsonl").read_text().splitlines()
    ]
    assert history[1]["validation"]["rolled_back"] is True

    tags = subprocess.run(
        ["git", "tag", "--list"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert {"candidate-evo-2", "rejected-evo-2", "evo-2"} <= set(tags)


def test_validation_gate_no_mutation_does_not_rollback(tmp_path):
    workspace = _validation_workspace(tmp_path / "no_mutation_workspace")
    benchmark = ValidationBenchmark()
    engine = RecordingEngine()
    config = EvolveConfig(
        batch_size=1,
        train_limit=1,
        validation_limit=1,
        max_cycles=1,
        egl_window=999,
    )
    loop = EvolutionLoop(ValidationAgent(workspace), benchmark, engine, config)
    loop.versioning = MagicMock()

    result = loop.run(cycles=1)

    assert result.details["validation_accuracy_history"] == [1.0]
    assert result.details["validation_accepted_history"] == [True]
    loop.versioning.rollback_to_tag.assert_not_called()
    assert not any(
        call.kwargs.get("tag", "").startswith("candidate-evo-")
        for call in loop.versioning.commit.call_args_list
    )


def _validation_workspace(path: Path) -> Path:
    (path / "prompts").mkdir(parents=True)
    (path / "skills").mkdir()
    (path / "memory").mkdir()
    (path / "tools").mkdir()
    (path / "prompts" / "system.md").write_text("baseline")
    return path
