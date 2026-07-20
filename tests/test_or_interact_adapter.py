from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from baseline.react.agent import AgentResult
from baseline.react.trace import TraceWriter
from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.agents.or_interact.react_agent import ORReactAgent
from agent_evolve.benchmarks.or_interact import ORInteractBenchmark
from agent_evolve.engine.versioning import VersionControl
from agent_evolve.engine.observer import Observer
from agent_evolve.types import Feedback, Observation, Task, Trajectory
from examples.or_interact_examples.harness_tree import (
    HarnessTreeRunner,
    sanitize_branch_name,
)


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


def test_validation_split_preserves_original_test_boundary() -> None:
    original = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        seed=42,
        train_size=50,
        val_size=0,
    )
    gated = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        seed=42,
        train_size=40,
        val_size=10,
    )

    original_train = [task.id for task in original.get_tasks("train", limit=50)]
    original_test = [task.id for task in original.get_tasks("test", limit=50)]
    gated_train = [task.id for task in gated.get_tasks("train", limit=40)]
    gated_val = [task.id for task in gated.get_tasks("validation", limit=10)]
    gated_test = [task.id for task in gated.get_tasks("test", limit=50)]

    assert gated_train == original_train[:40]
    assert gated_val == original_train[40:50]
    assert gated_test == original_test
    assert [task.id for task in gated.get_tasks("holdout", limit=10)] == gated_val
    assert set(gated_train).isdisjoint(gated_val)
    assert set(gated_train).isdisjoint(gated_test)
    assert set(gated_val).isdisjoint(gated_test)


def test_default_holdout_remains_test_alias() -> None:
    benchmark = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)

    assert [task.id for task in benchmark.get_tasks("holdout", limit=50)] == [
        task.id for task in benchmark.get_tasks("test", limit=50)
    ]


def test_validation_split_errors_when_requested_size_is_unavailable(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "OR-Interact-Bench"
    for index in range(3):
        _minimal_visible_task(benchmark_dir / "Tiny" / f"task_{index:03d}")

    with pytest.raises(ValueError, match="validation split cannot be satisfied"):
        ORInteractBenchmark(
            benchmark_dir=benchmark_dir,
            dataset="Tiny",
            train_size=2,
            val_size=2,
        )


def test_or_interact_clis_parse_validation_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.or_interact_examples import evaluate_or_interact, evolve_or_interact

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_or_interact.py",
            "--limit-train",
            "40",
            "--limit-val",
            "10",
            "--limit-test",
            "50",
        ],
    )
    evolve_args = evolve_or_interact.parse_args()
    assert (evolve_args.limit_train, evolve_args.limit_val, evolve_args.limit_test) == (
        40,
        10,
        50,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_or_interact.py",
            "--split",
            "val",
            "--limit-train",
            "40",
            "--limit-val",
            "10",
            "--limit-test",
            "50",
        ],
    )
    evaluate_args = evaluate_or_interact.parse_args()
    assert evaluate_args.split == "val"
    assert (evaluate_args.limit_train, evaluate_args.limit_val, evaluate_args.limit_test) == (
        40,
        10,
        50,
    )


def test_evolve_cli_selects_gepa_and_metric_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.or_interact_examples import evolve_or_interact

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_or_interact.py",
            "--algorithm",
            "gepa",
            "--limit-val",
            "10",
            "--gepa-max-metric-calls",
            "50",
        ],
    )

    args = evolve_or_interact.parse_args()

    assert args.algorithm == "gepa"
    assert args.gepa_max_metric_calls == 50


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (["--step-opsd"], "--step-opsd cannot be used"),
        (["--harness-tree"], "--harness-tree cannot be used"),
        (["--limit-val", "0"], "requires --limit-val greater than 0"),
    ],
)
def test_evolve_cli_rejects_invalid_gepa_combinations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    from examples.or_interact_examples import evolve_or_interact

    argv = ["evolve_or_interact.py", "--algorithm", "gepa"]
    if "--limit-val" not in extra_args:
        argv.extend(["--limit-val", "10"])
    monkeypatch.setattr(sys, "argv", [*argv, *extra_args])

    with pytest.raises(SystemExit):
        evolve_or_interact.parse_args()

    assert message in capsys.readouterr().err


def test_evolve_cli_rejects_non_positive_gepa_budget(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.or_interact_examples import evolve_or_interact

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_or_interact.py",
            "--algorithm",
            "gepa",
            "--limit-val",
            "10",
            "--gepa-max-metric-calls",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        evolve_or_interact.parse_args()

    assert "must be a positive integer" in capsys.readouterr().err


def test_gepa_engine_uses_evolver_openai_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    from examples.or_interact_examples import evolve_or_interact

    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="updated"))]
    )
    create = MagicMock(return_value=response)
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    openai_factory = MagicMock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", openai_factory)
    fake_gepa = types.SimpleNamespace(
        EngineConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ReflectionConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        GEPAConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        optimize_anything=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "gepa.optimize_anything", fake_gepa)
    config = EvolveConfig(
        batch_size=10,
        max_cycles=1,
        validation_limit=None,
        evolve_tools=False,
    )

    engine = evolve_or_interact._build_gepa_engine(
        config,
        max_metric_calls=50,
        validation_limit=10,
        model="openai:deepseek-v4-flash",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        temperature=0.25,
    )
    reflection_lm = engine.gepa_config.reflection.reflection_lm

    assert engine.gepa_config.engine.max_metric_calls == 50
    assert engine.validation_limit == 10
    assert config.validation_limit is None
    assert config.evolve_tools is False
    assert reflection_lm("reflect") == "updated"
    openai_factory.assert_called_once_with(
        base_url="https://example.invalid/v1",
        api_key="test-key",
    )
    create.assert_called_once_with(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "reflect"}],
        temperature=0.25,
    )


def test_loads_dataset_directory_when_index_lacks_dataset(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "OR-Interact-Bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "index.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task_001",
                        "source_instance_dir": "instance_1",
                        "path": "IndustryOR/task_001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _minimal_visible_task(benchmark_dir / "LargeScaleOR" / "task_001")
    _minimal_visible_task(benchmark_dir / "LargeScaleOR" / "task_002")

    benchmark = ORInteractBenchmark(
        benchmark_dir=benchmark_dir,
        dataset="LargeScaleOR",
        train_size=50,
    )
    tasks = benchmark.get_tasks("train", limit=10)

    assert sorted(task.id for task in tasks) == ["task_001", "task_002"]
    assert tasks[0].metadata["dataset"] == "LargeScaleOR"
    assert tasks[0].metadata["visible_roots"] == ["docs", "data"]


def test_task_metadata_includes_category_from_task_metadata_json(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "OR-Interact-Bench"
    task_dir = benchmark_dir / "RCO" / "task_001"
    _minimal_visible_task(task_dir)
    (task_dir / "metadata.json").write_text(
        json.dumps({"category": "logistics_distribution_and_routing_optimization"}),
        encoding="utf-8",
    )

    benchmark = ORInteractBenchmark(
        benchmark_dir=benchmark_dir,
        dataset="RCO",
        train_size=50,
    )
    task = benchmark.get_tasks("train", limit=1)[0]

    assert task.metadata["category"] == "logistics_distribution_and_routing_optimization"


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


def test_evaluate_records_grounded_ask_diagnostics(tmp_path: Path) -> None:
    benchmark = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        dataset="IndustryOR",
        interaction_enabled=True,
    )
    task = _task_by_id(benchmark, "task_001")
    expected = _oracle_value(task)
    _write_answer(tmp_path, expected)
    events = [
        {"type": "tool_call", "name": "ask_user", "arguments": {"question": "How should training work?"}},
        {
            "type": "tool_output",
            "name": "ask_user",
            "output": {
                "user_response": {
                    "answered": True,
                    "answer": "private",
                    "matched_file": "clarification.md",
                    "code": "answered",
                    "reason": "private matcher reason",
                }
            },
        },
        {"type": "tool_call", "name": "ask_user", "arguments": {"question": "Which route?"}},
        {
            "type": "tool_output",
            "name": "ask_user",
            "output": {
                "user_response": {
                    "answered": False,
                    "answer": "private refusal",
                    "matched_file": None,
                    "code": "no_match",
                    "reason": "private matcher reason",
                }
            },
        },
        {"type": "run_summary", "runtime_dir": str(tmp_path), "status": "success"},
    ]
    trajectory = Trajectory(
        task_id=task.id,
        output="",
        steps=events,
        conversation=events,
    )

    feedback = benchmark.evaluate(task, trajectory)

    evaluation = feedback.raw["evaluation"]
    assert evaluation["has_grounded"] is True
    assert evaluation["grounded_record_count"] > 0
    assert evaluation["has_answerable_grounded"] is True
    assert evaluation["ask_count"] == 2
    assert evaluation["answered_ask_count"] == 1
    assert evaluation["refused_ask_count"] == 1
    assert evaluation["no_match_ask_count"] == 1
    assert evaluation["no_grounded_records_ask_count"] == 0


def test_evaluate_omits_interaction_diagnostics_by_default(tmp_path: Path) -> None:
    benchmark = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        dataset="IndustryOR",
    )
    task = _task_by_id(benchmark, "task_001")
    expected = _oracle_value(task)
    _write_answer(tmp_path, expected)
    events = [
        {"type": "tool_call", "name": "ask_user", "arguments": {"question": "Hidden?"}},
        {"type": "run_summary", "runtime_dir": str(tmp_path)},
    ]

    feedback = benchmark.evaluate(
        task,
        Trajectory(task_id=task.id, output="", steps=events, conversation=events),
    )

    evaluation = feedback.raw["evaluation"]
    assert "has_grounded" not in evaluation
    assert "grounded_record_count" not in evaluation
    assert "ask_count" not in evaluation


def test_observer_redacts_grounded_answer_from_persisted_trajectory(tmp_path: Path) -> None:
    task = Task(id="task_x", input="", metadata={})
    events = [
        {
            "type": "assistant_message",
            "content": "I should ask.",
            "tool_calls": [
                {
                    "function": {
                        "name": "ask_user",
                        "arguments": json.dumps({"question": "PRIVATE DUPLICATE QUESTION"}),
                    }
                }
            ],
        },
        {"type": "tool_call", "name": "ask_user", "arguments": {"question": "Which rule?"}},
        {
            "type": "tool_output",
            "name": "ask_user",
            "output": {
                "user_response": {
                    "answered": False,
                    "answer": "PRIVATE GROUNDED ANSWER",
                    "matched_file": "private.md",
                    "code": "no_match",
                    "reason": "PRIVATE MATCHER REASON",
                }
            },
        },
    ]
    observation = Observation(
        task=task,
        trajectory=Trajectory(task_id=task.id, output="", steps=events, conversation=events),
        feedback=Feedback(success=False, score=0.0, detail="failed"),
    )

    record = Observer(tmp_path / "evolution").record_from_observation(observation)

    serialized = json.dumps(record)
    assert "Which rule?" not in serialized
    assert "PRIVATE DUPLICATE QUESTION" not in serialized
    assert "PRIVATE GROUNDED ANSWER" not in serialized
    assert "private.md" not in serialized
    assert "PRIVATE MATCHER REASON" not in serialized
    assert record["conversation"][0]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert record["conversation"][1]["arguments"] == {}
    assert record["conversation"][2]["output"] == {
        "user_response": {
            "answered": False,
            "code": "no_match",
            "reason": "No task-specific clarification matches this question.",
        }
    }


def test_evaluate_wrong_objective(tmp_path: Path) -> None:
    benchmark = ORInteractBenchmark(benchmark_dir=BENCHMARK_DIR)
    task = _task_by_id(benchmark, "task_001")
    _write_answer(tmp_path, -1)

    feedback = benchmark.evaluate(task, _trajectory(tmp_path))

    assert feedback.success is False
    assert feedback.score == 0.0
    assert "outside relative tolerance" in feedback.detail
    assert "Reference solution path:" in feedback.detail
    assert "oracle/reference_solution.py" in feedback.detail
    assert "factory_planning" in feedback.detail


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


def test_agent_prompt_uses_catalog_without_harness_tree_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    _write_registry(
        workspace,
        [
            {"name": "diagnose", "file": "diagnose.py", "function": "diagnose", "description": "diagnostic helper"},
        ],
    )

    captured: dict[str, object] = {}

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            captured["system_prompt"] = system_prompt
            captured["tools"] = registry.list_tools()
            captured["read_skill"] = registry.call("read_skill", name="modeling")
            captured["read_skill_by_path"] = registry.call("read_skill", name="skills/modeling/SKILL.md")
            captured["read_md_task"] = registry.call("read_md", file="docs/business_requirement.md")

        def run(self, **kwargs):
            from baseline.react.agent import AgentResult

            return AgentResult(status="success", turns=1, objective_value=1)

    monkeypatch.setattr("agent_evolve.agents.or_interact.react_agent.ReActAgent", FakeReActAgent)
    agent = ORReactAgent(workspace)
    task = Task(id="task_x", input="", metadata={"task_dir": str(_visible_task(tmp_path)), "dataset": "IndustryOR"})

    agent.solve(task)

    prompt = str(captured["system_prompt"])
    assert "type_router" not in prompt
    assert "answer_checker" not in prompt
    assert "Prefer explicit variable bounds." not in prompt
    assert "skills/modeling" not in prompt
    assert "modeling; description=Test skill" in prompt
    assert "Use `list_skills` to inspect available skills" in prompt
    assert "Skills live in the evolved workspace" in prompt
    assert "memory:1 category=memories; content=Check objective direction before finalizing." in prompt
    assert "diagnose" in captured["tools"]
    assert "list_skills" in captured["tools"]
    assert "read_skill" in captured["tools"]
    read_skill = captured["read_skill"]
    assert isinstance(read_skill, dict)
    assert read_skill["name"] == "modeling"
    assert "Prefer explicit variable bounds." in read_skill["content"]
    read_skill_by_path = captured["read_skill_by_path"]
    assert isinstance(read_skill_by_path, dict)
    assert read_skill_by_path["name"] == "modeling"
    assert "Prefer explicit variable bounds." in read_skill_by_path["content"]
    read_md_task = captured["read_md_task"]
    assert isinstance(read_md_task, dict)
    assert read_md_task["markdown"]["content"] == "Task"
    assert "type_router" not in captured["tools"]
    assert "answer_checker" not in captured["tools"]
    assert "run_heuristic" not in captured["tools"]
    assert "run_heuristic" not in prompt


def test_or_interact_heuristic_tool_is_enabled_only_by_workspace_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "or_interact_settings.json").write_text(
        json.dumps({"enable_heuristic_tool": True}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            captured["system_prompt"] = system_prompt
            captured["tools"] = registry.list_tools()

        def run(self, **kwargs):
            from baseline.react.agent import AgentResult

            return AgentResult(status="success", turns=1, objective_value=1)

    monkeypatch.setattr("agent_evolve.agents.or_interact.react_agent.ReActAgent", FakeReActAgent)
    agent = ORReactAgent(workspace)
    task = Task(id="task_x", input="", metadata={"task_dir": str(_visible_task(tmp_path)), "dataset": "IndustryOR"})

    agent.solve(task)

    prompt = str(captured["system_prompt"])
    assert "run_heuristic" in captured["tools"]
    assert "run_heuristic" in prompt
    assert "Heuristic Algorithm Evolution" in prompt
    assert "Do not" in prompt
    assert "tools/run_heuristic.py" in prompt


def test_or_interact_user_tool_is_enabled_only_by_workspace_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "or_interact_settings.json").write_text(
        json.dumps({"enable_user_tool": True}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OR_USER_SIM_MODEL", "fake-user-model")
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.chat = object()

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            self.registry = registry
            captured["system_prompt"] = system_prompt
            captured["tools"] = registry.list_tools()

        def run(self, **kwargs):
            from baseline.react.agent import AgentResult

            captured["user_response"] = self.registry.call(
                "ask_user",
                question="Which depot?",
            )
            return AgentResult(status="success", turns=1, objective_value=1)

    monkeypatch.setattr("user.simulator.OpenAI", FakeOpenAI)
    monkeypatch.setattr("agent_evolve.agents.or_interact.react_agent.ReActAgent", FakeReActAgent)
    task_dir = _visible_task(tmp_path)
    (task_dir / "grounded").mkdir()
    agent = ORReactAgent(workspace)
    task = Task(id="task_x", input="", metadata={"task_dir": str(task_dir), "dataset": "IndustryOR"})

    agent.solve(task)

    prompt = str(captured["system_prompt"])
    assert "ask_user" in captured["tools"]
    assert "ask_user" in prompt
    tool_response = captured["user_response"]
    assert isinstance(tool_response, dict)
    user_response = tool_response["user_response"]
    assert user_response["answered"] is False
    assert user_response["question"] == "Which depot?"
    assert user_response["matched_file"] is None
    assert user_response["code"] == "no_grounded_records"
    assert user_response["answer"]
    assert "run_heuristic" not in captured["tools"]


def test_or_interact_user_tool_disabled_hides_inherited_interaction_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "prompts" / "system.md").write_text(
        "Base prompt\n\n## User Interaction\nUse ask_user with grounded clarifications.\n",
        encoding="utf-8",
    )
    _write_skill(workspace, "modeling", "Prefer explicit variable bounds.")
    _write_skill(
        workspace,
        "clarification-policy",
        "Use ask_user when grounded clarification is needed.",
    )
    (workspace / "memory" / "memories.jsonl").write_text(
        json.dumps({"content": "Remember the ask_user interaction policy."}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            captured["system_prompt"] = system_prompt
            captured["tools"] = registry.list_tools()
            captured["skills"] = registry.call("list_skills")

        def run(self, **kwargs):
            return AgentResult(status="success", turns=1, objective_value=1)

    monkeypatch.setattr(
        "agent_evolve.agents.or_interact.react_agent.ReActAgent",
        FakeReActAgent,
    )
    agent = ORReactAgent(workspace)
    task = Task(
        id="task_x",
        input="",
        metadata={"task_dir": str(_visible_task(tmp_path)), "dataset": "IndustryOR"},
    )

    agent.solve(task)

    prompt = str(captured["system_prompt"])
    assert "Base prompt" in prompt
    assert "modeling; description=Test skill" in prompt
    assert "ask_user" not in prompt
    assert "User Interaction" not in prompt
    assert "grounded clarification" not in prompt
    assert "clarification-policy" not in prompt
    assert "interaction policy" not in prompt
    assert "ask_user" not in captured["tools"]
    assert captured["skills"] == {
        "skills": [
            {
                "name": "modeling",
                "path": "skills/modeling",
                "description": "Test skill",
            }
        ],
        "count": 1,
    }


@pytest.mark.parametrize(
    "tool_source",
    [
        "import subprocess\nfrom typing import Any\n\ndef run_heuristic() -> dict[str, Any]:\n    return {}\n",
        "from typing import Any\n\ndef run_heuristic() -> dict[str, Any]:\n    return {'error': '\n",
        "from typing import Any\n\ndef run_heuristic(*args: Any, **kwargs: Any) -> dict[str, Any]:\n    return {}\n",
    ],
)
def test_workspace_run_heuristic_entries_are_reserved_and_ignored(
    tmp_path: Path,
    tool_source: str,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "or_interact_settings.json").write_text(
        json.dumps({"enable_heuristic_tool": True}),
        encoding="utf-8",
    )
    _write_tool(workspace, "run_heuristic", tool_source)
    _write_registry(
        workspace,
        [
            {
                "name": "run_heuristic",
                "file": "run_heuristic.py",
                "function": "run_heuristic",
            }
        ],
    )

    agent = ORReactAgent(workspace)

    assert "run_heuristic" in agent.registry.list_tools()
    assert agent.registry.get("run_heuristic").kind == "dynamic"


def test_workspace_run_heuristic_entry_is_ignored_when_switch_is_off(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_tool(
        workspace,
        "run_heuristic",
        "from typing import Any\n\ndef run_heuristic() -> dict[str, Any]:\n    return {}\n",
    )
    _write_registry(
        workspace,
        [
            {
                "name": "run_heuristic",
                "file": "run_heuristic.py",
                "function": "run_heuristic",
            }
        ],
    )

    agent = ORReactAgent(workspace)

    assert "run_heuristic" not in agent.registry.list_tools()


def test_type_router_is_available_only_for_route_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    from baseline.react.agent import tool_schemas

    workspace = REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react"
    agent = ORReactAgent(workspace)

    assert "type_router" not in agent.registry.list_tools()
    assert "answer_checker" not in agent.registry.list_tools()

    route_registry = agent._build_registry(include_type_router=True)
    assert route_registry.get("type_router").kind == "evolved"
    assert "answer_checker" not in route_registry.list_tools()
    schemas = {schema["function"]["name"]: schema for schema in tool_schemas(route_registry)}
    router_parameters = schemas["type_router"]["function"]["parameters"]["properties"]
    assert router_parameters["existing_branches"]["type"] == "array"


def test_skill_tools_are_disabled_for_direct_phases_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(workspace, "modeling", "Prefer explicit variable bounds.")
    _copy_harness_tool(workspace, "type_router")
    _write_registry(workspace, [{"name": "type_router", "file": "type_router.py", "function": "type_router"}])
    captured: dict[str, object] = {}

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            captured["tools"] = registry.list_tools()
            captured["system_prompt"] = system_prompt

        def run(self, **kwargs):
            return AgentResult(status="success", turns=1, objective_value=1)

    monkeypatch.setattr("agent_evolve.agents.or_interact.react_agent.ReActAgent", FakeReActAgent)
    agent = ORReactAgent(workspace)
    task = Task(id="task_x", input="", metadata={"task_dir": str(_visible_task(tmp_path)), "dataset": "IndustryOR"})

    agent.run_phase(
        task,
        runtime_dir=tmp_path / "runs",
        trace=TraceWriter(tmp_path / "runs"),
        phase="train:route",
    )

    assert "type_router" in captured["tools"]
    assert "list_skills" not in captured["tools"]
    assert "read_skill" not in captured["tools"]
    assert "Use `list_skills` to inspect available skills" not in str(captured["system_prompt"])

    direct_registry = agent._build_registry(include_skill_tools=True)
    assert "type_router" not in direct_registry.list_tools()
    assert "list_skills" in direct_registry.list_tools()
    assert "read_skill" in direct_registry.list_tools()


def test_answer_checker_is_not_registered_even_when_env_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OR_REACT_ENABLE_ANSWER_CHECKER", "1")
    workspace = REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react"
    agent = ORReactAgent(workspace)

    assert "answer_checker" not in agent.registry.list_tools()
    assert "answer_checker" not in agent._build_registry(include_type_router=True).list_tools()


def test_type_router_uses_metadata_category_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react"
    agent = ORReactAgent(workspace)
    monkeypatch.setenv("OR_INTERACT_TASK_CATEGORY", "Known Category")

    result = agent._build_registry(include_type_router=True).call(
        "type_router",
        evidence_text="Visible routing evidence.",
        existing_branches=[],
    )

    assert result == {
        "branch_name": "branch/known-category",
        "confidence": 1.0,
        "rationale": "task metadata category: Known Category",
    }


def test_retailopt_task_can_index_and_call_read_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from baseline.react.agent import AgentResult, tool_schemas

    import agent_evolve.agents.or_interact.react_agent as react_module

    workspace = tmp_path / "workspace"
    shutil.copytree(REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react", workspace)
    benchmark = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        dataset="RetailOpt",
        train_size=999,
    )
    task = _task_by_id(benchmark, "task_001")
    captured: dict[str, object] = {}

    class FakeReActAgent:
        def __init__(self, *, config, trace, registry, system_prompt):
            captured["registry_tools"] = registry.list_tools()
            captured["schema_names"] = [tool["function"]["name"] for tool in tool_schemas(registry)]
            captured["system_prompt"] = system_prompt
            output = registry.call("read_json", file="data/instance.json")
            captured["read_json_file"] = output["json"]["file"]
            captured["read_json_content"] = output["json"]["content"]

        def run(self, **kwargs):
            return AgentResult(status="success", turns=1, objective_value="smoke")

    monkeypatch.setattr(react_module, "ReActAgent", FakeReActAgent)
    monkeypatch.setenv("OR_REACT_RESULTS_DIR", str(tmp_path / "runs"))

    trajectory = ORReactAgent(workspace).solve(task)

    assert "read_json" in captured["registry_tools"]
    assert "read_json" in captured["schema_names"]
    assert "read_json" in str(captured["system_prompt"])
    assert "answer_checker" not in captured["registry_tools"]
    assert "answer_checker" not in captured["schema_names"]
    assert "answer_checker" not in str(captured["system_prompt"])
    assert trajectory.steps[-1]["tools"] == captured["registry_tools"]
    assert captured["read_json_file"] == "data/instance.json"
    assert isinstance(captured["read_json_content"], dict)
    assert "products" in captured["read_json_content"]


def test_or_interact_cli_has_no_check_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.or_interact_examples import (
        evaluate_or_interact,
        evolve_or_interact,
        review_step_opsd_failures,
    )

    monkeypatch.setattr(sys, "argv", ["evaluate_or_interact.py"])
    assert not hasattr(evaluate_or_interact.parse_args(), "check")

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py"])
    args = evolve_or_interact.parse_args()
    assert not hasattr(args, "check")
    assert args.algorithm == "adaptive-skill"
    assert args.enable_heuristic_tool is False
    assert args.enable_user_tool is False
    assert args.step_opsd is False
    assert args.offline is False

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--enable-heuristic-tool"])
    assert evolve_or_interact.parse_args().enable_heuristic_tool is True

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--enable-user-tool"])
    assert evolve_or_interact.parse_args().enable_user_tool is True

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--step-opsd"])
    assert evolve_or_interact.parse_args().step_opsd is True

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--harness-tree", "--offline"])
    assert evolve_or_interact.parse_args().offline is True

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--offline"])
    with pytest.raises(SystemExit):
        evolve_or_interact.parse_args()

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--harness-tree", "--step-opsd"])
    with pytest.raises(SystemExit):
        evolve_or_interact.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_step_opsd_failures.py",
            "--dataset",
            "IndustryOR",
            "--limit",
            "2",
            "--output-dir",
            "/tmp/reviews",
        ],
    )
    review_args = review_step_opsd_failures.parse_args()
    assert review_args.dataset == "IndustryOR"
    assert review_args.limit == 2
    assert review_args.output_dir == "/tmp/reviews"


def test_or_interact_evolve_settings_writer_records_heuristic_switch(tmp_path: Path) -> None:
    from examples.or_interact_examples.evolve_or_interact import _write_or_interact_settings

    _write_or_interact_settings(tmp_path, enable_heuristic_tool=True)

    settings = json.loads((tmp_path / "or_interact_settings.json").read_text(encoding="utf-8"))
    assert settings == {"enable_heuristic_tool": True, "enable_user_tool": False}


def test_type_router_returns_only_branch_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(workspace, "legacy-skill", "Legacy skill body.")
    (workspace / "memory" / "memories.jsonl").write_text(
        json.dumps({"content": "Legacy memory."}) + "\n",
        encoding="utf-8",
    )
    _copy_harness_tool(workspace, "type_router")
    _write_registry(workspace, [{"name": "type_router", "file": "type_router.py", "function": "type_router"}])
    _install_fake_openai(
        monkeypatch,
        [
            {
                "branch_name": "branch/production-planning",
                "confidence": 0.8,
                "rationale": "production evidence",
            }
        ],
    )

    agent = ORReactAgent(workspace)
    result = agent._build_registry(include_type_router=True).call(
        "type_router",
        evidence_text="Visible production planning evidence.",
    )

    assert result == {
        "branch_name": "branch/production-planning",
        "confidence": 0.8,
        "rationale": "production evidence",
    }
    assert "selected_skills" not in result
    assert "selected_memories" not in result
    assert "task_types" not in result


def test_type_router_returns_branch_route_with_existing_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _copy_harness_tool(workspace, "type_router")
    _write_registry(workspace, [{"name": "type_router", "file": "type_router.py", "function": "type_router"}])
    _install_fake_openai(
        monkeypatch,
        [
            {
                "branch_name": "branch/Routing VRP",
                "confidence": 0.9,
                "rationale": "routing evidence matches existing branch",
            }
        ],
    )

    agent = ORReactAgent(workspace)
    result = agent._build_registry(include_type_router=True).call(
        "type_router",
        evidence_text="Visible vehicle route evidence.",
        existing_branches=[
            {
                "branch_name": "branch/routing-vrp",
                "rationale": "Vehicle routing tasks.",
            }
        ],
    )

    assert result["branch_name"] == "branch/routing-vrp"
    assert result["confidence"] == 0.9
    assert result["rationale"] == "routing evidence matches existing branch"


def test_type_router_judge_only_routes_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(workspace, "legacy-skill", "Legacy skill body.")
    _copy_harness_tool(workspace, "type_router")
    _write_registry(workspace, [{"name": "type_router", "file": "type_router.py", "function": "type_router"}])
    captured_requests: list[dict[str, Any]] = []
    _install_fake_openai(
        monkeypatch,
        [
            {
                "branch_name": "branch/general",
                "confidence": 0.7,
                "rationale": "general route",
            }
        ],
        captured_requests=captured_requests,
    )

    agent = ORReactAgent(workspace)
    agent._build_registry(include_type_router=True).call(
        "type_router",
        evidence_text="Visible production planning evidence.",
        existing_branches=[],
    )

    messages = captured_requests[0]["messages"]
    system = messages[0]["content"]
    user_payload = json.loads(messages[1]["content"])
    assert "branch_action" not in system
    assert "branch_label" not in system
    assert "confidence" in system
    assert "rationale" in system
    assert "task_types" not in system
    assert "selected_skill_paths" not in system
    assert "selected_memory_paths" not in system
    assert "broad OR problem family" in system
    assert "Do not select skills" in system
    assert "harness_catalog" not in user_payload
    assert user_payload["existing_branches"] == []


def test_sanitize_branch_name_handles_empty_unicode_and_special_chars() -> None:
    assert sanitize_branch_name("") == "branch/general"
    assert sanitize_branch_name("   ") == "branch/general"
    assert sanitize_branch_name("中文") == "branch/general"
    assert sanitize_branch_name("Routing VRP!") == "branch/routing-vrp"
    assert sanitize_branch_name("branch/Piecewise Discount") == "branch/piecewise-discount"
    assert sanitize_branch_name("production_planning") == "branch/production-planning"


def test_version_control_branch_api(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manifest.yaml").write_text("name: test\n", encoding="utf-8")

    vc = VersionControl(workspace)
    vc.init()

    assert vc.get_current_branch() == "main"
    assert vc.branch_exists("main")
    vc.create_branch("branch/routing-vrp", "main")
    assert vc.branch_exists("branch/routing-vrp")
    assert "branch/routing-vrp" in vc.list_branches()
    vc.checkout_branch("branch/routing-vrp")
    assert vc.get_current_branch() == "branch/routing-vrp"
    vc.checkout_branch("main")
    assert vc.get_current_branch() == "main"


def test_harness_tree_routes_buffers_and_final_eval_does_not_evolve(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    agent = FakeAgent(workspace)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=3, train_limit=3),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
    )

    progress_events: list[dict[str, Any]] = []
    result = runner.run_training(max_epochs=1, progress_callback=progress_events.append)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())

    assert result.details["tasks_completed"] == 3
    assert result.details["updates_completed"] == 3
    assert engine.evolved_scopes == ["alpha", "main", "beta"]
    assert engine.trajectory_profiles == ["branch", "main", "branch"]
    assert engine.prompt_log_dirs == [workspace / "evolution" / "evolver_prompts"] * 3
    assert engine.prompt_log_scopes == ["branch_alpha", "main", "branch_beta"]
    assert sorted(state["branches"]) == ["branch/alpha", "branch/beta"]
    assert state["main_pending"] == []
    assert state["branches"]["branch/alpha"]["pending"] == []
    assert state["branches"]["branch/beta"]["pending"] == []
    assert state["branches"]["branch/beta"]["solve_count"] == 1
    observation_files = sorted((workspace / "evolution" / "observations").glob("batch_*.jsonl"))
    assert [path.name for path in observation_files] == [
        "batch_0001_branch_alpha.jsonl",
        "batch_0002_main.jsonl",
        "batch_0003_branch_beta.jsonl",
    ]
    assert [len(path.read_text(encoding="utf-8").splitlines()) for path in observation_files] == [2, 3, 1]
    observation_records = [
        json.loads(line)
        for path in observation_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phase_skill_tool_flags = [
        (step.get("phase"), step.get("enable_skill_tools"))
        for record in observation_records
        for step in record["conversation"]
        if step.get("type") == "assistant_message"
    ]
    assert any(phase.endswith(":route") and enabled is False for phase, enabled in phase_skill_tool_flags)
    assert not any(phase.endswith(":route") and enabled is True for phase, enabled in phase_skill_tool_flags)
    assert any(phase.endswith(":solve") and enabled is True for phase, enabled in phase_skill_tool_flags)
    assert [event["event"] for event in progress_events].count("task_done") == 3
    assert any(event.get("event") == "evolve_done" and event.get("scope") == "main" for event in progress_events)
    assert (workspace / "memory" / "main.jsonl").is_file()
    alpha_overlay = workspace / "evolution" / "harness_tree" / "overlays" / "alpha" / "files"
    assert (alpha_overlay / "skills" / "domain-alpha" / "SKILL.md").is_file()
    assert not (alpha_overlay / "memory" / "main.jsonl").exists()

    pending_before = {
        name: list(branch["pending"])
        for name, branch in state["branches"].items()
    }
    main_pending_before = list(state["main_pending"])
    eval_summary = runner.run_final_evaluation(limit=1, output_dir=workspace / "evolution" / "final_test")
    state_after = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())

    assert eval_summary["total"] == 1
    assert eval_summary["per_branch"]["branch/alpha"]["total"] == 1
    assert engine.evolved_scopes == ["alpha", "main", "beta"]
    assert {
        name: branch["pending"]
        for name, branch in state_after["branches"].items()
    } == pending_before
    assert state_after["main_pending"] == main_pending_before


def test_harness_tree_final_eval_parallel_preserves_order(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.test_tasks = [
        _harness_task(tmp_path, f"alpha_eval_{index}", "alpha eval")
        for index in range(6)
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=10, train_limit=0),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
    )

    runner.run_final_evaluation(limit=6, output_dir=workspace / "evolution" / "final_test")

    with (workspace / "evolution" / "final_test" / "results.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == [f"alpha_eval_{index}" for index in range(6)]
    assert len({row["pid"] for row in rows}) > 1

    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())
    assert [item["task_id"] for item in state["final_eval"]["decisions"]] == [
        f"alpha_eval_{index}" for index in range(6)
    ]


def test_harness_tree_parallel_train_worker_error_is_recorded(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_error", "alpha explode"),
        _harness_task(tmp_path, "alpha_ok", "alpha model"),
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=10, train_limit=2),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
    )

    result = runner.run_training(max_epochs=1)
    observation_file = workspace / "evolution" / "observations" / "batch_0001_branch_alpha.jsonl"
    records = [
        json.loads(line)
        for line in observation_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.details["tasks_completed"] == 2
    assert engine.evolved_scopes == ["alpha", "main"]
    assert engine.trajectory_profiles == ["branch", "main"]
    assert engine.prompt_log_dirs == [workspace / "evolution" / "evolver_prompts"] * 2
    assert engine.prompt_log_scopes == ["branch_alpha", "main"]
    assert [record["task_id"] for record in records] == ["alpha_error", "alpha_ok"]
    assert records[0]["feedback_detail"] == "RuntimeError: boom"
    assert records[0]["score"] == 0.0
    assert records[1]["score"] == 1.0


def test_harness_tree_parallel_train_solve_uses_multiple_workers(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_one", "alpha model"),
        _harness_task(tmp_path, "alpha_two", "alpha model again"),
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=10, train_limit=2),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
    )

    runner.run_training(max_epochs=1)

    observation_file = workspace / "evolution" / "observations" / "batch_0001_branch_alpha.jsonl"
    records = [
        json.loads(line)
        for line in observation_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len({record["feedback"]["raw"]["evaluation"]["pid"] for record in records}) > 1


def test_harness_tree_offline_batches_evolve_after_parallel_solve(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_off_1", "alpha model"),
        _harness_task(tmp_path, "beta_off_1", "beta model"),
        _harness_task(tmp_path, "alpha_off_2", "alpha model again"),
        _harness_task(tmp_path, "beta_off_2", "beta model again"),
        _harness_task(tmp_path, "alpha_off_3", "alpha model third"),
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=2, train_limit=5),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
        offline=True,
    )

    progress_events: list[dict[str, Any]] = []
    result = runner.run_training(max_epochs=1, progress_callback=progress_events.append)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())
    branch_records = []
    for observation_file in sorted((workspace / "evolution" / "observations").glob("batch_*_branch_*.jsonl")):
        branch_records.extend(
            json.loads(line)
            for line in observation_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    assert result.details["offline"] is True
    assert result.details["tasks_completed"] == 5
    assert result.details["updates_completed"] == 6
    assert engine.evolved_scopes == ["main", "main", "main", "alpha", "alpha", "beta"]
    assert engine.records_per_evolve == [2, 2, 1, 2, 1, 2]
    assert engine.trajectory_profiles == ["main", "main", "main", "branch", "branch", "branch"]
    assert state["main_pending"] == []
    assert state["branches"]["branch/alpha"]["pending"] == []
    assert state["branches"]["branch/beta"]["pending"] == []
    assert len({record["feedback"]["raw"]["evaluation"]["pid"] for record in branch_records}) > 1
    evolve_starts = [
        event["scope"]
        for event in progress_events
        if event.get("event") == "evolve_start"
    ]
    assert evolve_starts == ["main", "main", "main", "branch/alpha", "branch/alpha", "branch/beta"]


def test_harness_tree_offline_route_uses_incremental_branch_table(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_slow", "alpha route_sleep"),
        _harness_task(tmp_path, "beta_slow", "beta route_sleep"),
        _harness_task(tmp_path, "reuse", "reuse_existing model"),
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=10, train_limit=3),
        type_buffer_size=10,
        router_confidence_threshold=0.5,
        offline=True,
    )

    runner.run_training(max_epochs=1)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())
    reuse_decision = next(
        item for item in state["router_decisions"] if item["task_id"] == "reuse"
    )

    assert reuse_decision["branch_name"] in {"branch/alpha", "branch/beta"}
    assert "branch/reuse-new" not in state["branches"]


def test_harness_tree_offline_disable_main_evolve_isolates_branch_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    original_prompt = (workspace / "prompts" / "system.md").read_text(encoding="utf-8")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_iso_1", "alpha model"),
        _harness_task(tmp_path, "alpha_iso_2", "alpha model again"),
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = LeakyEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=1, train_limit=2),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
        disable_main_evolve=True,
        offline=True,
    )

    result = runner.run_training(max_epochs=1)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())

    assert result.details["offline"] is True
    assert state["main_evolutions"] == []
    assert state["main_pending"] == []
    assert engine.evolved_scopes == ["alpha"]
    assert engine.records_per_evolve == [2]
    assert not (workspace / "skills" / "domain-alpha").exists()
    assert not (workspace / "skills" / "leaked-main").exists()
    assert (workspace / "prompts" / "system.md").read_text(encoding="utf-8") == original_prompt
    assert (
        workspace
        / "evolution"
        / "harness_tree"
        / "overlays"
        / "alpha"
        / "files"
        / "skills"
        / "domain-alpha"
        / "SKILL.md"
    ).is_file()


def test_harness_tree_route_phase_uses_metadata_category_in_router(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [_harness_task(tmp_path, "categorized_1", "alpha model")]
    benchmark.train_tasks[0].metadata["category"] = "Known Category"
    agent = FakeAgent(workspace, parallelism=2)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=2, train_limit=1),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
    )

    runner.run_training(max_epochs=1)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())

    assert sorted(state["branches"]) == ["branch/known-category"]
    assert state["router_decisions"][0]["branch_name"] == "branch/known-category"
    assert state["router_decisions"][0]["fallback_reason"] is None
    assert any(phase.endswith(":route") for phase in agent.phases)
    assert [event for event in state["branches"]["branch/known-category"]["pending"]] == []


def test_harness_tree_preserves_metadata_route_on_solve_error(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "categorized_error", "explode"),
        _harness_task(tmp_path, "categorized_ok", "alpha model"),
    ]
    benchmark.train_tasks[0].metadata["category"] = "Known Category"
    benchmark.train_tasks[1].metadata["category"] = "Known Category"
    agent = FakeAgent(workspace)
    engine = FakeEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=3, train_limit=2),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
    )

    progress_events: list[dict[str, Any]] = []
    runner.run_training(max_epochs=1, progress_callback=progress_events.append)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())
    task_done = next(event for event in progress_events if event["event"] == "task_done")
    observation_file = workspace / "evolution" / "observations" / "batch_0001_branch_known-category.jsonl"
    records = [
        json.loads(line)
        for line in observation_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert sorted(state["branches"]) == ["branch/known-category"]
    assert records[0]["feedback_detail"] == "RuntimeError: boom"
    assert records[0]["harness_tree"]["branch_name"] == "branch/known-category"
    assert task_done["branch_name"] == "branch/known-category"
    assert task_done["route_confidence"] == 1.0


def test_harness_tree_disable_main_evolve_isolates_branch_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    (workspace / "or_interact_settings.json").write_text(
        json.dumps({"enable_heuristic_tool": True}),
        encoding="utf-8",
    )
    original_prompt = (workspace / "prompts" / "system.md").read_text(encoding="utf-8")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_iso_1", "alpha model"),
        _harness_task(tmp_path, "alpha_iso_2", "alpha model again"),
    ]
    agent = FakeAgent(workspace, parallelism=2)
    engine = LeakyEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=1, train_limit=2),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
        disable_main_evolve=True,
    )

    result = runner.run_training(max_epochs=1)
    state = json.loads((workspace / "evolution" / "harness_tree" / "state.json").read_text())

    assert result.details["disable_main_evolve"] is True
    assert state["main_evolutions"] == []
    assert state["main_pending"] == []
    assert engine.evolved_scopes == ["alpha"]
    assert engine.trajectory_profiles == ["branch"]
    assert engine.prompt_log_dirs == [workspace / "evolution" / "evolver_prompts"]
    assert engine.prompt_log_scopes == ["branch_alpha"]
    assert not (workspace / "skills" / "domain-alpha").exists()
    assert not (workspace / "skills" / "leaked-main").exists()
    assert (workspace / "prompts" / "system.md").read_text(encoding="utf-8") == original_prompt
    assert (
        workspace
        / "evolution"
        / "harness_tree"
        / "overlays"
        / "alpha"
        / "files"
        / "skills"
        / "domain-alpha"
        / "SKILL.md"
    ).is_file()
    isolated_workspace = workspace.parent / "harness_tree_branch_workspaces" / "alpha"
    assert isolated_workspace.is_dir()
    assert workspace.resolve() not in isolated_workspace.resolve().parents
    assert json.loads(
        (isolated_workspace / "or_interact_settings.json").read_text(encoding="utf-8")
    ) == {"enable_heuristic_tool": True}


def test_harness_tree_filters_invalid_skill_artifacts_from_overlay(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    benchmark = FakeBenchmark(tmp_path)
    benchmark.train_tasks = [
        _harness_task(tmp_path, "alpha_noise_1", "alpha model"),
        _harness_task(tmp_path, "alpha_noise_2", "alpha model again"),
    ]
    agent = FakeAgent(workspace)
    engine = NoisySkillEngine(workspace)
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=EvolveConfig(batch_size=10, train_limit=2),
        type_buffer_size=2,
        router_confidence_threshold=0.5,
        disable_main_evolve=True,
    )

    runner.run_training(max_epochs=1)
    overlay_skills = workspace / "evolution" / "harness_tree" / "overlays" / "alpha" / "files" / "skills"

    assert (overlay_skills / "domain-alpha" / "SKILL.md").is_file()
    assert not (overlay_skills / "test.txt").exists()
    assert not (overlay_skills / "domain-alpha" / "SKILL.md.bak").exists()
    assert not (overlay_skills / "domain-alpha" / "notes.txt").exists()


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


def test_agent_parallelism_comes_from_react_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "react.yaml").write_text(
        "\n".join(
            [
                "api_key: test-key",
                "base_url: http://localhost/v1",
                "model: fake-model",
                "temperature: 0.0",
                "max_turns: 3",
                "parallelism: 4",
                f"benchmark_dir: {BENCHMARK_DIR}",
                "results_dir: results/react",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    agent = ORReactAgent(workspace)

    assert agent.config.parallelism == 4


def test_agent_temperature_can_be_omitted_for_compatible_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "react.yaml").write_text(
        "\n".join(
            [
                "api_key: test-key",
                "base_url: http://localhost/v1",
                "model: fake-model",
                "temperature: null",
                f"benchmark_dir: {BENCHMARK_DIR}",
                "results_dir: results/react",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    agent = ORReactAgent(workspace)

    assert agent.config.temperature is None


def test_agent_parallelism_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "react.yaml").write_text(
        "\n".join(
            [
                "api_key: test-key",
                "base_url: http://localhost/v1",
                "model: fake-model",
                "parallelism: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OR_REACT_PARALLELISM", "2")

    agent = ORReactAgent(workspace)

    assert agent.config.parallelism == 2


def test_agent_results_dir_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    results_dir = tmp_path / "evaluation" / "runs"
    monkeypatch.setenv("OR_REACT_RESULTS_DIR", str(results_dir))

    agent = ORReactAgent(workspace)

    assert agent.config.results_dir == results_dir.resolve()


@pytest.mark.parametrize("forbidden_path", ["grounded/clarification.md", "oracle/objective.json"])
def test_forbidden_evolved_tool_is_rejected(tmp_path: Path, forbidden_path: str) -> None:
    workspace = _workspace(tmp_path / "workspace")
    _write_tool(
        workspace,
        "bad_tool",
        f"from typing import Any\n\ndef bad_tool() -> dict[str, Any]:\n    return {{'path': {forbidden_path!r}}}\n",
    )
    _write_registry(workspace, [{"name": "bad_tool", "file": "bad_tool.py", "function": "bad_tool"}])

    with pytest.raises(ValueError, match="forbidden string"):
        ORReactAgent(workspace)


class FakeRegistry:
    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("harness-tree must not call type_router outside agent solve")


class FakeAgent:
    def __init__(self, workspace: Path, parallelism: int = 1) -> None:
        self.workspace = AgentWorkspace(workspace)
        self.registry = FakeRegistry()
        self.config = types.SimpleNamespace(task_timeout_seconds=0, parallelism=parallelism)
        self.phases: list[str] = []

    def reload_from_fs(self) -> None:
        self.registry = FakeRegistry()

    def export_to_fs(self) -> None:
        return None

    def start_task_run(self, task: Task) -> tuple[Path, TraceWriter]:
        runtime_dir = self.workspace.root / "runs" / task.id
        return runtime_dir, TraceWriter(runtime_dir)

    def run_phase(
        self,
        task: Task,
        *,
        runtime_dir: Path,
        trace: TraceWriter,
        phase: str = "solve",
        initial_messages: list[dict[str, Any]] | None = None,
        user_message: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        stop_after_tools: set[str] | None = None,
        enable_skill_tools: bool = False,
    ) -> AgentResult:
        self.phases.append(phase)
        messages = list(initial_messages or [{"role": "system", "content": "fake"}])
        label = "beta" if "beta" in task.input else "alpha"
        if phase.endswith(":route"):
            if "route_sleep" in task.input:
                time.sleep(0.15)
            category = task.metadata.get("category")
            if category:
                branch_name = sanitize_branch_name(category)
                confidence = 1.0
                rationale = f"task metadata category: {category}"
            elif "reuse_existing" in task.input and user_message and "branch/alpha" in user_message:
                branch_name = "branch/alpha"
                confidence = 0.9
                rationale = "reused alpha from existing branches"
            elif "reuse_existing" in task.input and user_message and "branch/beta" in user_message:
                branch_name = "branch/beta"
                confidence = 0.9
                rationale = "reused beta from existing branches"
            elif "reuse_existing" in task.input:
                branch_name = "branch/reuse-new"
                confidence = 0.9
                rationale = "no existing branch visible"
            else:
                branch_name = f"branch/{label}"
                confidence = 0.95
                rationale = f"{label} evidence"
            trace.event(
                "assistant_message",
                {
                    "phase": phase,
                    "content": "routing",
                    "tool_calls": [],
                    "enable_skill_tools": enable_skill_tools,
                },
            )
            trace.event(
                "tool_output",
                {
                    "phase": phase,
                    "name": "type_router",
                    "output": {
                        "branch_name": branch_name,
                        "confidence": confidence,
                        "rationale": rationale,
                    },
                },
            )
            messages.extend(
                [
                    {"role": "assistant", "content": "routing"},
                    {
                        "role": "tool",
                        "tool_call_id": "route",
                        "content": json.dumps({"branch_name": branch_name}),
                    },
                ]
            )
            return AgentResult(status="stopped_after_type_router", turns=1, messages=messages)

        if "explode" in task.input:
            raise RuntimeError("boom")
        assert initial_messages, "solve phase should inherit route messages"
        time.sleep(0.05)
        trace.event(
            "assistant_message",
            {
                "phase": phase,
                "content": "solving",
                "tool_calls": [],
                "enable_skill_tools": enable_skill_tools,
            },
        )
        return AgentResult(status="success", turns=1, objective_value=1, messages=messages)

    def finish_task_run(
        self,
        task: Task,
        *,
        runtime_dir: Path,
        result: AgentResult,
        elapsed: float,
    ) -> Trajectory:
        steps = []
        trace_path = runtime_dir / "trace.jsonl"
        if trace_path.is_file():
            steps = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        steps.append({"runtime_dir": str(runtime_dir), "status": result.status})
        return Trajectory(
            task_id=task.id,
            output=f"solved {task.id}",
            steps=steps,
            conversation=steps,
        )


class FakeBenchmark:
    def __init__(self, tmp_path: Path) -> None:
        self.train_tasks = [
            _harness_task(tmp_path, "alpha_1", "alpha model"),
            _harness_task(tmp_path, "beta_1", "beta model"),
            _harness_task(tmp_path, "alpha_2", "alpha model again"),
        ]
        self.test_tasks = [_harness_task(tmp_path, "alpha_test", "alpha eval")]

    def get_tasks(self, split: str = "train", limit: int | None = 10) -> list[Task]:
        tasks = self.train_tasks if split == "train" else self.test_tasks
        return tasks[:limit] if limit is not None else list(tasks)

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        return Feedback(
            success=True,
            score=1.0,
            detail="ok",
            raw={"evaluation": {"task_id": task.id, "correct": True, "pid": os.getpid()}},
        )


class FakeEngine:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.evolved_scopes: list[str] = []
        self.records_per_evolve: list[int] = []
        self.trajectory_profiles: list[str] = []
        self.prompt_log_dirs: list[Path | None] = []
        self.prompt_log_scopes: list[str | None] = []

    def evolve(
        self,
        workspace: AgentWorkspace,
        observation_logs: list[dict[str, Any]],
        evo_number: int = 0,
        trajectory_profile: str = "main",
        prompt_log_dir: Path | None = None,
        prompt_log_scope: str | None = None,
    ) -> dict[str, Any]:
        scope = "main" if workspace.root == self.workspace.resolve() else workspace.root.name
        self.evolved_scopes.append(scope)
        self.records_per_evolve.append(len(observation_logs))
        self.trajectory_profiles.append(trajectory_profile)
        self.prompt_log_dirs.append(prompt_log_dir)
        self.prompt_log_scopes.append(prompt_log_scope)
        if scope == "main":
            workspace.memory_dir.mkdir(parents=True, exist_ok=True)
            (workspace.memory_dir / "main.jsonl").write_text('{"content": "main"}\n', encoding="utf-8")
        else:
            skill_dir = workspace.skills_dir / f"domain-{scope}"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: domain-{scope}\ndescription: Domain {scope}\n---\n",
                encoding="utf-8",
            )
        return {"evo_number": evo_number, "tasks_analyzed": len(observation_logs)}


class LeakyEngine(FakeEngine):
    def evolve(
        self,
        workspace: AgentWorkspace,
        observation_logs: list[dict[str, Any]],
        evo_number: int = 0,
        trajectory_profile: str = "main",
        prompt_log_dir: Path | None = None,
        prompt_log_scope: str | None = None,
    ) -> dict[str, Any]:
        result = super().evolve(
            workspace,
            observation_logs,
            evo_number=evo_number,
            trajectory_profile=trajectory_profile,
            prompt_log_dir=prompt_log_dir,
            prompt_log_scope=prompt_log_scope,
        )
        leaked_skill = self.workspace / "skills" / "leaked-main"
        leaked_skill.mkdir(parents=True, exist_ok=True)
        (leaked_skill / "SKILL.md").write_text(
            "---\nname: leaked-main\ndescription: leaked\n---\n",
            encoding="utf-8",
        )
        (self.workspace / "prompts" / "system.md").write_text("leaked main prompt", encoding="utf-8")
        return result


class NoisySkillEngine(FakeEngine):
    def evolve(
        self,
        workspace: AgentWorkspace,
        observation_logs: list[dict[str, Any]],
        evo_number: int = 0,
        trajectory_profile: str = "main",
        prompt_log_dir: Path | None = None,
        prompt_log_scope: str | None = None,
    ) -> dict[str, Any]:
        result = super().evolve(
            workspace,
            observation_logs,
            evo_number=evo_number,
            trajectory_profile=trajectory_profile,
            prompt_log_dir=prompt_log_dir,
            prompt_log_scope=prompt_log_scope,
        )
        if workspace.root == self.workspace.resolve():
            return result
        scope = workspace.root.name
        (workspace.skills_dir / "test.txt").write_text("scratch", encoding="utf-8")
        skill_dir = workspace.skills_dir / f"domain-{scope}"
        (skill_dir / "SKILL.md.bak").write_text("backup", encoding="utf-8")
        (skill_dir / "notes.txt").write_text("scratch", encoding="utf-8")
        return result


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


def _write_skill(workspace: Path, name: str, body: str, frontmatter: str = "") -> None:
    skill_dir = workspace / "skills" / name
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill{frontmatter}\n---\n\n{body}\n",
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


def _copy_harness_tool(workspace: Path, name: str) -> None:
    source = REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react" / "tools" / f"{name}.py"
    shutil.copyfile(source, workspace / "tools" / f"{name}.py")


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, object]],
    *,
    captured_requests: list[dict[str, Any]] | None = None,
) -> None:
    remaining = list(payloads)

    class FakeCompletions:
        def create(self, **kwargs):
            if captured_requests is not None:
                captured_requests.append(kwargs)
            if not remaining:
                raise AssertionError("unexpected OpenAI call")
            content = json.dumps(remaining.pop(0))
            message = types.SimpleNamespace(content=content)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


def _visible_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "data").mkdir()
    (task_dir / "docs" / "business_requirement.md").write_text("Task", encoding="utf-8")
    return task_dir


def _harness_task(tmp_path: Path, task_id: str, overview: str) -> Task:
    task_dir = tmp_path / "tasks" / task_id
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "docs" / "overview.md").write_text(overview, encoding="utf-8")
    return Task(
        id=task_id,
        input=overview,
        metadata={"task_dir": str(task_dir), "visible_files": ["docs/overview.md"]},
    )


def _minimal_visible_task(task_dir: Path) -> None:
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "data").mkdir()
    (task_dir / "docs" / "business_requirement.md").write_text(
        "Task",
        encoding="utf-8",
    )
