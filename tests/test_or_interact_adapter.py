from __future__ import annotations

import csv
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from agent_evolve.agents.or_interact.react_agent import ANSWER_CHECKER_ENV, ORReactAgent
from agent_evolve.benchmarks.or_interact import (
    ORInteractBenchmark,
    evaluation_limit_for_split,
    train_size_from_limit,
)
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


def test_agent_prompt_uses_harness_catalog_and_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "0")
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
    _write_tool(
        workspace,
        "type_router",
        "from typing import Any\n\ndef type_router(evidence_text: str) -> dict[str, Any]:\n    return {'task_types': ['general']}\n",
    )
    _write_tool(
        workspace,
        "answer_checker",
        "from typing import Any\n\ndef answer_checker(compressed_trace: str, final_code: str, objective_value: str, task_types: str, harness_notes: str) -> dict[str, Any]:\n    return {'passed': True}\n",
    )
    _write_registry(
        workspace,
        [
            {"name": "diagnose", "file": "diagnose.py", "function": "diagnose", "description": "diagnostic helper"},
            {"name": "type_router", "file": "type_router.py", "function": "type_router"},
            {"name": "answer_checker", "file": "answer_checker.py", "function": "answer_checker"},
        ],
    )

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
    assert "Call type_router" in prompt
    assert "answer_checker" not in prompt
    assert "Prefer explicit variable bounds." not in prompt
    assert "Check objective direction before finalizing." not in prompt
    assert "modeling; types=general" in prompt
    assert "memory:1 category=memories types=general" in prompt
    assert "diagnose" in captured["tools"]
    assert "type_router" in captured["tools"]
    assert "answer_checker" not in captured["tools"]


def test_seed_workspace_skips_checker_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "0")
    workspace = REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react"
    agent = ORReactAgent(workspace)

    assert agent.registry.get("type_router").kind == "evolved"
    assert "answer_checker" not in agent.registry.list_tools()


def test_seed_workspace_loads_checker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "1")
    workspace = REPO_ROOT / "a-evolve" / "seed_workspaces" / "or_interact_react"
    agent = ORReactAgent(workspace)

    assert agent.registry.get("type_router").kind == "evolved"
    assert agent.registry.get("answer_checker").kind == "evolved"


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

        def run(self):
            return AgentResult(status="success", turns=1, objective_value="smoke")

    monkeypatch.setattr(react_module, "ReActAgent", FakeReActAgent)
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "0")
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


def test_check_flag_enables_checker_in_solve_prompt_and_schema(
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

        def run(self):
            return AgentResult(status="success", turns=1, objective_value="smoke")

    monkeypatch.setattr(react_module, "ReActAgent", FakeReActAgent)
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "1")
    monkeypatch.setenv("OR_REACT_RESULTS_DIR", str(tmp_path / "runs"))

    trajectory = ORReactAgent(workspace).solve(task)

    assert "answer_checker" in captured["registry_tools"]
    assert "answer_checker" in captured["schema_names"]
    assert "Call answer_checker" in str(captured["system_prompt"])
    assert trajectory.steps[-1]["tools"] == captured["registry_tools"]


def test_or_interact_cli_check_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.or_interact_examples import evaluate_or_interact, evolve_or_interact

    monkeypatch.setattr(sys, "argv", ["evaluate_or_interact.py"])
    assert evaluate_or_interact.parse_args().check is False
    monkeypatch.setattr(sys, "argv", ["evaluate_or_interact.py", "--check"])
    assert evaluate_or_interact.parse_args().check is True

    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py"])
    assert evolve_or_interact.parse_args().check is False
    monkeypatch.setattr(sys, "argv", ["evolve_or_interact.py", "--check"])
    assert evolve_or_interact.parse_args().check is True


def test_evaluate_and_evolve_use_matching_or_interact_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.or_interact_examples import evaluate_or_interact, evolve_or_interact

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_or_interact.py",
            "--dataset",
            "RCO-mini",
            "--split",
            "test",
            "--limit-train",
            "5",
            "--limit-test",
            "3",
        ],
    )
    evaluate_args = evaluate_or_interact.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_or_interact.py",
            "--dataset",
            "RCO-mini",
            "--limit-train",
            "5",
            "--limit-test",
            "3",
        ],
    )
    evolve_args = evolve_or_interact.parse_args()

    evaluate_benchmark = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        dataset=evaluate_args.dataset,
        train_size=train_size_from_limit(evaluate_args.limit_train),
    )
    evolve_benchmark = ORInteractBenchmark(
        benchmark_dir=BENCHMARK_DIR,
        dataset=evolve_args.dataset,
        train_size=train_size_from_limit(evolve_args.limit_train),
    )

    assert [task.id for task in evaluate_benchmark.get_tasks("train", limit=None)] == [
        task.id for task in evolve_benchmark.get_tasks("train", limit=None)
    ]
    assert [task.id for task in evaluate_benchmark.get_tasks("test", limit=None)] == [
        task.id for task in evolve_benchmark.get_tasks("test", limit=None)
    ]
    assert (
        evaluation_limit_for_split(
            evaluate_args.split,
            limit_train=evaluate_args.limit_train,
            limit_test=evaluate_args.limit_test,
        )
        == evaluation_limit_for_split(
            "test",
            limit_train=evolve_args.limit_train,
            limit_test=evolve_args.limit_test,
        )
        == 3
    )


def test_type_router_returns_selected_harness_content_and_old_skill_defaults(
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
                "task_types": ["general"],
                "selected_skill_paths": ["skills/legacy-skill/SKILL.md"],
                "selected_memory_paths": ["memory/memories.jsonl:1"],
                "rationale": "legacy general route",
            }
        ],
    )

    agent = ORReactAgent(workspace)
    result = agent.registry.call("type_router", evidence_text="Visible production planning evidence.")

    assert result["task_types"] == ["general"]
    assert result["selected_skills"][0]["path"] == "skills/legacy-skill/SKILL.md"
    assert "Legacy skill body." in result["selected_skills"][0]["content"]
    assert result["selected_memories"][0]["content"] == "Legacy memory."


def test_answer_checker_returns_failed_checklist_from_mocked_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "1")
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(
        workspace,
        "objective-check",
        "Check objective unit.",
        frontmatter="\ntypes: [general]\nchecklist:\n  - id: objective_unit\n    prompt: Verify unit.\n",
    )
    _copy_harness_tool(workspace, "answer_checker")
    _write_registry(workspace, [{"name": "answer_checker", "file": "answer_checker.py", "function": "answer_checker"}])
    _install_fake_openai(
        monkeypatch,
        [
            {
                "passed": False,
                "failed_items": ["objective_unit"],
                "check_results": [
                    {
                        "id": "objective_unit",
                        "passed": False,
                        "reason": "Unit mismatch.",
                        "evidence": ["objective_value=10; trace says quantity"],
                    },
                    {
                        "id": "visible_constraints",
                        "passed": True,
                        "reason": "Constraints are supported.",
                        "evidence": ["final_code contains stated model constraints"],
                    },
                    {
                        "id": "warning_resolution",
                        "passed": True,
                        "reason": "No warnings remain.",
                        "evidence": ["harness_notes=notes"],
                    },
                ],
                "required_fix": "Submit profit, not quantity.",
            }
        ],
    )

    agent = ORReactAgent(workspace)
    result = agent.registry.call(
        "answer_checker",
        compressed_trace="trace",
        final_code="print('model')",
        objective_value="10",
        task_types='["general"]',
        harness_notes="notes",
    )

    assert result["passed"] is False
    assert result["failed_items"] == ["objective_unit"]
    assert result["check_results"][0]["evidence"] == ["objective_value=10; trace says quantity"]
    assert result["required_fix"] == "Submit profit, not quantity."


def test_answer_checker_returns_pass_from_mocked_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "1")
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(
        workspace,
        "objective-check",
        "Check objective unit.",
        frontmatter="\ntypes: [general]\nchecklist:\n  - id: objective_unit\n    prompt: Verify unit.\n",
    )
    _copy_harness_tool(workspace, "answer_checker")
    _write_registry(workspace, [{"name": "answer_checker", "file": "answer_checker.py", "function": "answer_checker"}])
    _install_fake_openai(
        monkeypatch,
        [
            {
                "passed": True,
                "failed_items": [],
                "check_results": [
                    {
                        "id": "objective_unit",
                        "passed": True,
                        "reason": "Matches.",
                        "evidence": ["objective_value=10; trace says submitted profit"],
                    },
                    {
                        "id": "visible_constraints",
                        "passed": True,
                        "reason": "Constraints are supported.",
                        "evidence": ["final_code contains stated model constraints"],
                    },
                    {
                        "id": "warning_resolution",
                        "passed": True,
                        "reason": "Warnings resolved.",
                        "evidence": ["compressed_trace states no warnings"],
                    },
                ],
                "required_fix": "",
            }
        ],
    )

    agent = ORReactAgent(workspace)
    result = agent.registry.call(
        "answer_checker",
        compressed_trace="trace",
        final_code="print('model')",
        objective_value="10",
        task_types="general",
        harness_notes="notes",
    )

    assert result["passed"] is True
    assert result["failed_items"] == []
    assert result["check_results"][0]["evidence"] == ["objective_value=10; trace says submitted profit"]


def test_answer_checker_requires_per_item_explanation_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "1")
    workspace = _workspace(tmp_path / "workspace")
    _write_skill(
        workspace,
        "objective-check",
        "Check objective unit.",
        frontmatter="\ntypes: [general]\nchecklist:\n  - id: objective_unit\n    prompt: Verify unit.\n",
    )
    _copy_harness_tool(workspace, "answer_checker")
    _write_registry(workspace, [{"name": "answer_checker", "file": "answer_checker.py", "function": "answer_checker"}])
    _install_fake_openai(
        monkeypatch,
        [
            {
                "passed": True,
                "failed_items": [],
                "check_results": [
                    {"id": "objective_unit", "passed": True, "reason": "Matches."},
                    {
                        "id": "visible_constraints",
                        "passed": True,
                        "evidence": ["final_code contains stated model constraints"],
                    },
                    {
                        "id": "warning_resolution",
                        "passed": True,
                        "reason": "Warnings resolved.",
                        "evidence": ["compressed_trace states no warnings"],
                    },
                ],
                "required_fix": "",
            }
        ],
    )

    agent = ORReactAgent(workspace)
    result = agent.registry.call(
        "answer_checker",
        compressed_trace="trace",
        final_code="print('model')",
        objective_value="10",
        task_types="general",
        harness_notes="notes",
    )

    assert result["passed"] is False
    assert "objective_unit" in result["failed_items"]
    assert "visible_constraints" in result["failed_items"]
    objective_check = result["check_results"][0]
    assert objective_check["passed"] is False
    assert objective_check["reason"].endswith("Missing explicit evidence.")
    visible_check = result["check_results"][1]
    assert visible_check["passed"] is False
    assert visible_check["reason"] == "Missing per-item explanation."


def test_answer_checker_warns_and_blocks_after_three_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANSWER_CHECKER_ENV, "1")
    workspace = _workspace(tmp_path / "workspace")
    _write_tool(
        workspace,
        "answer_checker",
        "\n".join(
            [
                "from typing import Any",
                "",
                "CALLS = 0",
                "",
                "def answer_checker(",
                "    compressed_trace: str,",
                "    final_code: str,",
                "    objective_value: str,",
                "    task_types: str,",
                "    harness_notes: str,",
                ") -> dict[str, Any]:",
                "    global CALLS",
                "    CALLS += 1",
                "    return {",
                "        'passed': False,",
                "        'failed_items': ['objective_unit'],",
                "        'check_results': [],",
                "        'required_fix': f'revise attempt {CALLS}',",
                "        'calls': CALLS,",
                "    }",
            ]
        )
        + "\n",
    )
    _write_registry(workspace, [{"name": "answer_checker", "file": "answer_checker.py", "function": "answer_checker"}])

    agent = ORReactAgent(workspace)
    calls = [
        agent.registry.call(
            "answer_checker",
            compressed_trace="trace",
            final_code="print('model')",
            objective_value="10",
            task_types="general",
            harness_notes="notes",
        )
        for _ in range(4)
    ]

    assert calls[0]["calls"] == 1
    assert calls[1]["calls"] == 2
    assert calls[2]["calls"] == 3
    assert calls[2]["answer_checker_budget_exhausted"] is True
    assert calls[2]["answer_checker_call_allowed"] is False
    assert "call finalize" in calls[2]["warning"]
    assert "calls" not in calls[3]
    assert calls[3]["failed_items"] == ["answer_checker_budget_exhausted"]
    assert calls[3]["answer_checker_budget_exhausted"] is True
    assert calls[3]["answer_checker_call_allowed"] is False
    assert "Call finalize now" in calls[3]["warning"]


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


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, object]]) -> None:
    remaining = list(payloads)

    class FakeCompletions:
        def create(self, **kwargs):
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


def _minimal_visible_task(task_dir: Path) -> None:
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "data").mkdir()
    (task_dir / "docs" / "business_requirement.md").write_text(
        "Task",
        encoding="utf-8",
    )
