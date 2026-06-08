from __future__ import annotations

import json
from pathlib import Path

from agent_evolve.algorithms.adaptive_skill.engine import AdaptiveSkillEngine
from agent_evolve.algorithms.adaptive_skill.prompts import build_evolution_prompt
from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.engine.versioning import VersionControl


class _EmptyHistory:
    latest_cycle = 0

    def get_observations(self, last_n_cycles: int = 2) -> list[dict]:
        return []


def test_standard_prompt_includes_real_feedback_and_compressed_trajectory(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path)
    conversation = [
        {
            "type": "tool_call",
            "name": "read_csv",
            "arguments": {"relative_file": "data/table.csv"},
        },
        {
            "type": "tool_output",
            "name": "read_csv",
            "output": {"rows": 10},
        },
        {
            "type": "tool_call",
            "name": "run_solver",
            "arguments": {"code": "raise ValueError('bad model')"},
        },
        {
            "type": "tool_output",
            "name": "run_solver",
            "output": {"error": {"message": "Traceback: bad model"}},
        },
        {
            "type": "tool_call",
            "name": "finalize",
            "arguments": {"objective_value": 123},
        },
    ]
    logs = [
        {
            "task_id": "task_001",
            "success": False,
            "score": 0.0,
            "feedback_detail": "Failure reason: outside relative tolerance",
            "conversation": conversation,
        }
    ]

    prompt = build_evolution_prompt(
        workspace=workspace,
        logs=logs,
        drafts=[],
        evo_number=1,
        trajectory_only=False,
    )

    assert "real benchmark feedback" in prompt
    assert "Failure reason: outside relative tolerance" in prompt
    assert "private oracle information shown only to you" in prompt
    assert "Do NOT write evolved prompts, skills, memory, tools, or summaries" in prompt
    assert "compressed_trajectory" in prompt
    assert "signals" in prompt
    assert "read_csv" in prompt
    assert "run_solver" in prompt
    assert "Traceback: bad model" in prompt
    assert '"submitted": true' in prompt
    assert "[submitted] 123" in prompt


def test_standard_prompt_keeps_reference_solution_feedback(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path)
    reference_tail = "REFERENCE_SOLVER_MODEL_FORMULATION"
    logs = [
        {
            "task_id": "task_001",
            "success": False,
            "score": 0.0,
            "feedback_detail": (
                "Failure reason: outside relative tolerance\n"
                "Reference solution code:\n"
                "```python\n"
                + ("x = 1\n" * 300)
                + reference_tail
                + "\n```"
            ),
            "conversation": [],
        }
    ]

    prompt = build_evolution_prompt(
        workspace=workspace,
        logs=logs,
        drafts=[],
        evo_number=1,
        trajectory_only=False,
    )

    assert reference_tail in prompt


def test_adaptive_skill_step_marks_prompt_diff_as_mutation(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path)
    workspace.write_prompt("initial prompt")
    (tmp_path / "skills").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "manifest.yaml").write_text("name: test\n", encoding="utf-8")
    VersionControl(workspace.root).init()

    engine = AdaptiveSkillEngine(EvolveConfig())

    def mutate_prompt(_prompt: str, root: Path) -> dict:
        (root / "prompts" / "system.md").write_text("changed prompt", encoding="utf-8")
        return {"usage": {}}

    engine._run_llm = mutate_prompt  # type: ignore[method-assign]

    result = engine.step(
        workspace=workspace,
        observations=[],
        history=_EmptyHistory(),
        trial=None,
    )

    assert result.mutated is True
