from __future__ import annotations

import json
from pathlib import Path

from agent_evolve.algorithms.adaptive_skill.prompts import build_evolution_prompt
from agent_evolve.contract.workspace import AgentWorkspace


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
    assert "compressed_trajectory" in prompt
    assert "signals" in prompt
    assert "read_csv" in prompt
    assert "run_solver" in prompt
    assert "Traceback: bad model" in prompt
    assert '"submitted": true' in prompt
    assert "[submitted] 123" in prompt
