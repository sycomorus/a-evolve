from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from agent_evolve.algorithms.adaptive_skill.engine import AdaptiveSkillEngine
from agent_evolve.algorithms.adaptive_skill.prompts import build_evolution_prompt
from agent_evolve.algorithms.adaptive_skill.tools import (
    WORKSPACE_BASH_OUTPUT_CHAR_LIMIT,
    create_default_llm,
    make_workspace_bash,
)
from agent_evolve.algorithms.unified.openai_compat import OpenAICompatProvider
from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace
from agent_evolve.engine.versioning import VersionControl
from agent_evolve.llm.base import LLMResponse


class _EmptyHistory:
    latest_cycle = 0

    def get_observations(self, last_n_cycles: int = 2) -> list[dict]:
        return []


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(responses)


def _openai_compat_provider(responses) -> OpenAICompatProvider:
    provider = object.__new__(OpenAICompatProvider)
    provider.model = "fake-model"
    provider.client = _FakeClient(responses)
    provider.temperature = None
    provider.omit_temperature = True
    return provider


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


def test_trajectory_only_judge_uses_injected_evolver_llm(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path)

    class FakeJudge:
        calls = 0

        def complete(self, **_kwargs):
            self.calls += 1
            return LLMResponse(
                content=json.dumps({
                    "score": 2,
                    "category": "optimization",
                    "outcome": "failed before finalization",
                    "failure_reason": "no submitted answer",
                }),
                usage={},
            )

    judge = FakeJudge()
    prompt = build_evolution_prompt(
        workspace=workspace,
        logs=[{"task_id": "task_x", "conversation": []}],
        drafts=[],
        evo_number=1,
        trajectory_only=True,
        judge_llm=judge,
    )

    assert judge.calls == 1
    assert "judge_verdict" in prompt
    assert "no submitted answer" in prompt


def test_openai_compatible_evolver_can_omit_temperature() -> None:
    llm = create_default_llm(
        EvolveConfig(
            evolver_model="openai:gpt-5.5",
            extra={
                "evolver_base_url": "http://localhost/v1",
                "evolver_api_key": "test-key",
                "evolver_temperature": None,
            },
        )
    )

    assert getattr(llm, "omit_temperature") is True


def test_openai_compatible_evolver_accepts_string_responses() -> None:
    llm = _openai_compat_provider(["plain final response"])

    response = llm.converse_loop(
        system_prompt="system",
        user_message="user",
        tools=[],
        tool_executor={},
    )

    assert response.content == "plain final response"


def test_openai_compatible_evolver_accepts_json_string_tool_calls() -> None:
    first_response = json.dumps({
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "workspace_bash",
                                "arguments": json.dumps({"command": "pwd"}),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    })
    llm = _openai_compat_provider([first_response, "done"])

    response = llm.converse_loop(
        system_prompt="system",
        user_message="user",
        tools=[{"name": "workspace_bash", "input_schema": {"type": "object"}}],
        tool_executor={"workspace_bash": lambda command: f"ran {command}"},
    )

    requests = llm.client.chat.completions.requests
    assert response.content == "done"
    assert response.usage == {"input_tokens": 3, "output_tokens": 4}
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ran pwd",
    }


def test_workspace_bash_rejects_outside_workspace_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    bash = make_workspace_bash(workspace)

    assert bash("printf ok") == "ok"
    blocked = bash("cat ../outside.txt")

    assert "ERROR: workspace_bash may only access files under" in blocked
    assert "secret" not in blocked


def test_workspace_bash_truncates_large_output(tmp_path: Path) -> None:
    big_file = tmp_path / "big.txt"
    big_file.write_text("x" * (WORKSPACE_BASH_OUTPUT_CHAR_LIMIT + 100), encoding="utf-8")
    bash = make_workspace_bash(tmp_path)

    output = bash("cat big.txt")

    assert len(output) < WORKSPACE_BASH_OUTPUT_CHAR_LIMIT + 200
    assert "truncated 100 characters" in output


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
