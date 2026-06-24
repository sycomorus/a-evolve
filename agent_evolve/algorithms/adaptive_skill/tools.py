"""Bash tool spec and LLM provider factory for A-Evolve."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ...config import EvolveConfig
from ...llm.base import LLMProvider

BASH_TOOL_SPEC = {
    "name": "workspace_bash",
    "description": (
        "Execute a bash command inside the agent workspace directory. "
        "Use this only to read/write workspace files such as skills, prompts, memory, "
        "tools, and manifest.yaml; output is truncated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute in the workspace directory.",
            },
        },
        "required": ["command"],
    },
}

WORKSPACE_BASH_OUTPUT_CHAR_LIMIT = 20_000
_OUTSIDE_PATH_PATTERNS = (
    re.compile(r"(^|[\s\"'=])\.\.(?=/|$)"),
    re.compile(r"(^|[\s\"'=])~(?=/|$)"),
    re.compile(r"(^|[\s\"'=])/(?!dev/null(?:\s|$))"),
)


def make_workspace_bash(workspace_root: str | Path):
    """Create a bash callable scoped to the workspace directory."""
    root = Path(workspace_root).resolve()

    def bash(command: str) -> str:
        violation = _outside_workspace_violation(command)
        if violation:
            return f"ERROR: workspace_bash may only access files under {root}: {violation}"
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(root),
            )
            output = (result.stdout + result.stderr).strip()
            output = output if output else "(no output)"
            return _truncate_output(output)
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out."
        except Exception as e:
            return f"ERROR: {e}"

    return bash


def _outside_workspace_violation(command: str) -> str:
    for pattern in _OUTSIDE_PATH_PATTERNS:
        match = pattern.search(command)
        if match:
            return f"outside-workspace path reference {match.group(0).strip()!r}"
    return ""


def _truncate_output(output: str) -> str:
    if len(output) <= WORKSPACE_BASH_OUTPUT_CHAR_LIMIT:
        return output
    omitted = len(output) - WORKSPACE_BASH_OUTPUT_CHAR_LIMIT
    return (
        output[:WORKSPACE_BASH_OUTPUT_CHAR_LIMIT]
        + f"\n...[truncated {omitted} characters by workspace_bash output limit]"
    )


def create_default_llm(config: EvolveConfig) -> LLMProvider:
    """Create the default LLM provider based on the evolver_model config string."""
    model = config.evolver_model

    if (
        model.startswith("openai:")
        or config.extra.get("evolver_base_url")
        or os.environ.get("EVOLVER_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    ):
        from ..unified.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model=model.removeprefix("openai:"),
            api_key=config.extra.get("evolver_api_key"),
            base_url=config.extra.get("evolver_base_url"),
            temperature=config.extra.get("evolver_temperature"),
            omit_temperature=(
                "evolver_temperature" in config.extra
                and config.extra.get("evolver_temperature") is None
            ),
        )

    if "." in model and ("anthropic" in model or "amazon" in model or "meta" in model):
        from ...llm.bedrock import BedrockProvider

        region = config.extra.get("region", "us-west-2")
        return BedrockProvider(model_id=model, region=region)

    if model.startswith("claude"):
        from ...llm.anthropic import AnthropicProvider

        return AnthropicProvider(model=model)

    if model.startswith(("gpt-", "o1", "o3")):
        from ...llm.openai import OpenAIProvider

        return OpenAIProvider(model=model)

    from ...llm.bedrock import BedrockProvider

    return BedrockProvider(model_id=model)
