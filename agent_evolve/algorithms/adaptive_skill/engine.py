"""AdaptiveSkillEngine -- the core A-Evolve algorithm.

Uses an LLM with bash tool access to analyze observation logs and mutate
the agent workspace (prompts, skills, memory). This is the first and
default EvolutionEngine implementation.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from ...config import EvolveConfig
from ...contract.workspace import AgentWorkspace
from ...engine.base import EvolutionEngine
from ...engine.versioning import VersionControl
from ...llm.base import LLMMessage, LLMProvider
from ...types import Observation, StepResult
from .prompts import DEFAULT_EVOLVER_SYSTEM_PROMPT, build_evolution_prompt
from .tools import BASH_TOOL_SPEC, create_default_llm, make_workspace_bash

logger = logging.getLogger(__name__)

MUTATION_PATHS = ("prompts", "skills", "memory", "tools", "manifest.yaml")


class AdaptiveSkillEngine(EvolutionEngine):
    """LLM-driven workspace mutation engine."""

    def __init__(self, config: EvolveConfig, llm: LLMProvider | None = None):
        self.config = config
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = create_default_llm(self.config)
        return self._llm

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: Any,
        trial: Any,
    ) -> StepResult:
        """Analyze observations and mutate the workspace via LLM."""
        recent_logs = history.get_observations(last_n_cycles=2)
        cycle_num = history.latest_cycle + 1

        skills_before = [s.name for s in workspace.list_skills()]
        drafts = workspace.list_drafts()

        prompt = build_evolution_prompt(
            workspace,
            recent_logs,
            drafts,
            cycle_num,
            evolve_prompts=self.config.evolve_prompts,
            evolve_skills=self.config.evolve_skills,
            evolve_memory=self.config.evolve_memory,
            evolve_tools=self.config.evolve_tools,
            trajectory_only=self.config.trajectory_only,
            max_skills=self.config.extra.get("max_skills", 5),
            solver_proposed=self.config.extra.get("solver_proposed", False),
            prompt_only=self.config.extra.get("prompt_only", False),
            protect_skills=self.config.extra.get("protect_skills", False),
            judge_llm=self.llm if self.config.trajectory_only else None,
        )
        response = self._run_llm(prompt, workspace.root)

        workspace.clear_drafts()

        skills_after = [s.name for s in workspace.list_skills()]
        new_skills = len(set(skills_after) - set(skills_before))
        mutated = _workspace_has_mutation(workspace.root)

        return StepResult(
            mutated=mutated,
            summary=f"A-Evolve: {new_skills} new skills, {len(drafts)} drafts reviewed",
            metadata={
                "evo_number": cycle_num,
                "tasks_analyzed": len(recent_logs),
                "drafts_reviewed": len(drafts),
                "skills_before": len(skills_before),
                "skills_after": len(skills_after),
                "new_skills": new_skills,
                "usage": response.get("usage", {}),
            },
        )

    def evolve(
        self,
        workspace: AgentWorkspace,
        observation_logs: list[dict[str, Any]],
        evo_number: int = 0,
    ) -> dict[str, Any]:
        """Run one evolution pass outside the loop (for scripts/examples)."""
        vc = VersionControl(workspace.root)
        vc.init()

        skills_before = [s.name for s in workspace.list_skills()]
        drafts = workspace.list_drafts()

        vc.commit(
            message=f"pre-evo-{evo_number}: snapshot before evolution",
            tag=f"pre-evo-{evo_number}",
        )

        prompt = build_evolution_prompt(
            workspace,
            observation_logs,
            drafts,
            evo_number,
            evolve_prompts=self.config.evolve_prompts,
            evolve_skills=self.config.evolve_skills,
            evolve_memory=self.config.evolve_memory,
            evolve_tools=self.config.evolve_tools,
            trajectory_only=self.config.trajectory_only,
            max_skills=self.config.extra.get("max_skills", 5),
            solver_proposed=self.config.extra.get("solver_proposed", False),
            prompt_only=self.config.extra.get("prompt_only", False),
            protect_skills=self.config.extra.get("protect_skills", False),
            judge_llm=self.llm if self.config.trajectory_only else None,
        )
        response = self._run_llm(prompt, workspace.root)

        workspace.clear_drafts()

        skills_after = [s.name for s in workspace.list_skills()]
        new_skills = len(set(skills_after) - set(skills_before))
        mutated = _workspace_has_mutation(workspace.root)
        if mutated:
            vc.commit(
                message=f"evo-{evo_number}: {new_skills} new skills",
                tag=f"evo-{evo_number}",
            )
        else:
            vc.commit(
                message=f"evo-{evo_number}: no mutation",
                tag=f"evo-{evo_number}",
            )

        return {
            "evo_number": evo_number,
            "tasks_analyzed": len(observation_logs),
            "drafts_reviewed": len(drafts),
            "skills_before": len(skills_before),
            "skills_after": len(skills_after),
            "new_skills": new_skills,
            "usage": response.get("usage", {}),
        }

    def _run_llm(self, prompt: str, workspace_root: Path) -> dict[str, Any]:
        """Run the evolver LLM with bash access to the workspace."""
        bash_fn = make_workspace_bash(workspace_root)
        converse_loop = getattr(self.llm, "converse_loop", None)
        if callable(converse_loop):
            response = converse_loop(
                system_prompt=DEFAULT_EVOLVER_SYSTEM_PROMPT,
                user_message=prompt,
                tools=[BASH_TOOL_SPEC],
                tool_executor={"workspace_bash": lambda command: bash_fn(command)},
                max_tokens=self.config.evolver_max_tokens,
            )
            return {
                "content": response.content,
                "usage": response.usage,
            }

        messages = [
            LLMMessage(role="system", content=DEFAULT_EVOLVER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]
        response = self.llm.complete(
            messages, max_tokens=self.config.evolver_max_tokens
        )
        return {
            "content": response.content,
            "usage": response.usage,
        }


def _workspace_has_mutation(workspace_root: Path) -> bool:
    """Return whether evolvable workspace files changed since the last commit."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *MUTATION_PATHS],
        capture_output=True,
        text=True,
        cwd=str(workspace_root),
    )
    if result.returncode != 0:
        logger.warning("Could not inspect workspace mutation status: %s", result.stderr.strip())
        return False
    return bool(result.stdout.strip())
