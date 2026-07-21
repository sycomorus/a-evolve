"""GEPAEngine — GEPA-based evolution engine using optimize_anything.

Runs GEPA's full optimization pipeline inside a single step() call.
Returns StepResult(stop=True) so the loop exits after one cycle.

All imports from the `gepa` package are at module level — this file
should only be imported when GEPA is installed.
"""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

from ...engine.base import EvolutionEngine
from ...types import StepResult
from .evaluator import GEPADebugLog, make_evaluator, make_parallel_evaluator
from .prompts import COMPONENT_REFLECTION_PROMPTS
from .serialization import build_candidate, restore_candidate

if TYPE_CHECKING:
    from ...config import EvolveConfig
    from ...contract.workspace import AgentWorkspace
    from ...engine.history import EvolutionHistory
    from ...engine.trial import TrialRunner
    from ...types import Observation

logger = logging.getLogger(__name__)


class GEPAEngine(EvolutionEngine):
    """GEPA-based evolution engine using optimize_anything.

    Runs optimize_anything() to completion inside a single step() call,
    serializing A-Evolve workspace layers into GEPA's dict[str, str]
    candidate format. Returns StepResult(stop=True) to exit the loop.
    """

    def __init__(
        self,
        config: EvolveConfig,
        gepa_config: GEPAConfig | None = None,
        objective: str | None = None,
        background: str | None = None,
        parallel_workers: int = 1,
        validation_limit: int | None = None,
    ):
        self.config = config
        self.objective = objective
        self.background = background
        self.parallel_workers = parallel_workers
        self.validation_limit = validation_limit

        if gepa_config is not None:
            self.gepa_config = gepa_config
        else:
            self.gepa_config = GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=config.batch_size * config.max_cycles,
                ),
                reflection=ReflectionConfig(
                    reflection_lm=config.evolver_model,
                    reflection_prompt_template=(
                        COMPONENT_REFLECTION_PROMPTS.copy()
                        if not objective and not background
                        else None
                    ),
                ),
            )

    @property
    def manages_own_evaluation(self) -> bool:
        return True

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: EvolutionHistory,
        trial: TrialRunner,
    ) -> StepResult:
        debug_log = GEPADebugLog(workspace.root)
        agent_config = getattr(trial.agent, "config", None)
        debug_log.write(
            "engine_step_start",
            debug_log_path=str(debug_log.path),
            parallel_workers=self.parallel_workers,
            validation_limit=self.validation_limit,
            max_metric_calls=self.gepa_config.engine.max_metric_calls,
            agent_model=getattr(agent_config, "model", None),
            agent_base_url=getattr(agent_config, "base_url", None),
            agent_task_timeout_seconds=getattr(
                agent_config, "task_timeout_seconds", None
            ),
        )
        if self.gepa_config.engine.run_dir is None:
            self.gepa_config.engine.run_dir = str(
                workspace.root / "evolution" / "gepa"
            )

        seed_candidate = build_candidate(workspace, self.config)

        train_tasks = trial.get_tasks(
            split="train", limit=self.gepa_config.engine.max_metric_calls
        )
        try:
            val_tasks = trial.get_tasks(
                split="holdout",
                limit=(
                    self.validation_limit
                    if self.validation_limit is not None
                    else max(20, self.config.batch_size)
                ),
            )
        except Exception:
            val_tasks = None
        debug_log.write(
            "tasks_loaded",
            train_tasks=len(train_tasks),
            validation_tasks=len(val_tasks) if val_tasks is not None else None,
        )

        cleanup = None
        if self.parallel_workers > 1:
            evaluator, cleanup = make_parallel_evaluator(
                trial,
                workspace,
                self.parallel_workers,
                self.config,
                debug_log=debug_log,
            )
            self.gepa_config.engine.parallel = True
            self.gepa_config.engine.max_workers = self.parallel_workers
        else:
            evaluator = make_evaluator(trial, self.config, debug_log=debug_log)
            self.gepa_config.engine.parallel = False

        try:
            debug_log.write(
                "optimize_start",
                parallel=self.gepa_config.engine.parallel,
                max_workers=getattr(self.gepa_config.engine, "max_workers", None),
                run_dir=self.gepa_config.engine.run_dir,
            )
            result = optimize_anything(
                seed_candidate=seed_candidate,
                evaluator=evaluator,
                dataset=train_tasks,
                valset=val_tasks,
                objective=self.objective,
                background=self.background,
                config=self.gepa_config,
            )
            debug_log.write(
                "optimize_done",
                candidates=result.num_candidates,
                metric_calls=result.total_metric_calls,
            )
        except BaseException as exc:
            debug_log.write(
                "optimize_failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            if cleanup:
                cleanup()

        best = result.best_candidate
        if isinstance(best, str):
            best = {"system_prompt": best}
        restore_candidate(workspace, best, self.config)

        best_score = result.val_aggregate_scores[result.best_idx]
        return StepResult(
            mutated=True,
            stop=True,
            summary=f"GEPA: {result.num_candidates} candidates, best {best_score:.3f}",
            metadata={
                "gepa_best_idx": result.best_idx,
                "gepa_num_candidates": result.num_candidates,
                "gepa_total_metric_calls": result.total_metric_calls,
                "gepa_best_score": best_score,
                "gepa_run_dir": self.gepa_config.engine.run_dir,
            },
        )
