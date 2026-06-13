"""EvolutionLoop -- thin orchestrator that wires shared primitives to an engine.

The loop handles the expensive shared work that every engine needs:
  Solve -> Observe -> Snapshot -> engine.step() -> Snapshot -> Reload

The engine decides *how* to evolve; the loop decides *when* and provides
the infrastructure (versioning, observation logging, trial runner).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..config import EvolveConfig
from ..types import CycleRecord, EvolutionResult, Observation
from .history import EvolutionHistory
from .observer import Observer
from ..task_runner import run_task_evaluations
from .trial import TrialRunner
from .versioning import VersionControl

if TYPE_CHECKING:
    from ..benchmarks.base import BenchmarkAdapter
    from ..protocol.base_agent import BaseAgent
    from .base import EvolutionEngine

logger = logging.getLogger(__name__)


def _is_score_converged(
    scores: list[float], window: int = 3, epsilon: float = 0.01
) -> bool:
    """Generic convergence: score hasn't improved by more than *epsilon* in *window* cycles."""
    if len(scores) < window + 1:
        return False
    recent = scores[-window:]
    baseline = scores[-(window + 1)]
    return all(abs(s - baseline) < epsilon for s in recent)


class EvolutionLoop:
    """Orchestrates the full evolution loop with a pluggable engine."""

    def __init__(
        self,
        agent: BaseAgent,
        benchmark: BenchmarkAdapter,
        engine: EvolutionEngine,
        config: EvolveConfig | None = None,
    ):
        self.agent = agent
        self.benchmark = benchmark
        self.engine = engine
        self.config = config or EvolveConfig()

        workspace_root = self.agent.workspace.root
        evolution_dir = workspace_root / "evolution"
        evolution_dir.mkdir(parents=True, exist_ok=True)

        self.observer = Observer(evolution_dir)
        self.versioning = VersionControl(workspace_root)
        self.history = EvolutionHistory(self.observer, self.versioning)
        self.trial = TrialRunner(self.agent, self.benchmark)

    def run(
        self,
        cycles: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvolutionResult:
        """Run the evolution loop for the specified number of cycles."""
        max_epochs = cycles or self.config.max_cycles
        evolution_dir = self.agent.workspace.root / "evolution"

        self.versioning.init()

        score_history: list[float] = []
        completed_updates = 0
        completed_epochs = 0
        schedule = self._build_training_schedule(max_epochs)
        total_updates = len(schedule)

        for update_index, item in enumerate(schedule, start=1):
            cycle_num = update_index
            logger.info(
                "=== Evolution Cycle %d/%d (epoch %d/%d, batch %d/%d) ===",
                cycle_num,
                total_updates,
                item["epoch"],
                max_epochs,
                item["batch_index"],
                item["batch_count"],
            )

            # 1. SOLVE + 2. OBSERVE
            if self.engine.manages_own_evaluation:
                observations: list[Observation] = []
                self.agent.export_to_fs()
                batch_path = self.observer.collect(observations)
                cycle_score = 0.0
            else:
                tasks = item["tasks"]
                if tasks is None:
                    tasks = self.benchmark.get_tasks(
                        split="train",
                        limit=self.config.batch_size,
                    )
                task_results = run_task_evaluations(self.agent, self.benchmark, tasks)
                observations = []
                for result in task_results:
                    if result.trajectory is None or result.feedback is None:
                        logger.error("Error solving task %s: %s", result.task.id, result.error)
                        continue
                    observations.append(
                        Observation(
                            task=result.task,
                            trajectory=result.trajectory,
                            feedback=result.feedback,
                        )
                    )

                self.agent.export_to_fs()
                batch_path = self.observer.collect(observations)

                cycle_score = (
                    sum(o.feedback.score for o in observations) / len(observations)
                    if observations
                    else 0.0
                )
            score_history.append(cycle_score)
            completed_updates = cycle_num
            completed_epochs = max(completed_epochs, int(item["epoch"]))
            logger.info("Cycle %d score: %.3f", cycle_num, cycle_score)

            # 3. PRE-EVOLVE SNAPSHOT
            self.versioning.commit(
                message=f"pre-evo-{cycle_num}: score={cycle_score:.3f}",
                tag=f"pre-evo-{cycle_num}",
            )

            # 4. ENGINE STEP
            step_result = self.engine.step(
                workspace=self.agent.workspace,
                observations=observations,
                history=self.history,
                trial=self.trial,
            )

            # 5. POST-EVOLVE SNAPSHOT
            if step_result.mutated:
                self.versioning.commit(
                    message=f"evo-{cycle_num}: {step_result.summary}",
                    tag=f"evo-{cycle_num}",
                )
            else:
                self.versioning.commit(
                    message=f"evo-{cycle_num}: no mutation",
                    tag=f"evo-{cycle_num}",
                )

            # 6. RECORD CYCLE
            record = CycleRecord(
                cycle=cycle_num,
                score=cycle_score,
                mutated=step_result.mutated,
                engine_name=self.engine.__class__.__name__,
                summary=step_result.summary,
                observation_batch=batch_path.name,
                metadata=step_result.metadata,
            )
            self.history.record_cycle(record)

            # 7. RELOAD
            self.agent.reload_from_fs()
            self.engine.on_cycle_end(accepted=step_result.mutated, score=cycle_score)

            # 7b. STOP CHECK
            if step_result.stop:
                logger.info("Engine requested early stop after cycle %d.", cycle_num)
                self._append_history(evolution_dir, cycle_num, cycle_score, step_result.mutated)
                self._write_metrics(evolution_dir, score_history)
                self._notify_progress(
                    progress_callback,
                    cycle_num,
                    total_updates,
                    cycle_score,
                    step_result.mutated,
                    step_result.summary,
                    stopped=True,
                    converged=True,
                    epoch=int(item["epoch"]),
                    batch_index=int(item["batch_index"]),
                    batch_count=int(item["batch_count"]),
                )
                return EvolutionResult(
                    cycles_completed=completed_updates,
                    final_score=cycle_score,
                    score_history=score_history,
                    converged=True,
                    details={
                        "epochs_completed": completed_epochs,
                        "updates_completed": completed_updates,
                        "total_updates": total_updates,
                    },
                )

            # 8. LOGGING
            self._append_history(evolution_dir, cycle_num, cycle_score, step_result.mutated)
            self._write_metrics(evolution_dir, score_history)
            self._notify_progress(
                progress_callback,
                cycle_num,
                total_updates,
                cycle_score,
                step_result.mutated,
                step_result.summary,
                epoch=int(item["epoch"]),
                batch_index=int(item["batch_index"]),
                batch_count=int(item["batch_count"]),
            )

            # 9. CONVERGENCE CHECK
            if _is_score_converged(score_history, window=self.config.egl_window):
                logger.info("Score converged after %d cycles.", cycle_num)
                return EvolutionResult(
                    cycles_completed=completed_updates,
                    final_score=cycle_score,
                    score_history=score_history,
                    converged=True,
                    details={
                        "epochs_completed": completed_epochs,
                        "updates_completed": completed_updates,
                        "total_updates": total_updates,
                    },
                )

        return EvolutionResult(
            cycles_completed=completed_updates,
            final_score=score_history[-1] if score_history else 0.0,
            score_history=score_history,
            converged=False,
            details={
                "epochs_completed": completed_epochs,
                "updates_completed": completed_updates,
                "total_updates": total_updates,
            },
        )

    # ── Internal helpers ──────────────────────────────────────

    def _notify_progress(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cycle: int,
        total_cycles: int,
        score: float,
        mutated: bool,
        summary: str,
        *,
        stopped: bool = False,
        converged: bool = False,
        epoch: int | None = None,
        batch_index: int | None = None,
        batch_count: int | None = None,
    ) -> None:
        if progress_callback is None:
            return
        event = {
            "cycle": cycle,
            "total_cycles": total_cycles,
            "score": score,
            "mutated": mutated,
            "summary": summary,
            "stopped": stopped,
            "converged": converged,
        }
        if epoch is not None:
            event["epoch"] = epoch
        if batch_index is not None:
            event["batch_index"] = batch_index
        if batch_count is not None:
            event["batch_count"] = batch_count
        progress_callback(event)

    def _build_training_schedule(self, max_epochs: int) -> list[dict[str, Any]]:
        if self.engine.manages_own_evaluation or self.config.train_limit is None:
            return [
                {
                    "epoch": cycle + 1,
                    "batch_index": 1,
                    "batch_count": 1,
                    "tasks": None,
                }
                for cycle in range(max_epochs)
            ]

        train_tasks = self.benchmark.get_tasks(split="train", limit=self.config.train_limit)
        if not train_tasks:
            return []
        batch_size = max(1, int(self.config.batch_size))
        batches = [
            train_tasks[start : start + batch_size]
            for start in range(0, len(train_tasks), batch_size)
        ]
        batch_count = len(batches)
        return [
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_count": batch_count,
                "tasks": batch,
            }
            for epoch in range(1, max_epochs + 1)
            for batch_index, batch in enumerate(batches, start=1)
        ]

    def _append_history(
        self, evolution_dir: Path, cycle: int, score: float, mutated: bool
    ) -> None:
        history_file = evolution_dir / "history.jsonl"
        entry = {
            "cycle": cycle,
            "score": score,
            "mutated": mutated,
            "timestamp": datetime.now().isoformat(),
        }
        with open(history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _write_metrics(self, evolution_dir: Path, scores: list[float]) -> None:
        metrics_file = evolution_dir / "metrics.json"
        metrics = {
            "cycles_completed": len(scores),
            "latest_score": scores[-1] if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }
        metrics_file.write_text(json.dumps(metrics, indent=2))
