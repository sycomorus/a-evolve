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
from ..evaluation import run_evaluation
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
        validation_accuracy_history: list[float] = []
        validation_accepted_history: list[bool] = []
        validation_tasks = []
        initial_validation_accuracy: float | None = None
        accepted_validation_accuracy: float | None = None
        validation_limit = int(self.config.validation_limit or 0)
        if validation_limit > 0:
            if self.engine.manages_own_evaluation:
                raise ValueError("validation gating is not supported for self-managing engines")
            self.versioning.exclude_path("evolution/validation/")
            validation_tasks = self.benchmark.get_tasks(
                split="val",
                limit=validation_limit,
            )
            if len(validation_tasks) != validation_limit:
                raise ValueError(
                    "requested validation tasks cannot be satisfied: "
                    f"requested={validation_limit}, available={len(validation_tasks)}"
                )
            baseline_summary = run_evaluation(
                self.agent,
                self.benchmark,
                split="val",
                limit=validation_limit,
                tasks=validation_tasks,
                output_dir=evolution_dir / "validation" / "baseline",
            )
            baseline_summary["version_tag"] = "evo-0"
            (evolution_dir / "validation" / "baseline" / "summary.json").write_text(
                json.dumps(baseline_summary, indent=2),
                encoding="utf-8",
            )
            initial_validation_accuracy = float(baseline_summary["accuracy"])
            accepted_validation_accuracy = initial_validation_accuracy
            logger.info(
                "Initial validation accuracy: %.3f",
                initial_validation_accuracy,
            )
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
                if self.config.extra.get("step_opsd_enabled"):
                    from ..algorithms.step_opsd import build_step_opsd_records

                    base_records = [
                        self.observer.record_from_observation(observation)
                        for observation in observations
                    ]
                    records = build_step_opsd_records(
                        observations,
                        base_records,
                        evolution_dir=evolution_dir,
                        llm=getattr(self.engine, "llm", None),
                        failures_only=bool(
                            self.config.extra.get("step_opsd_review_failures_only", True)
                        ),
                        max_tokens=int(
                            self.config.extra.get(
                                "step_opsd_teacher_max_tokens",
                                min(self.config.evolver_max_tokens, 4096),
                            )
                        ),
                    )
                    batch_path = self.observer.collect_records(records)
                else:
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

            # 5. POST-EVOLVE SNAPSHOT + OPTIONAL VALIDATION GATE
            if validation_tasks and step_result.mutated:
                self.versioning.commit(
                    message=f"candidate-evo-{cycle_num}: {step_result.summary}",
                    tag=f"candidate-evo-{cycle_num}",
                )

            accepted = step_result.mutated
            rolled_back = False
            validation_metadata: dict[str, Any] = {}
            if validation_tasks:
                self.agent.reload_from_fs()
                comparison_baseline = float(accepted_validation_accuracy)
                validation_cycle_dir = (
                    evolution_dir / "validation" / f"cycle_{cycle_num:04d}"
                )
                validation_summary = run_evaluation(
                    self.agent,
                    self.benchmark,
                    split="val",
                    limit=validation_limit,
                    tasks=validation_tasks,
                    output_dir=validation_cycle_dir,
                )
                candidate_accuracy = float(validation_summary["accuracy"])
                accepted = candidate_accuracy >= comparison_baseline
                validation_accuracy_history.append(candidate_accuracy)
                validation_accepted_history.append(accepted)

                if step_result.mutated and not accepted:
                    self.versioning.commit(
                        message=(
                            f"rejected-evo-{cycle_num}: validation="
                            f"{candidate_accuracy:.3f} < {comparison_baseline:.3f}"
                        ),
                        tag=f"rejected-evo-{cycle_num}",
                    )
                    self.versioning.rollback_to_tag(f"pre-evo-{cycle_num}")
                    self.agent.reload_from_fs()
                    rolled_back = True
                elif accepted:
                    accepted_validation_accuracy = candidate_accuracy

                self.versioning.commit(
                    message=(
                        f"evo-{cycle_num}: accepted={accepted}, "
                        f"validation={candidate_accuracy:.3f}"
                    ),
                    tag=f"evo-{cycle_num}",
                )
                validation_metadata = {
                    "candidate_accuracy": candidate_accuracy,
                    "comparison_baseline": comparison_baseline,
                    "accepted": accepted,
                    "rolled_back": rolled_back,
                    "candidate_tag": (
                        f"candidate-evo-{cycle_num}" if step_result.mutated else None
                    ),
                    "rejected_tag": (
                        f"rejected-evo-{cycle_num}" if rolled_back else None
                    ),
                    "effective_tag": f"evo-{cycle_num}",
                }
                validation_summary.update(validation_metadata)
                validation_summary["accepted_accuracy_after_cycle"] = float(
                    accepted_validation_accuracy
                )
                (validation_cycle_dir / "summary.json").write_text(
                    json.dumps(validation_summary, indent=2),
                    encoding="utf-8",
                )
            else:
                self.versioning.commit(
                    message=(
                        f"evo-{cycle_num}: {step_result.summary}"
                        if step_result.mutated
                        else f"evo-{cycle_num}: no mutation"
                    ),
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
                metadata={**step_result.metadata, "validation": validation_metadata},
            )
            self.history.record_cycle(record)

            # 7. RELOAD
            if not validation_tasks:
                self.agent.reload_from_fs()
            self.engine.on_cycle_end(accepted=accepted, score=cycle_score)

            # 7b. STOP CHECK
            if step_result.stop:
                logger.info("Engine requested early stop after cycle %d.", cycle_num)
                self._append_history(
                    evolution_dir,
                    cycle_num,
                    cycle_score,
                    step_result.mutated,
                    accepted=accepted,
                    validation=validation_metadata,
                )
                self._write_metrics(
                    evolution_dir,
                    score_history,
                    initial_validation_accuracy=initial_validation_accuracy,
                    validation_accuracy_history=validation_accuracy_history,
                    validation_accepted_history=validation_accepted_history,
                    final_validation_accuracy=accepted_validation_accuracy,
                )
                self._notify_progress(
                    progress_callback,
                    cycle_num,
                    total_updates,
                    cycle_score,
                    step_result.mutated,
                    step_result.summary,
                    accepted=accepted,
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
                    initial_validation_accuracy=initial_validation_accuracy,
                    validation_accuracy_history=validation_accuracy_history,
                    validation_accepted_history=validation_accepted_history,
                    final_validation_accuracy=accepted_validation_accuracy,
                    converged=True,
                    details=self._result_details(
                        completed_epochs,
                        completed_updates,
                        total_updates,
                        initial_validation_accuracy,
                        validation_accuracy_history,
                        validation_accepted_history,
                        accepted_validation_accuracy,
                    ),
                )

            # 8. LOGGING
            self._append_history(
                evolution_dir,
                cycle_num,
                cycle_score,
                step_result.mutated,
                accepted=accepted,
                validation=validation_metadata,
            )
            self._write_metrics(
                evolution_dir,
                score_history,
                initial_validation_accuracy=initial_validation_accuracy,
                validation_accuracy_history=validation_accuracy_history,
                validation_accepted_history=validation_accepted_history,
                final_validation_accuracy=accepted_validation_accuracy,
            )
            self._notify_progress(
                progress_callback,
                cycle_num,
                total_updates,
                cycle_score,
                step_result.mutated,
                step_result.summary,
                accepted=accepted,
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
                    initial_validation_accuracy=initial_validation_accuracy,
                    validation_accuracy_history=validation_accuracy_history,
                    validation_accepted_history=validation_accepted_history,
                    final_validation_accuracy=accepted_validation_accuracy,
                    converged=True,
                    details=self._result_details(
                        completed_epochs,
                        completed_updates,
                        total_updates,
                        initial_validation_accuracy,
                        validation_accuracy_history,
                        validation_accepted_history,
                        accepted_validation_accuracy,
                    ),
                )

        return EvolutionResult(
            cycles_completed=completed_updates,
            final_score=score_history[-1] if score_history else 0.0,
            score_history=score_history,
            initial_validation_accuracy=initial_validation_accuracy,
            validation_accuracy_history=validation_accuracy_history,
            validation_accepted_history=validation_accepted_history,
            final_validation_accuracy=accepted_validation_accuracy,
            converged=False,
            details=self._result_details(
                completed_epochs,
                completed_updates,
                total_updates,
                initial_validation_accuracy,
                validation_accuracy_history,
                validation_accepted_history,
                accepted_validation_accuracy,
            ),
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
        accepted: bool | None = None,
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
            "accepted": mutated if accepted is None else accepted,
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
        self,
        evolution_dir: Path,
        cycle: int,
        score: float,
        mutated: bool,
        *,
        accepted: bool,
        validation: dict[str, Any],
    ) -> None:
        history_file = evolution_dir / "history.jsonl"
        entry = {
            "cycle": cycle,
            "score": score,
            "mutated": mutated,
            "accepted": accepted,
            "validation": validation,
            "timestamp": datetime.now().isoformat(),
        }
        with open(history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _write_metrics(
        self,
        evolution_dir: Path,
        scores: list[float],
        *,
        initial_validation_accuracy: float | None,
        validation_accuracy_history: list[float],
        validation_accepted_history: list[bool],
        final_validation_accuracy: float | None,
    ) -> None:
        metrics_file = evolution_dir / "metrics.json"
        metrics = {
            "cycles_completed": len(scores),
            "latest_score": scores[-1] if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "initial_validation_accuracy": initial_validation_accuracy,
            "validation_accuracy_history": validation_accuracy_history,
            "validation_accepted_history": validation_accepted_history,
            "final_validation_accuracy": final_validation_accuracy,
        }
        metrics_file.write_text(json.dumps(metrics, indent=2))

    @staticmethod
    def _result_details(
        epochs_completed: int,
        updates_completed: int,
        total_updates: int,
        initial_validation_accuracy: float | None,
        validation_accuracy_history: list[float],
        validation_accepted_history: list[bool],
        final_validation_accuracy: float | None,
    ) -> dict[str, Any]:
        return {
            "epochs_completed": epochs_completed,
            "updates_completed": updates_completed,
            "total_updates": total_updates,
            "initial_validation_accuracy": initial_validation_accuracy,
            "validation_accuracy_history": list(validation_accuracy_history),
            "validation_accepted_history": list(validation_accepted_history),
            "final_validation_accuracy": final_validation_accuracy,
        }
