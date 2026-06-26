"""OR-Interact type-local harness tree MVP."""

from __future__ import annotations

import csv
import importlib
import json
import re
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent_evolve.benchmarks.base import BenchmarkAdapter
from agent_evolve.config import EvolveConfig
from agent_evolve.engine.observer import Observer
from agent_evolve.engine.versioning import VersionControl
from agent_evolve.protocol.base_agent import BaseAgent
from agent_evolve.task_runner import resolve_agent_parallelism
from agent_evolve.types import EvolutionResult, Feedback, Observation, Task, Trajectory


STATE_DIR = Path("evolution") / "harness_tree"
STATE_FILE = STATE_DIR / "state.json"
MAIN_BRANCH = "main"
GENERAL_BRANCH = "branch/general"
MATERIALIZED_DIR = STATE_DIR / "materialized"
OVERLAYS_DIR = STATE_DIR / "overlays"
HARNESS_PATHS = (
    "prompts",
    "skills",
    "memory",
    "tools",
    "manifest.yaml",
    "or_interact_settings.json",
)
ROUTE_PHASE_TURNS = 8

ROUTE_PHASE_USER_MESSAGE = """\
Route this operations research task before solving it.

Use list_context and read_md/read_csv/read_json as needed to inspect visible docs/ and data/.
Summarize the visible evidence yourself, then call type_router(evidence_text, existing_branches).
The branch should be a broad OR problem family, not a narrow instance-specific subproblem.
Use categories like TSP, CVRP, vehicle routing, network flow, assignment, scheduling,
facility location, inventory planning, production planning, bin packing, knapsack, or
travel planning when supported by the evidence.
Do not build the optimization model, do not run solvers, and do not call finalize in this phase.

Existing branch summaries:
{existing_branches}
"""

SOLVE_PHASE_USER_MESSAGE = """\
Routing is complete.

Selected branch: {branch_name}
Route confidence: {confidence:.3f}
Route rationale: {rationale}

Continue from the context already gathered above and solve the optimization task using the selected domain harness.
Do not repeat context discovery unless required information is missing or ambiguous.
Call finalize with the best available objective value when done.
"""

@dataclass(frozen=True)
class RouteDecision:
    branch_name: str
    confidence: float
    rationale: str
    fallback_reason: str | None = None


@dataclass
class RoutedTask:
    task: Task
    route_messages: list[dict[str, Any]] | None
    runtime_dir: Path
    trace: Any
    decision: RouteDecision
    branch: str
    epoch: int
    cycle: int


def run_harness_tree(
    *,
    agent: BaseAgent,
    benchmark: BenchmarkAdapter,
    engine: Any,
    config: EvolveConfig,
    max_epochs: int,
    type_buffer_size: int,
    router_confidence_threshold: float,
    final_test_limit: int | None,
    final_dir: Path,
    disable_main_evolve: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[EvolutionResult, dict[str, Any]]:
    runner = HarnessTreeRunner(
        agent=agent,
        benchmark=benchmark,
        engine=engine,
        config=config,
        type_buffer_size=type_buffer_size,
        router_confidence_threshold=router_confidence_threshold,
        disable_main_evolve=disable_main_evolve,
    )
    result = runner.run_training(max_epochs=max_epochs, progress_callback=progress_callback)
    eval_summary = runner.run_final_evaluation(
        limit=final_test_limit,
        output_dir=final_dir,
        progress_callback=progress_callback,
    )
    return result, eval_summary


class HarnessTreeRunner:
    def __init__(
        self,
        *,
        agent: BaseAgent,
        benchmark: BenchmarkAdapter,
        engine: Any,
        config: EvolveConfig,
        type_buffer_size: int,
        router_confidence_threshold: float,
        disable_main_evolve: bool = False,
    ) -> None:
        self.agent = agent
        self.benchmark = benchmark
        self.engine = engine
        self.config = config
        self.type_buffer_size = max(1, int(type_buffer_size))
        self.router_confidence_threshold = float(router_confidence_threshold)
        self.disable_main_evolve = bool(disable_main_evolve)
        self.workspace_root = self.agent.workspace.root
        self.state_path = self.workspace_root / STATE_FILE
        self.observer = Observer(self.workspace_root / "evolution")
        self.versioning = VersionControl(self.workspace_root)
        self.state = _empty_state()
        self.evolve_number = 0
        self._base_harness_snapshot: dict[Path, bytes] | None = None
        self._external_materialized_root = self.workspace_root.parent / "harness_tree_branch_workspaces"
        self._worker_materialized_root = (
            self._external_materialized_root / "_workers"
            if self.disable_main_evolve
            else self.workspace_root / MATERIALIZED_DIR / "workers"
        )

    def run_training(
        self,
        *,
        max_epochs: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvolutionResult:
        self._prepare_repo()
        self.state = _load_state(self.state_path)
        self._capture_base_harness_snapshot()
        score_history: list[float] = []
        completed_tasks = 0
        schedule = self._training_schedule(max_epochs)
        execution_buffers: dict[str, list[RoutedTask]] = {}
        _emit_progress(
            progress_callback,
            {
                "phase": "train",
                "event": "start",
                "total": len(schedule),
                "max_epochs": max_epochs,
            },
        )

        for schedule_item in schedule:
            task = schedule_item["task"]
            _emit_progress(
                progress_callback,
                {
                    "phase": "train",
                    "event": "task_start",
                    "completed": completed_tasks,
                    "total": len(schedule),
                    "epoch": schedule_item["epoch"],
                    "cycle": schedule_item["cycle"],
                    "task_id": task.id,
                },
            )
            routed = self._run_route_phase(
                task,
                phase="train",
                epoch=schedule_item["epoch"],
                cycle=schedule_item["cycle"],
            )
            branch = routed.branch
            buffer = execution_buffers.setdefault(branch, [])
            buffer.append(routed)
            if len(buffer) >= self.type_buffer_size:
                completed_tasks = self._flush_training_buffer(
                    branch=branch,
                    buffer=buffer,
                    score_history=score_history,
                    completed_tasks=completed_tasks,
                    total_tasks=len(schedule),
                    progress_callback=progress_callback,
                )
                buffer.clear()

        total_evolves = len(self.state.get("main_evolutions", [])) + sum(
            int(item.get("evolve_count", 0)) for item in self.state["branches"].values()
        )
        return EvolutionResult(
            cycles_completed=total_evolves,
            final_score=(sum(score_history) / len(score_history)) if score_history else 0.0,
            score_history=score_history,
            converged=False,
            details={
                "tasks_completed": completed_tasks,
                "updates_completed": total_evolves,
                "epochs_completed": max_epochs,
                "branches": sorted(self.state["branches"]),
                "type_buffer_size": self.type_buffer_size,
                "disable_main_evolve": self.disable_main_evolve,
            },
        )

    def run_final_evaluation(
        self,
        *,
        limit: int | None,
        output_dir: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._prepare_repo()
        self.state = _load_state(self.state_path)
        self._capture_base_harness_snapshot()
        output_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        tasks = self.benchmark.get_tasks(split="test", limit=limit)
        workers = min(resolve_agent_parallelism(self.agent), len(tasks)) if tasks else 0
        _emit_progress(
            progress_callback,
            {
                "phase": "final_eval",
                "event": "start",
                "total": len(tasks),
                "workers": workers,
            },
        )
        evaluations = self._run_final_evaluations_parallel(tasks, progress_callback=progress_callback)
        for index, evaluation in enumerate(evaluations, start=1):
            decision = evaluation["route_decision"]
            self._record_route_decision(evaluation["task"], decision, phase="final_eval")
            branch = self._reported_eval_branch(decision)
            row = _row_from_evaluation(evaluation, branch, decision)
            rows.append(row)
            _emit_progress(
                progress_callback,
                {
                    "phase": "final_eval",
                    "event": "task_done",
                    "completed": index,
                    "total": len(tasks),
                    "task_id": evaluation["task"].id,
                    "branch_name": branch,
                    "route_confidence": decision.confidence,
                    "score": row.get("score", 0.0),
                    "success": _as_bool(row.get("success")),
                    "fallback_reason": decision.fallback_reason,
                    "feedback_detail": row.get("detail"),
                },
            )

        summary = _evaluation_summary(
            rows=rows,
            workspace=self.workspace_root,
            split="test",
            limit=limit,
            output_dir=output_dir,
        )
        _write_results_csv(output_dir / "results.csv", rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.state.setdefault("final_eval", {})["summary"] = summary
        _save_state(self.state_path, self.state)
        return summary

    def _run_final_evaluations_parallel(
        self,
        tasks: list[Task],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        if not tasks:
            return []
        workers = min(resolve_agent_parallelism(self.agent), len(tasks))
        if workers <= 1:
            results = []
            for index, task in enumerate(tasks):
                _emit_progress(
                    progress_callback,
                    {
                        "phase": "final_eval",
                        "event": "task_start",
                        "completed": index,
                        "total": len(tasks),
                        "task_id": task.id,
                    },
                )
                results.append(self._run_two_phase_task(task, phase="final_eval", record_route=False))
            return results

        agent_class = _class_path(self.agent.__class__)
        state_snapshot = json.loads(json.dumps(self.state))
        results: list[dict[str, Any] | None] = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, task in enumerate(tasks):
                _emit_progress(
                    progress_callback,
                    {
                        "phase": "final_eval",
                        "event": "task_start",
                        "completed": index,
                        "total": len(tasks),
                        "task_id": task.id,
                    },
                )
                future = executor.submit(
                    _run_final_two_phase_worker,
                    agent_class,
                    self.workspace_root,
                    self.benchmark,
                    task,
                    state_snapshot,
                    self.router_confidence_threshold,
                    self._worker_materialized_root,
                )
                futures[future] = index
            for future in as_completed(futures):
                index = futures[future]
                task = tasks[index]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failed_evaluation(
                        task,
                        decision=_normalize_route_decision(
                            {},
                            threshold=self.router_confidence_threshold,
                            fallback_reason=f"{type(exc).__name__}: {exc}",
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results[index] = result
        return [result for result in results if result is not None]

    def _prepare_repo(self) -> None:
        if not (self.workspace_root / ".git").exists():
            self.versioning.init()
        _exclude_state_dir(self.workspace_root)
        if not self.versioning.branch_exists(MAIN_BRANCH):
            current = self.versioning.get_current_branch()
            if current:
                self.versioning.create_branch(MAIN_BRANCH, current)
        self.versioning.checkout_branch(MAIN_BRANCH)
        self.agent.reload_from_fs()

    def _capture_base_harness_snapshot(self) -> None:
        if self.disable_main_evolve and self._base_harness_snapshot is None:
            self._base_harness_snapshot = _snapshot_harness_workspace(self.workspace_root)

    def _restore_base_harness_snapshot(self) -> None:
        if self.disable_main_evolve and self._base_harness_snapshot is not None:
            _restore_harness_workspace(self.workspace_root, self._base_harness_snapshot)
            self.agent.reload_from_fs()

    def _training_schedule(self, max_epochs: int) -> list[dict[str, Any]]:
        if self.config.train_limit is None:
            tasks = self.benchmark.get_tasks(split="train", limit=self.config.batch_size)
            return [
                {"epoch": epoch, "cycle": index + 1, "task": task}
                for epoch in range(1, max_epochs + 1)
                for index, task in enumerate(tasks)
            ]

        train_tasks = self.benchmark.get_tasks(split="train", limit=self.config.train_limit)
        return [
            {"epoch": epoch, "cycle": index + 1, "task": task}
            for epoch in range(1, max_epochs + 1)
            for index, task in enumerate(train_tasks)
        ]

    def _route_from_observation(
        self,
        observation: Observation,
        *,
        phase: str,
        record: bool = True,
    ) -> RouteDecision:
        raw, extraction_reason = _extract_type_router_output(observation.trajectory)
        decision = _normalize_route_decision(
            raw,
            threshold=self.router_confidence_threshold,
            fallback_reason=extraction_reason,
        )
        if record:
            self._record_route_decision(observation.task, decision, phase=phase)
        return decision

    def _record_route_decision(self, task: Task, decision: RouteDecision, *, phase: str) -> None:
        self.state["router_decisions"].append(
            {
                "phase": phase,
                "task_id": task.id,
                "branch_name": decision.branch_name,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
                "fallback_reason": decision.fallback_reason,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if phase == "final_eval":
            self.state.setdefault("final_eval", {}).setdefault("decisions", []).append(
                self.state["router_decisions"][-1]
            )
        _save_state(self.state_path, self.state)

    def _ensure_training_branch(self, decision: RouteDecision, cycle: int) -> str:
        branch = decision.branch_name
        branch_state = self.state["branches"].setdefault(
            branch,
            {
                "name": branch,
                "rationale": decision.rationale,
                "overlay": str(self._overlay_dir(branch).relative_to(self.workspace_root)),
                "created_cycle": cycle,
                "solve_count": 0,
                "success_count": 0,
                "evolve_count": 0,
                "pending": [],
                "evolutions": [],
            },
        )
        branch_state.setdefault("overlay", str(self._overlay_dir(branch).relative_to(self.workspace_root)))
        return branch

    def _reported_eval_branch(self, decision: RouteDecision) -> str:
        branch = decision.branch_name
        if branch in self.state.get("branches", {}):
            return branch
        if GENERAL_BRANCH in self.state.get("branches", {}):
            return GENERAL_BRANCH
        return GENERAL_BRANCH

    def _run_route_phase(self, task: Task, *, phase: str, epoch: int, cycle: int) -> RoutedTask:
        self.versioning.checkout_branch(MAIN_BRANCH)
        self.agent.reload_from_fs()
        runtime_dir, trace = self.agent.start_task_run(task)
        try:
            route_result = self.agent.run_phase(
                task,
                runtime_dir=runtime_dir,
                trace=trace,
                phase=f"{phase}:route",
                user_message=ROUTE_PHASE_USER_MESSAGE.format(
                    existing_branches=json.dumps(_branch_summaries(self.state), ensure_ascii=False, indent=2)
                ),
                max_turns=ROUTE_PHASE_TURNS,
                stop_after_tools={"type_router"},
            )
            route_trajectory = _trajectory_from_runtime(task, runtime_dir)
            decision = self._route_from_observation(
                Observation(task=task, trajectory=route_trajectory, feedback=Feedback(False, 0.0, "")),
                phase=phase,
            )
            branch = self._ensure_training_branch(decision, cycle)
            _save_state(self.state_path, self.state)
            return RoutedTask(
                task=task,
                route_messages=route_result.messages or None,
                runtime_dir=runtime_dir,
                trace=trace,
                decision=decision,
                branch=branch,
                epoch=epoch,
                cycle=cycle,
            )
        except Exception as exc:
            decision = _normalize_route_decision(
                {},
                threshold=self.router_confidence_threshold,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
            self._record_route_decision(task, decision, phase=phase)
            branch = self._ensure_training_branch(decision, cycle)
            _save_state(self.state_path, self.state)
            return RoutedTask(
                task=task,
                route_messages=None,
                runtime_dir=runtime_dir,
                trace=trace,
                decision=decision,
                branch=branch,
                epoch=epoch,
                cycle=cycle,
            )

    def _run_two_phase_task(self, task: Task, *, phase: str, record_route: bool = True) -> dict[str, Any]:
        self.versioning.checkout_branch(MAIN_BRANCH)
        self.agent.reload_from_fs()
        runtime_dir, trace = self.agent.start_task_run(task)
        started_at = datetime.now()
        decision: RouteDecision | None = None
        branch: str | None = None
        try:
            route_result = self.agent.run_phase(
                task,
                runtime_dir=runtime_dir,
                trace=trace,
                phase=f"{phase}:route",
                user_message=ROUTE_PHASE_USER_MESSAGE.format(
                    existing_branches=json.dumps(_branch_summaries(self.state), ensure_ascii=False, indent=2)
                ),
                max_turns=ROUTE_PHASE_TURNS,
                stop_after_tools={"type_router"},
            )
            route_messages = route_result.messages or None
            route_trajectory = _trajectory_from_runtime(task, runtime_dir)
            decision = self._route_from_observation(
                Observation(task=task, trajectory=route_trajectory, feedback=Feedback(False, 0.0, "")),
                phase=phase,
                record=record_route,
            )
            branch = decision.branch_name if phase == "train" else self._reported_eval_branch(decision)
            materialized = self._materialize_branch_workspace(branch)
            phase_agent = self.agent.__class__(materialized)
            solve_result = phase_agent.run_phase(
                task,
                runtime_dir=runtime_dir,
                trace=trace,
                phase=f"{phase}:solve",
                initial_messages=route_messages,
                user_message=SOLVE_PHASE_USER_MESSAGE.format(
                    branch_name=branch,
                    confidence=decision.confidence,
                    rationale=decision.rationale or "n/a",
                ),
            )
            elapsed = (datetime.now() - started_at).total_seconds()
            trajectory = phase_agent.finish_task_run(
                task,
                runtime_dir=runtime_dir,
                result=solve_result,
                elapsed=elapsed,
            )
            feedback = self.benchmark.evaluate(task, trajectory)
            phase_agent.export_to_fs()
            return {
                "task": task,
                "trajectory": trajectory,
                "feedback": feedback,
                "route_decision": decision,
                "branch_name": branch,
                "materialized_workspace": str(materialized),
            }
        except Exception as exc:
            trajectory = _trajectory_from_runtime(task, runtime_dir)
            fallback_decision = decision or _normalize_route_decision(
                {},
                threshold=self.router_confidence_threshold,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
            return {
                "task": task,
                "trajectory": trajectory,
                "feedback": Feedback(False, 0.0, f"{type(exc).__name__}: {exc}"),
                "route_decision": fallback_decision,
                "branch_name": branch or fallback_decision.branch_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _flush_training_buffer(
        self,
        *,
        branch: str,
        buffer: list[RoutedTask],
        score_history: list[float],
        completed_tasks: int,
        total_tasks: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> int:
        evaluations = self._run_solve_buffer_parallel(buffer)
        for routed, evaluation in zip(buffer, evaluations, strict=False):
            observation = self._observation_from_evaluation(evaluation)
            decision = evaluation["route_decision"]
            branch = evaluation.get("branch_name") or routed.branch
            completed_tasks += 1
            score_history.append(observation.feedback.score)

            record = self._record_from_observation(observation)
            record["harness_tree"] = {
                "branch_name": branch,
                "route_confidence": decision.confidence,
                "route_rationale": decision.rationale,
                "fallback_reason": decision.fallback_reason,
            }
            if self.disable_main_evolve:
                self.state["main_pending"] = []
            else:
                self.state.setdefault("main_pending", []).append(record)
            branch_state = self.state["branches"][branch]
            branch_state["pending"].append(record)
            branch_state["solve_count"] += 1
            if observation.feedback.success:
                branch_state["success_count"] += 1
            branch_state["last_task_id"] = routed.task.id
            _save_state(self.state_path, self.state)
            _emit_progress(
                progress_callback,
                {
                    "phase": "train",
                    "event": "task_done",
                    "completed": completed_tasks,
                    "total": total_tasks,
                    "epoch": routed.epoch,
                    "cycle": routed.cycle,
                    "task_id": routed.task.id,
                    "branch_name": branch,
                    "route_confidence": decision.confidence,
                    "score": observation.feedback.score,
                    "success": observation.feedback.success,
                    "fallback_reason": decision.fallback_reason,
                    "feedback_detail": observation.feedback.detail,
                    "main_pending": None
                    if self.disable_main_evolve
                    else len(self.state.get("main_pending", [])),
                    "branch_pending": len(branch_state["pending"]),
                    "type_buffer_size": self.type_buffer_size,
                },
            )

            if (
                not self.disable_main_evolve
                and len(self.state.get("main_pending", [])) >= max(1, int(self.config.batch_size))
            ):
                _emit_progress(
                    progress_callback,
                    {
                        "phase": "train",
                        "event": "evolve_start",
                        "scope": "main",
                        "records": len(self.state.get("main_pending", [])),
                    },
                )
                self._evolve_main()
                _emit_progress(
                    progress_callback,
                    {
                        "phase": "train",
                        "event": "evolve_done",
                        "scope": "main",
                        "updates_completed": len(self.state.get("main_evolutions", [])),
                    },
                )

        branch_state = self.state["branches"][branch]
        if len(branch_state["pending"]) >= self.type_buffer_size:
            _emit_progress(
                progress_callback,
                {
                    "phase": "train",
                    "event": "evolve_start",
                    "scope": branch,
                    "records": len(branch_state["pending"]),
                },
            )
            self._evolve_branch(branch)
            _emit_progress(
                progress_callback,
                {
                    "phase": "train",
                    "event": "evolve_done",
                    "scope": branch,
                    "updates_completed": branch_state["evolve_count"],
                },
            )
        return completed_tasks

    def _run_solve_buffer_parallel(self, buffer: list[RoutedTask]) -> list[dict[str, Any]]:
        if not buffer:
            return []
        workers = min(resolve_agent_parallelism(self.agent), len(buffer))
        if workers <= 1:
            return [self._run_solve_for_routed_task(routed) for routed in buffer]

        agent_class = _class_path(self.agent.__class__)
        results: list[dict[str, Any] | None] = [None] * len(buffer)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, routed in enumerate(buffer):
                materialized = self._materialize_worker_workspace(routed.branch, routed.task.id)
                future = executor.submit(
                    _run_solve_worker,
                    agent_class,
                    materialized,
                    self.benchmark,
                    routed.task,
                    routed.runtime_dir,
                    routed.trace,
                    routed.route_messages,
                    routed.decision,
                    routed.branch,
                    "train",
                )
                futures[future] = (index, routed)
            for future in as_completed(futures):
                index, routed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failed_evaluation(
                        routed.task,
                        decision=routed.decision,
                        error=f"{type(exc).__name__}: {exc}",
                        branch=routed.branch,
                        runtime_dir=routed.runtime_dir,
                    )
                results[index] = result
        return [result for result in results if result is not None]

    def _run_solve_for_routed_task(self, routed: RoutedTask) -> dict[str, Any]:
        materialized = self._materialize_worker_workspace(routed.branch, routed.task.id)
        return _run_solve_worker(
            _class_path(self.agent.__class__),
            materialized,
            self.benchmark,
            routed.task,
            routed.runtime_dir,
            routed.trace,
            routed.route_messages,
            routed.decision,
            routed.branch,
            "train",
        )

    def _observation_from_evaluation(self, evaluation: dict[str, Any]) -> Observation:
        task = evaluation["task"]
        trajectory = evaluation.get("trajectory") or Trajectory(task_id=task.id, output="", steps=[])
        feedback = evaluation.get("feedback") or Feedback(
            success=False,
            score=0.0,
            detail=str(evaluation.get("error") or "missing feedback"),
        )
        return Observation(task=task, trajectory=trajectory, feedback=feedback)

    def _record_from_observation(self, observation: Observation) -> dict[str, Any]:
        return self.observer.record_from_observation(observation)

    def _evolve_main(self) -> None:
        buffer = list(self.state.get("main_pending", []))
        if not buffer:
            return
        self.observer.collect_records(buffer, suffix="main")
        self.versioning.checkout_branch(MAIN_BRANCH)
        self.agent.reload_from_fs()
        self.evolve_number += 1
        with _scope_instruction(
            self.config,
            "Update only generally reusable harness behavior in main. Do not encode branch-specific details.",
        ):
            result = self.engine.evolve(
                self.agent.workspace,
                observation_logs=buffer,
                evo_number=self.evolve_number,
            )
        self.state["main_pending"] = []
        self.state.setdefault("main_evolutions", []).append(result)
        _save_state(self.state_path, self.state)
        self.agent.reload_from_fs()

    def _evolve_branch(self, branch: str) -> None:
        branch_state = self.state["branches"][branch]
        buffer = list(branch_state["pending"])
        if not buffer:
            return
        self.observer.collect_records(buffer, suffix=f"branch_{_branch_slug(branch)}")
        materialized = self._materialize_branch_workspace(branch)
        phase_agent = self.agent.__class__(materialized)
        self.evolve_number += 1
        try:
            with _scope_instruction(
                self.config,
                (
                    f"Update only specialization for {branch}. Keep changes domain-specific; "
                    "do not duplicate generic main harness behavior unless it must be overridden."
                ),
            ):
                result = self.engine.evolve(
                    phase_agent.workspace,
                    observation_logs=buffer,
                    evo_number=self.evolve_number,
                )
        finally:
            self._restore_base_harness_snapshot()
        _save_overlay_from_workspace(
            materialized_root=materialized,
            main_root=self.workspace_root,
            overlay_dir=self._overlay_dir(branch),
        )
        branch_state["pending"] = []
        branch_state["evolve_count"] += 1
        branch_state["evolutions"].append(result)
        _save_state(self.state_path, self.state)
        phase_agent.reload_from_fs()

    def _materialize_branch_workspace(self, branch: str) -> Path:
        self._restore_base_harness_snapshot()
        destination = (
            self._external_materialized_root / _branch_slug(branch)
            if self.disable_main_evolve
            else self.workspace_root / MATERIALIZED_DIR / _branch_slug(branch)
        )
        _copy_harness_workspace(self.workspace_root, destination)
        _apply_overlay(destination, self._overlay_dir(branch))
        return destination

    def _materialize_worker_workspace(self, branch: str, task_id: str) -> Path:
        destination = self._worker_materialized_root / f"{_branch_slug(branch)}-{_safe_task_slug(task_id)}-{uuid.uuid4().hex[:8]}"
        _materialize_branch_workspace_at(
            main_root=self.workspace_root,
            branch=branch,
            destination=destination,
        )
        return destination

    def _overlay_dir(self, branch: str) -> Path:
        return self.workspace_root / OVERLAYS_DIR / _branch_slug(branch)


def sanitize_branch_name(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("branch/"):
        text = text[len("branch/") :]
    text = text.lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"[-.]+$", "", text).strip("-._")
    if not text:
        text = "general"
    return f"branch/{text[:80]}"


def _branch_slug(branch: str) -> str:
    return sanitize_branch_name(branch).removeprefix("branch/").replace("/", "__")


def _copy_harness_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative in HARNESS_PATHS:
        src = source / relative
        dst = destination / relative
        if src.is_dir():
            shutil.copytree(src, dst, ignore=_copy_ignore)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _materialize_branch_workspace_at(*, main_root: Path, branch: str, destination: Path) -> None:
    _copy_harness_workspace(main_root, destination)
    _apply_overlay(destination, main_root / OVERLAYS_DIR / _branch_slug(branch))


def _safe_task_slug(task_id: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(task_id)).strip("-._")
    return text[:80] or "task"


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def _apply_overlay(workspace: Path, overlay_dir: Path) -> None:
    deleted_path = overlay_dir / "deleted.json"
    if deleted_path.is_file():
        deleted = json.loads(deleted_path.read_text(encoding="utf-8"))
        if isinstance(deleted, list):
            for relative in deleted:
                target = workspace / str(relative)
                if _is_harness_relative(Path(str(relative))) and target.exists():
                    target.unlink()
    files_dir = overlay_dir / "files"
    if not files_dir.is_dir():
        return
    for source in files_dir.rglob("*"):
        if source.is_file():
            relative = source.relative_to(files_dir)
            if _is_harness_relative(relative):
                target = workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def _save_overlay_from_workspace(*, materialized_root: Path, main_root: Path, overlay_dir: Path) -> None:
    files_dir = overlay_dir / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    deleted: list[str] = []
    materialized_files = _harness_files(materialized_root)
    main_files = _harness_files(main_root)
    for relative in sorted(materialized_files | main_files):
        materialized_file = materialized_root / relative
        main_file = main_root / relative
        if materialized_file.is_file() and main_file.is_file():
            if materialized_file.read_bytes() == main_file.read_bytes():
                continue
            destination = files_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(materialized_file, destination)
        elif materialized_file.is_file():
            destination = files_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(materialized_file, destination)
        elif main_file.is_file():
            deleted.append(str(relative))

    if deleted:
        (overlay_dir / "deleted.json").write_text(
            json.dumps(deleted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        (overlay_dir / "deleted.json").unlink(missing_ok=True)


def _harness_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for relative in HARNESS_PATHS:
        path = root / relative
        if path.is_file():
            files.add(Path(relative))
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not _ignored_generated_file(child):
                    files.add(child.relative_to(root))
    return files


def _snapshot_harness_workspace(root: Path) -> dict[Path, bytes]:
    return {relative: (root / relative).read_bytes() for relative in _harness_files(root)}


def _restore_harness_workspace(root: Path, snapshot: dict[Path, bytes]) -> None:
    current_files = _harness_files(root)
    for relative in sorted(current_files - set(snapshot)):
        target = root / relative
        if target.is_file():
            target.unlink()
    for relative, content in snapshot.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for relative in HARNESS_PATHS:
        path = root / relative
        if not path.is_dir():
            continue
        for child in sorted((item for item in path.rglob("*") if item.is_dir()), reverse=True):
            try:
                child.rmdir()
            except OSError:
                pass


def _ignored_generated_file(path: Path) -> bool:
    return "__pycache__" in path.parts or path.name.endswith(".pyc")


def _is_harness_relative(relative: Path) -> bool:
    if relative.is_absolute() or ".." in relative.parts:
        return False
    return bool(relative.parts) and relative.parts[0] in HARNESS_PATHS


def _extract_type_router_output(trajectory: Trajectory) -> tuple[dict[str, Any], str | None]:
    for step in reversed(trajectory.steps):
        if step.get("type") != "tool_output" or step.get("name") != "type_router":
            continue
        output = step.get("output")
        if isinstance(output, dict):
            return output, None
        return {}, "type_router output was not a JSON object"
    return {}, "agent did not call type_router"


def _trajectory_from_runtime(task: Task, runtime_dir: Path) -> Trajectory:
    trace_path = runtime_dir / "trace.jsonl"
    steps: list[dict[str, Any]] = []
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                steps.append(json.loads(line))
    return Trajectory(task_id=task.id, output="", steps=steps, conversation=steps)


def _normalize_route_decision(
    raw: dict[str, Any],
    *,
    threshold: float,
    fallback_reason: str | None = None,
) -> RouteDecision:
    branch_name = sanitize_branch_name(raw.get("branch_name") or GENERAL_BRANCH)
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    if confidence < threshold:
        confidence_reason = f"router confidence {confidence:.3f} below threshold {threshold:.3f}"
        fallback_reason = f"{fallback_reason}; {confidence_reason}" if fallback_reason else confidence_reason
        branch_name = GENERAL_BRANCH
    return RouteDecision(
        branch_name=branch_name,
        confidence=confidence,
        rationale=str(raw.get("rationale") or ""),
        fallback_reason=fallback_reason,
    )


def _branch_summaries(state: dict[str, Any]) -> list[dict[str, Any]]:
    branches = state.get("branches", {})
    return [
        {
            "branch_name": item.get("name") or name,
            "rationale": item.get("rationale") or "",
            "solve_count": item.get("solve_count", 0),
            "evolve_count": item.get("evolve_count", 0),
        }
        for name, item in sorted(branches.items())
    ]


def _empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        "branches": {},
        "main_pending": [],
        "main_evolutions": [],
        "router_decisions": [],
        "final_eval": {"decisions": []},
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    data.setdefault("version", 1)
    data.setdefault("branches", {})
    data.setdefault("main_pending", [])
    data.setdefault("main_evolutions", [])
    data.setdefault("router_decisions", [])
    data.setdefault("final_eval", {"decisions": []})
    return data


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def _scope_instruction(config: EvolveConfig, instruction: str):
    sentinel = object()
    previous = config.extra.get("scope_instruction", sentinel)
    config.extra["scope_instruction"] = instruction
    try:
        yield
    finally:
        if previous is sentinel:
            config.extra.pop("scope_instruction", None)
        else:
            config.extra["scope_instruction"] = previous


def _exclude_state_dir(workspace_root: Path) -> None:
    info_dir = workspace_root / ".git" / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    exclude_path = info_dir / "exclude"
    line = f"/{STATE_DIR.as_posix()}/"
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.is_file() else ""
    if line not in existing.splitlines():
        suffix = "" if existing.endswith("\n") or not existing else "\n"
        exclude_path.write_text(f"{existing}{suffix}{line}\n", encoding="utf-8")


def _row_from_evaluation(
    result: dict[str, Any],
    branch: str,
    decision: RouteDecision,
) -> dict[str, Any]:
    task = result["task"]
    feedback = result.get("feedback")
    row = {
        "task_id": task.id,
        "branch_name": branch,
        "route_confidence": decision.confidence,
        "route_rationale": decision.rationale,
        "route_fallback_reason": decision.fallback_reason or "",
        "success": False,
        "score": 0.0,
        "detail": result.get("error") or "missing feedback",
    }
    if feedback is not None:
        row.update(
            {
                "success": feedback.success,
                "score": feedback.score,
                "detail": feedback.detail,
            }
        )
        evaluation = feedback.raw.get("evaluation") if isinstance(feedback.raw, dict) else None
        if isinstance(evaluation, dict):
            row.update(evaluation)
    return row


def _evaluation_summary(
    *,
    rows: list[dict[str, Any]],
    workspace: Path,
    split: str,
    limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    per_branch: dict[str, dict[str, Any]] = {}
    for row in rows:
        branch = str(row.get("branch_name") or GENERAL_BRANCH)
        item = per_branch.setdefault(branch, {"total": 0, "success": 0, "score_sum": 0.0})
        item["total"] += 1
        item["success"] += 1 if _as_bool(row.get("success")) else 0
        item["score_sum"] += float(row.get("score") or 0.0)
    for item in per_branch.values():
        total = int(item["total"])
        item["accuracy"] = (int(item["success"]) / total) if total else 0.0
        item["avg_score"] = (float(item["score_sum"]) / total) if total else 0.0
        item.pop("score_sum", None)

    total = len(rows)
    success = sum(1 for row in rows if _as_bool(row.get("success")))
    score_sum = sum(float(row.get("score") or 0.0) for row in rows)
    return {
        "workspace": str(workspace),
        "split": split,
        "limit": limit,
        "total": total,
        "success": success,
        "accuracy": (success / total) if total else 0.0,
        "avg_score": (score_sum / total) if total else 0.0,
        "per_branch": per_branch,
        "results_csv": str(output_dir / "results.csv"),
    }


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["task_id", "branch_name", "route_confidence", "route_rationale", "success", "score", "detail"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _run_final_two_phase_worker(
    agent_class: str,
    workspace_root: Path,
    benchmark: BenchmarkAdapter,
    task: Task,
    state: dict[str, Any],
    router_confidence_threshold: float,
    worker_materialized_root: Path,
) -> dict[str, Any]:
    cls = _import_class(agent_class)
    route_agent = cls(workspace_root)
    runtime_dir, trace = route_agent.start_task_run(task)
    decision: RouteDecision | None = None
    branch: str | None = None
    try:
        route_result = route_agent.run_phase(
            task,
            runtime_dir=runtime_dir,
            trace=trace,
            phase="final_eval:route",
            user_message=ROUTE_PHASE_USER_MESSAGE.format(
                existing_branches=json.dumps(_branch_summaries(state), ensure_ascii=False, indent=2)
            ),
            max_turns=ROUTE_PHASE_TURNS,
            stop_after_tools={"type_router"},
        )
        route_messages = route_result.messages or None
        route_trajectory = _trajectory_from_runtime(task, runtime_dir)
        raw, extraction_reason = _extract_type_router_output(route_trajectory)
        decision = _normalize_route_decision(
            raw,
            threshold=router_confidence_threshold,
            fallback_reason=extraction_reason,
        )
        branch = _reported_eval_branch_from_state(state, decision)
        materialized = (
            worker_materialized_root / f"{_branch_slug(branch)}-{_safe_task_slug(task.id)}-{uuid.uuid4().hex[:8]}"
        )
        _materialize_branch_workspace_at(
            main_root=workspace_root,
            branch=branch,
            destination=materialized,
        )
        return _run_solve_worker(
            agent_class,
            materialized,
            benchmark,
            task,
            runtime_dir,
            trace,
            route_messages,
            decision,
            branch,
            "final_eval",
        )
    except Exception as exc:
        fallback_decision = decision or _normalize_route_decision(
            {},
            threshold=router_confidence_threshold,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
        return _failed_evaluation(
            task,
            decision=fallback_decision,
            error=f"{type(exc).__name__}: {exc}",
            branch=branch or fallback_decision.branch_name,
            runtime_dir=runtime_dir,
        )


def _run_solve_worker(
    agent_class: str,
    materialized: Path,
    benchmark: BenchmarkAdapter,
    task: Task,
    runtime_dir: Path,
    trace: Any,
    route_messages: list[dict[str, Any]] | None,
    decision: RouteDecision,
    branch: str,
    phase: str,
) -> dict[str, Any]:
    cls = _import_class(agent_class)
    phase_agent = cls(materialized)
    started_at = datetime.now()
    try:
        solve_result = phase_agent.run_phase(
            task,
            runtime_dir=runtime_dir,
            trace=trace,
            phase=f"{phase}:solve",
            initial_messages=route_messages,
            user_message=SOLVE_PHASE_USER_MESSAGE.format(
                branch_name=branch,
                confidence=decision.confidence,
                rationale=decision.rationale or "n/a",
            ),
        )
        elapsed = (datetime.now() - started_at).total_seconds()
        trajectory = phase_agent.finish_task_run(
            task,
            runtime_dir=runtime_dir,
            result=solve_result,
            elapsed=elapsed,
        )
        feedback = benchmark.evaluate(task, trajectory)
        phase_agent.export_to_fs()
        return {
            "task": task,
            "trajectory": trajectory,
            "feedback": feedback,
            "route_decision": decision,
            "branch_name": branch,
            "materialized_workspace": str(materialized),
        }
    except Exception as exc:
        return _failed_evaluation(
            task,
            decision=decision,
            error=f"{type(exc).__name__}: {exc}",
            branch=branch,
            runtime_dir=runtime_dir,
        )


def _failed_evaluation(
    task: Task,
    *,
    decision: RouteDecision,
    error: str,
    branch: str | None = None,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    trajectory = _trajectory_from_runtime(task, runtime_dir) if runtime_dir is not None else Trajectory(
        task_id=task.id,
        output="",
        steps=[],
    )
    return {
        "task": task,
        "trajectory": trajectory,
        "feedback": Feedback(False, 0.0, error),
        "route_decision": decision,
        "branch_name": branch or decision.branch_name,
        "error": error,
    }


def _reported_eval_branch_from_state(state: dict[str, Any], decision: RouteDecision) -> str:
    branch = decision.branch_name
    if branch in state.get("branches", {}):
        return branch
    if GENERAL_BRANCH in state.get("branches", {}):
        return GENERAL_BRANCH
    return GENERAL_BRANCH


def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _import_class(dotted_path: str) -> type:
    module_name, qualname = dotted_path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
