"""MetaHarnessEngine -- evolution via Claude Code as proposer.

Implements the Meta-Harness search framework (Lee et al., 2026):
  - Proposer is Claude Code CLI with Opus 4.6 via Bedrock
  - Growing filesystem archive stores every candidate's source code,
    evaluation scores, and execution traces
  - The proposer browses this archive with grep/cat/ls (~10M tokens)
    rather than receiving compressed summaries in the prompt
  - k candidates per iteration with Pareto-aware selection
  - Interface validation before expensive evaluation
  - Automatic rollback when score regresses
  - Candidate archive persists across runs for cross-run transfer
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import yaml

from ...config import EvolveConfig
from ...contract.workspace import AgentWorkspace, read_jsonl
from ...engine.base import EvolutionEngine
from ...engine.history import EvolutionHistory
from ...engine.trial import TrialRunner
from ...types import Observation, StepResult
from .prompts import build_proposer_prompt

logger = logging.getLogger(__name__)

# Default model: Opus 4.6 via Claude Code CLI (same as the paper)
# Note: Claude Code CLI uses raw Bedrock model IDs without the "bedrock:" prefix
DEFAULT_MODEL = "us.anthropic.claude-opus-4-6-v1"

# Workspace files to snapshot into each candidate archive
_SNAPSHOT_DIRS = ("prompts", "skills", "memory", "tools", ".claude")
_SNAPSHOT_FILES = ("harness.py", "CLAUDE.md")
_ISOLATED_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "CLAUDE_CODE_DISABLE_TELEMETRY",
)
_CONFLICTING_PROVIDER_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
_PROPOSER_KILL_GRACE_SEC = 5


class MetaHarnessEngine(EvolutionEngine):
    """Evolution engine that uses Claude Code CLI as the proposer.

    Maintains a growing candidate archive in ``evolution/candidates/``
    that the proposer browses via filesystem access — matching the
    paper's design of full trace + source code access per candidate.

    Each evaluated candidate gets its own directory::

        evolution/candidates/cycle_003_cand_1/
        ├── snapshot/          # workspace files at time of proposal
        │   ├── prompts/
        │   ├── skills/
        │   ├── harness.py
        │   └── ...
        ├── scores.json        # evaluation results {score, cost, selected, valid, ...}
        └── traces/            # symlink or copy of observation batch

    Features matching the paper:
      - Full benchmark evaluation per candidate (eval_sample_size=0 → all tasks)
      - Interface validation before expensive evaluation (Algorithm 1 line 11)
      - Pareto frontier tracking across (score, cost) objectives
      - Candidate archive persists across runs (cross-run knowledge transfer)
      - Initial population evaluation handled by A-Evolve's loop (cycle 0)
    """

    def __init__(
        self,
        config: EvolveConfig,
        proposer_env_factory: Callable[[], dict[str, str]] | None = None,
        eval_factory: Callable[[Path], TrialRunner] | None = None,
    ):
        self.config = config
        self.harness_enabled: bool = config.extra.get("harness_enabled", False)
        self.model: str = config.extra.get("proposer_model", DEFAULT_MODEL)
        self.max_turns: int = config.extra.get("proposer_max_turns", 50)
        self.timeout_sec: int = config.extra.get("proposer_timeout_sec", 900)
        self.tool_timeout_sec: int = int(
            config.extra.get("proposer_tool_timeout_sec", 60)
        )
        if self.tool_timeout_sec <= 0:
            raise ValueError("proposer_tool_timeout_sec must be positive")
        proposer_config_dir = config.extra.get("proposer_config_dir")
        self.proposer_config_dir = (
            Path(proposer_config_dir).expanduser().absolute()
            if proposer_config_dir
            else None
        )
        self.proposer_provider: str = config.extra.get("proposer_provider", "default")
        self._proposer_env_factory = proposer_env_factory
        self._eval_factory = eval_factory
        # Multi-candidate: generate k variants per cycle (paper: typically 2)
        self.num_candidates: int = config.extra.get("num_candidates", 2)
        # Evaluation sample size: 0 = all tasks (paper default), >0 = subsample
        self.eval_sample_size: int = config.extra.get("eval_sample_size", 0)
        self.eval_split: str = config.extra.get("eval_split", "train")
        # Rollback: revert if best candidate scores below current best.
        # Default False to match the paper — Meta-Harness stores all
        # candidates and allows temporary regressions for exploration.
        # The Pareto frontier tracks the best across all cycles.
        self.rollback_on_regression: bool = config.extra.get(
            "rollback_on_regression", False
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: EvolutionHistory,
        trial: TrialRunner,
        tasks: list | None = None,
        eval_factory: Callable[[Path], TrialRunner] | None = None,
    ) -> StepResult:
        """Run one Meta-Harness evolution step (Algorithm 1 inner loop).

        Phase A — Propose (serial): for each of k candidates, run the
        Claude Code proposer, capture the diff and snapshot, then reset.

        Phase B — Evaluate in isolated workspace copies.  An explicit
        eval_factory enables parallel evaluation; otherwise candidates use
        isolated copies sequentially with the current agent type.

        Phase C — Select: Pareto-aware selection, apply or rollback.

        Args:
            eval_factory: Optional callable ``(workspace_path) -> TrialRunner``.
                When provided and num_candidates > 1, candidates are evaluated
                in parallel.  Otherwise isolated candidates are evaluated
                sequentially using the current agent type and benchmark.

        Note: Initial population evaluation (Algorithm 1 lines 3-5) is
        handled by the A-Evolve loop's first SOLVE→OBSERVE cycle before
        engine.step() is called.
        """
        cycle_num = history.latest_cycle + 1
        score_curve = history.get_score_curve()
        current_best = max(score_curve) if score_curve else 0.0
        evaluation_tasks = self._resolve_evaluation_tasks(trial, tasks)

        candidates_dir = workspace.root / "evolution" / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        # Count existing candidates (includes prior runs — cross-run archive)
        existing = len([
            d for d in candidates_dir.iterdir() if d.is_dir()
        ])

        parallel = eval_factory is not None and self.num_candidates > 1

        # ==============================================================
        # Phase A — Propose all candidates (serial)
        # ==============================================================
        proposed: list[dict[str, Any]] = []

        try:
            for i in range(self.num_candidates):
                if i > 0:
                    self._git_reset(workspace.root)

                cand_label = f"cycle_{cycle_num:03d}_cand_{i}"

                prompt = build_proposer_prompt(
                    workspace,
                    cycle_num,
                    score_curve,
                    turn_budget=self.max_turns,
                    harness_enabled=self.harness_enabled,
                    candidate_index=i,
                    num_candidates=self.num_candidates,
                    num_archived=existing + len(proposed),
                )

                result = self._run_claude_code(prompt, workspace.root)
                if result.get("error") or result.get("timed_out"):
                    error = str(result.get("error") or (
                        f"Claude Code proposer timed out after {self.timeout_sec}s: "
                        f"{result.get('stderr') or 'no diagnostic output'}"
                    ))
                    failure_stage = str(
                        result.get("failure_stage")
                        or ("proposer_timeout" if result.get("timed_out") else "proposer_failure")
                    )
                    self._mark_archived_cycle_skipped(
                        candidates_dir,
                        proposed,
                        error,
                    )
                    self._archive_candidate_from_snapshot(
                        workspace,
                        candidates_dir / cand_label,
                        {},
                        0.0,
                        0.0,
                        cycle_num,
                        i,
                        result,
                        valid=False,
                        validation_err=error,
                        diagnostics={
                            "proposal_valid": False,
                            "proposal_validation_error": error,
                            "failure_stage": failure_stage,
                            "error": error,
                            "selection_attempted": False,
                            "final_apply_succeeded": False,
                            "final_apply_error": "",
                        },
                    )
                    logger.warning(
                        "Skipping Meta-Harness cycle %d after %s failed: %s",
                        cycle_num, cand_label, error,
                    )
                    metadata = {
                        "cycle": cycle_num,
                        "cycle_skipped": True,
                        "failure_stage": failure_stage,
                        "failed_candidate": cand_label,
                        "proposer_exit_code": result.get("exit_code"),
                        "proposer_error": error,
                        "proposed_before_failure": len(proposed),
                    }
                    if result.get("timed_out"):
                        metadata["proposer_timeout_sec"] = self.timeout_sec
                    return StepResult(
                        mutated=False,
                        summary=(
                            f"MetaHarness cycle {cycle_num}: skipped after "
                            f"{cand_label} proposer failure: {error}"
                        ),
                        metadata=metadata,
                    )
                diff = self._git_diff(workspace.root)
                valid, validation_err = self._validate_candidate(workspace)

                # Regex audit for task-specific string leakage (paper §4.3)
                task_ids = [task.id for task in evaluation_tasks]
                leakage = self._audit_leakage(workspace, task_ids)
                if leakage:
                    logger.warning(
                        "Leakage audit for %s: %s", cand_label, "; ".join(leakage),
                    )
                    valid = False
                    validation_err = (
                        (validation_err + "; " if validation_err else "")
                        + "leakage: " + "; ".join(leakage)
                    )

                # Snapshot workspace state for archiving (before reset)
                snapshot_files = self._capture_snapshot(workspace)

                proposal = {
                    "index": i,
                    "label": cand_label,
                    "diff": diff,
                    "valid": valid,
                    "validation_err": validation_err,
                    "proposer_result": result,
                    "snapshot_files": snapshot_files,
                }
                proposed.append(proposal)
                self._archive_candidate_from_snapshot(
                    workspace,
                    candidates_dir / cand_label,
                    snapshot_files,
                    0.0,
                    0.0,
                    cycle_num,
                    i,
                    result,
                    valid=valid,
                    validation_err=validation_err,
                    diff=diff,
                )
                logger.info(
                    "Proposed %s: valid=%s (%d chars diff)",
                    cand_label, valid, len(diff),
                )
        finally:
            self._git_reset(workspace.root)

        isolated_eval_factory = eval_factory or self._eval_factory
        if isolated_eval_factory is None:
            agent_type = type(trial.agent)

            def isolated_eval_factory(path: Path) -> TrialRunner:
                return TrialRunner(agent_type(path), trial.benchmark)

        candidates = self._evaluate_candidates(
            proposed,
            workspace,
            candidates_dir,
            isolated_eval_factory,
            evaluation_tasks,
            parallel=parallel,
        )

        # -- Selection --
        # Filter to valid candidates with finite scores
        valid_candidates = [
            c for c in candidates
            if c["valid"]
            and math.isfinite(c["score"])
            and math.isfinite(c["cost"])
        ]

        if not valid_candidates:
            logger.warning("All %d candidates failed validation", len(candidates))
            self._git_reset(workspace.root)
            return StepResult(
                mutated=False,
                summary=(
                    f"MetaHarness cycle {cycle_num}: "
                    f"all {self.num_candidates} candidates failed validation"
                ),
                metadata={
                    "cycle": cycle_num,
                    "num_candidates": self.num_candidates,
                    "all_invalid": True,
                    "validation_errors": [c["validation_err"] for c in candidates],
                    "proposer_model": self.model,
                    "evaluation_split": self.eval_split,
                },
            )

        # Compute Pareto frontier across (score↑, cost↓)
        frontier = _pareto_frontier(valid_candidates)
        # Select the highest-scoring candidate from the frontier
        best = max(frontier, key=lambda c: c["score"])

        logger.info(
            "Selected %s (score=%.3f, cost=%d) from %d candidates "
            "(%d on Pareto frontier)",
            best["label"], best["score"], best["cost"],
            len(candidates), len(frontier),
        )

        self._git_reset(workspace.root)
        for candidate in candidates:
            self._update_candidate_selection(
                candidates_dir / candidate["label"],
                selected=False,
                pareto_optimal=candidate in frontier,
            )

        if self.rollback_on_regression and best["score"] < current_best:
            logger.info(
                "Best candidate %.3f < current best %.3f — rolling back",
                best["score"], current_best,
            )
            return StepResult(
                mutated=False,
                summary=(
                    f"MetaHarness cycle {cycle_num}: "
                    f"{len(valid_candidates)} valid candidates evaluated, "
                    f"rolled back (best={best['score']:.3f}, "
                    f"current={current_best:.3f})"
                ),
                metadata={
                    "cycle": cycle_num,
                    "rolled_back": True,
                    "num_candidates": self.num_candidates,
                    "num_valid": len(valid_candidates),
                    "candidate_scores": [c["score"] for c in candidates],
                    "pareto_frontier": [c["label"] for c in frontier],
                    "selected": None,
                    "current_best": current_best,
                    "proposer_model": self.model,
                    "evaluation_split": self.eval_split,
                },
            )

        final_apply_error = ""
        try:
            self._apply_diff(workspace.root, best["diff"])
            deployed_valid, deployed_error = self._validate_candidate(workspace)
            if not deployed_valid:
                raise RuntimeError(deployed_error)
        except Exception as exc:
            final_apply_error = str(exc)
            self._git_reset(workspace.root)
            trial.agent.reload_from_fs()
            self._update_candidate_selection(
                candidates_dir / best["label"],
                selected=False,
                pareto_optimal=True,
                selection_attempted=True,
                final_apply_succeeded=False,
                final_apply_error=final_apply_error,
            )
            return StepResult(
                mutated=False,
                summary=(
                    f"MetaHarness cycle {cycle_num}: selected candidate "
                    f"failed final apply"
                ),
                metadata={
                    "cycle": cycle_num,
                    "num_candidates": self.num_candidates,
                    "num_valid": len(valid_candidates),
                    "selected": None,
                    "selection_attempted": best["label"],
                    "final_apply_error": final_apply_error,
                    "candidate_scores": [c["score"] for c in candidates],
                    "pareto_frontier": [c["label"] for c in frontier],
                    "proposer_model": self.model,
                    "evaluation_split": self.eval_split,
                },
            )

        self._update_candidate_selection(
            candidates_dir / best["label"],
            selected=True,
            pareto_optimal=True,
            selection_attempted=True,
            final_apply_succeeded=True,
        )
        changes_summary = (
            f"selected {best['label']} "
            f"(score={best['score']:.3f}, cost={best['cost']})"
            if best["diff"]
            else "no mutation"
        )

        return StepResult(
            mutated=bool(best["diff"]),
            summary=f"MetaHarness cycle {cycle_num}: {changes_summary}",
            metadata={
                "cycle": cycle_num,
                "num_candidates": self.num_candidates,
                "num_valid": len(valid_candidates),
                "selected": best["label"],
                "candidate_scores": [c["score"] for c in candidates],
                "candidate_costs": [c["cost"] for c in candidates],
                "pareto_frontier": [c["label"] for c in frontier],
                "best_score": best["score"],
                "best_cost": best["cost"],
                "harness_enabled": self.harness_enabled,
                "proposer_model": self.model,
                "evaluation_split": self.eval_split,
                "total_archived": existing + len(candidates),
            },
        )

    # ------------------------------------------------------------------
    # Phase B helpers — isolated candidate evaluation
    # ------------------------------------------------------------------

    def _failed_candidate(
        self,
        proposal: dict[str, Any],
        stage: str,
        error: str,
    ) -> dict[str, Any]:
        return {
            "index": proposal["index"],
            "label": proposal["label"],
            "score": 0.0,
            "cost": 0.0,
            "diff": proposal["diff"],
            "valid": False,
            "validation_err": error,
            "failure_stage": stage,
            "exit_code": proposal["proposer_result"].get("exit_code"),
            "output_chars": len(proposal["proposer_result"].get("output", "")),
        }

    @staticmethod
    def _mark_archived_cycle_skipped(
        candidates_dir: Path,
        proposed: list[dict[str, Any]],
        error: str,
    ) -> None:
        for proposal in proposed:
            cand_dir = candidates_dir / proposal["label"]
            diagnostics_path = cand_dir / "diagnostics.json"
            if diagnostics_path.exists():
                diagnostics = json.loads(diagnostics_path.read_text())
                diagnostics.update({
                    "failure_stage": "cycle_skipped",
                    "error": error,
                })
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

            scores_path = cand_dir / "scores.json"
            if scores_path.exists():
                scores = json.loads(scores_path.read_text())
                scores.update({
                    "failure_stage": "cycle_skipped",
                    "validation_error": error,
                })
                scores_path.write_text(json.dumps(scores, indent=2))

    def _evaluate_isolated_candidate(
        self,
        proposal: dict[str, Any],
        workspace: AgentWorkspace,
        eval_factory: Callable[[Path], TrialRunner],
        tasks: list | None,
    ) -> dict[str, Any]:
        if not proposal["valid"]:
            return self._failed_candidate(
                proposal,
                "proposal_validation",
                proposal["validation_err"],
            )

        with tempfile.TemporaryDirectory(prefix=f"mh_{proposal['label']}_") as tmp:
            tmp_workspace = Path(tmp) / "workspace"
            shutil.copytree(
                workspace.root,
                tmp_workspace,
                symlinks=True,
                ignore=shutil.ignore_patterns("evolution"),
            )

            try:
                self._assert_workspace_clean(tmp_workspace)
                self._apply_diff(tmp_workspace, proposal["diff"])
            except Exception as exc:
                return self._failed_candidate(proposal, "apply", str(exc))

            replay_workspace = AgentWorkspace(tmp_workspace)
            valid, validation_err = self._validate_candidate(replay_workspace)
            if not valid:
                return self._failed_candidate(
                    proposal,
                    "post_apply_validation",
                    validation_err,
                )

            try:
                eval_trial = eval_factory(tmp_workspace)
                eval_result = self._evaluate_candidate(eval_trial, tasks=tasks)
            except Exception as exc:
                return self._failed_candidate(proposal, "evaluation", str(exc))

            return {
                "index": proposal["index"],
                "label": proposal["label"],
                "score": eval_result["score"],
                "cost": eval_result["cost"],
                "diff": proposal["diff"],
                "valid": True,
                "validation_err": "",
                "failure_stage": None,
                "exit_code": proposal["proposer_result"].get("exit_code"),
                "output_chars": len(proposal["proposer_result"].get("output", "")),
            }

    def _evaluate_candidates(
        self,
        proposed: list[dict[str, Any]],
        workspace: AgentWorkspace,
        candidates_dir: Path,
        eval_factory: Callable[[Path], TrialRunner],
        tasks: list | None,
        *,
        parallel: bool,
    ) -> list[dict[str, Any]]:
        if parallel:
            logger.info("Evaluating %d candidates in parallel", len(proposed))
            with ThreadPoolExecutor(max_workers=len(proposed)) as pool:
                futures = {
                    pool.submit(
                        self._evaluate_isolated_candidate,
                        proposal,
                        workspace,
                        eval_factory,
                        tasks,
                    ): proposal
                    for proposal in proposed
                }
                results = []
                for future in as_completed(futures):
                    proposal = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(
                            self._failed_candidate(proposal, "evaluation", str(exc))
                        )
        else:
            results = [
                self._evaluate_isolated_candidate(
                    proposal,
                    workspace,
                    eval_factory,
                    tasks,
                )
                for proposal in proposed
            ]

        results.sort(key=lambda candidate: candidate["index"])
        for result in results:
            cand_dir = candidates_dir / result["label"]
            if not cand_dir.exists():
                continue
            diagnostics = {
                "failure_stage": result["failure_stage"],
                "error": result["validation_err"],
                "selection_attempted": False,
                "final_apply_succeeded": False,
                "final_apply_error": "",
            }
            diagnostics_path = cand_dir / "diagnostics.json"
            if diagnostics_path.exists():
                existing_diag = json.loads(diagnostics_path.read_text())
                existing_diag.update(diagnostics)
                diagnostics = existing_diag
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

            scores_path = cand_dir / "scores.json"
            if scores_path.exists():
                scores = json.loads(scores_path.read_text())
                scores["score"] = result["score"]
                scores["cost"] = result["cost"]
                scores["valid"] = result["valid"]
                scores["validation_error"] = result["validation_err"]
                scores["failure_stage"] = result["failure_stage"]
                scores_path.write_text(json.dumps(scores, indent=2))
            logger.info(
                "Candidate %s: valid=%s, score=%.3f, cost=%d",
                result["label"],
                result["valid"],
                result["score"],
                result["cost"],
            )
        return results

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _capture_snapshot(self, workspace: AgentWorkspace) -> dict[str, bytes]:
        """Capture mutable workspace files as in-memory bytes for later archiving."""
        snapshot: dict[str, bytes] = {}
        for dirname in _SNAPSHOT_DIRS:
            src = workspace.root / dirname
            if src.exists():
                for f in src.rglob("*"):
                    if f.is_file():
                        rel = str(f.relative_to(workspace.root))
                        snapshot[rel] = f.read_bytes()
        for fname in _SNAPSHOT_FILES:
            src = workspace.root / fname
            if src.exists():
                snapshot[fname] = src.read_bytes()
        return snapshot

    def _archive_candidate_from_snapshot(
        self,
        workspace: AgentWorkspace,
        cand_dir: Path,
        snapshot_files: dict[str, bytes],
        score: float,
        cost: int | float,
        cycle: int,
        cand_index: int,
        proposer_result: dict[str, Any],
        *,
        valid: bool = True,
        validation_err: str = "",
        diff: str = "",
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Archive candidate source, patch, proposer output, and diagnostics."""
        cand_dir.mkdir(parents=True, exist_ok=True)

        snapshot_dir = cand_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        for rel_path, content in snapshot_files.items():
            dest = snapshot_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        (cand_dir / "candidate.patch").write_text(diff)
        (cand_dir / "proposer.json").write_text(
            json.dumps(
                {
                    "exit_code": proposer_result.get("exit_code"),
                    "output": proposer_result.get("output", ""),
                    "stderr": proposer_result.get("stderr", ""),
                    "failure_stage": proposer_result.get("failure_stage"),
                    "error": proposer_result.get("error", ""),
                    "model": self.model,
                    "provider": self.proposer_provider,
                },
                indent=2,
            )
        )
        diagnostics_data = diagnostics or {
            "proposal_valid": valid,
            "proposal_validation_error": validation_err,
            "failure_stage": "proposal_validation" if not valid else None,
            "error": validation_err,
            "selection_attempted": False,
            "final_apply_succeeded": False,
            "final_apply_error": "",
        }
        (cand_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics_data, indent=2)
        )

        scores_data = {
            "cycle": cycle,
            "candidate_index": cand_index,
            "score": score,
            "cost": cost,
            "valid": valid,
            "validation_error": validation_err,
            "failure_stage": diagnostics_data.get("failure_stage"),
            "selected": False,
            "pareto_optimal": False,
            "proposer_model": self.model,
            "evaluation_split": self.eval_split,
            "proposer_exit_code": proposer_result.get("exit_code"),
        }
        (cand_dir / "scores.json").write_text(json.dumps(scores_data, indent=2))

        traces_dir = cand_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        obs_dir = workspace.root / "evolution" / "observations"
        if obs_dir.exists():
            batches = sorted(obs_dir.glob("batch_*.jsonl"))
            if batches:
                latest_batch = batches[-1]
                link_target = traces_dir / latest_batch.name
                if not link_target.exists() and not link_target.is_symlink():
                    try:
                        link_target.symlink_to(latest_batch.resolve())
                    except OSError:
                        shutil.copy2(latest_batch, link_target)

        logger.debug("Archived candidate to %s", cand_dir)

    def _update_candidate_selection(
        self,
        cand_dir: Path,
        *,
        selected: bool,
        pareto_optimal: bool,
        selection_attempted: bool = False,
        final_apply_succeeded: bool = False,
        final_apply_error: str = "",
    ) -> None:
        scores_path = cand_dir / "scores.json"
        scores = json.loads(scores_path.read_text())
        scores["selected"] = selected
        scores["pareto_optimal"] = pareto_optimal
        scores["final_apply_succeeded"] = final_apply_succeeded
        scores_path.write_text(json.dumps(scores, indent=2))

        diagnostics_path = cand_dir / "diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["selection_attempted"] = selection_attempted
        diagnostics["final_apply_succeeded"] = final_apply_succeeded
        diagnostics["final_apply_error"] = final_apply_error
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    # ------------------------------------------------------------------
    # Interface validation (Algorithm 1 line 11)
    # ------------------------------------------------------------------

    def _validate_candidate(
        self, workspace: AgentWorkspace,
    ) -> tuple[bool, str]:
        """Validate a candidate before evaluation or deployment."""
        errors: list[str] = []

        try:
            self._assert_index_clean(workspace.root)
        except RuntimeError as exc:
            errors.append(str(exc))

        python_files = [workspace.root / "harness.py"]
        tools_dir = workspace.root / "tools"
        if tools_dir.exists():
            python_files.extend(tools_dir.rglob("*.py"))
        for py_file in python_files:
            if not py_file.exists():
                continue
            try:
                source = py_file.read_text()
                compile(source, str(py_file), "exec")
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                rel = py_file.relative_to(workspace.root)
                errors.append(f"{rel}: {exc}")

        prompt_path = workspace.root / "prompts" / "system.md"
        try:
            if not prompt_path.exists() or not prompt_path.read_text().strip():
                errors.append("prompts/system.md is empty")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"prompts/system.md: {exc}")

        text_files: list[Path] = []
        for dirname in _SNAPSHOT_DIRS:
            directory = workspace.root / dirname
            if directory.exists():
                text_files.extend(path for path in directory.rglob("*") if path.is_file())
        for filename in _SNAPSHOT_FILES:
            path = workspace.root / filename
            if path.is_file():
                text_files.append(path)

        for path in text_files:
            try:
                lines = path.read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if (
                    line.startswith("<<<<<<< ")
                    or line == "======="
                    or line.startswith(">>>>>>> ")
                ):
                    rel = path.relative_to(workspace.root)
                    errors.append(f"{rel}:{line_number}: git conflict marker")
                    break

        memory_dir = workspace.root / "memory"
        if memory_dir.exists():
            for jsonl_path in sorted(memory_dir.rglob("*.jsonl")):
                try:
                    read_jsonl(jsonl_path)
                except ValueError as exc:
                    errors.append(str(exc))

        if tools_dir.exists():
            yaml_paths = sorted(tools_dir.rglob("*.yaml")) + sorted(
                tools_dir.rglob("*.yml")
            )
            for yaml_path in yaml_paths:
                rel = yaml_path.relative_to(workspace.root)
                try:
                    parsed = yaml.safe_load(yaml_path.read_text())
                except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                    errors.append(f"{rel}: invalid YAML: {exc}")
                    continue
                if parsed is not None and not isinstance(parsed, dict):
                    errors.append(f"{rel}: expected YAML mapping")
                    continue
                if rel.as_posix() == "tools/registry.yaml":
                    tools = (parsed or {}).get("tools", [])
                    if not isinstance(tools, list):
                        errors.append("tools/registry.yaml: 'tools' must be a list")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _audit_leakage(
        self,
        workspace: AgentWorkspace,
        task_ids: list[str],
    ) -> list[str]:
        """Regex audit for task-specific string leakage (paper §4.3).

        Scans workspace files (prompts, skills, harness, tools) for
        hardcoded task IDs.  Returns list of warnings (empty = clean).
        """
        if not task_ids:
            return []

        import re

        # Collect text from all mutable workspace files
        texts: list[tuple[str, str]] = []  # (filename, content)
        for d in _SNAPSHOT_DIRS:
            d_path = workspace.root / d
            if d_path.exists():
                for f in d_path.rglob("*"):
                    if f.is_file():
                        try:
                            texts.append((str(f.relative_to(workspace.root)), f.read_text()))
                        except (UnicodeDecodeError, OSError):
                            pass
        for f_name in _SNAPSHOT_FILES:
            f_path = workspace.root / f_name
            if f_path.exists():
                try:
                    texts.append((f_name, f_path.read_text()))
                except (UnicodeDecodeError, OSError):
                    pass

        warnings: list[str] = []
        for task_id in task_ids:
            # Skip very short IDs that would cause false positives
            if len(task_id) < 8:
                continue
            pattern = re.escape(task_id)
            for filename, content in texts:
                if re.search(pattern, content):
                    warnings.append(f"task ID '{task_id}' found in {filename}")
                    break  # one warning per task ID

        return warnings

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def _resolve_evaluation_tasks(
        self, trial: TrialRunner | None, tasks: list | None = None,
    ) -> list:
        if tasks is not None:
            return tasks
        if trial is None:
            return []
        limit = self.eval_sample_size if self.eval_sample_size > 0 else 10000
        return trial.get_tasks(split=self.eval_split, limit=limit)

    def _evaluate_candidate(
        self, trial: TrialRunner | None, tasks: list | None = None,
    ) -> dict[str, Any]:
        """Evaluate a candidate on benchmark tasks.

        Returns dict with 'score' and 'cost' (total tokens).
        If tasks are provided, uses them directly. Otherwise falls back
        to loading from trial runner (eval_sample_size controls limit).
        """
        if trial is None:
            raise RuntimeError("candidate evaluation requires a TrialRunner")

        tasks = self._resolve_evaluation_tasks(trial, tasks)
        if not tasks:
            raise RuntimeError("candidate evaluation has no tasks")

        obs = trial.run_tasks(tasks)
        if len(obs) != len(tasks):
            raise RuntimeError(
                f"candidate evaluation produced {len(obs)} observations "
                f"for {len(tasks)} tasks"
            )

        score = sum(o.feedback.score for o in obs) / len(obs)

        total_tokens = 0
        for observation in obs:
            for step in observation.trajectory.steps:
                usage = step.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)

        return {"score": score, "cost": total_tokens}



    # ------------------------------------------------------------------
    # Claude Code CLI invocation
    # ------------------------------------------------------------------

    def _isolated_proposer_env(self) -> tuple[dict[str, str] | None, tuple[str, ...]]:
        if self.proposer_config_dir is None:
            if self._proposer_env_factory is not None:
                raise ValueError(
                    "proposer_env_factory requires proposer_config_dir"
                )
            return None, ()
        if not self.model.strip():
            raise ValueError("proposer_model must not be empty")
        if (
            self.proposer_config_dir.is_symlink()
            or not self.proposer_config_dir.is_dir()
        ):
            raise ValueError(
                f"proposer_config_dir must be a real directory: {self.proposer_config_dir}"
            )
        if self._proposer_env_factory is None:
            raise ValueError(
                "isolated proposer_config_dir requires proposer_env_factory"
            )

        provider_env = self._proposer_env_factory()
        if not isinstance(provider_env, dict):
            raise TypeError("proposer_env_factory must return a dict")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in provider_env.items()
        ):
            raise TypeError("proposer environment keys and values must be strings")
        for key in (
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
        ):
            if not provider_env.get(key, "").strip():
                raise ValueError(f"isolated proposer environment requires {key}")
        if provider_env["ANTHROPIC_MODEL"] != self.model:
            raise ValueError(
                "proposer model changed after startup; restart the evolution run"
            )

        env = {
            key: os.environ[key]
            for key in _ISOLATED_ENV_KEYS
            if key in os.environ
        }
        env.update(provider_env)
        env["CLAUDE_CONFIG_DIR"] = str(self.proposer_config_dir)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        for key in _CONFLICTING_PROVIDER_ENV_KEYS:
            env.pop(key, None)

        sensitive_values = tuple(
            value
            for key, value in provider_env.items()
            if value
            and any(
                marker in key.upper()
                for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD")
            )
        )
        return env, sensitive_values

    def _audit_isolated_candidate(
        self,
        workspace_root: Path,
        sensitive_values: tuple[str, ...],
    ) -> None:
        if not sensitive_values:
            return

        for dirname in _SNAPSHOT_DIRS:
            source = workspace_root / dirname
            if source.is_symlink():
                raise RuntimeError("Claude Code proposer created a symlink")
            if not source.exists():
                continue
            for path in source.rglob("*"):
                if path.is_symlink():
                    raise RuntimeError("Claude Code proposer created a symlink")
                if path.is_file():
                    content = path.read_bytes()
                    if any(value.encode() in content for value in sensitive_values):
                        raise RuntimeError(
                            "Claude Code proposer wrote sensitive provider data"
                        )

        for filename in _SNAPSHOT_FILES:
            path = workspace_root / filename
            if path.is_symlink():
                raise RuntimeError("Claude Code proposer created a symlink")
            if path.is_file():
                content = path.read_bytes()
                if any(value.encode() in content for value in sensitive_values):
                    raise RuntimeError(
                        "Claude Code proposer wrote sensitive provider data"
                    )

        diff = self._git_diff(workspace_root)
        if any(value in diff for value in sensitive_values):
            raise RuntimeError("Claude Code proposer wrote sensitive provider data")
        if any(line.endswith("mode 120000") for line in diff.splitlines()):
            raise RuntimeError("Claude Code proposer created a symlink")

    @staticmethod
    def _redact(text: str, sensitive_values: tuple[str, ...]) -> str:
        redacted = text
        for value in sorted(set(sensitive_values), key=len, reverse=True):
            redacted = redacted.replace(value, "[REDACTED]")
        return redacted

    def _run_claude_code(self, prompt: str, workspace_root: Path) -> dict[str, Any]:
        """Invoke Claude Code CLI as the proposer."""
        proposer_env, sensitive_values = self._isolated_proposer_env()
        cmd = [
            "claude",
            "-p", prompt,
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--no-session-persistence",
        ]
        if proposer_env is not None:
            cmd.extend(["--setting-sources", "user"])

        logger.info(
            "Running Claude Code proposer (model=%s, provider=%s, config_dir=%s)",
            self.model,
            self.proposer_provider,
            self.proposer_config_dir or "default",
        )

        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "cwd": str(workspace_root),
            "start_new_session": True,
        }
        process_env = dict(proposer_env or os.environ)
        tool_timeout_ms = str(self.tool_timeout_sec * 1000)
        process_env["BASH_DEFAULT_TIMEOUT_MS"] = tool_timeout_ms
        process_env["BASH_MAX_TIMEOUT_MS"] = tool_timeout_ms
        popen_kwargs["env"] = process_env

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
            try:
                stdout, raw_stderr = proc.communicate(timeout=self.timeout_sec)
            except subprocess.TimeoutExpired as exc:
                stdout, raw_stderr = self._terminate_process_group(proc)
                if not stdout:
                    stdout = self._subprocess_text(exc.stdout)
                if not raw_stderr:
                    raw_stderr = self._subprocess_text(exc.stderr)
                stderr = self._redact(raw_stderr.strip(), sensitive_values)
                detail = stderr or "no diagnostic output"
                logger.error(
                    "Claude Code timed out after %ds: %s",
                    self.timeout_sec,
                    detail[-1000:],
                )
                return {
                    "output": self._redact(stdout.strip(), sensitive_values),
                    "stderr": stderr or "TIMEOUT",
                    "exit_code": -1,
                    "timed_out": True,
                    "failure_stage": "proposer_timeout",
                    "error": (
                        f"Claude Code proposer timed out after {self.timeout_sec}s: "
                        f"{detail}"
                    ),
                }

            output = stdout.strip()
            stderr = self._redact(raw_stderr.strip(), sensitive_values)

            parsed: Any = None
            result_text: Any = output
            try:
                parsed = json.loads(output)
                result_text = parsed.get("result", output)
            except (json.JSONDecodeError, TypeError):
                pass

            reported_error = isinstance(parsed, dict) and parsed.get("is_error") is True
            if proc.returncode != 0 or reported_error:
                logger.warning("Claude Code exited with code %d", proc.returncode)
                detail = stderr or self._redact(output, sensitive_values)
                error = (
                    f"Claude Code proposer failed with exit code {proc.returncode}: "
                    f"{detail[-1000:] or 'no diagnostic output'}"
                )
                return {
                    "output": self._redact(str(result_text), sensitive_values),
                    "stderr": stderr,
                    "exit_code": proc.returncode,
                    "failure_stage": "proposer_exit",
                    "error": error,
                }
            elif proposer_env is not None:
                self._audit_isolated_candidate(workspace_root, sensitive_values)

            if proposer_env is not None:
                result_text = self._redact(str(result_text), sensitive_values)

            logger.info("Claude Code finished (exit=%d)", proc.returncode)
            return {
                "output": result_text,
                "stderr": stderr,
                "exit_code": proc.returncode,
            }

        except OSError as exc:
            error = (
                "Claude Code CLI not found"
                if isinstance(exc, FileNotFoundError)
                else f"Claude Code proposer failed to start: {exc}"
            )
            logger.error(error)
            return {
                "output": "",
                "stderr": str(exc),
                "exit_code": -1,
                "failure_stage": "proposer_launch",
                "error": error,
            }

    @staticmethod
    def _subprocess_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    @staticmethod
    def _terminate_process_group(
        proc: subprocess.Popen[str],
    ) -> tuple[str, str]:
        """Terminate Claude Code and any tool processes it spawned."""
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            return proc.communicate(timeout=_PROPOSER_KILL_GRACE_SEC)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return proc.communicate()

    # ------------------------------------------------------------------
    # Git helpers for multi-candidate workflow
    # ------------------------------------------------------------------

    def _run_git(
        self,
        root: Path,
        *args: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            ["git", *args],
            input=input_text,
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "unknown git error"
            raise RuntimeError(
                f"git {' '.join(args)} failed in {root}: {detail[:1000]}"
            )
        return proc

    def _assert_index_clean(self, root: Path) -> None:
        staged = self._run_git(
            root,
            "diff",
            "--cached",
            "--name-only",
            "--",
            ".",
            ":(exclude)evolution/",
        ).stdout.strip()
        unmerged = self._run_git(root, "ls-files", "--unmerged").stdout.strip()
        if staged or unmerged:
            details = []
            if staged:
                details.append(f"staged files: {staged}")
            if unmerged:
                details.append(f"unmerged files: {unmerged}")
            raise RuntimeError(f"workspace index is not clean: {'; '.join(details)}")

    def _assert_workspace_clean(self, root: Path) -> None:
        status = self._run_git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude)evolution/",
        ).stdout.strip()
        if status:
            raise RuntimeError(f"workspace is not clean: {status[:1000]}")
        self._assert_index_clean(root)

    def _git_reset(self, root: Path) -> None:
        """Restore non-evolution paths to HEAD while preserving run artifacts."""
        self._run_git(
            root,
            "reset",
            "--quiet",
            "HEAD",
            "--",
            ".",
            ":(exclude)evolution/",
        )
        self._run_git(
            root,
            "checkout",
            "--",
            ".",
            ":(exclude)evolution/",
        )
        self._run_git(root, "clean", "-fd", "--exclude=evolution/")
        self._assert_workspace_clean(root)

    def _git_diff(self, root: Path) -> str:
        """Capture non-evolution workspace changes without retaining index state."""
        self._run_git(
            root,
            "add",
            "-A",
            "--",
            ".",
            ":(exclude)evolution/",
        )
        try:
            return self._run_git(
                root,
                "diff",
                "--cached",
                "--",
                ".",
                ":(exclude)evolution/",
            ).stdout
        finally:
            self._run_git(
                root,
                "reset",
                "--quiet",
                "HEAD",
                "--",
                ".",
                ":(exclude)evolution/",
            )
            self._assert_index_clean(root)

    def _apply_diff(self, root: Path, diff: str) -> None:
        """Apply a candidate only when it applies cleanly to the current baseline."""
        if not diff.strip():
            return
        self._run_git(root, "apply", "--check", "-", input_text=diff)
        self._run_git(root, "apply", "-", input_text=diff)
        self._assert_index_clean(root)


# ----------------------------------------------------------------------
# Pareto frontier computation
# ----------------------------------------------------------------------

def _pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return Pareto-optimal candidates (maximize score, minimize cost).

    A candidate is Pareto-optimal if no other candidate has both a
    higher (or equal) score AND a lower (or equal) cost, with at least
    one strict inequality.
    """
    frontier = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            # 'other' dominates 'c' if:
            #   other.score >= c.score AND other.cost <= c.cost
            #   with at least one strict inequality
            if (other["score"] >= c["score"]
                    and other["cost"] <= c["cost"]
                    and (other["score"] > c["score"]
                         or other["cost"] < c["cost"])):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return frontier
