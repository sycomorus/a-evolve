"""Run OR-Interact tasks and write full Step-OPSD teacher reviews for failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.algorithms.adaptive_skill.tools import create_default_llm  # noqa: E402
from agent_evolve.algorithms.step_opsd import review_observation_for_audit  # noqa: E402
from agent_evolve.agents.or_interact.react_agent import ORReactAgent  # noqa: E402
from agent_evolve.benchmarks.or_interact import (  # noqa: E402
    ORInteractBenchmark,
    train_size_from_limit,
)
from agent_evolve.config import EvolveConfig  # noqa: E402
from agent_evolve.runs import create_run_workspace, update_run_metadata  # noqa: E402
from agent_evolve.task_runner import run_task_evaluations  # noqa: E402
from agent_evolve.types import Observation  # noqa: E402
from examples.or_interact_examples.evolve_or_interact import (  # noqa: E402
    _write_or_interact_settings,
    resolve_evolver_llm,
)


def main() -> int:
    args = parse_args()
    console = Console()
    evolver_model, evolver_base_url, evolver_api_key, evolver_temperature = resolve_evolver_llm()
    config = EvolveConfig(
        evolver_model=evolver_model,
        evolver_max_tokens=args.teacher_max_tokens,
        extra={
            "evolver_base_url": evolver_base_url,
            "evolver_api_key": evolver_api_key,
            "evolver_temperature": evolver_temperature,
        },
    )
    llm = create_default_llm(config)
    benchmark = ORInteractBenchmark(
        benchmark_dir=args.benchmark_dir,
        dataset=args.dataset,
        seed=42,
        train_size=train_size_from_limit(args.limit_train),
    )
    seed_workspace = ROOT / "seed_workspaces" / "or_interact_react"
    run = create_run_workspace(
        args.work_dir,
        args.from_source,
        seed_workspace,
        agent="or-interact",
        benchmark=f"or-interact:{args.dataset}",
    )
    _write_or_interact_settings(
        run.workspace_dir,
        enable_heuristic_tool=args.enable_heuristic_tool,
        enable_user_tool=args.enable_user_tool,
    )

    agent = ORReactAgent(run.workspace_dir)
    tasks = benchmark.get_tasks(split=args.split, limit=args.limit)
    output_dir = Path(args.output_dir) if args.output_dir else run.workspace_dir / "evolution" / "step_opsd_teacher_reviews"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Solving {len(tasks)} {args.split} tasks from {args.dataset}")
    task_results = run_task_evaluations(agent, benchmark, tasks)
    observations: list[Observation] = []
    errors: list[dict[str, Any]] = []
    for result in task_results:
        if result.trajectory is None or result.feedback is None:
            errors.append({"task_id": result.task.id, "error": result.error})
            continue
        observations.append(
            Observation(
                task=result.task,
                trajectory=result.trajectory,
                feedback=result.feedback,
            )
        )

    reviewed: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    for obs in observations:
        observation_rows.append({
            "task_id": obs.task.id,
            "success": obs.feedback.success,
            "score": obs.feedback.score,
            "feedback_detail": obs.feedback.detail,
        })
        if obs.feedback.success:
            continue
        console.log(f"Teacher reviewing failed task {obs.task.id}")
        reviewed.append(
            review_observation_for_audit(
                obs,
                output_dir=output_dir,
                llm=llm,
                max_tokens=args.teacher_max_tokens,
            )
        )

    _write_jsonl(output_dir / "observations.jsonl", observation_rows)
    _write_jsonl(output_dir / "solve_errors.jsonl", errors)
    _write_jsonl(output_dir / "teacher_review_records.jsonl", reviewed)
    summary = {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "workspace": str(run.workspace_dir),
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "total": len(observations),
        "success": sum(1 for obs in observations if obs.feedback.success),
        "failed": sum(1 for obs in observations if not obs.feedback.success),
        "reviewed": len(reviewed),
        "solve_errors": len(errors),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_run_metadata(run.run_dir, step_opsd_teacher_review=summary)
    console.log(f"Wrote review artifacts to {output_dir}")
    console.print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve OR-Interact tasks and run Step-OPSD teacher review for failures.",
    )
    parser.add_argument("--benchmark-dir", default=str(REPO_ROOT / "OR-Interact-Bench"))
    parser.add_argument("--dataset", default="IndustryOR")
    parser.add_argument("--split", choices=["train", "holdout", "test"], default="train")
    parser.add_argument("--limit", type=int, default=10, help="Number of tasks to solve.")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Train split size used before selecting holdout/test tasks.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / "a-evolve" / "step_opsd_review_workdir"),
    )
    parser.add_argument(
        "--from",
        dest="from_source",
        default="empty",
        help="Run source: empty, latest, run id/path, or workspace path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for teacher contexts and responses. Defaults under the run workspace.",
    )
    parser.add_argument("--teacher-max-tokens", type=int, default=4096)
    parser.add_argument("--enable-heuristic-tool", action="store_true")
    parser.add_argument("--enable-user-tool", action="store_true")
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
