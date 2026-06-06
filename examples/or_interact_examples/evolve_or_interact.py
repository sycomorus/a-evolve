"""Run a-evolve on OR-Interact-Bench IndustryOR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.api import Evolver
from agent_evolve.algorithms.adaptive_skill import AdaptiveSkillEngine
from agent_evolve.benchmarks.or_interact import ORInteractBenchmark
from agent_evolve.config import EvolveConfig


def main() -> int:
    args = parse_args()
    benchmark = ORInteractBenchmark(
        benchmark_dir=args.benchmark_dir,
        dataset="IndustryOR",
        seed=42,
        train_size=50,
    )
    train_batch = args.limit_train if args.limit_train is not None else args.batch_size
    config = EvolveConfig(
        batch_size=train_batch,
        max_cycles=args.max_cycles,
        evolve_prompts=True,
        evolve_skills=True,
        evolve_memory=True,
        evolve_tools=True,
        trajectory_only=False,
        extra={"max_skills": args.max_skills},
    )
    engine = AdaptiveSkillEngine(config)
    seed_workspace = ROOT / "seed_workspaces" / "or_interact_react"
    evolver = Evolver(
        agent=seed_workspace,
        benchmark=benchmark,
        config=config,
        engine=engine,
        work_dir=args.work_dir,
    )

    result = evolver.run(cycles=args.max_cycles)
    workspace = evolver.agent.workspace.root
    final_dir = workspace / "evolution" / "final_test"
    final_dir.mkdir(parents=True, exist_ok=True)

    test_limit = args.limit_test if args.limit_test is not None else 50
    rows = []
    for task in benchmark.get_tasks(split="test", limit=test_limit):
        trajectory = evolver.agent.solve(task)
        feedback = benchmark.evaluate(task, trajectory)
        evaluation = feedback.raw.get("evaluation", {})
        rows.append(
            {
                "task_id": task.id,
                "success": feedback.success,
                "score": feedback.score,
                **evaluation,
            }
        )

    summary = {
        "workspace": str(workspace),
        "cycles_completed": result.cycles_completed,
        "score_history": result.score_history,
        "test_total": len(rows),
        "test_success": sum(1 for row in rows if row["success"]),
        "test_accuracy": (sum(1 for row in rows if row["success"]) / len(rows)) if rows else 0.0,
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (final_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve the OR-Interact ReAct harness.")
    parser.add_argument("--benchmark-dir", default=str(REPO_ROOT / "OR-Interact-Bench"))
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "a-evolve" / "evolution_workdir_or_interact"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--max-skills", type=int, default=8)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
