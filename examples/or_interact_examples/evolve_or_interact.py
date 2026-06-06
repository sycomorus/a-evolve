"""Run a-evolve on OR-Interact-Bench IndustryOR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
REACT_CONFIG = REPO_ROOT / "config" / "react.yaml"
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
    evolver_model, evolver_base_url, evolver_api_key = resolve_evolver_llm()
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
        evolver_model=evolver_model,
        evolve_prompts=True,
        evolve_skills=True,
        evolve_memory=True,
        evolve_tools=True,
        trajectory_only=False,
        extra={
            "max_skills": args.max_skills,
            "evolver_base_url": evolver_base_url,
            "evolver_api_key": evolver_api_key,
        },
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


def resolve_evolver_llm() -> tuple[str, str, str | None]:
    config = load_yaml(REACT_CONFIG)
    model = config.get("model")
    base_url = config.get("base_url")
    api_key = config.get("api_key")
    if not model or not base_url:
        raise ValueError(
            f"{REACT_CONFIG} must define model and base_url for the OpenAI-compatible evolver."
        )
    if base_url and not model.startswith("openai:"):
        model = f"openai:{model}"
    return model, base_url, api_key


def load_yaml(path: str | Path) -> dict[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
