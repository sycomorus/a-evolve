"""Run a-evolve on OR-Interact-Bench IndustryOR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

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
from agent_evolve.display import print_evolve_summary, print_run_header
from agent_evolve.evaluation import run_evaluation
from agent_evolve.runs import create_run_workspace, update_run_metadata


def main() -> int:
    args = parse_args()
    console = Console()
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
    run = create_run_workspace(
        args.work_dir,
        args.from_source,
        seed_workspace,
        agent="or-interact",
        benchmark="industry-or",
    )
    print_run_header(
        title="OR-Interact Evolution",
        run_dir=run.run_dir,
        workspace=run.workspace_dir,
        source=args.from_source,
        console=console,
    )
    evolver = Evolver(
        agent=run.workspace_dir,
        benchmark=benchmark,
        config=config,
        engine=engine,
        work_dir=run.run_dir,
    )

    with Progress(
        TextColumn("[bold magenta]Evolving"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]score={task.fields[score]} mutated={task.fields[mutated]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task(
            "evolve",
            total=args.max_cycles,
            score="n/a",
            mutated="n/a",
        )

        def update_progress(event: dict[str, object]) -> None:
            progress.update(
                progress_task,
                completed=int(event["cycle"]),
                score=f"{float(event['score']):.3f}",
                mutated="yes" if event["mutated"] else "no",
            )

        result = evolver.run(cycles=args.max_cycles, progress_callback=update_progress)

    workspace = evolver.agent.workspace.root
    final_dir = workspace / "evolution" / "final_test"
    test_limit = args.limit_test if args.limit_test is not None else 50
    console.rule("[bold cyan]Final test evaluation")
    eval_summary = run_evaluation(
        evolver.agent,
        benchmark,
        split="test",
        limit=test_limit,
        output_dir=final_dir,
        show_progress=True,
        console=console,
    )
    summary = {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "workspace": str(workspace),
        "cycles_completed": result.cycles_completed,
        "score_history": result.score_history,
        "test_total": eval_summary["total"],
        "test_success": eval_summary["success"],
        "test_accuracy": eval_summary["accuracy"],
    }
    (final_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_run_metadata(
        run.run_dir,
        cycles_completed=result.cycles_completed,
        final_score=result.final_score,
        score_history=result.score_history,
        final_test=summary,
    )

    print_evolve_summary(summary, console=console)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve the OR-Interact ReAct harness.")
    parser.add_argument("--benchmark-dir", default=str(REPO_ROOT / "OR-Interact-Bench"))
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / "a-evolve" / "evolution_workdir_or_interact"),
    )
    parser.add_argument(
        "--from",
        dest="from_source",
        default="empty",
        help=(
            "Run source: empty, latest, a run id under work-dir/runs, "
            "a run path, or a workspace path."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-cycles", "--cycles", dest="max_cycles", type=int, default=5)
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
