"""Run a-evolve on an OR-Interact-Bench dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
EVOLVER_CONFIG = REPO_ROOT / "config" / "evolver.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.api import Evolver  # noqa: E402
from agent_evolve.algorithms.adaptive_skill import AdaptiveSkillEngine  # noqa: E402
from agent_evolve.benchmarks.or_interact import ORInteractBenchmark  # noqa: E402
from agent_evolve.config import EvolveConfig  # noqa: E402
from agent_evolve.display import print_evolve_summary, print_run_header  # noqa: E402
from agent_evolve.evaluation import run_evaluation  # noqa: E402
from agent_evolve.runs import create_run_workspace, update_run_metadata  # noqa: E402

ANSWER_CHECKER_ENV = "OR_REACT_ENABLE_ANSWER_CHECKER"


def main() -> int:
    args = parse_args()
    console = Console()
    evolver_model, evolver_base_url, evolver_api_key, evolver_temperature = resolve_evolver_llm()
    benchmark = ORInteractBenchmark(
        benchmark_dir=args.benchmark_dir,
        dataset=args.dataset,
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
            "evolver_temperature": evolver_temperature,
        },
    )
    engine = AdaptiveSkillEngine(config)
    seed_workspace = ROOT / "seed_workspaces" / "or_interact_react"
    run = create_run_workspace(
        args.work_dir,
        args.from_source,
        seed_workspace,
        agent="or-interact",
        benchmark=f"or-interact:{args.dataset}",
    )
    print_run_header(
        title="OR-Interact Evolution",
        run_dir=run.run_dir,
        workspace=run.workspace_dir,
        source=args.from_source,
        console=console,
    )
    with _answer_checker_setting(args.check):
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
        "dataset": args.dataset,
        "check": args.check,
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
        check=args.check,
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
        "--dataset",
        default="IndustryOR",
        help="Dataset directory under OR-Interact-Bench, for example IndustryOR or LargeScaleOR.",
    )
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Expose the optional answer_checker tool during evolution and final evaluation.",
    )
    return parser.parse_args()


def resolve_evolver_llm() -> tuple[str, str, str | None, float | None]:
    config_path = Path(os.environ.get("OR_EVOLVER_CONFIG", EVOLVER_CONFIG))
    config = load_yaml(config_path)
    model = config.get("model")
    base_url = config.get("base_url")
    api_key = config.get("api_key")
    temperature = config.get("temperature")
    if not model or not base_url:
        raise ValueError(
            f"{config_path} must define model and base_url for the OpenAI-compatible evolver."
        )
    if base_url and not model.startswith("openai:"):
        model = f"openai:{model}"
    return model, base_url, api_key, float(temperature) if temperature is not None else None


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


@contextmanager
def _answer_checker_setting(enabled: bool):
    previous = os.environ.get(ANSWER_CHECKER_ENV)
    os.environ[ANSWER_CHECKER_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ANSWER_CHECKER_ENV, None)
        else:
            os.environ[ANSWER_CHECKER_ENV] = previous


if __name__ == "__main__":
    raise SystemExit(main())
