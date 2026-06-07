"""Evaluate an OR-Interact ReAct workspace on a benchmark split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_evolve.agents.or_interact.react_agent import ORReactAgent
from agent_evolve.benchmarks.or_interact import ORInteractBenchmark
from agent_evolve.display import print_evaluation_summary, print_run_header
from agent_evolve.evaluation import run_evaluation
from agent_evolve.runs import RUNS_DIRNAME, resolve_workspace_source


def main() -> int:
    args = parse_args()
    console = Console()
    seed_workspace = ROOT / "seed_workspaces" / "or_interact_react"
    source_type, workspace = resolve_workspace_source(
        args.work_dir,
        args.from_source,
        seed_workspace,
    )
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(
        args.work_dir,
        workspace,
        args.split,
    )
    print_run_header(
        title="OR-Interact Evaluation",
        workspace=workspace,
        source=f"{args.from_source} ({source_type})",
        console=console,
    )

    benchmark = ORInteractBenchmark(
        benchmark_dir=args.benchmark_dir,
        dataset="IndustryOR",
        seed=42,
        train_size=50,
    )
    previous_results_dir = os.environ.get("OR_REACT_RESULTS_DIR")
    os.environ["OR_REACT_RESULTS_DIR"] = str((output_dir / "runs").resolve())
    try:
        agent = ORReactAgent(workspace)
        summary = run_evaluation(
            agent,
            benchmark,
            split=args.split,
            limit=args.limit,
            output_dir=output_dir,
            show_progress=True,
            console=console,
        )
    finally:
        if previous_results_dir is None:
            os.environ.pop("OR_REACT_RESULTS_DIR", None)
        else:
            os.environ["OR_REACT_RESULTS_DIR"] = previous_results_dir
    summary["source_type"] = source_type
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_evaluation_summary(summary, title="OR-Interact Evaluation Summary", console=console)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an OR-Interact ReAct workspace.")
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
            "Evaluation source: empty, latest, a run id under work-dir/runs, "
            "a run path, or a workspace path."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _default_output_dir(work_dir: str | Path, workspace: Path, split: str) -> Path:
    if workspace.name == "workspace" and workspace.parent.parent.name == RUNS_DIRNAME:
        return workspace.parent / "evaluation" / split
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(work_dir) / "evaluations" / stamp / split


if __name__ == "__main__":
    raise SystemExit(main())
