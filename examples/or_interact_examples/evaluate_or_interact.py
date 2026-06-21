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

from agent_evolve.agents.or_interact.react_agent import ORReactAgent  # noqa: E402
from agent_evolve.benchmarks.or_interact import (  # noqa: E402
    ORInteractBenchmark,
    evaluation_limit_for_split,
    train_size_from_limit,
)
from agent_evolve.display import print_evaluation_summary, print_run_header  # noqa: E402
from agent_evolve.evaluation import run_evaluation  # noqa: E402
from agent_evolve.runs import resolve_workspace_source  # noqa: E402

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
        dataset=args.dataset,
        seed=42,
        train_size=train_size_from_limit(args.limit_train),
    )
    evaluation_limit = evaluation_limit_for_split(
        args.split,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        legacy_limit=args.limit,
    )
    previous_results_dir = os.environ.get("OR_REACT_RESULTS_DIR")
    os.environ["OR_REACT_RESULTS_DIR"] = str((output_dir / "runs").resolve())
    try:
        agent = ORReactAgent(workspace)
        summary = run_evaluation(
            agent,
            benchmark,
            split=args.split,
            limit=evaluation_limit,
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
    summary["dataset"] = args.dataset
    summary["limit_train"] = args.limit_train
    summary["limit_test"] = args.limit_test
    summary["check"] = False
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
            "Evaluation source: empty, latest, a run id under work-dir/runs, "
            "a run path, or a workspace path."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        help="Backward-compatible alias for the active split limit when split-specific limits are omitted.",
    )
    parser.add_argument(
        "--limit-train",
        "--limit_train",
        dest="limit_train",
        type=int,
        help="Train split size and evaluation limit when --split train.",
    )
    parser.add_argument(
        "--limit-test",
        "--limit_test",
        dest="limit_test",
        type=int,
        help="Evaluation limit when --split test or holdout.",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _default_output_dir(work_dir: str | Path, _workspace: Path, split: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(work_dir) / "evaluations" / stamp / split


if __name__ == "__main__":
    raise SystemExit(main())
