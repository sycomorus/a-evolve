from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

from agent_evolve.runs import (
    create_run_workspace,
    latest_run_dir,
    resolve_workspace_source,
)


def test_empty_source_creates_timestamped_run_from_seed(tmp_path: Path) -> None:
    seed = _workspace(tmp_path / "seed")

    run = create_run_workspace(
        tmp_path / "work",
        "empty",
        seed,
        agent="or-interact",
        benchmark="industry-or",
        created_at=datetime(2026, 6, 7, 12, 0, 0),
    )

    assert run.run_id == "20260607-120000"
    assert run.workspace_dir == tmp_path / "work" / "runs" / "20260607-120000" / "workspace"
    assert (run.workspace_dir / "manifest.yaml").is_file()
    metadata = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["source"]["type"] == "empty"
    assert metadata["workspace"] == str(run.workspace_dir)


def test_duplicate_timestamp_uses_numeric_suffix_and_latest_order(tmp_path: Path) -> None:
    seed = _workspace(tmp_path / "seed")
    now = datetime(2026, 6, 7, 12, 0, 0)

    first = create_run_workspace(
        tmp_path / "work",
        "empty",
        seed,
        agent="a",
        benchmark="b",
        created_at=now,
    )
    latest = first
    for _ in range(9):
        latest = create_run_workspace(
            tmp_path / "work",
            "empty",
            seed,
            agent="a",
            benchmark="b",
            created_at=now,
        )

    assert first.run_id == "20260607-120000"
    assert latest.run_id == "20260607-120000-10"
    assert latest_run_dir(tmp_path / "work") == latest.run_dir


def test_latest_run_id_and_path_sources_resolve_workspace(tmp_path: Path) -> None:
    seed = _workspace(tmp_path / "seed")
    older = create_run_workspace(
        tmp_path / "work",
        "empty",
        seed,
        agent="a",
        benchmark="b",
        created_at=datetime(2026, 6, 7, 12, 0, 0),
    )
    newer = create_run_workspace(
        tmp_path / "work",
        "empty",
        seed,
        agent="a",
        benchmark="b",
        created_at=datetime(2026, 6, 7, 12, 0, 1),
    )
    (older.workspace_dir / "marker.txt").write_text("older", encoding="utf-8")
    (newer.workspace_dir / "marker.txt").write_text("newer", encoding="utf-8")

    assert resolve_workspace_source(tmp_path / "work", "latest", seed) == (
        "latest",
        newer.workspace_dir,
    )
    assert resolve_workspace_source(tmp_path / "work", older.run_id, seed) == (
        "run",
        older.workspace_dir,
    )
    assert resolve_workspace_source(tmp_path / "work", newer.run_dir, seed) == (
        "run",
        newer.workspace_dir,
    )
    assert resolve_workspace_source(tmp_path / "work", newer.workspace_dir, seed) == (
        "workspace",
        newer.workspace_dir,
    )


def test_simple_run_id_preferred_over_cwd_relative_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed = _workspace(tmp_path / "seed")
    cwd_workspace = _workspace(tmp_path / "manual")
    run_workspace = _workspace(tmp_path / "work" / "runs" / "manual" / "workspace")

    monkeypatch.chdir(tmp_path)

    assert resolve_workspace_source(tmp_path / "work", "manual", seed) == (
        "run",
        run_workspace,
    )
    assert resolve_workspace_source(tmp_path / "work", cwd_workspace, seed) == (
        "workspace",
        cwd_workspace,
    )


def test_evaluate_run_workspace_default_output_dir_uses_evaluations_root() -> None:
    module = _load_evaluate_module()
    work_dir = Path("/tmp/work")
    workspace = work_dir / "runs" / "manual" / "workspace"

    output_dir = module._default_output_dir(work_dir, workspace, "train")

    assert output_dir.name == "train"
    assert output_dir.parent.parent == work_dir / "evaluations"
    assert output_dir != work_dir / "runs" / "manual" / "evaluation" / "train"


def _workspace(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.yaml").write_text("name: fake\n", encoding="utf-8")
    return path


def _load_evaluate_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "or_interact_examples"
        / "evaluate_or_interact.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_or_interact_script", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
