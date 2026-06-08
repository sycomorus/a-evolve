"""Timestamped run directory helpers for A-Evolve workspaces."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RUNS_DIRNAME = "runs"
WORKSPACE_DIRNAME = "workspace"
RUN_METADATA = "run.json"


@dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    run_dir: Path
    workspace_dir: Path
    source_type: str
    source_path: Path
    metadata: dict[str, Any]


def create_run_workspace(
    work_dir: str | Path,
    source: str | Path,
    seed_workspace: str | Path,
    *,
    agent: str,
    benchmark: str,
    created_at: datetime | None = None,
) -> RunWorkspace:
    """Create ``<work-dir>/runs/<timestamp>/workspace`` from a resolved source.

    ``source="empty"`` copies the seed workspace. For OR-Interact this matches
    the existing baseline behavior: no evolved skills/tools, only the built-in
    seed tools registered by the agent implementation.
    """

    root = Path(work_dir)
    source_type, source_workspace = resolve_workspace_source(root, source, seed_workspace)
    run_dir = _create_unique_run_dir(root, created_at)
    workspace_dir = run_dir / WORKSPACE_DIRNAME
    shutil.copytree(source_workspace, workspace_dir)

    created = created_at or datetime.now()
    metadata = {
        "run_id": run_dir.name,
        "created_at": created.isoformat(timespec="seconds"),
        "agent": agent,
        "benchmark": benchmark,
        "workspace": str(workspace_dir),
        "source": {
            "spec": str(source),
            "type": source_type,
            "path": str(source_workspace),
        },
    }
    write_run_metadata(run_dir, metadata)
    return RunWorkspace(
        run_id=run_dir.name,
        run_dir=run_dir,
        workspace_dir=workspace_dir,
        source_type=source_type,
        source_path=source_workspace,
        metadata=metadata,
    )


def resolve_workspace_source(
    work_dir: str | Path,
    source: str | Path,
    seed_workspace: str | Path,
) -> tuple[str, Path]:
    """Resolve ``empty``, ``latest``, run id, run path, or workspace path."""

    source_text = str(source)
    if source_text == "empty":
        workspace = Path(seed_workspace)
        _require_workspace(workspace)
        return "empty", workspace
    if source_text == "latest":
        run_dir = latest_run_dir(work_dir)
        return "latest", _workspace_from_run(run_dir)

    candidate = Path(source)
    run_id_path = Path(work_dir) / RUNS_DIRNAME / source_text
    is_simple_run_id = not candidate.is_absolute() and len(candidate.parts) == 1
    if is_simple_run_id and run_id_path.exists():
        candidate = run_id_path
    elif not candidate.exists() and run_id_path.exists():
        candidate = run_id_path

    if candidate.exists():
        if (candidate / WORKSPACE_DIRNAME).is_dir():
            return "run", _workspace_from_run(candidate)
        _require_workspace(candidate)
        return "workspace", candidate

    raise FileNotFoundError(f"cannot resolve workspace source: {source!r}")


def latest_run_dir(work_dir: str | Path) -> Path:
    runs = list_run_dirs(work_dir)
    if not runs:
        raise FileNotFoundError(f"no runs found under {Path(work_dir) / RUNS_DIRNAME}")
    return runs[-1]


def list_run_dirs(work_dir: str | Path) -> list[Path]:
    runs_root = Path(work_dir) / RUNS_DIRNAME
    if not runs_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and (path / WORKSPACE_DIRNAME).is_dir()
        ),
        key=_run_sort_key,
    )


def read_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / RUN_METADATA
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> None:
    (Path(run_dir) / RUN_METADATA).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_run_metadata(run_dir: str | Path, **updates: Any) -> dict[str, Any]:
    metadata = read_run_metadata(run_dir)
    metadata.update(updates)
    write_run_metadata(run_dir, metadata)
    return metadata


def _create_unique_run_dir(work_dir: Path, created_at: datetime | None) -> Path:
    runs_root = work_dir / RUNS_DIRNAME
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = (created_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = runs_root / stamp
    if not candidate.exists():
        candidate.mkdir()
        return candidate

    suffix = 2
    while True:
        candidate = runs_root / f"{stamp}-{suffix}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        suffix += 1


def _workspace_from_run(run_dir: Path) -> Path:
    workspace = run_dir / WORKSPACE_DIRNAME
    _require_workspace(workspace)
    return workspace


def _require_workspace(path: Path) -> None:
    if not path.is_dir() or not (path / "manifest.yaml").is_file():
        raise FileNotFoundError(f"missing workspace manifest: {path / 'manifest.yaml'}")


def _run_sort_key(path: Path) -> tuple[str, int]:
    parts = path.name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return path.name, 1
