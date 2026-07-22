from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evolve.algorithms.meta_harness import MetaHarnessEngine
from agent_evolve.config import EvolveConfig
from agent_evolve.contract.workspace import AgentWorkspace


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _workspace(root: Path) -> AgentWorkspace:
    (root / "prompts").mkdir(parents=True)
    (root / "memory").mkdir()
    (root / "tools").mkdir()
    (root / "evolution").mkdir()
    (root / "prompts" / "system.md").write_text("Base prompt\n")
    (root / "memory" / "memories.jsonl").write_text('{"content": "base"}\n')
    (root / "tools" / "registry.yaml").write_text("tools: []\n")
    (root / "manifest.yaml").write_text("name: test-agent\n")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "baseline")
    return AgentWorkspace(root)


def _engine() -> MetaHarnessEngine:
    return MetaHarnessEngine(EvolveConfig(extra={"num_candidates": 1}))


def test_apply_diff_uses_clean_git_apply_only(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    calls: list[tuple[str, ...]] = []

    def run_git(
        _root: Path,
        *args: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr(engine, "_run_git", run_git)
    engine._apply_diff(Path("/tmp/workspace"), "diff --git a/x b/x\n")

    assert calls[:2] == [("apply", "--check", "-"), ("apply", "-")]
    assert all("--allow-empty" not in call for call in calls)
    assert all("--3way" not in call for call in calls)


def test_git_reset_restores_index_and_preserves_evolution(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    engine = _engine()
    root = workspace.root

    prompt = root / "prompts" / "system.md"
    prompt.write_text("Candidate prompt\n")
    _git(root, "add", "prompts/system.md")
    (root / "skills").mkdir()
    (root / "skills" / "new.md").write_text("untracked\n")
    artifact = root / "evolution" / "candidate.json"
    artifact.write_text("{}\n")

    engine._git_reset(root)

    assert prompt.read_text() == "Base prompt\n"
    assert not (root / "skills" / "new.md").exists()
    assert artifact.exists()
    assert _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)evolution/",
    ) == ""


def test_candidate_patches_replay_independently(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    engine = _engine()
    root = workspace.root
    prompt = root / "prompts" / "system.md"

    prompt.write_text("Candidate zero\n")
    candidate_zero = engine._git_diff(root)
    engine._git_reset(root)
    prompt.write_text("Candidate one\n")
    candidate_one = engine._git_diff(root)
    engine._git_reset(root)

    engine._apply_diff(root, candidate_zero)
    assert prompt.read_text() == "Candidate zero\n"
    assert _git(root, "diff", "--cached", "--name-only") == ""

    engine._git_reset(root)
    engine._apply_diff(root, candidate_one)
    assert prompt.read_text() == "Candidate one\n"
    assert "<<<<<<<" not in prompt.read_text()
    assert _git(root, "ls-files", "--unmerged") == ""


def test_stale_patch_fails_without_conflict_markers(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    engine = _engine()
    root = workspace.root
    prompt = root / "prompts" / "system.md"

    prompt.write_text("Candidate\n")
    patch = engine._git_diff(root)
    engine._git_reset(root)
    prompt.write_text("New baseline\n")
    _git(root, "add", "prompts/system.md")
    _git(root, "commit", "-m", "new baseline")

    with pytest.raises(RuntimeError, match="git apply --check"):
        engine._apply_diff(root, patch)

    assert prompt.read_text() == "New baseline\n"
    assert _git(root, "ls-files", "--unmerged") == ""
    assert "<<<<<<<" not in prompt.read_text()


@pytest.mark.parametrize(
    ("relative_path", "content", "expected"),
    [
        ("memory/memories.jsonl", '{"ok": true}\nnot-json\n', ":2: Expecting value"),
        ("memory/memories.jsonl", "[]\n", "expected object, got list"),
        (
            "prompts/system.md",
            "<<<<<<< ours\nleft\n=======\nright\n>>>>>>> theirs\n",
            "git conflict marker",
        ),
        ("tools/registry.yaml", "tools: [\n", "invalid YAML"),
    ],
)
def test_candidate_validation_rejects_structural_corruption(
    tmp_path: Path,
    relative_path: str,
    content: str,
    expected: str,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    (workspace.root / relative_path).write_text(content)

    valid, error = _engine()._validate_candidate(workspace)

    assert valid is False
    assert expected in error


def test_evaluation_failure_is_invalid_and_archived(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "workspace")
    engine = _engine()
    proposal = {
        "index": 0,
        "label": "cycle_001_cand_0",
        "diff": "",
        "valid": True,
        "validation_err": "",
        "proposer_result": {"output": "done", "stderr": "", "exit_code": 0},
        "snapshot_files": engine._capture_snapshot(workspace),
    }

    candidates_dir = workspace.root / "evolution" / "candidates"
    engine._archive_candidate_from_snapshot(
        workspace,
        candidates_dir / proposal["label"],
        proposal["snapshot_files"],
        0.0,
        0.0,
        1,
        0,
        proposal["proposer_result"],
        valid=True,
        validation_err="",
        diff="",
    )

    def failing_factory(_path: Path):
        raise RuntimeError("factory failed")

    results = engine._evaluate_candidates(
        [proposal],
        workspace,
        candidates_dir,
        failing_factory,
        [SimpleNamespace(id="task")],
        parallel=False,
    )

    assert results[0]["valid"] is False
    assert results[0]["failure_stage"] == "evaluation"
    candidate_dir = candidates_dir / proposal["label"]
    assert (candidate_dir / "candidate.patch").exists()
    assert (candidate_dir / "proposer.json").exists()
    diagnostics = json.loads((candidate_dir / "diagnostics.json").read_text())
    assert diagnostics["failure_stage"] == "evaluation"
    assert "factory failed" in diagnostics["error"]


def test_final_apply_failure_restores_main_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    engine = _engine()
    prompt = workspace.root / "prompts" / "system.md"

    class FakeAgent:
        def __init__(self) -> None:
            self.reloads: list[str] = []

        def reload_from_fs(self) -> None:
            self.reloads.append(prompt.read_text())

    class FakeTrial:
        def __init__(self) -> None:
            self.agent = FakeAgent()
            self.benchmark = object()

    def propose(_prompt: str, _root: Path) -> dict[str, object]:
        prompt.write_text("Candidate prompt\n")
        return {"output": "done", "stderr": "", "exit_code": 0}

    def evaluate(proposed, *_args, **_kwargs):
        proposal = proposed[0]
        return [
            {
                "index": 0,
                "label": proposal["label"],
                "score": 1.0,
                "cost": 1,
                "diff": proposal["diff"],
                "valid": True,
                "validation_err": "",
                "failure_stage": None,
                "exit_code": 0,
                "output_chars": 4,
            }
        ]

    monkeypatch.setattr(engine, "_run_claude_code", propose)
    monkeypatch.setattr(engine, "_evaluate_candidates", evaluate)
    monkeypatch.setattr(
        engine,
        "_apply_diff",
        lambda _root, _diff: (_ for _ in ()).throw(RuntimeError("final apply failed")),
    )

    history = SimpleNamespace(latest_cycle=0, get_score_curve=lambda: [])
    trial = FakeTrial()
    result = engine.step(workspace, [], history, trial, tasks=[])

    assert result.mutated is False
    assert "final apply failed" in result.metadata["final_apply_error"]
    assert prompt.read_text() == "Base prompt\n"
    engine._assert_workspace_clean(workspace.root)
    assert trial.agent.reloads == ["Base prompt\n"]
    diagnostics = json.loads(
        (
            workspace.root
            / "evolution"
            / "candidates"
            / "cycle_001_cand_0"
            / "diagnostics.json"
        ).read_text()
    )
    assert diagnostics["selection_attempted"] is True
    assert diagnostics["final_apply_succeeded"] is False
