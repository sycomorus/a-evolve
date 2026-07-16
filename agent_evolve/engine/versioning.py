"""Git-based version control for agent workspaces.

Ported from CodeDojo/swe-agent/swe_agent/evolve/state_repo.py and adapted
to work with the AgentWorkspace file system contract.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class VersionControl:
    """Manages git history on an agent workspace directory."""

    def __init__(self, workspace_root: str | Path):
        self.root = Path(workspace_root).resolve()

    def init(self) -> None:
        """Initialize a git repo in the workspace (idempotent)."""
        new_repo = not (self.root / ".git").exists()
        if new_repo:
            logger.info("Initializing git repo at %s", self.root)
            self._git("init")
            self._git("config", "user.email", "evolver@agent-evolve")
            self._git("config", "user.name", "Agent Evolve")

        self._git("add", "-A")
        try:
            self._git("commit", "-m", "Initial workspace state")
            self._git("tag", "evo-0")
            if new_repo:
                self._git("branch", "-M", "main")
            logger.info("Created initial commit with tag evo-0")
        except RuntimeError:
            pass  # already committed

    def commit(self, message: str, tag: str | None = None) -> None:
        self._git("add", "-A")
        try:
            self._git("commit", "-m", message)
            logger.info("Committed: %s", message)
        except RuntimeError:
            logger.debug("Nothing to commit: %s", message)
        if tag:
            self._git("tag", "-f", tag)
            logger.debug("Tagged: %s", tag)

    def exclude_path(self, path: str) -> None:
        """Keep generated runtime artifacts outside workspace snapshots."""
        exclude_file = self.root / ".git" / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        normalized = path.strip()
        existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
        if normalized and normalized not in existing.splitlines():
            with exclude_file.open("a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write(f"{normalized}\n")

    def rollback(self, ref: str = "HEAD~1") -> None:
        """Restore workspace content from *ref* as a NEW commit.

        Unlike ``git reset --hard``, this preserves the rejected version
        in git history so it can be inspected or reused later.
        """
        logger.info("Rolling back workspace to %s", ref)
        self._git("checkout", ref, "--", ".")
        self._git("add", "-A")
        try:
            self._git("commit", "-m", f"rollback to {ref}")
        except RuntimeError:
            pass  # nothing changed

    def rollback_to_tag(self, tag: str) -> None:
        """Restore workspace to the state at *tag* (history preserved)."""
        self.rollback(tag)

    def get_diff(self, from_ref: str = "HEAD~1", to_ref: str = "HEAD") -> str:
        return self._git("diff", from_ref, to_ref)

    def get_diff_stat(self, from_ref: str = "HEAD~1", to_ref: str = "HEAD") -> str:
        return self._git("diff", "--stat", from_ref, to_ref)

    def get_log(self, n: int = 20) -> str:
        return self._git("log", "--oneline", f"-{n}")

    def list_tags(self) -> list[str]:
        output = self._git("tag", "-l", "evo-*", "--sort=-version:refname")
        return [t.strip() for t in output.splitlines() if t.strip()]

    def create_branch(self, branch: str, start_point: str = "HEAD", *, checkout: bool = False) -> None:
        """Create *branch* from *start_point* if it does not already exist."""
        if self.branch_exists(branch):
            if checkout:
                self.checkout_branch(branch)
            return
        args = ["checkout", "-b", branch, start_point] if checkout else ["branch", branch, start_point]
        self._git(*args)

    def checkout_branch(self, branch: str) -> None:
        """Checkout an existing branch."""
        self._git("checkout", branch)

    def branch_exists(self, branch: str) -> bool:
        try:
            self._git("rev-parse", "--verify", f"refs/heads/{branch}")
        except RuntimeError:
            return False
        return True

    def list_branches(self) -> list[str]:
        output = self._git("branch", "--format=%(refname:short)")
        return [line.strip() for line in output.splitlines() if line.strip()]

    def get_current_branch(self) -> str:
        return self._git("branch", "--show-current")

    def show_file_at(self, ref: str, filepath: str) -> str:
        return self._git("show", f"{ref}:{filepath}")

    def checkout_copy(self, ref: str, dest: Path) -> None:
        """Create a separate working copy of the workspace at *ref*.

        Uses ``git worktree`` so the copy shares the object store but has
        its own working tree.  Use :meth:`remove_copy` to clean up.
        """
        self._git("worktree", "add", "--detach", str(dest), ref)

    def remove_copy(self, dest: Path) -> None:
        """Remove a working copy created by :meth:`checkout_copy`."""
        self._git("worktree", "remove", str(dest), "--force")

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(self.root),
        )
        if result.returncode != 0 and "nothing to commit" not in result.stderr:
            stderr = result.stderr.strip()
            if stderr:
                raise RuntimeError(f"git {' '.join(args)}: {stderr}")
        return result.stdout.strip()
