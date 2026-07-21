"""Workspace ↔ GEPA candidate serialization.

Converts A-Evolve workspace layers (system prompt, fragments, skills, memory)
to GEPA's dict[str, str] candidate format and back.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import EvolveConfig
    from ...contract.workspace import AgentWorkspace

_FRAGMENT_DELIM = "=== FRAGMENT: {} ==="
_SKILL_DELIM = "=== SKILL: {} ==="
_FRAGMENT_RE = re.compile(r"^=== FRAGMENT: (.+?) ===$", re.MULTILINE)
_SKILL_RE = re.compile(r"^=== SKILL: (.+?) ===$", re.MULTILINE)


class CandidateFormatError(ValueError):
    """Raised when a GEPA candidate cannot be safely restored."""


def build_candidate(workspace: AgentWorkspace, config: EvolveConfig) -> dict[str, str]:
    """Read workspace layers into a GEPA candidate dict."""
    candidate: dict[str, str] = {}
    if config.evolve_prompts:
        candidate["system_prompt"] = workspace.read_prompt()
        candidate["prompt_fragments"] = serialize_fragments(workspace)
    if config.evolve_skills:
        candidate["skills"] = serialize_skills(workspace)
    if config.evolve_memory:
        candidate["memory"] = serialize_memory(workspace)
    return candidate


def serialize_fragments(workspace: AgentWorkspace) -> str:
    parts: list[str] = []
    for name in workspace.list_fragments():
        content = workspace.read_fragment(name)
        parts.append(_FRAGMENT_DELIM.format(name))
        parts.append(content)
    return "\n".join(parts)


def serialize_skills(workspace: AgentWorkspace) -> str:
    parts: list[str] = []
    for skill_meta in workspace.list_skills():
        content = workspace.read_skill(skill_meta.name)
        parts.append(_SKILL_DELIM.format(skill_meta.name))
        parts.append(content)
    return "\n".join(parts)


def serialize_memory(workspace: AgentWorkspace) -> str:
    all_memories = workspace.read_all_memories(limit=10000)
    lines: list[str] = []
    for entry in all_memories:
        lines.append(json.dumps(entry, default=str))
    return "\n".join(lines)


def parse_fragments(blob: str) -> list[tuple[str, str]]:
    return _parse_delimited(blob, _FRAGMENT_RE)


def parse_skills(blob: str) -> list[tuple[str, str]]:
    return _parse_delimited(blob, _SKILL_RE)


def _parse_delimited(blob: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    matches = list(pattern.finditer(blob))
    if not matches:
        return []
    result: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
        content = blob[start:end].strip()
        result.append((name, content))
    return result


def restore_candidate(
    workspace: AgentWorkspace, candidate: dict[str, str], config: EvolveConfig
) -> None:
    system_prompt = _candidate_text(candidate, "system_prompt")
    fragments = (
        _parse_candidate_sections(
            _candidate_text(candidate, "prompt_fragments"),
            _FRAGMENT_RE,
            "prompt_fragments",
        )
        if "prompt_fragments" in candidate
        else None
    )
    skills = (
        _parse_candidate_sections(
            _candidate_text(candidate, "skills"),
            _SKILL_RE,
            "skills",
        )
        if "skills" in candidate
        else None
    )
    memories = (
        _parse_memory_entries(_candidate_text(candidate, "memory"))
        if "memory" in candidate
        else None
    )

    if system_prompt is not None:
        workspace.write_prompt(system_prompt)
    if fragments is not None:
        for name in workspace.list_fragments():
            (workspace.prompts_dir / "fragments" / name).unlink()
        for name, content in fragments:
            workspace.write_fragment(name, content)
    if skills is not None:
        for skill in workspace.list_skills():
            workspace.delete_skill(skill.name)
        for name, content in skills:
            workspace.write_skill(name, content)
    if memories is not None:
        _replace_memory(workspace, memories)


def restore_memory(workspace: AgentWorkspace, memory_blob: str) -> None:
    entries = _parse_memory_entries(memory_blob)
    _replace_memory(workspace, entries)


def _candidate_text(candidate: dict[str, str], component: str) -> str | None:
    if component not in candidate:
        return None
    value = candidate[component]
    if not isinstance(value, str):
        raise CandidateFormatError(
            f"{component} must be text, got {type(value).__name__}"
        )
    return value


def _parse_candidate_sections(
    blob: str | None,
    pattern: re.Pattern,
    component: str,
) -> list[tuple[str, str]]:
    if blob is None or not blob.strip():
        return []
    matches = list(pattern.finditer(blob))
    if not matches or blob[: matches[0].start()].strip():
        raise CandidateFormatError(
            f"{component} must use its required section delimiter"
        )
    sections = _parse_delimited(blob, pattern)
    seen: set[str] = set()
    for name, _ in sections:
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise CandidateFormatError(
                f"{component} contains an invalid section name: {name!r}"
            )
        if name in seen:
            raise CandidateFormatError(
                f"{component} contains a duplicate section name: {name!r}"
            )
        seen.add(name)
    return sections


def _parse_memory_entries(memory_blob: str | None) -> list[tuple[dict, str]]:
    if memory_blob is None:
        return []
    entries: list[tuple[dict, str]] = []
    for line_number, line in enumerate(memory_blob.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateFormatError(
                f"memory line {line_number} must be a valid JSON object: {exc.msg}"
            ) from exc
        if not isinstance(entry, dict):
            raise CandidateFormatError(
                f"memory line {line_number} must be a JSON object, "
                f"got {type(entry).__name__}"
            )
        category = entry.pop("_category", "episodic")
        if not isinstance(category, str) or not category.strip():
            raise CandidateFormatError(
                f"memory line {line_number} has an invalid _category"
            )
        if category in {".", ".."} or "/" in category or "\\" in category:
            raise CandidateFormatError(
                f"memory line {line_number} has an unsafe _category: {category!r}"
            )
        entries.append((entry, category))
    return entries


def _replace_memory(
    workspace: AgentWorkspace, entries: list[tuple[dict, str]]
) -> None:
    if workspace.memory_dir.exists():
        for f in workspace.memory_dir.glob("*.jsonl"):
            f.unlink()
    for entry, category in entries:
        workspace.add_memory(entry, category=category)
