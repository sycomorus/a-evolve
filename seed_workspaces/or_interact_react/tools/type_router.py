from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def type_router(evidence_text: str, existing_branches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Route visible task evidence to relevant workspace skills and memory."""
    root = Path(__file__).resolve().parents[1]
    skills = _load_skills(root)
    memories = _load_memories(root)
    branch_summaries = existing_branches or []
    judge = _judge_route(
        evidence_text=evidence_text,
        existing_branches=branch_summaries,
    )
    branch_route = _branch_route(judge)
    task_types = _normalize_types(judge.get("task_types") or _task_type_from_branch(branch_route["branch_name"]))
    selected_skills = _select_skills(skills, task_types)
    selected_memories = _select_memories(memories, task_types)
    checklist = _merge_checklists(selected_skills + selected_memories)
    return {
        "task_types": task_types,
        "selected_skills": [
            {
                "path": item["path"],
                "name": item["name"],
                "content": item["content"],
                "checklist": item["checklist"],
            }
            for item in selected_skills
        ],
        "selected_memories": [
            {
                "path": item["path"],
                "content": item["content"],
                "checklist": item["checklist"],
            }
            for item in selected_memories
        ],
        "required_checklist": checklist,
        "rationale": str(judge.get("rationale") or ""),
        **branch_route,
    }


def _judge_route(
    evidence_text: str,
    existing_branches: list[dict[str, Any]],
) -> dict[str, Any]:
    system = (
        "You are a task-type router for an operations-research harness tree. "
        "Use only the supplied visible evidence and existing branch summaries. "
        "Return JSON with only branch_name, confidence, and rationale. "
        "Use an existing branch_name only when it clearly matches the visible evidence; "
        "otherwise propose a new branch/<slug> name. "
        "Do not select skills or memories; branch-local harness content is managed "
        "downstream by the workspace overlay."
    )
    user = json.dumps(
        {
            "visible_evidence": evidence_text[:12000],
            "existing_branches": existing_branches,
        },
        ensure_ascii=False,
    )
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("OR_REACT_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OR_REACT_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        )
        response = client.chat.completions.create(
            model=os.environ.get("OR_REACT_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _parse_json_object(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        fallback_types = _heuristic_types(evidence_text)
        return {
            "branch_name": _first_specific_type(fallback_types),
            "confidence": 0.5,
            "rationale": f"LLM routing unavailable; used keyword fallback: {type(exc).__name__}.",
            "task_types": fallback_types,
        }
    fallback_types = _heuristic_types(evidence_text)
    return {
        "branch_name": _first_specific_type(fallback_types),
        "confidence": 0.5,
        "rationale": "Router returned invalid JSON; used keyword fallback.",
        "task_types": fallback_types,
    }


def _branch_route(judge: dict[str, Any]) -> dict[str, Any]:
    raw_name = str(judge.get("branch_name") or "").strip()
    try:
        confidence = float(judge.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "branch_name": _sanitize_branch_name(raw_name or "general"),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _task_type_from_branch(branch_name: str) -> str:
    return _sanitize_branch_name(branch_name).removeprefix("branch/") or "general"


def _first_specific_type(task_types: list[str]) -> str:
    for task_type in task_types:
        if task_type and task_type != "general":
            return task_type
    return task_types[0] if task_types else "general"


def _sanitize_branch_name(value: str) -> str:
    text = value.strip()
    if text.startswith("branch/"):
        text = text[len("branch/") :]
    text = text.lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"[-.]+$", "", text).strip("-._")
    if not text:
        text = "general"
    return f"branch/{text[:80]}"


def _load_skills(root: Path) -> list[dict[str, Any]]:
    skills_dir = root / "skills"
    if not skills_dir.is_dir():
        return []
    skills: list[dict[str, Any]] = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        meta = _frontmatter(text)
        skills.append(
            {
                "path": str(skill_file.relative_to(root)),
                "name": str(meta.get("name") or skill_file.parent.name),
                "description": str(meta.get("description") or ""),
                "types": _normalize_types(meta.get("types")),
                "checklist": _normalize_checklist(meta.get("checklist")),
                "content": text,
            }
        )
    return skills


def _load_memories(root: Path) -> list[dict[str, Any]]:
    memory_dir = root / "memory"
    if not memory_dir.is_dir():
        return []
    memories: list[dict[str, Any]] = []
    for jsonl in sorted(memory_dir.glob("*.jsonl")):
        for index, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            memories.append(
                {
                    "path": f"{jsonl.relative_to(root)}:{index}",
                    "content": content,
                    "types": _normalize_types(entry.get("types")),
                    "checklist": _normalize_checklist(entry.get("checklist")),
                }
            )
    return memories


def _frontmatter(text: str) -> dict[str, Any]:
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    parsed = yaml.safe_load(match.group(1)) or {}
    return parsed if isinstance(parsed, dict) else {}


def _select_skills(skills: list[dict[str, Any]], task_types: list[str]) -> list[dict[str, Any]]:
    wanted = set(task_types) | {"general"}
    return [item for item in skills if wanted.intersection(item["types"])]


def _select_memories(memories: list[dict[str, Any]], task_types: list[str]) -> list[dict[str, Any]]:
    wanted = set(task_types) | {"general"}
    return [item for item in memories if wanted.intersection(item["types"])]


def _merge_checklists(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for item in items:
        for check in item.get("checklist", []):
            check_id = check["id"]
            if check_id not in seen:
                seen.add(check_id)
                merged.append(check)
    return merged


def _heuristic_types(text: str) -> list[str]:
    lowered = text.lower()
    types = ["general"]
    keyword_types = [
        ("piecewise_discount", ("piecewise", "discount", "tier", "segment", "range")),
        ("routing_vrp", ("route", "routing", "vehicle", "tour", "tsp", "travel")),
        ("multi_period", ("period", "inventory", "workforce", "schedule", "week", "month")),
        ("logic_activation", ("binary", "activation", "setup", "fixed cost", "implies")),
    ]
    for type_name, keywords in keyword_types:
        if any(keyword in lowered for keyword in keywords):
            types.append(type_name)
    return types


def _normalize_types(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = ["general"]
    normalized = [str(item).strip() for item in items if str(item).strip()]
    return normalized or ["general"]


def _normalize_checklist(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            prompt = str(item.get("prompt", "")).strip()
            check_id = str(item.get("id", f"check_{index}")).strip()
        else:
            prompt = str(item).strip()
            check_id = f"check_{index}"
        if prompt:
            items.append({"id": check_id or f"check_{index}", "prompt": prompt})
    return items


def _parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
