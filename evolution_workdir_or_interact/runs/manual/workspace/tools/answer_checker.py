from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CHECKLIST = [
    {"id": "objective_unit", "prompt": "Verify the submitted value has the same unit as the visible request."},
    {"id": "visible_constraints", "prompt": "Verify each non-obvious constraint is supported by visible evidence."},
    {"id": "warning_resolution", "prompt": "Verify known warnings were resolved before final submission."},
]


def answer_checker(
    compressed_trace: str,
    final_code: str,
    objective_value: str,
    task_types: str,
    harness_notes: str,
) -> dict[str, Any]:
    """Check final OR modeling work against selected workspace harness checklists."""
    root = Path(__file__).resolve().parents[1]
    types = _parse_task_types(task_types)
    checks = _checks_for_types(root, types)
    judge = _judge_answer(
        compressed_trace=compressed_trace,
        final_code=final_code,
        objective_value=objective_value,
        task_types=types,
        harness_notes=harness_notes,
        checklist=checks,
    )
    return _normalize_result(judge, checks)


def _judge_answer(
    compressed_trace: str,
    final_code: str,
    objective_value: str,
    task_types: list[str],
    harness_notes: str,
    checklist: list[dict[str, str]],
) -> dict[str, Any]:
    system = (
        "You are a skeptical, adversarial pre-submit reviewer for optimization-modeling work. "
        "Use only the supplied trace, code, submitted value, task types, harness notes, "
        "and checklist. Check every checklist item independently. A checklist item passes "
        "only when the supplied materials contain explicit supporting evidence; the agent's "
        "unsupported assertion is not enough. If evidence is missing, ambiguous, or a warning "
        "was not resolved with concrete support, mark that item failed. Return JSON with "
        "passed, failed_items, check_results, and required_fix. Each check_results item must "
        "include id, passed, reason, and a non-empty evidence list of short snippets or "
        "locations from compressed_trace, final_code, objective_value, task_types, or "
        "harness_notes."
    )
    user = json.dumps(
        {
            "compressed_trace": compressed_trace[:12000],
            "final_code": final_code[:20000],
            "objective_value": objective_value,
            "task_types": task_types,
            "harness_notes": harness_notes[:6000],
            "checklist": checklist,
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
        return {
            "passed": False,
            "failed_items": [check["id"] for check in checklist],
            "check_results": [
                {
                    "id": check["id"],
                    "passed": False,
                    "reason": f"LLM answer check unavailable: {type(exc).__name__}.",
                    "evidence": [f"answer_checker exception: {type(exc).__name__}"],
                }
                for check in checklist
            ],
            "required_fix": "Run the checklist manually, revise any issue, then call answer_checker again.",
        }
    return {
        "passed": False,
        "failed_items": [check["id"] for check in checklist],
        "check_results": [
            {
                "id": check["id"],
                "passed": False,
                "reason": "answer_checker returned invalid JSON.",
                "evidence": ["No valid JSON check_results were returned."],
            }
            for check in checklist
        ],
        "required_fix": "answer_checker returned invalid JSON; rerun with clearer trace and harness notes.",
    }


def _checks_for_types(root: Path, task_types: list[str]) -> list[dict[str, str]]:
    wanted = set(task_types) | {"general"}
    checks: list[dict[str, str]] = []
    for item in _load_skills(root) + _load_memories(root):
        if wanted.intersection(item["types"]):
            checks.extend(item["checklist"])
    checks.extend(DEFAULT_CHECKLIST)
    return _dedupe_checks(checks)


def _load_skills(root: Path) -> list[dict[str, Any]]:
    skills_dir = root / "skills"
    if not skills_dir.is_dir():
        return []
    skills: list[dict[str, Any]] = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        meta = _frontmatter(skill_file.read_text(encoding="utf-8"))
        skills.append(
            {
                "types": _normalize_types(meta.get("types")),
                "checklist": _normalize_checklist(meta.get("checklist")),
            }
        )
    return skills


def _load_memories(root: Path) -> list[dict[str, Any]]:
    memory_dir = root / "memory"
    if not memory_dir.is_dir():
        return []
    memories: list[dict[str, Any]] = []
    for jsonl in sorted(memory_dir.glob("*.jsonl")):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            memories.append(
                {
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


def _parse_task_types(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return ["general"]
    try:
        parsed = json.loads(text)
        return _normalize_types(parsed)
    except Exception:
        return _normalize_types([part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()])


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


def _dedupe_checks(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for check in checks:
        check_id = check["id"]
        if check_id not in seen:
            seen.add(check_id)
            deduped.append(check)
    return deduped


def _normalize_result(judge: dict[str, Any], checklist: list[dict[str, str]]) -> dict[str, Any]:
    check_results = judge.get("check_results")
    if not isinstance(check_results, list):
        check_results = [
            {
                "id": check["id"],
                "passed": False,
                "reason": "Missing per-item check_result from reviewer.",
                "evidence": ["No per-item evidence was returned for this checklist item."],
            }
            for check in checklist
        ]
    results_by_id: dict[str, dict[str, Any]] = {}
    for result in check_results:
        if isinstance(result, dict):
            check_id = str(result.get("id") or "").strip()
            if check_id:
                results_by_id[check_id] = result

    normalized_results: list[dict[str, Any]] = []
    failed_items: list[str] = []
    for check in checklist:
        check_id = check["id"]
        result = results_by_id.get(check_id)
        if result is None:
            normalized_results.append(
                {
                    "id": check_id,
                    "passed": False,
                    "reason": "Reviewer omitted this checklist item.",
                    "evidence": ["No evidence was returned for this checklist item."],
                }
            )
            failed_items.append(check_id)
            continue

        reason = str(result.get("reason") or "").strip()
        evidence = _normalize_evidence(result.get("evidence"))
        passed = bool(result.get("passed")) and bool(reason) and bool(evidence)
        if not reason:
            reason = "Missing per-item explanation."
        if not evidence:
            reason = (reason + " " if reason else "") + "Missing explicit evidence."
        normalized_results.append(
            {
                "id": check_id,
                "passed": passed,
                "reason": reason,
                "evidence": evidence,
            }
        )
        if not passed:
            failed_items.append(check_id)
    passed = bool(judge.get("passed", False)) and not failed_items
    return {
        "passed": passed,
        "failed_items": failed_items,
        "check_results": normalized_results,
        "required_fix": "" if passed else str(judge.get("required_fix") or "Fix failed checklist items."),
    }


def _normalize_evidence(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]


def _parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
