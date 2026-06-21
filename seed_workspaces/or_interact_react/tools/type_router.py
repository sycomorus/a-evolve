from __future__ import annotations

import json
import os
import re
from typing import Any

TASK_CATEGORY_ENV = "OR_INTERACT_TASK_CATEGORY"


def type_router(evidence_text: str, existing_branches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Route visible task evidence to a broad OR harness-tree branch."""
    metadata_category = _metadata_category()
    if metadata_category:
        return {
            "branch_name": _sanitize_branch_name(metadata_category),
            "confidence": 1.0,
            "rationale": f"task metadata category: {metadata_category}",
        }

    branch_summaries = existing_branches or []
    judge = _judge_route(
        evidence_text=evidence_text,
        existing_branches=branch_summaries,
    )
    branch_route = _branch_route(judge)
    return {
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
        "Choose a broad OR problem family branch, not a narrow instance-specific "
        "subproblem label. Good branch families include TSP, CVRP, vehicle routing, "
        "network flow, assignment, scheduling, facility location, inventory planning, "
        "production planning, bin packing, knapsack, and travel planning. "
        "Use an existing branch_name only when it clearly matches the visible evidence; "
        "otherwise propose a new branch/<slug> name. "
        "Do not select skills, memories, tools, or checklists; downstream code only "
        "uses the returned branch route."
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
        fallback_branch = _heuristic_branch(evidence_text)
        return {
            "branch_name": fallback_branch,
            "confidence": 0.5,
            "rationale": f"LLM routing unavailable; used keyword fallback: {type(exc).__name__}.",
        }
    fallback_branch = _heuristic_branch(evidence_text)
    return {
        "branch_name": fallback_branch,
        "confidence": 0.5,
        "rationale": "Router returned invalid JSON; used keyword fallback.",
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


def _metadata_category() -> str | None:
    text = str(os.environ.get(TASK_CATEGORY_ENV) or "").strip()
    return text or None


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


def _heuristic_branch(text: str) -> str:
    lowered = text.lower()
    branch_keywords = [
        ("cvrp", ("cvrp", "vehicle capacity", "capacitated vehicle", "capacity vehicle")),
        ("tsp", ("tsp", "traveling salesman", "travelling salesman", "tour")),
        ("vehicle-routing", ("route", "routing", "vehicle", "vrp")),
        ("network-flow", ("network flow", "flow", "arc", "node", "source", "sink")),
        ("scheduling", ("schedule", "scheduling", "machine", "job", "shift")),
        ("facility-location", ("facility", "location", "warehouse", "depot")),
        ("inventory-planning", ("inventory", "stock", "replenishment")),
        ("production-planning", ("production", "plant", "manufacturing")),
        ("assignment", ("assignment", "assign", "matching")),
        ("bin-packing", ("bin packing", "packing", "bin")),
        ("knapsack", ("knapsack", "budget", "select items")),
        ("travel-planning", ("travel planning", "itinerary", "trip")),
    ]
    for branch_name, keywords in branch_keywords:
        if any(keyword in lowered for keyword in keywords):
            return f"branch/{branch_name}"
    return "branch/general"


def _parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
