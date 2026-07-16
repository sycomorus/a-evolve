"""Interaction diagnostics for OR-Interact trajectories and teacher reviews."""

from __future__ import annotations

from typing import Any


REFUSAL_CODES = (
    "empty_question",
    "no_grounded_records",
    "no_match",
    "matcher_unavailable",
    "invalid_decision",
    "invalid_match",
)


def summarize_interaction_metrics(
    rows: list[dict[str, Any]],
    review_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = [_normalize_row(row) for row in rows]
    grounded = [row for row in normalized if row["has_answerable_grounded"]]
    non_grounded = [row for row in normalized if not row["has_answerable_grounded"]]
    asked_tasks = [row for row in normalized if row["ask_count"] > 0]
    total_asks = sum(row["ask_count"] for row in normalized)
    answered_asks = sum(row["answered_ask_count"] for row in normalized)
    refused_asks = sum(row["refused_ask_count"] for row in normalized)
    refusal_code_totals = {
        code: sum(row[f"{code}_ask_count"] for row in normalized)
        for code in REFUSAL_CODES
    }

    grounded_recalled = sum(
        1 for row in grounded if row["answered_ask_count"] > 0
    )
    precise_ask_tasks = sum(
        1
        for row in asked_tasks
        if row["has_answerable_grounded"] and row["answered_ask_count"] > 0
    )
    correct_abstentions = sum(
        1 for row in non_grounded if row["ask_count"] == 0
    )

    answer_use_values = _answer_use_values(review_records or [])
    used_correctly = sum(1 for value in answer_use_values if value == "used_correctly")

    metrics = {
        "total_tasks": len(normalized),
        "grounded_tasks": len(grounded),
        "non_grounded_tasks": len(non_grounded),
        "ask_precision": _ratio(precise_ask_tasks, len(asked_tasks)),
        "ask_recall": _ratio(grounded_recalled, len(grounded)),
        "grounded_match_rate": _ratio(answered_asks, total_asks),
        "correct_abstention_rate": _ratio(correct_abstentions, len(non_grounded)),
        "answer_utilization_rate": (
            _ratio(used_correctly, len(answer_use_values))
            if answer_use_values
            else None
        ),
        "mean_ask_count": _ratio(total_asks, len(normalized)),
        "refusal_rate": _ratio(refused_asks, total_asks),
        "grounded_accuracy": _accuracy(grounded),
        "non_grounded_accuracy": _accuracy(non_grounded),
        "total_asks": total_asks,
        "answered_asks": answered_asks,
        "refused_asks": refused_asks,
        "reviewed_answer_uses": len(answer_use_values),
    }
    for code, count in refusal_code_totals.items():
        metrics[f"{code}_asks"] = count
        metrics[f"{code}_rate"] = _ratio(count, total_asks)
    return metrics


def _answer_use_values(records: list[dict[str, Any]]) -> list[str]:
    values = []
    for record in records:
        step_opsd = record.get("step_opsd", {})
        if not isinstance(step_opsd, dict) or step_opsd.get("redaction_status") != "passed":
            continue
        review = step_opsd.get("teacher_review", {})
        interaction = review.get("interaction_review", {}) if isinstance(review, dict) else {}
        value = str(interaction.get("answer_use") or "") if isinstance(interaction, dict) else ""
        if value in {"used_correctly", "ignored", "misused"}:
            values.append(value)
    return values


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    has_grounded = _as_bool(row.get("has_grounded"))
    answerable_value = row.get("has_answerable_grounded")
    has_answerable_grounded = (
        _as_bool(answerable_value)
        if answerable_value is not None and answerable_value != ""
        else has_grounded
    )
    normalized = {
        "success": _as_bool(row.get("success")),
        "has_grounded": has_grounded,
        "has_answerable_grounded": has_answerable_grounded,
        "grounded_record_count": _as_int(row.get("grounded_record_count")),
        "ask_count": _as_int(row.get("ask_count")),
        "answered_ask_count": _as_int(row.get("answered_ask_count")),
        "refused_ask_count": _as_int(row.get("refused_ask_count")),
    }
    normalized.update(
        {
            f"{code}_ask_count": _as_int(row.get(f"{code}_ask_count"))
            for code in REFUSAL_CODES
        }
    )
    return normalized


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return _ratio(sum(1 for row in rows if row["success"]), len(rows))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
