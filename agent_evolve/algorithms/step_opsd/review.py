"""Step-OPSD teacher review and redaction helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ...llm.base import LLMMessage, LLMProvider
from ...types import Observation

MAX_TEXT_CHARS = 20000
MAX_OUTPUT_CHARS = 1200
MAX_ARG_CHARS = 600

TEACHER_SYSTEM_PROMPT = """\
You are a training-time retrospective teacher for an OR agent.

You may use privileged oracle/reference information to diagnose the student trajectory.
Do not solve the task again.
Do not reveal oracle/reference content in your output, including oracle objective values,
reference solution code, reference formulation code, paths, or task-specific answer parameters.

Return JSON only with:
- task_id
- overall_diagnosis
- step_reviews: list of {step_id, phase, credit, support, error_type, reason,
  better_next_action, harness_update_hint}
- missed_steps: list of {after_step_id, phase, expected_action, why_it_matters,
  harness_update_hint}
- leakage_check: {contains_oracle_value, contains_reference_code}
"""


def build_step_opsd_records(
    observations: list[Observation],
    base_records: list[dict[str, Any]],
    *,
    evolution_dir: Path,
    llm: LLMProvider | None,
    failures_only: bool = True,
    max_tokens: int = 2048,
) -> list[dict[str, Any]]:
    """Attach Step-OPSD trace views and redacted teacher reviews to records."""
    full_cleaned_dir = evolution_dir / "step_opsd" / "full_cleaned"
    full_cleaned_dir.mkdir(parents=True, exist_ok=True)

    enriched: list[dict[str, Any]] = []
    for obs, base_record in zip(observations, base_records):
        record = dict(base_record)
        full_cleaned = _clean_trace(obs.trajectory.conversation or obs.trajectory.steps)
        steps = _segment_steps(full_cleaned)
        full_cleaned_path = _write_full_cleaned(
            full_cleaned_dir,
            obs.task.id,
            full_cleaned,
        )
        privileged_packet = _build_privileged_packet(obs, steps)

        should_review = bool(llm is not None) and (
            not failures_only or not bool(obs.feedback.success)
        )
        teacher_review: dict[str, Any]
        redaction_status = "skipped"
        if should_review:
            teacher_review, redaction_status = _run_teacher_review(
                llm=llm,
                obs=obs,
                steps=steps,
                privileged_packet=privileged_packet,
                max_tokens=max_tokens,
            )
        else:
            teacher_review = {
                "task_id": obs.task.id,
                "overall_diagnosis": "Teacher review skipped.",
                "step_reviews": [],
                "missed_steps": [],
            }

        record["trace_views"] = {
            "teacher_full_cleaned_path": str(full_cleaned_path),
            "evolve_compressed": _compress_for_evolver(
                obs=obs,
                steps=steps,
                teacher_review=teacher_review,
                redaction_status=redaction_status,
            ),
        }
        record["step_opsd"] = {
            "teacher_review": teacher_review,
            "redaction_status": redaction_status,
        }
        enriched.append(record)
    return enriched


def sanitize_feedback_detail(value: Any) -> str:
    """Remove oracle/reference content from benchmark feedback before Evolver sees it."""
    text = _jsonish(value)
    if not text:
        return ""

    sanitized_lines: list[str] = []
    skipping_reference_block = False
    reference_fence_open = False
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if skipping_reference_block:
            if stripped.startswith("```"):
                if reference_fence_open:
                    skipping_reference_block = False
                    reference_fence_open = False
                else:
                    reference_fence_open = True
            continue
        if lowered.startswith("reference solution code"):
            skipping_reference_block = True
            reference_fence_open = False
            continue
        if lowered.startswith("reference solution path"):
            continue
        if "reference_solution.py" in lowered or "/oracle/" in lowered:
            continue
        if lowered.startswith("expected objective:"):
            sanitized_lines.append("Expected objective: <redacted>")
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines).strip()


def redacted_step_opsd_for_evolver(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the Step-OPSD subset that may be shown to the Evolver."""
    step_opsd = record.get("step_opsd")
    if not isinstance(step_opsd, dict):
        return None
    status = step_opsd.get("redaction_status")
    review = step_opsd.get("teacher_review")
    if status != "passed" or not isinstance(review, dict):
        return {
            "redaction_status": status or "missing",
            "teacher_review": {
                "overall_diagnosis": "Teacher review unavailable or withheld.",
                "step_reviews": [],
                "missed_steps": [],
            },
        }
    return {
        "redaction_status": "passed",
        "teacher_review": {
            "overall_diagnosis": review.get("overall_diagnosis", ""),
            "step_reviews": _redacted_step_reviews(review.get("step_reviews")),
            "missed_steps": _redacted_missed_steps(review.get("missed_steps")),
        },
    }


def summarize_step_opsd_batch(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate redacted Step-OPSD reviews into a compact batch summary."""
    step_records = [redacted_step_opsd_for_evolver(record) for record in records]
    step_records = [record for record in step_records if record is not None]
    if not step_records:
        return None

    negative = Counter()
    missed = Counter()
    positive = Counter()
    for record in step_records:
        review = record.get("teacher_review", {})
        if not isinstance(review, dict):
            continue
        for step in review.get("step_reviews", []) or []:
            if not isinstance(step, dict):
                continue
            key = (str(step.get("phase", "")), str(step.get("error_type", "")))
            credit = step.get("credit")
            if credit == "negative":
                negative[key] += 1
            elif credit == "positive":
                positive[key] += 1
        for item in review.get("missed_steps", []) or []:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("phase", "")), str(item.get("expected_action", "")))
            missed[key] += 1

    return {
        "records": len(records),
        "reviewed_records": sum(1 for record in step_records if record.get("redaction_status") == "passed"),
        "top_negative_patterns": _counter_items(negative),
        "top_missed_steps": _counter_items(missed),
        "positive_patterns": _counter_items(positive),
    }


def _run_teacher_review(
    *,
    llm: LLMProvider,
    obs: Observation,
    steps: list[dict[str, Any]],
    privileged_packet: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    prompt = json.dumps(
        {
            "visible_task_summary": {
                "task_id": obs.task.id,
                "task_input": obs.task.input,
                "metadata": _visible_metadata(obs.task.metadata),
            },
            "teacher_full_cleaned_steps": steps,
            "privileged_feedback_packet": privileged_packet,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    try:
        response = llm.complete(
            [
                LLMMessage(role="system", content=TEACHER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        review = _parse_json_response(response.content)
    except Exception as exc:
        return (
            {
                "task_id": obs.task.id,
                "overall_diagnosis": f"Teacher review failed: {type(exc).__name__}: {exc}",
                "step_reviews": [],
                "missed_steps": [],
            },
            "failed",
        )

    leaked, reasons = _contains_leakage(review, privileged_packet)
    if leaked:
        return (
            {
                "task_id": obs.task.id,
                "overall_diagnosis": "Teacher review withheld by leakage check.",
                "step_reviews": [],
                "missed_steps": [],
                "leakage_check": {"blocked": True, "reasons": reasons},
            },
            "failed",
        )
    review.setdefault("task_id", obs.task.id)
    review.setdefault("step_reviews", [])
    review.setdefault("missed_steps", [])
    review.setdefault("leakage_check", {})
    return review, "passed"


def _build_privileged_packet(obs: Observation, steps: list[dict[str, Any]]) -> dict[str, Any]:
    raw = obs.feedback.raw or {}
    task_dir = Path(str(raw.get("task_dir") or obs.task.metadata.get("task_dir", "")))
    evaluation = raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {}
    return {
        "task_id": obs.task.id,
        "dataset": obs.task.metadata.get("dataset"),
        "success": obs.feedback.success,
        "score": obs.feedback.score,
        "evaluation_detail": obs.feedback.detail,
        "runtime_summary": _read_runtime_summary(raw.get("runtime_dir")),
        "solver_feedback": _extract_solver_feedback(steps),
        "answer_check_feedback": evaluation.get("error") or obs.feedback.detail,
        "oracle_feedback": {
            "available": bool(evaluation),
            "expected_objective": evaluation.get("expected"),
            "predicted_objective": evaluation.get("predicted"),
            "relative_error": evaluation.get("relative_error"),
            "absolute_error": evaluation.get("absolute_error"),
            "tolerance": evaluation.get("relative_tolerance"),
        },
        "reference_solution": _read_reference_solution(task_dir),
        "reference_formulation": _read_reference_formulation(task_dir),
        "grounded_clarifications": _read_grounded_clarifications(task_dir),
        "step_records": steps,
    }


def _compress_for_evolver(
    *,
    obs: Observation,
    steps: list[dict[str, Any]],
    teacher_review: dict[str, Any],
    redaction_status: str,
) -> dict[str, Any]:
    safe_review = redacted_step_opsd_for_evolver({
        "step_opsd": {
            "teacher_review": teacher_review,
            "redaction_status": redaction_status,
        }
    })
    return {
        "key_tool_sequence": [
            step.get("action", {}).get("tool")
            for step in steps
            if step.get("action", {}).get("tool")
        ],
        "failure_summary": sanitize_feedback_detail(obs.feedback.detail),
        "final_submitted_quantity": _final_submitted_quantity(steps),
        "teacher_marked_steps": (
            safe_review.get("teacher_review", {}).get("step_reviews", [])
            if isinstance(safe_review, dict)
            else []
        ),
    }


def _clean_trace(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen_consecutive = ""
    for event in events:
        if not isinstance(event, dict):
            continue
        if _is_noise_event(event):
            continue
        compact = _compact_event(event)
        signature = json.dumps(compact, sort_keys=True, ensure_ascii=False, default=str)
        if signature == seen_consecutive:
            continue
        seen_consecutive = signature
        cleaned.append(compact)
    return cleaned


def _segment_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        if event_type in {"tool_call", "assistant_tool_call"} or "name" in event and "arguments" in event:
            tool = str(event.get("name") or event.get("tool") or "")
            args = event.get("arguments", {})
            if not isinstance(args, dict):
                args = _parse_jsonish(args)
            step = {
                "step_id": f"t{len(steps) + 1:03d}",
                "phase": _phase_for_tool(tool),
                "history_summary": "",
                "action": {
                    "tool": tool,
                    "arguments_summary": _short_json(args, MAX_ARG_CHARS),
                },
                "observation": {},
            }
            steps.append(step)
            pending.append(step)
            continue
        if event_type == "tool_output" or "output" in event:
            output = event.get("output", event.get("content", ""))
            step = _match_pending_step(event, pending)
            if step is not None:
                step["observation"] = {
                    "status": _status_from_output(output),
                    "summary": _short_json(output, MAX_OUTPUT_CHARS),
                }
    if steps:
        return steps
    for event in events:
        steps.append({
            "step_id": f"t{len(steps) + 1:03d}",
            "phase": str(event.get("phase") or "context_reading"),
            "history_summary": "",
            "action": {"tool": str(event.get("tool") or event.get("status") or "step")},
            "observation": {"summary": _short_json(event, MAX_OUTPUT_CHARS)},
        })
    return steps


def _write_full_cleaned(directory: Path, task_id: str, events: list[dict[str, Any]]) -> Path:
    path = _unique_path(directory / f"{_safe_name(task_id)}.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return path


def _contains_leakage(
    review: dict[str, Any],
    privileged_packet: dict[str, Any],
) -> tuple[bool, list[str]]:
    text = json.dumps(review, ensure_ascii=False, default=str)
    reasons: list[str] = []
    oracle = privileged_packet.get("oracle_feedback", {})
    if isinstance(oracle, dict):
        for key in ("expected_objective",):
            value = oracle.get(key)
            if value is not None:
                for variant in _number_variants(value):
                    if variant and variant in text:
                        reasons.append(key)
                        break
    forbidden = (
        "reference_solution.py",
        "Reference solution code",
        "reference solution code",
        "oracle/",
        "/oracle",
    )
    for item in forbidden:
        if item in text:
            reasons.append(item)
    return bool(reasons), reasons


def _redacted_step_reviews(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "step_id",
        "phase",
        "credit",
        "support",
        "error_type",
        "reason",
        "better_next_action",
        "harness_update_hint",
    }
    return [
        {key: item.get(key) for key in allowed if key in item}
        for item in value
        if isinstance(item, dict)
    ]


def _redacted_missed_steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    allowed = {
        "after_step_id",
        "phase",
        "expected_action",
        "why_it_matters",
        "harness_update_hint",
    }
    return [
        {key: item.get(key) for key in allowed if key in item}
        for item in value
        if isinstance(item, dict)
    ]


def _extract_solver_feedback(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feedback = []
    for step in steps:
        observation = step.get("observation", {})
        status = observation.get("status")
        if status and status != "ok":
            feedback.append({
                "step_id": step.get("step_id"),
                "status": status,
                "message": observation.get("summary", ""),
            })
    return feedback


def _final_submitted_quantity(steps: list[dict[str, Any]]) -> str:
    for step in reversed(steps):
        if step.get("action", {}).get("tool") == "finalize":
            return str(step.get("action", {}).get("arguments_summary", ""))
    return ""


def _read_runtime_summary(value: Any) -> Any:
    if not value:
        return None
    path = Path(str(value)) / "run_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _read_reference_solution(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / "oracle" / "reference_solution.py"
    if not path.is_file():
        return None
    return {"path": str(path), "code_or_summary": _read_text(path)}


def _read_reference_formulation(task_dir: Path) -> str | None:
    oracle_dir = task_dir / "oracle"
    for pattern in ("reference_formulation.*", "source_prompt.*", "reference_solution.md"):
        for path in sorted(oracle_dir.glob(pattern)):
            if path.is_file():
                return _read_text(path)
    return None


def _read_grounded_clarifications(task_dir: Path) -> list[dict[str, str]]:
    grounded_dir = task_dir / "grounded"
    if not grounded_dir.is_dir():
        return []
    items = []
    for path in sorted(grounded_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        items.append({"path": str(path), "content": _read_text(path)})
    return items


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + f"...[truncated {len(text) - MAX_TEXT_CHARS} chars]"


def _visible_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"task_dir", "source_instance_dir"}
    }


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    compact = dict(event)
    for key in ("content", "output"):
        if key in compact:
            compact[key] = _short_json(compact[key], MAX_TEXT_CHARS)
    if "arguments" in compact:
        compact["arguments"] = _parse_jsonish(compact["arguments"])
    return compact


def _is_noise_event(event: dict[str, Any]) -> bool:
    text = _jsonish(event)
    lowered = text.lower()
    return (
        "tool registry" in lowered
        or "heartbeat" in lowered
        or "tools/schema" in lowered
    )


def _match_pending_step(
    event: dict[str, Any],
    pending: list[dict[str, Any]],
) -> dict[str, Any] | None:
    name = str(event.get("name") or "")
    if name:
        for step in list(pending):
            if step.get("action", {}).get("tool") == name:
                pending.remove(step)
                return step
    return pending.pop(0) if pending else None


def _phase_for_tool(tool: str) -> str:
    if tool in {"read_csv", "read_json"}:
        return "data_inspection"
    if tool.startswith("read_"):
        return "context_reading"
    if tool in {"run_solver", "execute_python", "run_heuristic"}:
        return "solver_execution"
    if tool == "finalize":
        return "finalization"
    return "model_formulation"


def _status_from_output(output: Any) -> str:
    text = _jsonish(output).lower()
    if "infeasible" in text:
        return "infeasible"
    if "traceback" in text or "error" in text or "exception" in text:
        return "error"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "ok"


def _parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("teacher response must be a JSON object")
    return parsed


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except Exception:
        return value
    return parsed


def _short_json(value: Any, limit: int) -> str:
    text = _jsonish(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _counter_items(counter: Counter[tuple[str, str]], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"phase": phase, "pattern": pattern, "count": count}
        for (phase, pattern), count in counter.most_common(limit)
    ]


def _safe_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("._-")
    return text or "task"


def _number_variants(value: Any) -> set[str]:
    variants = {str(value)}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return variants
    variants.add(f"{number:g}")
    variants.add(f"{number:.10g}")
    if number.is_integer():
        variants.add(str(int(number)))
    return variants


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique path for {path}")
