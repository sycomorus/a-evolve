"""Step-OPSD teacher review and redaction helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from user.simulator import safe_user_response_summary

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
Most steps are expected to be acceptable. Do not review every step. In step_reviews,
include only the earliest causally wrong step, or at most 1-2 steps with the largest
impact on the final failure. Omit correct or minor steps. Keep each field concise.

Return JSON only with:
- task_id
- overall_diagnosis
- step_reviews: 1-2 item list of {step_id, phase, credit, support, error_type, reason,
  better_next_action, harness_update_hint}
- missed_steps: list of {after_step_id, phase, expected_action, why_it_matters,
  harness_update_hint}
- leakage_check: {contains_oracle_value, contains_reference_code}
"""
TEACHER_INTERACTION_PROMPT_EXTENSION = """\

Grounded clarifications are also privileged. Do not repeat or paraphrase a grounded answer,
the original grounded question, its file name, or task-specific values from it. Describe only
the abstract information need, the visible evidence that should trigger a question, a reusable
question template, and how the answer should affect the formulation.

Also return interaction_review: {requirement, observed_behavior, decision,
evidence_before_decision, recommended_timing, information_need, question_template,
answer_use, expected_answer_use, harness_update_hint}. Use requirement
required|helpful|unnecessary; observed_behavior answered_ask|refused_ask|no_ask;
decision correct_ask|missed_ask|unnecessary_ask|poor_question|correct_abstention;
answer_use used_correctly|ignored|misused|unavailable. Keep reusable fields abstract
and answer-free.
"""


def build_step_opsd_records(
    observations: list[Observation],
    base_records: list[dict[str, Any]],
    *,
    evolution_dir: Path,
    llm: LLMProvider | None,
    failures_only: bool = True,
    max_tokens: int = 2048,
    interaction_enabled: bool = True,
    parallelism: int = 1,
) -> list[dict[str, Any]]:
    """Attach Step-OPSD trace views and redacted teacher reviews to records."""
    full_cleaned_dir = evolution_dir / "step_opsd" / "full_cleaned"
    full_cleaned_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[
        tuple[Observation, dict[str, Any], list[dict[str, Any]], dict[str, Any], str]
    ] = []
    for obs, base_record in zip(observations, base_records):
        record = _redact_record_for_evolver(base_record)
        full_cleaned = _clean_trace(obs.trajectory.conversation or obs.trajectory.steps)
        steps = _segment_steps(full_cleaned)
        full_cleaned_path = _write_full_cleaned(
            full_cleaned_dir,
            obs.task.id,
            _redact_user_answers(full_cleaned),
        )
        privileged_packet = _build_privileged_packet(
            obs,
            steps,
            interaction_enabled=interaction_enabled,
        )
        prepared.append((obs, record, steps, privileged_packet, str(full_cleaned_path)))

    reviews: list[tuple[dict[str, Any], str] | None] = [None] * len(prepared)
    review_indices = [
        index
        for index, (obs, _record, _steps, _packet, _path) in enumerate(prepared)
        if bool(llm is not None) and (
            not failures_only or not bool(obs.feedback.success)
        )
    ]
    workers = min(max(1, int(parallelism)), len(review_indices)) if review_indices else 0
    if workers <= 1:
        for index in review_indices:
            obs, _record, steps, privileged_packet, _path = prepared[index]
            reviews[index] = _run_teacher_review(
                llm=llm,
                obs=obs,
                steps=steps,
                privileged_packet=privileged_packet,
                max_tokens=max_tokens,
                interaction_enabled=interaction_enabled,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_teacher_review,
                    llm=llm,
                    obs=prepared[index][0],
                    steps=prepared[index][2],
                    privileged_packet=prepared[index][3],
                    max_tokens=max_tokens,
                    interaction_enabled=interaction_enabled,
                ): index
                for index in review_indices
            }
            for future in as_completed(futures):
                reviews[futures[future]] = future.result()

    enriched: list[dict[str, Any]] = []
    for index, (obs, record, steps, _packet, full_cleaned_path) in enumerate(prepared):
        review_result = reviews[index]
        if review_result is None:
            teacher_review = {
                "task_id": obs.task.id,
                "overall_diagnosis": "Teacher review skipped.",
                "step_reviews": [],
                "missed_steps": [],
            }
            if interaction_enabled:
                teacher_review["interaction_review"] = {}
            redaction_status = "skipped"
        else:
            teacher_review, redaction_status = review_result

        record["trace_views"] = {
            "teacher_full_cleaned_path": full_cleaned_path,
            "evolve_compressed": _compress_for_evolver(
                obs=obs,
                steps=steps,
                teacher_review=teacher_review,
                redaction_status=redaction_status,
                interaction_enabled=interaction_enabled,
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


def redacted_step_opsd_for_evolver(
    record: dict[str, Any],
    *,
    interaction_enabled: bool = True,
) -> dict[str, Any] | None:
    """Return the Step-OPSD subset that may be shown to the Evolver."""
    step_opsd = record.get("step_opsd")
    if not isinstance(step_opsd, dict):
        return None
    status = step_opsd.get("redaction_status")
    review = step_opsd.get("teacher_review")
    if status != "passed" or not isinstance(review, dict):
        withheld_review = {
            "overall_diagnosis": "Teacher review unavailable or withheld.",
            "step_reviews": [],
            "missed_steps": [],
        }
        if interaction_enabled:
            withheld_review["interaction_review"] = {}
        return {
            "redaction_status": status or "missing",
            "teacher_review": withheld_review,
        }
    safe_review = {
        "overall_diagnosis": review.get("overall_diagnosis", ""),
        "step_reviews": _redacted_step_reviews(review.get("step_reviews")),
        "missed_steps": _redacted_missed_steps(review.get("missed_steps")),
    }
    if interaction_enabled:
        safe_review["interaction_review"] = _redacted_interaction_review(
            review.get("interaction_review")
        )
    return {
        "redaction_status": "passed",
        "teacher_review": safe_review,
    }


def summarize_step_opsd_batch(
    records: list[dict[str, Any]],
    *,
    interaction_enabled: bool = True,
) -> dict[str, Any] | None:
    """Aggregate redacted Step-OPSD reviews into a compact batch summary."""
    step_records = [
        redacted_step_opsd_for_evolver(
            record,
            interaction_enabled=interaction_enabled,
        )
        for record in records
    ]
    step_records = [record for record in step_records if record is not None]
    if not step_records:
        return None

    negative = Counter()
    missed = Counter()
    positive = Counter()
    interaction = Counter()
    answer_use = Counter()
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
        interaction_review = review.get("interaction_review", {})
        if interaction_enabled and isinstance(interaction_review, dict):
            decision = str(interaction_review.get("decision", ""))
            if decision:
                interaction[decision] += 1
            use = str(interaction_review.get("answer_use", ""))
            if use:
                answer_use[use] += 1

    summary = {
        "records": len(records),
        "reviewed_records": sum(1 for record in step_records if record.get("redaction_status") == "passed"),
        "top_negative_patterns": _counter_items(negative),
        "top_missed_steps": _counter_items(missed),
        "positive_patterns": _counter_items(positive),
    }
    if interaction_enabled:
        summary["interaction_decisions"] = _label_counter_items(interaction)
        summary["answer_use_patterns"] = _label_counter_items(answer_use)
    return summary


def review_observation_for_audit(
    obs: Observation,
    *,
    output_dir: Path,
    llm: LLMProvider,
    max_tokens: int = 4096,
    interaction_enabled: bool = True,
) -> dict[str, Any]:
    """Run one teacher review and write the exact teacher context/response."""
    task_dir = output_dir / _safe_name(obs.task.id)
    full_cleaned_dir = task_dir / "full_cleaned"
    task_dir.mkdir(parents=True, exist_ok=True)
    full_cleaned_dir.mkdir(parents=True, exist_ok=True)

    full_cleaned = _clean_trace(obs.trajectory.conversation or obs.trajectory.steps)
    steps = _segment_steps(full_cleaned)
    full_cleaned_path = _write_full_cleaned(full_cleaned_dir, obs.task.id, full_cleaned)
    privileged_packet = _build_privileged_packet(
        obs,
        steps,
        interaction_enabled=interaction_enabled,
    )
    prompt_payload = _teacher_prompt_payload(obs, steps, privileged_packet)
    context_text = json.dumps(prompt_payload, ensure_ascii=False, indent=2, default=str)

    raw_response = ""
    response_metadata: dict[str, Any] = {
        "max_tokens": max_tokens,
        "content_chars": 0,
        "usage": {},
        "finish_reason": None,
        "raw_api_response": None,
    }
    parsed_review: dict[str, Any] | None = None
    redaction_status = "failed"
    leakage_reasons: list[str] = []
    error: str | None = None
    teacher_system_prompt = (
        TEACHER_SYSTEM_PROMPT + TEACHER_INTERACTION_PROMPT_EXTENSION
        if interaction_enabled
        else TEACHER_SYSTEM_PROMPT
    )
    try:
        response = llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=teacher_system_prompt,
                ),
                LLMMessage(role="user", content=context_text),
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw_response = response.content
        response_metadata = _response_metadata(response, max_tokens=max_tokens)
        parsed_review = _parse_json_response(raw_response)
        if not interaction_enabled:
            parsed_review.pop("interaction_review", None)
        leaked, leakage_reasons = _contains_leakage(parsed_review, privileged_packet)
        redaction_status = "withheld" if leaked else "passed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    (task_dir / "teacher_system_prompt.txt").write_text(
        teacher_system_prompt,
        encoding="utf-8",
    )
    (task_dir / "teacher_context.json").write_text(context_text, encoding="utf-8")
    (task_dir / "teacher_raw_response.txt").write_text(raw_response, encoding="utf-8")
    (task_dir / "teacher_response_metadata.json").write_text(
        json.dumps(response_metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if parsed_review is not None:
        (task_dir / "teacher_parsed_review.json").write_text(
            json.dumps(parsed_review, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    record = {
        "task_id": obs.task.id,
        "success": obs.feedback.success,
        "score": obs.feedback.score,
        "redaction_status": redaction_status,
        "leakage_reasons": leakage_reasons,
        "error": error,
        "teacher_system_prompt_path": str(task_dir / "teacher_system_prompt.txt"),
        "teacher_context_path": str(task_dir / "teacher_context.json"),
        "teacher_raw_response_path": str(task_dir / "teacher_raw_response.txt"),
        "teacher_response_metadata_path": str(task_dir / "teacher_response_metadata.json"),
        "teacher_parsed_review_path": (
            str(task_dir / "teacher_parsed_review.json")
            if parsed_review is not None
            else None
        ),
        "teacher_full_cleaned_path": str(full_cleaned_path),
    }
    (task_dir / "review_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return record


def _run_teacher_review(
    *,
    llm: LLMProvider,
    obs: Observation,
    steps: list[dict[str, Any]],
    privileged_packet: dict[str, Any],
    max_tokens: int,
    interaction_enabled: bool,
) -> tuple[dict[str, Any], str]:
    prompt = json.dumps(
        _teacher_prompt_payload(obs, steps, privileged_packet),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    try:
        response = llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        TEACHER_SYSTEM_PROMPT + TEACHER_INTERACTION_PROMPT_EXTENSION
                        if interaction_enabled
                        else TEACHER_SYSTEM_PROMPT
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        review = _parse_json_response(response.content)
    except Exception as exc:
        failure_review = {
            "task_id": obs.task.id,
            "overall_diagnosis": f"Teacher review failed: {type(exc).__name__}: {exc}",
            "step_reviews": [],
            "missed_steps": [],
        }
        if interaction_enabled:
            failure_review["interaction_review"] = {}
        return failure_review, "failed"

    review.setdefault("task_id", obs.task.id)
    review.setdefault("step_reviews", [])
    review.setdefault("missed_steps", [])
    if interaction_enabled:
        review.setdefault("interaction_review", {})
    else:
        review.pop("interaction_review", None)
    review.setdefault("leakage_check", {})
    leaked, reasons = _contains_leakage(review, privileged_packet)
    if leaked:
        withheld_review = {
            "task_id": obs.task.id,
            "overall_diagnosis": "Teacher review withheld because privileged content leaked.",
            "step_reviews": [],
            "missed_steps": [],
            "leakage_check": {"reasons": reasons},
        }
        if interaction_enabled:
            withheld_review["interaction_review"] = {}
        return withheld_review, "withheld"
    return review, "passed"


def _teacher_prompt_payload(
    obs: Observation,
    steps: list[dict[str, Any]],
    privileged_packet: dict[str, Any],
) -> dict[str, Any]:
    return {
        "visible_task_summary": {
            "task_id": obs.task.id,
            "task_input": obs.task.input,
            "metadata": _visible_metadata(obs.task.metadata),
        },
        "teacher_full_cleaned_steps": steps,
        "privileged_feedback_packet": privileged_packet,
    }


def _build_privileged_packet(
    obs: Observation,
    steps: list[dict[str, Any]],
    *,
    interaction_enabled: bool,
) -> dict[str, Any]:
    raw = obs.feedback.raw or {}
    task_dir = Path(str(raw.get("task_dir") or obs.task.metadata.get("task_dir", "")))
    evaluation = raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {}
    packet = {
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
    }
    if interaction_enabled:
        packet["grounded_clarifications"] = _read_grounded_clarifications(task_dir)
    return packet


def _compress_for_evolver(
    *,
    obs: Observation,
    steps: list[dict[str, Any]],
    teacher_review: dict[str, Any],
    redaction_status: str,
    interaction_enabled: bool,
) -> dict[str, Any]:
    safe_review = redacted_step_opsd_for_evolver(
        {
            "step_opsd": {
                "teacher_review": teacher_review,
                "redaction_status": redaction_status,
            }
        },
        interaction_enabled=interaction_enabled,
    )
    compressed = {
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
    if interaction_enabled:
        compressed["teacher_interaction_review"] = (
            safe_review.get("teacher_review", {}).get("interaction_review", {})
            if isinstance(safe_review, dict)
            else {}
        )
    return compressed


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
    recent_assistant = ""
    for event in events:
        event_type = event.get("type")
        if event_type == "assistant_message":
            recent_assistant = _short_json(event.get("content", ""), MAX_OUTPUT_CHARS)
            continue
        if event_type in {"tool_call", "assistant_tool_call"} or "name" in event and "arguments" in event:
            tool = str(event.get("name") or event.get("tool") or "")
            args = event.get("arguments", {})
            if not isinstance(args, dict):
                args = _parse_jsonish(args)
            step = {
                "step_id": f"t{len(steps) + 1:03d}",
                "phase": _phase_for_tool(tool),
                "history_summary": recent_assistant,
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
                step["observation"] = _step_observation(
                    str(step.get("action", {}).get("tool") or ""),
                    output,
                )
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
    review_for_check = {
        key: value
        for key, value in review.items()
        if key not in {"task_id", "leakage_check"}
    }
    text = json.dumps(review_for_check, ensure_ascii=False, default=str)
    folded_text = text.casefold()
    reasons: list[str] = []
    oracle = privileged_packet.get("oracle_feedback", {})
    if isinstance(oracle, dict):
        for key in ("expected_objective",):
            value = oracle.get(key)
            if value is not None:
                for variant in _number_variants(value):
                    if variant and re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])",
                        text,
                    ):
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
        if item.casefold() in folded_text:
            reasons.append(item)
    for grounded in privileged_packet.get("grounded_clarifications", []) or []:
        if not isinstance(grounded, dict):
            continue
        path = str(grounded.get("path") or "")
        if path and (
            path.casefold() in folded_text
            or Path(path).name.casefold() in folded_text
        ):
            reasons.append("grounded_path")
        content = str(grounded.get("content") or "")
        for secret in _grounded_secrets(content):
            if secret.casefold() in folded_text:
                reasons.append("grounded_content")
                break
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


def _redacted_interaction_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "requirement",
        "observed_behavior",
        "decision",
        "evidence_before_decision",
        "recommended_timing",
        "information_need",
        "question_template",
        "answer_use",
        "expected_answer_use",
        "harness_update_hint",
    }
    return {key: value.get(key) for key in allowed if key in value}


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


def _redact_user_answers(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted: list[dict[str, Any]] = []
    for event in events:
        compact = dict(event)
        if compact.get("name") == "ask_user":
            if compact.get("type") == "tool_output":
                output = compact.get("output", compact.get("content", {}))
                compact["output"] = _safe_user_response(output)
                compact.pop("content", None)
            elif compact.get("type") in {"tool_call", "assistant_tool_call"}:
                compact["arguments"] = {}
        tool_calls = compact.get("tool_calls")
        if isinstance(tool_calls, list):
            safe_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    safe_calls.append(tool_call)
                    continue
                safe_call = dict(tool_call)
                function = safe_call.get("function")
                if isinstance(function, dict) and function.get("name") == "ask_user":
                    safe_function = dict(function)
                    safe_function["arguments"] = (
                        "{}" if isinstance(safe_function.get("arguments"), str) else {}
                    )
                    safe_call["function"] = safe_function
                elif safe_call.get("name") == "ask_user":
                    safe_call["arguments"] = (
                        "{}" if isinstance(safe_call.get("arguments"), str) else {}
                    )
                safe_calls.append(safe_call)
            compact["tool_calls"] = safe_calls
        redacted.append(compact)
    return redacted


def _redact_record_for_evolver(record: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(record)
    if "feedback_detail" in redacted:
        redacted["feedback_detail"] = sanitize_feedback_detail(
            redacted.get("feedback_detail")
        )
    for key in ("conversation", "steps"):
        value = redacted.get(key)
        if isinstance(value, list):
            redacted[key] = _redact_user_answers(value)
    trajectory = redacted.get("trajectory")
    if isinstance(trajectory, dict):
        safe_trajectory = dict(trajectory)
        steps = safe_trajectory.get("steps")
        if isinstance(steps, list):
            safe_trajectory["steps"] = _redact_user_answers(steps)
        redacted["trajectory"] = safe_trajectory
    task = redacted.get("task")
    if isinstance(task, dict):
        safe_task = dict(task)
        metadata = safe_task.get("metadata")
        if isinstance(metadata, dict):
            safe_task["metadata"] = _visible_metadata(metadata)
        redacted["task"] = safe_task
    feedback = redacted.get("feedback")
    if isinstance(feedback, dict):
        safe_feedback = dict(feedback)
        safe_feedback["detail"] = sanitize_feedback_detail(
            safe_feedback.get("detail")
        )
        safe_feedback.pop("raw", None)
        redacted["feedback"] = safe_feedback
    return redacted


def _safe_user_response(output: Any) -> dict[str, Any]:
    return {"user_response": safe_user_response_summary(output)}


def _step_observation(tool: str, output: Any) -> dict[str, Any]:
    if tool == "ask_user":
        safe = _safe_user_response(output)
        answered = bool(safe["user_response"]["answered"])
        return {
            "status": "answered" if answered else "refused",
            "summary": safe,
        }
    return {
        "status": _status_from_output(output),
        "summary": _short_json(output, MAX_OUTPUT_CHARS),
    }


def _grounded_secrets(content: str) -> list[str]:
    secrets: list[str] = []
    for heading in ("Question", "Answer"):
        match = re.search(
            rf"^##\s+{heading}\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
            content,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if match:
            body = match.group("body").strip()
            if len(body) >= 8:
                secrets.append(body)
    return secrets


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
    if tool == "ask_user":
        return "user_interaction"
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


def _response_metadata(response: Any, *, max_tokens: int) -> dict[str, Any]:
    raw_api_response = _jsonable(response.raw)
    return {
        "max_tokens": max_tokens,
        "content_chars": len(response.content or ""),
        "usage": response.usage,
        "finish_reason": _finish_reason(raw_api_response),
        "raw_api_response": raw_api_response,
    }


def _finish_reason(raw_api_response: Any) -> str | None:
    if not isinstance(raw_api_response, dict):
        return None
    choices = raw_api_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    finish_reason = choices[0].get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    return str(value)


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


def _label_counter_items(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"pattern": pattern, "count": count}
        for pattern, count in counter.most_common(limit)
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
