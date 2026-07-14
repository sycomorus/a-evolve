"""Prompt templates for A-Evolve."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ...contract.workspace import AgentWorkspace
from ..step_opsd import (
    redacted_step_opsd_for_evolver,
    sanitize_feedback_detail,
    summarize_step_opsd_batch,
)

logger = logging.getLogger(__name__)

STANDARD_FEEDBACK_CHAR_LIMIT = 9000
MAIN_CODE_SUMMARY_LIMIT = 500
BRANCH_CODE_SUMMARY_LIMIT = 1400
MAIN_OUTPUT_LIMIT = 500
BRANCH_OUTPUT_LIMIT = 1200
ARG_SUMMARY_LIMIT = 500

DEFAULT_EVOLVER_SYSTEM_PROMPT = """\
You are a meta-learning agent that improves another agent by modifying its workspace files.

The workspace follows a standard directory structure:
- prompts/system.md  -- the agent's system prompt
- skills/*/SKILL.md  -- reusable skill definitions
- skills/_drafts/    -- draft skills from the solver
- memory/*.jsonl     -- episodic and semantic memory
- tools/             -- tool implementations

Your job each cycle:
1. Analyze task observation logs -- identify patterns, common failures, recurring themes
2. Review draft skills -- refine into real skills, merge with existing, or discard
3. Improve the system prompt if needed
4. Update memory with high-level insights, prune redundant entries
5. Use the provided bash tool to read/write files in the workspace
6. Verify your changes with `git diff` before finishing

Guidelines:
- Quality over quantity. Only create skills that genuinely help future tasks.
- The task traces you need are already provided in the evolution prompt. Do not inspect raw
  evolution logs, run traces, or files outside the current workspace.
- Modify only files in the current workspace's prompts, skills, memory, tools, or manifest.
- Skills live at `skills/<name>/SKILL.md`, where `<name>` exactly matches the YAML `name`.
- Skill names must match `^[a-z0-9]+(-[a-z0-9]+)*$`: lowercase kebab-case only; no spaces, underscores, or uppercase.
- Skills use SKILL.md format with YAML frontmatter (name, description).
- Keep memory concise and actionable.
- When modifying files, use precise edits.
"""


def _extract_trajectory_signals(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract structured behavioral signals from a conversation trajectory.

    Analyzes the conversation to produce a compact summary of what happened,
    including success/failure indicators that can be inferred without test results.
    """
    n_turns = 0
    n_tool_calls = 0
    n_errors = 0
    n_timeouts = 0
    tools_used: dict[str, int] = {}
    commands_run: list[str] = []
    repeated_commands: list[str] = []
    submitted = False
    submit_value = ""
    error_messages: list[str] = []

    for msg in conversation:
        role = msg.get("role", "")
        event_type = msg.get("type", "")
        if role == "assistant" or event_type == "assistant_message":
            n_turns += 1
            for tc in msg.get("tool_calls", []):
                n_tool_calls += 1
                fn, args = _tool_call_name_args(tc)
                tools_used[fn] = tools_used.get(fn, 0) + 1
                cmd = args.get("cmd", "") or args.get("command", "")
                if cmd:
                    cmd_short = cmd[:80]
                    commands_run.append(cmd_short)
                if fn in {"submit", "task_submit", "finalize"}:
                    submitted = True
                    submit_value = str(args.get("answer", args.get("objective_value", "")))
        elif event_type == "tool_call":
            n_tool_calls += 1
            fn = str(msg.get("name", ""))
            tools_used[fn] = tools_used.get(fn, 0) + 1
            args = msg.get("arguments", {}) if isinstance(msg.get("arguments", {}), dict) else {}
            cmd = args.get("cmd", "") or args.get("command", "") or args.get("code", "")
            if cmd:
                commands_run.append(str(cmd)[:80])
            if fn in {"submit", "task_submit", "finalize"}:
                submitted = True
                submit_value = str(args.get("answer", args.get("objective_value", "")))
        elif role == "tool" or event_type == "tool_output":
            content = msg.get("content") or _jsonish(msg.get("output", ""))
            if "ERROR:" in content or "error:" in content.lower()[:50]:
                n_errors += 1
                error_messages.append(content[:100])
            if "timed out" in content.lower() or "timeout" in content.lower():
                n_timeouts += 1

    # Detect repeated commands (same command run 3+ times)
    cmd_counts: dict[str, int] = {}
    for c in commands_run:
        cmd_counts[c] = cmd_counts.get(c, 0) + 1
    repeated_commands = [c for c, cnt in cmd_counts.items() if cnt >= 3]

    return {
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "n_errors": n_errors,
        "n_timeouts": n_timeouts,
        "tools_used": tools_used,
        "submitted": submitted,
        "submit_value": submit_value,
        "repeated_commands": repeated_commands,
        "error_snippets": error_messages[:5],
    }


def _summarize_conversation(conversation: list[dict[str, Any]]) -> str:
    """Condense a conversation into a readable trajectory summary.

    Keeps assistant reasoning and full tool call/result pairs.  Skips the
    initial user message (task prompt) since the evolver already has the
    task_id.
    """
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "")
        if role == "assistant":
            text = msg.get("content") or ""
            tool_calls = msg.get("tool_calls", [])
            cmds = []
            for tc in tool_calls:
                fn = tc.get("function", "")
                args = tc.get("arguments", {})
                cmd = args.get("cmd", "") or args.get("command", "")
                if cmd:
                    cmds.append(f"  {fn}({cmd})")
            if text.strip():
                parts.append(f"[assistant] {text.strip()}")
            if cmds:
                parts.append("\n".join(cmds))
        elif role == "tool":
            content = msg.get("content") or ""
            parts.append(f"[tool output] {content.strip()}")
    return "\n".join(parts)


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _shorten_text(value: Any, limit: int) -> str:
    text = _jsonish(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}...[truncated {omitted} chars]"


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _canonical_tool_events(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = [msg for msg in conversation if msg.get("type") in {"tool_call", "tool_output"}]
    if events:
        return events

    # Backward compatibility for older observations that only stored chat-shaped
    # assistant/tool turns. Canonical traces skip this branch and avoid double
    # counting assistant_message.tool_calls.
    converted: list[dict[str, Any]] = []
    for msg in conversation:
        role = msg.get("role", "")
        if role == "assistant":
            for tc in msg.get("tool_calls", []):
                name, args = _tool_call_name_args(tc)
                converted.append({
                    "type": "tool_call",
                    "id": str(tc.get("id", "")),
                    "name": name,
                    "arguments": args,
                })
        elif role == "tool":
            converted.append({
                "type": "tool_output",
                "tool_call_id": str(msg.get("tool_call_id", "")),
                "name": str(msg.get("name", "")),
                "output": msg.get("content", ""),
            })
    return converted


def _code_arg_name(tool_name: str, args: dict[str, Any]) -> str | None:
    preferred = {
        "run_solver": ("solver_code", "code"),
        "execute_python": ("python_code", "code"),
    }.get(tool_name, ())
    for key in preferred:
        if isinstance(args.get(key), str):
            return key
    return None


def _summarize_args(tool_name: str, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    summarized: dict[str, Any] = {}
    code_summary = None
    code_key = _code_arg_name(tool_name, args)
    for key, value in args.items():
        if key == code_key and isinstance(value, str):
            code_summary = _summarize_code(value, profile="main")
            summarized[key] = f"<omitted {len(value)} chars; see code_summary>"
            continue
        if isinstance(value, (dict, list)):
            summarized[key] = _shorten_text(value, ARG_SUMMARY_LIMIT)
        elif isinstance(value, str):
            summarized[key] = _shorten_text(value, ARG_SUMMARY_LIMIT)
        else:
            summarized[key] = value
    return summarized, code_summary


def _extract_evolve_summary(code: str, *, limit: int) -> dict[str, Any] | None:
    marker = "# EVOLVE_SUMMARY:"
    index = code.rfind(marker)
    if index < 0:
        return None

    fields: dict[str, str] = {}
    lines: list[str] = []
    for raw_line in code[index:].splitlines()[1:]:
        stripped = raw_line.strip()
        if not stripped.startswith("#"):
            if stripped:
                break
            continue
        text = stripped[1:].strip()
        if not text:
            continue
        lines.append(text)
        if ":" in text:
            key, value = text.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            if normalized_key in {"purpose", "data_inputs", "model_or_check", "objective_or_output"}:
                fields[normalized_key] = value.strip()

    summary_text = "\n".join(lines)
    return {
        "source": "EVOLVE_SUMMARY",
        "chars": len(code),
        "fields": fields,
        "text": _shorten_text(summary_text, limit),
    }


def _summarize_code(code: str, *, profile: str) -> dict[str, Any]:
    limit = BRANCH_CODE_SUMMARY_LIMIT if profile == "branch" else MAIN_CODE_SUMMARY_LIMIT
    explicit = _extract_evolve_summary(code, limit=limit)
    if explicit is not None:
        return explicit

    lines = code.splitlines()
    imports = [
        line.strip()
        for line in lines
        if line.lstrip().startswith(("import ", "from "))
    ][: 12 if profile == "branch" else 5]
    context_reads = _regex_unique(
        r"['\"]((?:docs|data)/[^'\"]+)['\"]",
        code,
        limit=12 if profile == "branch" else 5,
    )
    modeling_patterns = (
        "LpVariable",
        "LpProblem",
        "addVar",
        "addVars",
        "Model(",
        ".addConstr",
        ".add_constraint",
        "constraint",
        "objective",
        "minimize",
        "maximize",
        ".optimize",
        ".solve",
    )
    modeling = [
        line.strip()
        for line in lines
        if any(pattern in line for pattern in modeling_patterns)
    ][: 12 if profile == "branch" else 5]
    outputs = [
        line.strip()
        for line in lines
        if "print(" in line or "finalize" in line or "submitted_answer" in line
    ][: 10 if profile == "branch" else 4]
    return {
        "source": "fallback_static",
        "chars": len(code),
        "imports": [_shorten_text(line, 160) for line in imports],
        "context_reads": context_reads,
        "modeling": [_shorten_text(line, 220) for line in modeling],
        "outputs": [_shorten_text(line, 220) for line in outputs],
    }


def _regex_unique(pattern: str, text: str, *, limit: int) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for match in re.findall(pattern, text):
        value = str(match)
        if value in seen:
            continue
        seen.add(value)
        items.append(value)
        if len(items) >= limit:
            break
    return items


def _is_error_output(output: Any) -> bool:
    text = _jsonish(output)
    lowered = text.lower()
    if any(marker in lowered[:1000] for marker in ("traceback", "error", "exception", "timed out", "timeout")):
        return True
    if isinstance(output, dict):
        for key in ("solver_run", "python_run"):
            run = output.get(key)
            if isinstance(run, dict) and int(run.get("exit_code") or 0) != 0:
                return True
    return False


def _output_status(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    status: dict[str, Any] = {}
    for key in ("solver_run", "python_run"):
        run = output.get(key)
        if isinstance(run, dict):
            if "exit_code" in run:
                status["exit_code"] = run.get("exit_code")
            if "status" in run:
                status["status"] = run.get("status")
            if isinstance(run.get("files_written"), list):
                status["files_written"] = run.get("files_written")[:20]
    for key in ("status", "exit_code", "files_written"):
        if key in output and key not in status:
            status[key] = output[key]
    return status


def _summarize_output(output: Any, *, profile: str) -> tuple[Any, str]:
    limit = BRANCH_OUTPUT_LIMIT if profile == "branch" else MAIN_OUTPUT_LIMIT
    if _is_error_output(output):
        return _output_status(output), _shorten_text(output, limit)

    status = _output_status(output)
    if isinstance(output, dict):
        snippets: dict[str, str] = {}
        for run_key in ("solver_run", "python_run"):
            run = output.get(run_key)
            if not isinstance(run, dict):
                continue
            for stream in ("stdout", "stderr"):
                text = str(run.get(stream) or "")
                if text:
                    snippets[f"{run_key}.{stream}_tail"] = _shorten_text(text[-limit:], limit)
        if snippets:
            return {**status, **snippets}, ""
        if status:
            return status, ""
    text = _shorten_text(output, limit)
    return text, ""


def build_tool_trace(conversation: list[dict[str, Any]], profile: str = "main") -> list[dict[str, Any]]:
    """Build a structured, bounded trace from canonical tool_call/tool_output events."""
    if profile not in {"main", "branch"}:
        profile = "main"

    trace: list[dict[str, Any]] = []
    pending_by_id: dict[str, dict[str, Any]] = {}
    pending_without_id: list[dict[str, Any]] = []

    for event in _canonical_tool_events(conversation):
        event_type = event.get("type")
        if event_type == "tool_call":
            name = str(event.get("name", ""))
            arguments = event.get("arguments", {})
            args = arguments if isinstance(arguments, dict) else _parse_arguments(arguments)
            summarized_args, code_summary = _summarize_args(name, args)
            if code_summary is not None and profile == "branch":
                code_key = _code_arg_name(name, args)
                if code_key and isinstance(args.get(code_key), str):
                    code_summary = _summarize_code(args[code_key], profile="branch")
            entry: dict[str, Any] = {
                "step": len(trace) + 1,
                "tool": name,
                "args": summarized_args,
                "output": None,
                "error": "",
            }
            if code_summary is not None:
                entry["code_summary"] = code_summary
            trace.append(entry)
            event_id = str(event.get("id") or event.get("tool_call_id") or "")
            if event_id:
                pending_by_id[event_id] = entry
            pending_without_id.append(entry)
            continue

        if event_type != "tool_output":
            continue
        output = event.get("output", event.get("content", ""))
        entry = _matching_tool_entry(event, pending_by_id, pending_without_id)
        if entry is None:
            continue
        if entry.get("tool") == "ask_user":
            entry["output"] = _summarize_ask_user_output(output)
            entry["error"] = ""
            continue
        entry["output"], entry["error"] = _summarize_output(output, profile=profile)

    return trace


def _summarize_ask_user_output(output: Any) -> dict[str, Any]:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            output = {}
    response = output.get("user_response", {}) if isinstance(output, dict) else {}
    return {"answered": bool(response.get("answered"))}


def _matching_tool_entry(
    event: dict[str, Any],
    pending_by_id: dict[str, dict[str, Any]],
    pending_without_id: list[dict[str, Any]],
) -> dict[str, Any] | None:
    tool_call_id = str(event.get("tool_call_id") or event.get("id") or "")
    if tool_call_id and tool_call_id in pending_by_id:
        entry = pending_by_id.pop(tool_call_id)
        if entry in pending_without_id:
            pending_without_id.remove(entry)
        return entry

    name = str(event.get("name", ""))
    if name:
        for entry in list(pending_without_id):
            if entry.get("tool") == name:
                pending_without_id.remove(entry)
                return entry
    if pending_without_id:
        return pending_without_id.pop(0)
    return None


def _tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function", {})
    if isinstance(function, dict):
        name = str(function.get("name", ""))
        args = _parse_arguments(function.get("arguments", {}))
        return name, args
    return (
        str(tool_call.get("name", function or "")),
        _parse_arguments(tool_call.get("arguments", {})),
    )


def _event_command(name: str, args: dict[str, Any]) -> str:
    command = args.get("cmd", "") or args.get("command", "") or args.get("code", "")
    if command:
        return str(command)
    if name:
        compact_args = _jsonish(args)
        return f"{name}({compact_args[:180]})"
    return ""


def _compress_trajectory(conversation: list[dict[str, Any]]) -> str:
    """Compress a trajectory into a failure-focused summary.

    Instead of the full conversation, extracts:
    - The task approach (first few commands)
    - All errors and their context (command that caused them)
    - Loops/thrashing (repeated commands)
    - The final few commands and their results
    - Whether submit was called

    This gives the evolver specific failure patterns to address rather than
    a wall of text to parse.
    """
    events: list[dict[str, str]] = []
    prev_cmd = ""

    for msg in conversation:
        role = msg.get("role", "")
        event_type = msg.get("type", "")
        if role == "assistant" or event_type == "assistant_message":
            for tc in msg.get("tool_calls", []):
                fn, args = _tool_call_name_args(tc)
                cmd = _event_command(fn, args)
                answer = args.get("answer", args.get("objective_value", ""))
                if fn in ("submit", "task_submit", "finalize"):
                    events.append({"type": "submit", "value": answer})
                elif cmd:
                    prev_cmd = cmd[:200]
                    events.append({"type": "cmd", "fn": fn, "cmd": prev_cmd})
        elif event_type == "tool_call":
            fn = str(msg.get("name", ""))
            args = msg.get("arguments", {}) if isinstance(msg.get("arguments", {}), dict) else {}
            cmd = _event_command(fn, args)
            if fn in ("submit", "task_submit", "finalize"):
                events.append({
                    "type": "submit",
                    "value": str(args.get("answer", args.get("objective_value", ""))),
                })
            elif cmd:
                prev_cmd = cmd[:200]
                events.append({"type": "cmd", "fn": fn, "cmd": prev_cmd})
        elif role == "tool" or event_type == "tool_output":
            content = (msg.get("content") or _jsonish(msg.get("output", ""))).strip()
            is_error = (
                "ERROR:" in content
                or "error:" in content[:80].lower()
                or "Traceback" in content[:200]
                or "TIMEOUT" in content.upper()[:50]
                or "timed out" in content.lower()[:80]
                or "No such file" in content[:100]
                or "command not found" in content[:100]
            )
            if is_error:
                events.append({
                    "type": "error",
                    "cmd": prev_cmd,
                    "output": content[:300],
                })

    # Build compressed summary
    parts: list[str] = []
    n_cmds = sum(1 for e in events if e["type"] == "cmd")
    n_errors = sum(1 for e in events if e["type"] == "error")
    submitted = any(e["type"] == "submit" for e in events)

    parts.append(f"Commands: {n_cmds}, Errors: {n_errors}, Submitted: {submitted}")

    # First 3 commands (approach)
    cmds_seen = 0
    for e in events:
        if e["type"] == "cmd":
            cmds_seen += 1
            if cmds_seen <= 3:
                parts.append(f"[start] {e['fn']}({e['cmd']})")

    # All errors with context
    if n_errors > 0:
        parts.append(f"\n--- Errors ({n_errors}) ---")
        for e in events:
            if e["type"] == "error":
                parts.append(f"  cmd: {e.get('cmd', '?')}")
                parts.append(f"  err: {e['output'][:200]}")

    # Detect loops
    cmd_list = [e["cmd"] for e in events if e["type"] == "cmd"]
    cmd_counts: dict[str, int] = {}
    for c in cmd_list:
        cmd_counts[c] = cmd_counts.get(c, 0) + 1
    loops = {c: n for c, n in cmd_counts.items() if n >= 3}
    if loops:
        parts.append("\n--- Repeated commands ---")
        for c, n in loops.items():
            parts.append(f"  {c} (x{n})")

    # Last 3 commands
    last_cmds = [e for e in events if e["type"] == "cmd"][-3:]
    if last_cmds:
        parts.append("\n--- Final commands ---")
        for e in last_cmds:
            parts.append(f"  {e['fn']}({e['cmd']})")

    if submitted:
        submit_events = [e for e in events if e["type"] == "submit"]
        if submit_events:
            parts.append(f"\n[submitted] {submit_events[-1].get('value', '')}")

    return "\n".join(parts)


JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether an AI agent successfully completed a command-line task.
You can ONLY see the agent's actions (commands run and their outputs). You do NOT have access to the actual test results.
Based on the trajectory, estimate whether the task was completed successfully."""

JUDGE_USER_TEMPLATE = """\
Task: {task_id}

Agent trajectory:
{trajectory}

Based on this trajectory, evaluate the agent's performance:
1. Score (0-10): 0=complete failure, 5=partial progress, 10=likely fully solved
2. Category: What type of task is this? (build, debug, data-science, security, scientific, system-admin, software-engineering, etc.)
3. Outcome: One sentence describing what happened.
4. Failure reason: If score < 7, what specific thing went wrong? Be concrete.

Respond in JSON format:
{{"score": N, "category": "...", "outcome": "...", "failure_reason": "..."}}"""


def judge_trajectories(
    logs: list[dict[str, Any]],
    llm: Any | None = None,
    model_id: str = "us.anthropic.claude-opus-4-6-v1",
    region: str = "us-west-2",
) -> list[dict[str, Any]]:
    """Use an LLM to score each trajectory as a proxy for success/failure.

    Returns a list of judge verdicts, one per log entry.
    Each verdict has: score (0-10), category, outcome, failure_reason.
    """
    try:
        from ...llm.base import LLMMessage
        if llm is None:
            from ...llm.bedrock import BedrockProvider
            llm = BedrockProvider(model_id=model_id, region=region)
    except ImportError:
        logger.warning("LLM judge provider not available, skipping judge")
        return [{"score": -1, "category": "unknown", "outcome": "judge unavailable", "failure_reason": ""} for _ in logs]

    verdicts = []

    for log in logs:
        conversation = log.get("conversation", [])
        task_id = log.get("task_id", "unknown")
        compressed = _compress_trajectory(conversation)

        prompt = JUDGE_USER_TEMPLATE.format(task_id=task_id, trajectory=compressed)

        try:
            response = llm.complete(
                messages=[
                    LLMMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=prompt),
                ],
                max_tokens=300,
                temperature=0.0,
            )
            # Parse JSON from response
            text = response.content.strip()
            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            verdict = json.loads(text)
            verdicts.append(verdict)
            logger.info("Judge: %s → score=%s, category=%s", task_id, verdict.get("score"), verdict.get("category"))
        except Exception as e:
            logger.warning("Judge failed for %s: %s", task_id, str(e)[:100])
            verdicts.append({"score": -1, "category": "unknown", "outcome": f"judge error: {e}", "failure_reason": ""})

    return verdicts


def build_evolution_prompt(
    workspace: AgentWorkspace,
    logs: list[dict[str, Any]],
    drafts: list[dict[str, str]],
    evo_number: int,
    *,
    evolve_prompts: bool = True,
    evolve_skills: bool = True,
    evolve_memory: bool = True,
    evolve_tools: bool = False,
    trajectory_only: bool = False,
    max_skills: int = 5,
    solver_proposed: bool = False,
    prompt_only: bool = False,
    protect_skills: bool = False,
    judge_llm: Any | None = None,
    scope_instruction: str | None = None,
    evolution_instruction: str | None = None,
    trajectory_profile: str = "main",
) -> str:
    """Build the user-message prompt for one evolution cycle.

    When ``trajectory_only`` is True, the evolver sees only the agent's
    trajectory (tool calls and outputs) — no pass/fail, score, or test
    feedback.  This forces the meta-learner to infer improvement
    opportunities from behavior alone.
    """
    summaries = []
    recent_logs = logs[-30:]

    # In trajectory-only mode, run the LLM judge to get proxy outcome signals
    verdicts = []
    if trajectory_only and recent_logs:
        try:
            verdicts = judge_trajectories(recent_logs, llm=judge_llm)
        except Exception as e:
            logger.warning("Judge step failed, proceeding without verdicts: %s", str(e)[:100])
            verdicts = []

    for i, log in enumerate(recent_logs):
        if trajectory_only:
            conversation = log.get("conversation", [])
            signals = _extract_trajectory_signals(conversation)
            entry: dict[str, Any] = {
                "task_id": log.get("task_id", ""),
                "signals": signals,
                "tool_trace": build_tool_trace(conversation, trajectory_profile),
            }
            if i < len(verdicts) and verdicts[i].get("score", -1) >= 0:
                entry["judge_verdict"] = verdicts[i]
            summaries.append(entry)
        else:
            conversation = log.get("conversation", [])
            entry = {
                "task_id": log.get("task_id", ""),
                "success": log.get("success", False),
                "score": log.get("score", 0.0),
                "feedback": sanitize_feedback_detail(
                    log.get("feedback_detail", "")
                )[:STANDARD_FEEDBACK_CHAR_LIMIT],
                "signals": _extract_trajectory_signals(conversation),
                "tool_trace": build_tool_trace(conversation, trajectory_profile),
            }
            trace_views = log.get("trace_views", {})
            compressed = (
                trace_views.get("evolve_compressed")
                if isinstance(trace_views, dict)
                else None
            )
            if isinstance(compressed, dict):
                entry["evolve_compressed"] = compressed
            step_opsd = redacted_step_opsd_for_evolver(log)
            if step_opsd is not None:
                entry["step_opsd"] = step_opsd
            summaries.append(entry)

    skills = workspace.list_skills()
    skill_names = [s.name for s in skills]

    draft_section = "No draft skills this batch."
    if drafts:
        parts = []
        for d in drafts:
            # Show full content for solver-proposed drafts (they're the primary input)
            max_len = 4000 if solver_proposed else 1000
            content = d['content'][:max_len]
            parts.append(f"#### Draft: {d['name']}\n```markdown\n{content}\n```")
        draft_section = "\n\n".join(parts)

    permission_lines = []
    if evolve_prompts:
        permission_lines.append("- You CAN modify prompts/system.md")
    if evolve_skills:
        if protect_skills:
            permission_lines.append(
                "- You CAN create NEW skills in skills/<kebab-name>/SKILL.md but MUST NOT modify or delete existing skills"
            )
        else:
            permission_lines.append("- You CAN create/modify/delete skills in skills/<kebab-name>/SKILL.md")
    if evolve_memory:
        permission_lines.append("- You CAN add/prune entries in memory/*.jsonl")
    if evolve_tools:
        permission_lines.append("- You CAN create/modify tools in tools/")

    if trajectory_only:
        summary_heading = _build_trajectory_only_heading()
        if prompt_only:
            instruction_lines = _build_prompt_only_instructions()
        elif solver_proposed and drafts:
            instruction_lines = _build_solver_proposed_instructions(len(skill_names), max_skills=max_skills)
        else:
            instruction_lines = _build_trajectory_only_instructions(len(skill_names), max_skills=max_skills, protect_skills=protect_skills)
    else:
        summary_heading = _build_standard_heading()
        instruction_lines = _build_standard_instructions()

    scope_section = f"### Scope\n{scope_instruction}" if scope_instruction else ""
    evolution_instruction_section = ""
    if evolution_instruction:
        evolution_instruction_section = (
            f"\n\n### Run-Specific Evolution Guidance\n{evolution_instruction}"
        )
    step_opsd_batch = summarize_step_opsd_batch(recent_logs)
    step_opsd_section = ""
    if step_opsd_batch is not None:
        step_opsd_section = (
            "\n\n### Step-OPSD Batch Summary\n"
            "This section contains only redacted teacher-derived signals. "
            "Do not infer or write task-specific oracle/reference content.\n"
            "```json\n"
            f"{json.dumps(step_opsd_batch, indent=2, ensure_ascii=False)}\n"
            "```"
        )

    return f"""\
## Evolution Cycle #{evo_number}

### Permissions
{chr(10).join(permission_lines)}

{scope_section}{evolution_instruction_section}

{summary_heading}
```json
{json.dumps(summaries, indent=2)}
```
{step_opsd_section}

### Draft Skills
{draft_section}

### Current Skills ({len(skill_names)})
{chr(10).join(f'- {s}' for s in skill_names) if skill_names else 'No skills yet.'}

### Instructions
{instruction_lines}

When done, summarize what you changed and why.
"""


def _build_trajectory_only_heading() -> str:
    return """\
### Agent Behavior Analysis (this batch)

You can ONLY see the agent's actions. You do NOT have access to actual test results.

Each task includes:
- `signals`: automated behavioral metrics (turns, errors, timeouts, submission status, loops)
- `tool_trace`: structured tool calls and bounded outputs. Code tools include compact summaries
  instead of full source.
- `judge_verdict`: An LLM judge's assessment of whether the agent likely succeeded. Includes:
  - `score` (0-10): 0=complete failure, 5=partial, 10=likely solved
  - `category`: task type (build, debug, data-science, security, etc.)
  - `outcome`: what happened
  - `failure_reason`: specific thing that went wrong (if score < 7)

**Use judge scores to prioritize your work:**
- **Score 0-3 (FAILED)**: Agent clearly failed. Analyze the failure_reason and trajectory to understand WHY.
- **Score 4-6 (PARTIAL)**: Agent made progress but likely didn't finish. Look for what blocked it.
- **Score 7-10 (LIKELY SOLVED)**: Agent probably succeeded. Skip these — do not create skills from them.

**Group failures by category.** If multiple tasks in the same category failed for similar reasons, that's a pattern worth addressing with a category-specific skill."""


def _build_standard_heading() -> str:
    return """\
### Task Summaries (this batch)

Each task includes:
- `success`, `score`, and `feedback`: real benchmark feedback from evaluation.
- `signals`: automated behavior metrics extracted from the trajectory.
- `tool_trace`: structured tool calls and bounded outputs. Code tools include compact summaries
  instead of full source.
- When Step-OPSD is enabled, `evolve_compressed`, `step_opsd`, and the batch summary contain
  redacted teacher-derived diagnosis for reusable harness evolution.
- Interaction reviews describe when clarification was required, whether the question was useful,
  and whether the answer was applied. They never contain the grounded answer. Improve reusable
  prompt/skill policy: ask only for material information unavailable from docs/data, ask one
  concise evidence-based question, never request the oracle/final objective, apply an answer to
  the formulation, and do not repeat a refused synonymous question.

Oracle/reference values, reference paths, and reference code are not available to you. Use only
the redacted feedback, trajectory fields, and redacted Teacher output to identify reusable
patterns before changing prompts, skills, memory, or tools."""


def _build_trajectory_only_instructions(current_skill_count: int, max_skills: int = 5, protect_skills: bool = False) -> str:
    skill_budget_note = ""
    if current_skill_count >= max_skills:
        skill_budget_note = f"""
**SKILL BUDGET REACHED ({current_skill_count}/{max_skills}).** You MUST NOT create new skills.
Instead: refine existing skills with new patterns from this batch's failures."""
    elif current_skill_count > 0:
        remaining = max_skills - current_skill_count
        skill_budget_note = f"""
**Skill budget: {current_skill_count}/{max_skills} used ({remaining} remaining).**"""

    protect_note = ""
    if protect_skills:
        protect_note = """
**EXISTING SKILLS ARE READ-ONLY.** You MUST NOT modify or delete any existing skill files. \
You may ONLY create NEW skills. Existing skills have been validated and optimized — do not touch them."""

    modify_or_create = """3. **For each pattern with 2+ failed tasks**, either:
   - **Refine an existing skill** if it covers that category but missed the specific failure.
   - **Create a new skill** if no existing skill covers that failure category."""
    if protect_skills:
        modify_or_create = """3. **For each pattern with 2+ failed tasks**, create a NEW skill targeting that failure category. \
Do NOT modify existing skills."""

    return f"""\
**You may ONLY modify skills.** Do NOT modify prompts/system.md, memory, or tools.
{protect_note}
Skills are loaded on demand by the agent via `read_skill(name)`. The agent sees skill \
names and descriptions in its system prompt, and decides which to read. Good skills are \
ones the agent will actually choose to read and benefit from.

**Analysis steps:**
1. **Sort tasks by judge score.** List each task with its score, category, and failure_reason.
2. **Identify failure patterns.** Group failed tasks (score < 7) by category and failure reason.
{modify_or_create}
4. **If failures are diverse** (all different categories/reasons), focus on the lowest-scoring tasks.
5. **Skip tasks with score >= 7** — the agent likely solved them without help.
{skill_budget_note}

**Skill quality requirements:**
- Store each skill at `skills/<name>/SKILL.md`; the directory `<name>` and YAML frontmatter `name` MUST be identical
- `name` must be short, descriptive lowercase kebab-case matching `^[a-z0-9]+(-[a-z0-9]+)*$`
- Never use spaces, underscores, uppercase letters, slashes, or display-title names in `name`
- `description` must clearly say WHEN this skill applies (the agent decides to read based on this)
- Body must contain domain-specific knowledge the agent couldn't infer on its own
- Max 2000 chars per skill — concise and actionable
- Include verification steps (how to confirm the task is solved)

**FORBIDDEN — do NOT write any of the following (the agent already knows these):**
- Timeout handling, package installation tips, session persistence warnings
- Generic debugging advice, command chaining tips
- Any advice about HOW to use bash/python tools

**REQUIRED — only write domain knowledge the agent does NOT already have:**
- Specific libraries/tools/commands needed for a task category
- Verification steps that prove a task category is solved
- Common domain-specific pitfalls and how to avoid them

Use the workspace_bash tool to read/write files. Verify with `git diff`."""


def _build_prompt_only_instructions() -> str:
    return """\
**You may ONLY modify `prompts/system.md`.** Do NOT create, modify, or delete skills. Do NOT modify memory.

The agent's system prompt controls HOW it approaches tasks — its problem-solving strategy, \
not domain knowledge. Analyze the trajectories to identify BEHAVIORAL PATTERNS that lead to \
failure, then add concise strategy rules to the system prompt.

**Good strategy rules (change HOW the agent works):**
- "When a build fails, read the full error message before attempting a fix"
- "For unfamiliar file formats, use `file` and `head` to inspect before writing parsers"
- "If a command produces no output, verify it ran correctly before proceeding"
- "When multiple approaches exist, try the simplest one first"
- "Before submitting, verify the solution meets ALL requirements in the task description"

**BAD rules (domain knowledge — the agent already knows these):**
- Specific library names, API calls, or package versions
- Task-category-specific review procedures
- Tool installation instructions

**Constraints:**
1. Read the current `prompts/system.md` first.
2. ADD at most 3-5 short rules (one line each) per evolution cycle. Do not rewrite the prompt.
3. Keep total prompt under 2000 characters. Brevity is critical — every extra character dilutes attention.
4. Rules must be GENERAL (apply to many task types), not specific to any one task.
5. Use the workspace_bash tool to read/write files.
6. Verify your changes with `git diff` before finishing."""


def _build_solver_proposed_instructions(current_skill_count: int, max_skills: int = 5) -> str:
    skill_budget_note = ""
    if current_skill_count >= max_skills:
        skill_budget_note = f"""
**SKILL BUDGET REACHED ({current_skill_count}/{max_skills}).** You MUST NOT create new skills.
You may only UPDATE existing skills by merging in useful content from drafts."""
    elif current_skill_count > 0:
        remaining = max_skills - current_skill_count
        skill_budget_note = f"""
**Skill budget: {current_skill_count}/{max_skills} used ({remaining} remaining).**"""

    return f"""\
The **Draft Skills** above were proposed by the solver agent after completing each task. \
The solver had full access to the task environment (files, errors, tools) — it knows \
what specific knowledge would have helped.

**Your job is to GENERALIZE these drafts into reusable skills.**

The solver drafts are task-specific. Your job is to extract the GENERAL PRINCIPLES that \
apply to a whole CATEGORY of tasks, not just the specific task the solver saw.

For each draft:
1. **Extract the general principle.** E.g., "pMARS needs X11 flags removed" → "When building legacy C projects, check for optional graphics/GUI dependencies and disable them if not needed."
2. **Merge into the right category skill.** ADD the generalized knowledge to the appropriate existing skill. Do NOT replace existing content — append new sections.
3. **Reject task-specific details.** Specific file paths, specific package versions, specific API calls that only apply to one task — skip these.

**CRITICAL RULES:**
- **NEVER shrink existing skills.** Only ADD content. If a skill is 3500 chars, it should be ≥3500 chars after your changes.
- **NEVER replace existing skill content** with draft content. Existing content was already validated. Only append new sections.
- If no draft contains generalizable knowledge for a category, leave that skill unchanged.
{skill_budget_note}
5. **Do NOT modify prompts/system.md.** The base prompt is optimal. Only update skills.
6. Use the workspace_bash tool to read/write files in the workspace.
7. Verify your changes with `git diff` before finishing. Confirm skills did not shrink.

**FORBIDDEN — do NOT keep any of the following in skills (the agent already knows these):**
- Timeout handling advice (background processes, nohup, checking if processes are running)
- Package installation tips (apt-get, pip, --break-system-packages)
- Generic debugging advice (read error messages, check logs)
- Command chaining advice (use &&, combine commands)
- Any advice about HOW to use bash/python tools

**REQUIRED — only keep domain knowledge the agent does NOT already have:**
- What specific libraries/tools are needed for a task category
- What verification steps prove a task category is solved
- What common domain-specific mistakes to avoid

**SIZE GUIDANCE: Each skill should be 2000-4000 characters.** Prefer the upper end."""


def _build_standard_instructions() -> str:
    return """\
1. Review the task summaries -- identify patterns, common failures, recurring themes
2. Review draft skills -- decide: refine into a real skill, merge with existing, or discard
3. Review current skills -- any need updating based on new evidence?
4. Review memory -- prune redundant entries, add high-level insights
5. Use the workspace_bash tool to read/write files in the workspace
6. Verify your changes with `git diff` before finishing"""
