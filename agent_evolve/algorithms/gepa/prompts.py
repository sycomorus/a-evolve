"""Component-specific reflection prompts for structured GEPA candidates."""

from __future__ import annotations


def _reflection_prompt(component: str, format_requirements: str) -> str:
    return f"""You are improving the `{component}` component of an agent workspace.

Current component:
```
<curr_param>
```

Evaluation results:
```
<side_info>
```

Propose a complete drop-in replacement that improves task performance while preserving
useful existing behavior.

Required output format:
{format_requirements}

Return only the replacement inside one triple-backtick block. Do not include explanations
outside the block."""


COMPONENT_REFLECTION_PROMPTS = {
    "system_prompt": _reflection_prompt(
        "system_prompt",
        "Output the complete system prompt as plain text or Markdown.",
    ),
    "prompt_fragments": _reflection_prompt(
        "prompt_fragments",
        "Output either an empty block or one or more sections. Every section must start "
        "with an exact header line `=== FRAGMENT: <filename> ===`, followed by that "
        "fragment's complete text. Do not output unsectioned text.",
    ),
    "skills": _reflection_prompt(
        "skills",
        "Output either an empty block or one or more sections. Every section must start "
        "with an exact header line `=== SKILL: <name> ===`, followed by the complete "
        "SKILL.md content. Do not output booleans or unsectioned text.",
    ),
    "memory": _reflection_prompt(
        "memory",
        "Output either an empty block or JSONL with exactly one JSON object per non-empty "
        "line. Each object may contain an optional string `_category` field. Do not output "
        "JSON arrays, booleans, null, comments, prose, or nested code fences.",
    ),
}
