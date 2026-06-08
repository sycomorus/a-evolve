"""Enhanced prompt templates using per-claim feedback and task-type analysis.

Builds on code_evolve prompts with:
1. Claim-type performance breakdowns
2. Task-type-specific guidance
3. Judge feedback pattern summaries
4. Meta-evolution history integration
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...contract.workspace import AgentWorkspace
from .base_analysis import BatchAnalysis
from .analyzer import AdaptiveAnalysisResult, ClaimStats, FailurePattern, TaskTypeStats


@dataclass
class AdaptivePromptConfig:
    """Configuration for adaptive evolution prompts."""

    prompt_max_chars: int = 4000
    skill_max_chars: int = 2000
    max_skills: int = 15
    include_claim_details: bool = True
    include_judge_patterns: bool = True
    include_task_type_stats: bool = True
    include_evolution_history: bool = True
    extra_constraints: str = ""
    extra_instructions: str = ""


def build_adaptive_system_prompt(config: AdaptivePromptConfig | None = None) -> str:
    """Build evolver's system prompt with adaptive guidance."""
    cfg = config or AdaptivePromptConfig()

    extra = ""
    if cfg.extra_constraints:
        extra = "\n\n" + cfg.extra_constraints

    return f"""\
You are a meta-learning agent that improves another agent by modifying its workspace files.

The workspace follows a standard directory structure:
- prompts/system.md  -- the agent's system prompt
- skills/*/SKILL.md  -- reusable skill definitions
- memory/*.jsonl     -- episodic and semantic memory

Your job each cycle:
1. Review the DETAILED ANALYSIS provided — includes per-claim failures, task-type performance, and judge feedback patterns.
2. Make TARGETED, EVIDENCE-BASED changes. Focus on the weakest claim types and task types.
3. Use the provided bash tool to read/write files in the workspace.
4. Verify changes with `git diff` before finishing.

## CRITICAL CONSTRAINTS

### System Prompt Rules:
- MUST stay under {cfg.prompt_max_chars} characters.
- NEVER add batch-specific data, task IDs, or exact counts.
- First paragraph (agent identity) must be preserved exactly.
- If prompt is already effective (>85% pass rate), make minimal changes.
- Prefer creating/modifying skills over changing system prompt.

### Skill Rules:
- Each skill under {cfg.skill_max_chars} characters.
- Use YAML frontmatter (name, description, optionally: triggers).
- Store each skill at `skills/<name>/SKILL.md`; the directory `<name>` and YAML frontmatter `name` MUST be identical.
- Skill names must match `^[a-z0-9]+(-[a-z0-9]+)*$`: lowercase kebab-case only; no spaces, underscores, uppercase letters, slashes, or display-title names.
- One skill per concept — avoid overlap.
- Maximum {cfg.max_skills} skills total.
- Name skills by their PURPOSE, not by batch number.

### Memory Rules:
- Keep under 15 entries total. Prune aggressively.
- Store only high-level insights, not task specifics.
- Remove stale or redundant entries every cycle.

### Evidence-Based Evolution:
- **Claim-type analysis**: If "calculate" claims fail at 40%, create calculation skill.
- **Task-type analysis**: If multi-requirement tasks fail at 60%, add requirement extraction protocol.
- **Judge feedback**: If judge consistently says "missing X", add explicit X verification.
- **Failure patterns**: Address systematic issues (multi-req misses, wrong entity, etc.).

### Surgical Scope:
- DON'T change everything every cycle.
- IF pass rate > 85%: Make 0-1 changes maximum.
- IF specific claim type weak: Target ONLY that claim type.
- IF specific task type weak: Create skill for ONLY that task type.
- Quality over quantity. One targeted fix beats five generic changes.

### Meta-Learning:
- Learn from evolution history: what types of changes worked/failed.
- If a previous change improved performance, note the pattern.
- If a previous change hurt performance, avoid similar changes.
- Build on successes, don't redo what already works.{extra}
"""


def build_adaptive_evolution_prompt(
    workspace: AgentWorkspace,
    observations: list[dict[str, Any]],
    analysis: AdaptiveAnalysisResult,
    evo_number: int,
    *,
    evolve_prompts: bool = True,
    evolve_skills: bool = True,
    evolve_memory: bool = True,
    prompt_config: AdaptivePromptConfig | None = None,
    evolution_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build adaptive evolution prompt with rich analysis.

    Args:
        workspace: Agent workspace
        observations: Raw observation logs
        analysis: AdaptiveAnalysisResult with all analysis layers
        evo_number: Current evolution cycle number
        evolve_prompts: Whether to allow prompt modifications
        evolve_skills: Whether to allow skill modifications
        evolve_memory: Whether to allow memory modifications
        prompt_config: Optional prompt configuration
        evolution_history: Optional evolution change history

    Returns:
        Complete evolution prompt string
    """
    cfg = prompt_config or AdaptivePromptConfig()
    base = analysis.base_analysis

    # ── Header ──────────────────────────────────────────────────
    sections = [f"# Evolution Cycle {evo_number}\n"]

    # ── Batch Summary ───────────────────────────────────────────
    sections.append(f"""
## Batch Summary

- **Tasks**: {base.total_tasks} total, {base.passed} passed, {base.failed} failed
- **Pass Rate**: {base.pass_rate:.1%}
- **Code Execution**: {analysis.code_stats.tasks_used_code}/{base.total_tasks} tasks used it
""")

    # ── Claim-Type Performance ──────────────────────────────────
    if cfg.include_claim_details and analysis.claim_stats:
        sections.append("\n## Claim-Type Performance\n")
        sections.append("Which specific types of requirements are failing:\n\n")

        # Show weakest claim types
        for claim_type, pass_rate in analysis.weakest_claim_types:
            stats = analysis.claim_stats[claim_type]
            sections.append(
                f"### {claim_type}: {pass_rate:.0%} pass rate "
                f"({stats.fulfilled} fulfilled, {stats.partial} partial, "
                f"{stats.failed} failed out of {stats.total} claims)\n\n"
            )

            if stats.examples:
                sections.append("**Failed Examples**:\n")
                for ex in stats.examples[:2]:  # Show up to 2
                    sections.append(f"- Task `{ex['task_id']}`: \"{ex['claim']}\"\n")
                    sections.append(f"  *Why failed*: {ex['justification']}\n")
                sections.append("\n")

        sections.append(
            "*Action*: Create skills or prompt sections targeting the weakest claim types.\n\n"
        )

    # ── Task-Type Performance ───────────────────────────────────
    if cfg.include_task_type_stats and analysis.task_type_stats:
        sections.append("\n## Task-Type Performance\n")
        sections.append("How agent performs on different types of tasks:\n\n")

        for task_type, stats in sorted(
            analysis.task_type_stats.items(),
            key=lambda x: x[1].pass_rate
        ):
            sections.append(
                f"- **{task_type}**: {stats.pass_rate:.0%} pass rate "
                f"({stats.passed}/{stats.total} tasks)\n"
            )

        if analysis.weakest_task_types:
            sections.append("\n**Weakest Types**: ")
            sections.append(
                ", ".join(f"{t} ({pr:.0%})" for t, pr in analysis.weakest_task_types[:3])
            )
            sections.append("\n\n*Action*: Create type-specific handling skills for weak types.\n\n")

    # ── Judge Feedback Patterns ─────────────────────────────────
    if cfg.include_judge_patterns and analysis.judge_patterns:
        sections.append("\n## Judge Feedback Patterns\n")
        sections.append("Common reasons judges gave for failing claims:\n\n")

        for pattern_name, examples in sorted(
            analysis.judge_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        ):
            if not examples:
                continue

            sections.append(f"### {pattern_name}: {len(examples)} occurrences\n\n")

            # Show example justification
            if examples:
                sections.append(f"*Example*: \"{examples[0]['justification']}\"\n\n")

        sections.append(
            "*Action*: Address common judge feedback systematically in prompt or skills.\n\n"
        )

    # ── Failure Patterns ────────────────────────────────────────
    if analysis.failure_patterns:
        sections.append("\n## Detected Failure Patterns\n")
        sections.append("Systematic issues requiring targeted fixes:\n\n")

        for pattern in analysis.failure_patterns:
            sections.append(f"### {pattern.pattern_name}\n\n")
            sections.append(f"- **Count**: {pattern.count} tasks\n")
            sections.append(f"- **Description**: {pattern.description}\n")
            sections.append(f"- **Suggested Fix**: {pattern.suggested_fix}\n")
            sections.append(f"- **Affected**: {', '.join(pattern.task_ids[:3])}")
            if len(pattern.task_ids) > 3:
                sections.append(f" (and {len(pattern.task_ids) - 3} more)")
            sections.append("\n\n")

    # ── Code Execution Analysis ─────────────────────────────────
    code_stats = analysis.code_stats
    if code_stats.tasks_used_code > 0 or code_stats.missed_opportunities:
        sections.append("\n## Code Execution Analysis\n\n")
        sections.append(f"- **Usage**: {code_stats.tasks_used_code}/{base.total_tasks} tasks\n")

        if code_stats.tasks_used_code > 0:
            sections.append(f"- **Code exec pass rate**: {code_stats.code_pass_rate:.0%}\n")
        if code_stats.tasks_no_code > 0:
            sections.append(f"- **Direct calls pass rate**: {code_stats.no_code_pass_rate:.0%}\n")

        if code_stats.missed_opportunities:
            sections.append(
                f"\n*Missed opportunities*: {len(code_stats.missed_opportunities)} "
                f"tasks with 15+ tool calls but no code execution (likely search/iteration tasks).\n\n"
            )

    # ── Tool Errors ─────────────────────────────────────────────
    if base.tool_error_counts:
        sections.append("\n## Tool Errors\n\n")
        for tool, count in sorted(
            base.tool_error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            sections.append(f"- `{tool}`: {count} errors\n")
        sections.append("\n")

    # ── Strategy Issues ─────────────────────────────────────────
    if base.strategy_issue_counts:
        sections.append("\n## Strategy Issues\n\n")
        for issue, count in sorted(
            base.strategy_issue_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            sections.append(f"- {issue}: {count} occurrences\n")
        sections.append("\n")

    # ── Evolution History ───────────────────────────────────────
    if cfg.include_evolution_history and evolution_history:
        sections.append("\n## Evolution History (What Worked / Didn't Work)\n\n")

        # Separate successful and harmful changes
        successful = [h for h in evolution_history if h.get("impact", 0) > 0.02]
        harmful = [h for h in evolution_history if h.get("impact", 0) < -0.02]

        if successful:
            sections.append("### ✓ Successful Changes (to build on):\n\n")
            for change in successful[-3:]:  # Last 3 successful
                sections.append(
                    f"- Cycle {change.get('cycle')}: {change.get('description', 'N/A')}\n"
                )
                sections.append(f"  *Impact*: +{change.get('impact', 0):.1%}\n")
            sections.append("\n")

        if harmful:
            sections.append("### ✗ Harmful Changes (to avoid):\n\n")
            for change in harmful[-2:]:  # Last 2 harmful
                sections.append(
                    f"- Cycle {change.get('cycle')}: {change.get('description', 'N/A')}\n"
                )
                sections.append(f"  *Impact*: {change.get('impact', 0):.1%}\n")
            sections.append("\n")

    # ── Recommendations ─────────────────────────────────────────
    if analysis.evolution_recommendations:
        sections.append("\n## Recommended Actions\n\n")
        for i, rec in enumerate(analysis.evolution_recommendations[:5], 1):
            sections.append(f"{i}. {rec}\n")
        sections.append("\n")

    # ── Workspace State ─────────────────────────────────────────
    sections.append("\n## Current Workspace\n\n")

    prompt = workspace.read_prompt()
    sections.append(f"**System Prompt**: {len(prompt)} chars")
    if len(prompt) > cfg.prompt_max_chars * 0.9:
        sections.append(f" ⚠️ (close to {cfg.prompt_max_chars} limit)")
    sections.append("\n\n")

    skills = workspace.list_skills()
    sections.append(f"**Skills**: {len(skills)}/{cfg.max_skills} ")
    if skills:
        sections.append(f"({', '.join(s.name for s in skills)})")
    sections.append("\n\n")

    # ── Evolution Scope ─────────────────────────────────────────
    sections.append("\n## Your Task\n\n")

    # Determine scope based on performance
    if base.pass_rate >= 0.85:
        sections.append(
            "**Pass rate is high (≥85%).** Make MINIMAL changes:\n"
            "- If a specific claim/task type is weak, create ONE targeted skill\n"
            "- Otherwise, make NO changes (preserve what works)\n\n"
        )
    elif analysis.failure_patterns:
        sections.append(
            "**Failure patterns detected.** Make SURGICAL fixes:\n"
            "- Address the top 1-2 failure patterns only\n"
            "- Create/modify skills for weak claim/task types\n"
            "- Don't change things that are already working\n\n"
        )
    else:
        sections.append(
            "**General improvement needed.** Focus on:\n"
            "- Weakest claim types and task types\n"
            "- Common judge feedback patterns\n"
            "- Create 1-2 targeted skills, avoid prompt bloat\n\n"
        )

    # Permissions
    permissions = []
    if evolve_prompts:
        permissions.append("✓ Modify system prompt")
    if evolve_skills:
        permissions.append("✓ Create/modify skills")
    if evolve_memory:
        permissions.append("✓ Modify memory")

    sections.append("**Allowed actions**: " + ", ".join(permissions) + "\n\n")

    # Final reminder
    sections.append(
        "**Remember**: \n"
        "- Make TARGETED changes based on SPECIFIC weaknesses\n"
        "- ONE good fix beats five mediocre ones\n"
        "- If pass rate is high, change LESS, not more\n"
        "- Use `git diff` to verify changes before finishing\n"
    )

    return "".join(sections)


def build_multi_req_skill() -> str:
    """Return multi-requirement handler skill template."""
    return """\
---
name: multi-requirement-handler
description: Handle tasks with multiple requirements systematically
triggers: ["and", "also", "additionally"]
---

# Multi-Requirement Task Handler

## Detection

Task has multiple requirements if it contains:
- " and " (e.g., "Get X and also Y")
- " also " (e.g., "Return A, also B")
- " additionally " (e.g., "Find X. Additionally, ...")
- Bullet points or numbered lists

## Protocol

### STEP 1: Extract ALL Requirements (BEFORE any tool calls)

Read task carefully. Write numbered list:

```
Requirements:
1. [First requirement]
2. [Second requirement]
3. [Third requirement if present]
...
```

### STEP 2: Plan Tool Calls Per Requirement

```
Requirement 1 needs: [list tools]
Requirement 2 needs: [list tools]
...
```

### STEP 3: Execute IN ORDER, Mark Completion

After completing requirement 1:
- ✓ Requirement 1: [brief result]
- Remaining: 2, 3, ...

After completing requirement 2:
- ✓ Requirement 1: [result]
- ✓ Requirement 2: [result]
- Remaining: 3, ...

### STEP 4: Final Verification (BEFORE answering)

For EACH requirement:
- "Do I have SPECIFIC data for this?" (not just "looked for it")
- If ANY = "No", make more tool calls NOW

### STEP 5: Structure Answer

Address each requirement explicitly:

```
1. [Requirement 1]: [specific answer with exact values]
2. [Requirement 2]: [specific answer with exact values]
...
```

## Common Pitfall

❌ Spending 80% effort on requirement 1, then rushing/skipping requirement 2
✓ Budget effort EVENLY: 1/N effort per requirement
"""


def build_entity_verification_skill() -> str:
    """Return entity verification skill template."""
    return """\
---
name: entity-verification
description: Verify working with correct entity before proceeding
triggers: ["quoted names", "specific IDs", "dates"]
---

# Entity Verification Protocol

## When Task Mentions Specific Entity

If task includes ANY of:
- Quoted names: "TensorFlow", "AssaultCube"
- Specific IDs: "Task ID: 12345", "Issue #42"
- Exact dates: "May 9, 2007"
- Unique identifiers: "repository tensorflow/tensorflow"

## Mandatory Checkpoint (After FIRST Tool Call)

**STOP and verify**:

1. Does the returned data include the EXACT identifier from the task?
   - Task said "TensorFlow" → response has "TensorFlow"? ✓/✗
   - Task said "May 9, 2007" → response has this date? ✓/✗
   - Task said "repository X/Y" → response has owner=X, name=Y? ✓/✗

2. **If identifier NOT found**:
   - Your tool call retrieved the WRONG entity
   - STOP immediately, do NOT proceed
   - Try more specific query with exact identifier
   - Use different tool/endpoint if needed

3. **Never proceed with wrong entity**
   - Wrong entity = 0.0 score (complete failure)
   - Better to try 3 different tools than proceed with wrong data

## Examples

### ❌ Wrong Approach
```
Task: "Get creation date of repository tensorflow/tensorflow"
Action: search_repos("tensorflow") → picks first result
Problem: Could be wrong repository (many contain "tensorflow")
```

### ✓ Correct Approach
```
Task: "Get creation date of repository tensorflow/tensorflow"
Action: get_repo(owner="tensorflow", name="tensorflow") → exact match
Verify: Response has owner="tensorflow" AND name="tensorflow"? ✓
Continue: Safe to extract creation date
```

## Critical Rule

**After first tool call, always ask**: "Does this data match the task's EXACT entity?"

If NO → PIVOT immediately, don't continue with wrong data.
"""


def build_claim_type_skill(claim_type: str, examples: list[dict]) -> str:
    """Generate a skill for handling a specific weak claim type.

    Args:
        claim_type: The claim type (e.g., "calculate", "provide_fact")
        examples: List of failed examples for this claim type

    Returns:
        Skill content as string
    """
    # Create description based on claim type
    descriptions = {
        "calculate": "Handle calculation and numerical difference requirements",
        "compare": "Handle comparison between two or more entities",
        "aggregate": "Handle requirements for totals, lists, and all items",
        "provide_fact": "Handle fact retrieval and information provision",
        "identify_entity": "Handle entity identification and finding",
        "entity_property": "Handle extraction of entity properties and attributes",
    }

    desc = descriptions.get(claim_type, f"Handle {claim_type} type requirements")

    skill_content = f"""\
---
name: {claim_type}-handler
description: {desc}
---

# {claim_type.replace('_', ' ').title()} Handler

## Common Failures for This Claim Type

Based on recent failures, this claim type often fails when:
"""

    # Add specific failure patterns from examples
    if examples:
        for ex in examples[:2]:
            skill_content += f"- \"{ex.get('claim', '')}\" - {ex.get('justification', '')}\n"

    skill_content += f"""
## Guidelines for {claim_type.replace('_', ' ').title()} Claims

"""

    # Add type-specific guidance
    if claim_type == "calculate":
        skill_content += """
1. **Don't just mention the numbers** - perform the actual calculation
2. **State the operation explicitly**: "difference", "sum", "ratio"
3. **Show the result clearly**: "X - Y = Z" or "The difference is Z"
4. **Include units** if applicable (years, dollars, items, etc.)

Example:
- ❌ "The repo was created in 2013 and domain in 2007"
- ✓ "The repo was created in 2013, domain in 2007. Difference: 2013 - 2007 = 6 years"
"""
    elif claim_type == "compare":
        skill_content += """
1. **Fetch BOTH entities explicitly** - don't skip one
2. **Present side-by-side**: "Entity A: [properties] vs Entity B: [properties]"
3. **State the comparison result**: "A is larger/older/different than B"
4. **Include specific values**: Not just "different" but "A=10, B=20, difference=10"
"""
    elif claim_type == "aggregate":
        skill_content += """
1. **Don't stop at first page** - handle pagination to get ALL items
2. **State the total count**: "Found 47 items total"
3. **If listing all, actually list them** (not just "there are many")
4. **For totals/sums, show the calculation**: "10 + 20 + 30 = 60"
"""
    else:
        skill_content += """
1. **Retrieve the specific information** required by the claim
2. **Present it explicitly** in your answer (don't assume it's implied)
3. **Use exact values** from tool responses (copy, don't paraphrase)
4. **Verify you answered the EXACT question** asked
"""

    skill_content += """
## Verification Checklist

Before answering, verify:
- [ ] Did I retrieve the necessary data?
- [ ] Did I perform any required calculations/comparisons?
- [ ] Is the result EXPLICITLY stated in my answer?
- [ ] Are exact values included (not just descriptions)?
"""

    return skill_content
