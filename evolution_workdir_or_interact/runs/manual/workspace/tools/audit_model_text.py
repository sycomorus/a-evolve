from __future__ import annotations

import re
from typing import Any


def audit_model_text(model_text: str, problem_notes: str = "") -> dict[str, Any]:
    """Return generic warnings for common OR formulation mismatches."""
    text = f"{problem_notes}\n{model_text}".lower()
    warnings: list[str] = []

    def has_any(patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    if has_any([r"\binteger\b", r"grb\.integer", r"cat=['\"]integer", r"\bintvar\b"]):
        warnings.append(
            "Integer variables detected. Confirm the visible task requires indivisible quantities; otherwise use continuous variables."
        )
    if has_any([r"\bbinary\b", r"grb\.binary", r"cat=['\"]binary"]):
        warnings.append(
            "Binary variables detected. Confirm each binary represents a stated selection, activation, segment, or logical relation."
        )
    if has_any([r"idle", r"lexicographic", r"priority", r"phase", r"first.*then", r"then.*max", r"then.*min"]):
        warnings.append(
            "Multiple objective-like quantities may be present. Verify the finalized value is the requested final metric, not an intermediate phase."
        )
    if has_any([r"piecewise", r"segment", r"tier", r"range", r"discount"]):
        warnings.append(
            "Piecewise ranges detected. Decide all-units versus incremental block pricing before coding segment variables."
        )
    if has_any([r"activation", r"setup", r"fixed cost", r"fixed_cost", r"open", r"prepare"]):
        warnings.append(
            "Activation or fixed cost detected. Check whether activation is optional or forced, and include both upper links and minimum batches when needed."
        )
    if has_any([r"makespan", r"rental", r"machine", r"project", r"precedence"]):
        warnings.append(
            "Scheduling or machine terms detected. Map each cost to its stated interval; do not assume all costs use global makespan."
        )
    if has_any([r"assignment", r"facility", r"location", r"factory"]):
        warnings.append(
            "Assignment/location terms detected. Use linear pair costs unless the visible data defines pairwise flows and pairwise distances."
        )
    if has_any([r"requires", r"accompan", r"with", r"mutual", r"exclusive", r"implies"]):
        warnings.append(
            "Logical relationships detected. Do not add reverse implications unless both directions are stated."
        )
    if has_any([r"gdp", r"net export", r"import", r"commodity", r"input.output", r"intermediate"]):
        warnings.append(
            "Economic production terms detected. Add net or balance constraints only when visible text requires them, and define net expressions explicitly."
        )

    return {"warnings": warnings, "warning_count": len(warnings)}
