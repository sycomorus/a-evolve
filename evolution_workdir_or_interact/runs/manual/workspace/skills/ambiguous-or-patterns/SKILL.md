---
name: ambiguous-or-patterns
description: Use when a familiar OR template may not match the visible task statement exactly.
types: [general, piecewise_discount, routing_vrp, logic_activation]
checklist:
  - id: no_template_substitution
    prompt: Verify the model did not replace the stated problem with a familiar variant.
  - id: piecewise_disambiguation
    prompt: Verify any piecewise ranges were modeled as all-units or incremental blocks according to visible evidence.
  - id: implication_direction
    prompt: Verify logical implications were not made bidirectional unless both directions were stated.
---

When a task resembles a known pattern, first disambiguate it from the visible files.

- Do not turn a one-way implication into a two-way relationship unless the text says both directions.
- Do not assume assignment/location costs are quadratic unless the data provides pairwise flows and pairwise distances.
- For piecewise costs, prices, or capacities, identify whether tiers are all-units or incremental blocks before modeling segments.
- For cutting-stock style tasks, check whether waste includes overproduction as well as trim loss.
- For process, furnace, or method rows, map variables to the stated unit before choosing continuous hours, integer uses, or binary selections.

If a modeling choice is ambiguous, document the visible-field basis in the final code.
