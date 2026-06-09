---
name: formulation-replication-discipline
description: Use before building the model to ensure the formulation replicates the visible business requirement.
types: [general]
checklist:
  - id: objective_sense_and_unit
    prompt: Verify the objective sense and unit match the visible business requirement.
  - id: supported_constraints_only
    prompt: Verify every non-obvious constraint or cost term is supported by visible text or CSV headers.
  - id: variable_domains
    prompt: Verify integer or binary domains are used only when required by visible evidence.
---

Model only what the visible requirement and data files support.

- Preserve the requested objective sense and unit.
- Use continuous variables unless the task requires indivisible counts, binary selections, or logical activation.
- Do not add textbook constraints, extra cost terms, symmetry assumptions, alternate routes, or reverse implications.
- Match every non-obvious constraint to a sentence in the requirement or to CSV headers.
- After solving, inspect key variables to confirm the model behavior matches the business description.

If the visible files do not state a restriction, leave it out.
