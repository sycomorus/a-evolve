---
name: objective-reporting-check
description: Use immediately before finalizing to verify the submitted value is the requested objective quantity.
types: [general]
checklist:
  - id: objective_unit
    prompt: Verify the submitted value has the requested unit.
  - id: final_phase_metric
    prompt: Verify multi-phase or priority solves submit the final requested metric rather than an intermediate proxy.
  - id: recompute_submitted_value
    prompt: Verify the submitted objective was recomputed from solved variables and matches the final code.
---

Before calling finalize, check the reported quantity separately from solver status.

- State the quantity and unit being submitted: profit, cost, inventory, idle time, project duration cost, assignment cost, GDP, or another requested metric.
- If the task has priorities or phases, submit the final requested phase result rather than the first-stage proxy.
- If the solver objective differs from the requested report, compute the reported value explicitly from solved variables.
- Recompute the submitted objective from key variable values and compare it with the solver result.

Finalize only after the submitted value and the visible request have the same unit.
