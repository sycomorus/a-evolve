---
name: port-berth-allocation
description: Use when the task involves berth allocation, quay-crane assignment, or discrete space-time vessel scheduling at a container terminal.
types: [berth_allocation, quay_crane_scheduling, space_time_discretization, vessel_scheduling]
checklist:
  - id: space_time_markers
    prompt: Verify discrete space markers (pi/sigma for berth start/occupancy) and time markers (alpha/beta/gamma for service start/duration/end) are defined with proper linking to continuous position b_k and time t_k/c_k variables.
  - id: non_overlap_separation
    prompt: Verify both temporal (x_kl) and spatial (y_kl) separation binaries exist for every vessel pair, with at-least-one constraint x_kl + x_lk + y_kl + y_lk >= 1 and mutual exclusivity x_kl + x_lk <= 1, y_kl + y_lk <= 1.
  - id: release_and_fit
    prompt: Verify each vessel start t_k >= release time R_k and berth position b_k <= P - H_k + 1.
  - id: quay_crane_positioning
    prompt: Verify each crane has exactly one position per period (sum_p S_{gpj}=1), each position has at most one crane (sum_g S_{gpj} <= 1), and non-crossing order is enforced.
  - id: crane_work_vs_position
    prompt: Verify active work indicator Q_{gpj} <= B_{gj} * S_{gpj} (crane works only where positioned), and Q_{gpj} links to worker assignment and truck deployment.
  - id: vessel_position_time_zone
    prompt: Verify Z_{kpj} linking variable equals sigma_{kp} AND beta_{kj} (vessel k occupies position p at time j) with three constraints: Z <= sigma, Z <= beta, Z >= sigma + beta - 1.
---

Port berth allocation problems use a **space-time discretization** approach where binary marker variables indicate vessel start/end in both space (berth position) and time (service periods).

## Space (Berth Position) Discretization

- **pi_{kp}** = 1 if vessel k's leftmost berth position is p (start marker)
- **sigma_{kp}** = 1 if vessel k occupies berth position p
- **b_k** = sum_p (p * pi_{kp}) — continuous berth start position
- **sum_p sigma_{kp} = H_k** — vessel length
- **sum_p pi_{kp} = 1** — exactly one start position
- pi marks the start of a consecutive sigma block: pi_{kp} >= sigma_{kp} - sigma_{k,p-1}

## Time (Service Period) Discretization

- **alpha_{kj}** = 1 if period j is the first service period of vessel k (start marker)
- **beta_{kj}** = 1 if vessel k is being served during period j (active)
- **gamma_{kj}** = 1 if period j is the last service period of vessel k (end marker)
- **t_k** = sum_j (j * alpha_{kj}) — continuous start time
- **c_k** = sum_j ((j+1) * gamma_{kj}) — continuous completion time
- **sum_j alpha_{kj} = 1**, **sum_j gamma_{kj} = 1**
- alpha marks beta start: alpha_{kj} >= beta_{kj} - beta_{k,j-1}
- gamma marks beta end: gamma_{kj} >= beta_{kj} - beta_{k,j+1}

## Non-Overlap

For every pair (k,l) with k < l:
- At least one of x_kl, x_lk, y_kl, y_lk must be 1
- x_kl + x_lk <= 1 (only one temporal order)
- y_kl + y_lk <= 1 (only one spatial order)
- Temporal: if x_kl=1 then beta_{k,r} + beta_{l,j} <= 1 when r >= j-1
- Spatial: if y_kl=1 then vessel k's sigma block and vessel l's pi position must not overlap

## QC Crane Modeling

- Each crane g at exactly one position per period: sum_p S_{gpj} = 1
- At most one crane per position per period: sum_g S_{gpj} <= 1
- Non-crossing: if g1 < g2, position of g1 <= position of g2
- Work indicator Q_{gpj} <= S_{gpj} (work only where positioned)
- Work only within vessel occupied zones: Q_{gpj} <= sum_k Z_{kpj}

## Common Mistakes

- Forgetting the at-least-one non-overlap constraint (x + reverse_x + y + reverse_y >= 1)
- Missing the continuous-to-discrete linking (b_k = sum p*pi_kp, t_k = sum j*alpha_kj)
- Not enforcing vessel fit within quay length (b_k <= P - H_k + 1 implicitly via pi domain)
- Omitting the Z linking variable for vessel-position-time union
- Using Big-M for time separation when discrete beta markers provide exact separation
