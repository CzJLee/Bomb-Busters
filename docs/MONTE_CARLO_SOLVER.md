# Monte Carlo Solver

The Monte Carlo (MC) solver approximates wire probability distributions by generating random valid wire assignments and aggregating statistics. It serves as a fallback for game states where the exact solver is intractable (typically >22 hidden positions). Unlike the exact solver, the MC solver produces approximate probabilities that converge to the true values as the sample count increases.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Algorithm Architecture](#algorithm-architecture)
  - [Solver Context Reuse](#solver-context-reuse)
  - [Per-Sample Generation](#per-sample-generation)
  - [Importance Weighting](#importance-weighting)
  - [Aggregation](#aggregation)
- [Joint Probability Queries](#joint-probability-queries)
  - [MCSamples and Raw Assignment Storage](#mcsamples-and-raw-assignment-storage)
  - [Double Detector Probability](#double-detector-probability)
  - [Red Wire Risk for DD](#red-wire-risk-for-dd)
- [Auto-Switching Between Solvers](#auto-switching-between-solvers)
- [Constraints and Deductions](#constraints-and-deductions)
- [Historical Context: Rejected Approaches](#historical-context-rejected-approaches)
- [Advantages](#advantages)
- [Limitations](#limitations)
- [Ideas for Further Optimization](#ideas-for-further-optimization)

## Overview

The MC solver generates thousands of random wire-to-position assignments that satisfy all game constraints (sort order, pool composition, must-have deductions). Each valid sample is an assignment of every wire in the unknown pool to a hidden position. By counting how often each wire appears at each position across all samples, the solver estimates the marginal probability distribution for every hidden slot.

The algorithm uses **backward-guided composition sampling** — for each player in each sample, it builds a lightweight single-player dynamic programming table and samples from it proportionally. This guarantees that every sample is a valid ascending sequence with no dead ends. Samples are then weighted via **self-normalized importance sampling** to correct for the sequential per-player sampling not accounting for downstream feasibility.

The MC solver returns the same data format as the exact solver (`{(player_index, slot_index): Counter({Wire: count})}`), making it a drop-in replacement for all downstream functions.

## Motivation

The exact solver's runtime grows combinatorially with the number of hidden positions. For a typical 5-player game with blue wires only:

| Hidden positions | Exact solver time |
|-----------------|-------------------|
| ~20 | < 1 second |
| ~24 | ~2 seconds |
| ~28 | 60+ seconds |
| ~36 (full early game) | Intractable |

Early-game states with most wires still hidden push the exact solver well beyond acceptable wait times. The MC solver provides useful probability estimates in seconds regardless of game complexity, making it viable for early-game analysis.

The default threshold for auto-switching is `MC_POSITION_THRESHOLD = 22`, set in `compute_probabilities.py`.

## Algorithm Architecture

### Solver Context Reuse

The MC solver reuses the same `_setup_solver()` infrastructure as the exact solver to build a `SolverContext`. This gives it the same inputs: position constraints grouped by player, player processing order, distinct wire types with pool counts, and must-have constraints. The MC solver does not call `_solve_backward()` — it replaces the memo-building phase with sampling.

### Per-Sample Generation

The core sampling loop is in `_guided_mc_sample()`. For each sample attempt:

**1. Initialize the remaining pool** to `ctx.initial_pool` (a tuple of counts for each distinct wire type).

**2. Process players sequentially** in the same order as the exact solver (`ctx.player_order`).

**3. For each player**, build a lightweight backward DP table and sample a composition:

**Backward pass** (per-player, per-sample):

Compute `f[d][pi]` — the total combinatorial weight of valid compositions that fill positions `pi..N-1` using wire types `d..D-1` from the *current remaining pool*. This is the same recurrence as the exact solver's backward pass, but only for a single player with the current pool state:

$$f[d][\pi] = \sum_{k=0}^{k_{\max}} \binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

where $k_{\max} = \min(c_d, N - \pi, \text{max\_run}[d][\pi])$ and `max_run` prunes impossible placements.

If `f[0][0] == 0`, no valid composition exists for this player with the current pool — the sample is a dead end (this is extremely rare with guided sampling, but can happen due to must-have constraint interactions).

**Forward sampling** (per-player):

Starting from wire type `d=0`, position `pi=0`, sample how many copies `k` of each wire type to place. The probability of choosing `k` copies of type `d` is proportional to:

$$\binom{c_d}{k} \cdot f[d{+}1][\pi{+}k]$$

This weights each choice by both the number of ways to select `k` copies from the pool *and* the number of valid completions for the remaining positions. The sampling uses a cumulative distribution function over the valid options and a uniform random draw.

**4. Check must-have constraints**: After sampling a composition for each player, verify that any required values (from failed dual cut deductions) are present. If not, the sample is rejected. This is the only source of rejection — dead ends from infeasible pools are caught by `f[0][0] == 0` at step 3.

**5. Record the assignment**: Map the composition to an ascending wire sequence and record `(player_index, slot_index, wire)` for each position.

**6. Update the remaining pool** by subtracting the composition counts.

### Importance Weighting

The per-player backward-guided sampling guarantees valid compositions for each player individually, but doesn't account for whether the remaining pool after one player's assignment will be feasible for subsequent players. A composition that happens to leave a "good" pool for later players should carry more weight than one that leaves a "bad" pool.

To correct for this, each sample's weight is the **product of per-player normalization constants** — the `f[0][0]` values from each player's backward DP table:

$$w_s = \prod_{p \in \text{player\_order}} f_p[0][0]$$

This implements **self-normalized importance sampling**. The intuition: `f_p[0][0]` measures the total number of valid completions for player `p` given the pool at the time player `p` is processed. A larger value means more valid options were available, so that sample is more "representative" of the true distribution. The product across all players captures the joint feasibility.

When computing probabilities, the weighted sum is normalized:

$$P(\text{condition}) = \frac{\sum_{s : \text{condition}} w_s}{\sum_s w_s}$$

This produces **unbiased probability estimates** in the limit of many samples.

### Aggregation

For each valid sample, the solver aggregates wire assignments into per-position `Counter` objects:

```python
result[(player_index, slot_index)][wire] += sample_weight
```

The resulting dict has the same structure as `_forward_pass_positions()` output, so all downstream functions (`rank_all_moves`, `guaranteed_actions`, `probability_of_dual_cut`, etc.) work identically.

## Joint Probability Queries

### MCSamples and Raw Assignment Storage

Marginal per-position distributions (which wire is most likely at slot X?) can be computed from the aggregated counters. But **joint probability queries** — like "what's the probability that *both* slot A and slot B have specific properties?" — cannot be recovered from marginals alone, because the positions share a pool and are not independent.

To support joint queries, `monte_carlo_analysis()` returns an `MCSamples` object alongside the aggregated marginals. `MCSamples` stores:

- `samples`: A list of per-sample assignment dicts. Each dict maps `(player_index, slot_index)` to the `Wire` assigned in that sample. Discard player entries (from uncertain wire groups) are excluded.
- `weights`: Importance weight for each sample (parallel to `samples`).

### Double Detector Probability

`mc_dd_probability()` computes P(at least one of two target slots has the guessed value) by iterating over raw samples:

```python
for sample, weight in zip(mc_samples.samples, mc_samples.weights):
    total_weight += weight
    wire1 = sample.get((target_player, slot1))
    wire2 = sample.get((target_player, slot2))
    if wire1.gameplay_value == value or wire2.gameplay_value == value:
        match_weight += weight
return match_weight / total_weight
```

This correctly captures the joint distribution — the correlation between the two slots sharing a pool is reflected in each sample's consistent assignment.

### Red Wire Risk for DD

`mc_red_dd_probability()` computes P(both target slots are red wires) using the same weighted iteration, checking `wire.color == RED` for both positions simultaneously.

## Auto-Switching Between Solvers

`print_probability_analysis()` automatically chooses between the exact and MC solvers based on `count_hidden_positions()`:

1. Count the total hidden positions on other players' stands (HIDDEN slots + unknown-identity INFO_REVEALED slots + discard positions from uncertain wire groups).
2. If the count exceeds `MC_POSITION_THRESHOLD` (default 22), use `monte_carlo_analysis()`.
3. Otherwise, use `build_solver()` + `_forward_pass_positions()`.

The threshold is configurable via the `mc_threshold` parameter. Set to 0 to always use MC, or a very large number to always use the exact solver.

When MC is used, the terminal output includes a note: `(Monte Carlo: N unknown wires, M samples)`.

## Constraints and Deductions

The MC solver incorporates the same constraints as the exact solver:

| Constraint | How it's handled |
|-----------|-----------------|
| Sort-order bounds | `PositionConstraint` bounds + `max_run` pruning in per-player backward DP |
| Pool composition | Remaining pool updated per-player; `f[0][0] == 0` detects infeasible states |
| Must-have (failed dual cuts) | Post-composition check; sample rejected if required values are missing |
| `MustHaveValue` constraints | Merged into `must_have` during solver setup (same rejection-based enforcement as failed dual cut deductions) |
| `MustNotHaveValue` constraints | Post-composition check; sample rejected if any forbidden value appears in the player's assignment |
| `AdjacentNotEqual` constraints | Post-composition check; sample rejected if two adjacent hidden slots are assigned wires with the same gameplay value. When one slot is known, bounds are tightened during setup instead |
| `AdjacentEqual` constraints | Post-composition check; sample rejected if two adjacent hidden slots are assigned wires with different gameplay values. When one slot is known, bounds are tightened during setup instead |
| Uncertain wire groups | Discard slots with `required_color` — same as exact solver |
| Info tokens (blue) | Wire removed from pool; bounds tightened |
| Info tokens (yellow) | `required_color` constraint on position |

The MC solver uses the same `_setup_solver()` and `compute_position_constraints()` as the exact solver, so all constraint-handling logic is shared. The `MustNotHaveValue`, `AdjacentNotEqual`, and `AdjacentEqual` constraints add additional rejection checks after each player's composition is sampled. Since these constraints are rare (typically 0-2 per game), the increase in rejection rate is negligible.

## Historical Context: Rejected Approaches

The backward-guided sampler was not the first MC approach attempted. An earlier approach used **shuffle-sort-validate rejection sampling**:

1. Randomly shuffle the entire unknown pool.
2. Deal wires to each player's hidden positions.
3. Check if each player's assigned wires satisfy the sort-order constraints.
4. If all constraints are satisfied, accept the sample; otherwise, reject and retry.

This approach had an acceptance rate of roughly **0.5%** — only about 1 in 200 random shuffles produced a valid assignment. With the overhead of checking constraints on each attempt, throughput was approximately **25 valid samples per second**. At that rate, reaching 1,000 samples would take 40 seconds, which is slower than the exact solver for many game states and undermines the purpose of having an MC fallback.

The backward-guided approach eliminated nearly all rejection by construction: the per-player DP table ensures that sampled compositions are always valid ascending sequences. The only remaining rejection source is must-have constraints, which reject roughly 0-5% of samples. Throughput improved to **1,000-8,000 valid samples per second**, a 40-300x improvement.

## Advantages

- **Predictable runtime**: Generates N valid samples in roughly N/throughput seconds, regardless of the number of hidden positions or pool diversity. Typical: 1,000 samples in 0.1-1.0 seconds, 10,000 samples in 1-10 seconds.
- **Scalable to early game**: Works on 36+ hidden positions where the exact solver is intractable.
- **Same output format**: Drop-in replacement for the exact solver — all downstream functions work identically.
- **Joint probability support**: Raw per-sample assignments enable any joint query (DD, red DD, or future multi-position queries) without needing specialized forward passes.
- **No large memo**: Memory usage is O(num_samples × positions_per_sample), typically a few MB. The exact solver's memo can grow much larger for complex states.

## Limitations

**Approximation error**: MC results are estimates, not exact values. The standard error decreases as `1/√N` with sample count N. At 10,000 samples, a probability of 0.5 has a standard error of ~0.5%. At 1,000 samples, ~1.6%. This means:

- Rare events (probability < 1%) may not appear in the sample at all with small sample counts.
- Probabilities very close to 0% or 100% are less accurately estimated.
- Rankings of moves with similar probabilities may occasionally swap.

**No exact guarantees**: The exact solver can identify 100% guaranteed actions (a wire that appears in every valid distribution). The MC solver can only approximate this as "appeared in 100% of samples," which may miss edge cases with very few valid distributions.

**Must-have rejection**: In game states with many must-have constraints (many failed dual cuts), the rejection rate increases, reducing effective throughput. The `max_attempts` parameter (default `num_samples * 5`) prevents infinite loops, but if the rejection rate is very high, the solver may return fewer than the requested number of samples.

**Sequential bias from player ordering**: The importance weighting corrects for the bias introduced by sequential per-player sampling, but the correction assumes the `f[0][0]` normalization constants are representative. In extreme cases (highly constrained pools with few valid global assignments), the importance weights may have high variance, reducing the effective sample size. This manifests as a few samples dominating the weighted sum.

**No progress reporting**: Unlike the exact solver (which has a tqdm progress bar), the MC solver doesn't currently report progress during sampling. For large sample counts, there's no visual indication of progress.

## Ideas for Further Optimization

### Stratified Sampling

Instead of purely random sampling, partition the composition space and sample from each partition. For example, stratify by the first player's composition — ensure that each valid composition at the root level is sampled roughly equally. This reduces variance without increasing sample count, at the cost of more complex bookkeeping.

### Effective Sample Size Monitoring

Track the effective sample size (ESS) during sampling:

$$\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$$

If ESS drops significantly below the actual sample count, the importance weights have high variance and the estimates are unreliable. The solver could warn the user or automatically increase the sample count.

### Adaptive Sample Count

Instead of a fixed sample count, sample until a convergence criterion is met — for example, until the top-ranked move's probability estimate has a 95% confidence interval narrower than some threshold (e.g., ±2%). This would automatically use fewer samples for easy game states and more for complex ones.

### Parallel Sampling

Each sample is independent, so sampling is embarrassingly parallel. A multiprocessing approach could generate samples across multiple cores. The per-player backward DP table is lightweight enough that the overhead of spawning workers would be offset by throughput gains for large sample counts.

### Rao-Blackwellization

Instead of recording only the sampled wire at each position, use the per-player backward DP table to compute the exact conditional distribution for each position given the sampled compositions of upstream players. This replaces the binary "wire X was sampled at position Y" with a smooth conditional probability, reducing variance without increasing sample count. The per-player backward table is already computed during sampling, so the additional cost is just the forward-pass probability extraction per player per sample.

### Caching Per-Player DP Tables

The per-player backward DP table depends only on the player's position constraints and the current remaining pool. If two samples reach the same player with the same remaining pool, the DP table is identical. A cache mapping `(player_index, remaining_pool_tuple)` to the DP table could avoid redundant computation. This is most effective for later players in the processing order, where the remaining pool is more constrained and more likely to repeat across samples.
