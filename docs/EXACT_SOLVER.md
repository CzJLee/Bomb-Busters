# Exact Constraint Solver

The exact solver computes precise probability distributions for every hidden wire position on other players' stands. Given the observable game state from a specific player's perspective, it enumerates all valid wire distributions consistent with the constraints (sort order, known wires, info tokens, validation tokens, failed dual cut deductions) and counts how many place each wire at each position. The result is exact probabilities with no approximation error.

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Algorithm Architecture](#algorithm-architecture)
  - [Knowledge Extraction](#knowledge-extraction)
  - [Unknown Pool Computation](#unknown-pool-computation)
  - [Position Constraints](#position-constraints)
  - [Solver Context Setup](#solver-context-setup)
  - [Backward Solve (Memo Builder)](#backward-solve-memo-builder)
  - [Forward Passes](#forward-passes)
- [Key Optimizations](#key-optimizations)
  - [Composition-Based Enumeration](#composition-based-enumeration)
  - [Combinatorial Weights](#combinatorial-weights)
  - [Memoization at Player Boundaries](#memoization-at-player-boundaries)
  - [Max-Run Pruning](#max-run-pruning)
  - [Player Ordering](#player-ordering)
  - [Build Once, Query Many](#build-once-query-many)
- [Uncertain (X of Y) Wire Groups](#uncertain-x-of-y-wire-groups)
- [Constraints and Deductions](#constraints-and-deductions)
- [Limitations](#limitations)
- [Ideas for Further Optimization](#ideas-for-further-optimization)

## Overview

The exact solver answers the fundamental question: "From my perspective as the active player, what is the probability that each hidden wire on other players' stands is a specific wire?" This enables computing:

- **Dual cut probability**: P(target slot has guessed value)
- **Double Detector probability**: P(at least one of two target slots has guessed value) — computed via joint enumeration, not naive P(A) + P(B) - P(A)P(B)
- **Red wire risk**: P(target slot is red) and P(both DD target slots are red)
- **Guaranteed actions**: Moves with 100% success probability

The solver produces exact integer counts (no floating point rounding), so probabilities are mathematically precise.

## Problem Formulation

The solver must distribute a pool of unknown wires across hidden positions on other players' stands such that:

1. **Sort-order constraint**: Each player's stand is sorted in ascending order by `sort_value`. A wire at position `i` must have `sort_value` between the nearest known left neighbor and the nearest known right neighbor.
2. **Pool constraint**: The unknown pool contains exactly the wires that could occupy hidden positions — all wires in play minus the observer's own wires, minus other players' cut and info-revealed wires.
3. **Must-have constraint**: If a player failed a dual cut for value V and hasn't since cut a wire of value V, they must still have at least one wire of value V. This deduction from turn history constrains which distributions are valid.
4. **Pool exhaustion**: Every wire in the pool must be placed somewhere. The total number of hidden positions equals the pool size.

The output is, for each hidden position `(player_index, slot_index)`, a `Counter` mapping each possible `Wire` to the number of valid distributions that place it there.

## Algorithm Architecture

The solver has two main phases: a **backward solve** that builds a reusable memo, and **forward passes** that extract specific probability queries from the memo. The architecture is designed so the expensive backward solve runs once, and multiple cheap forward passes can answer different questions instantly.

### Knowledge Extraction

`extract_known_info()` collects all information visible to the observing player into a `KnownInfo` object:

- **Observer's wires**: All wires on the observer's own stand (all slots, including cut ones). The observer must know all their own wires for probability calculations to work.
- **Cut wires**: All wires cut across all players (publicly visible).
- **Info tokens**: Slots with info tokens from failed dual cuts, revealing the wire's gameplay value and position.
- **Validation tokens**: Blue values where all 4 copies have been cut (these values cannot appear on anyone's stand).
- **Must-have deductions**: From `_compute_must_have()`, which scans turn history for failed dual cuts. If player P failed a dual cut guessing value V, P has at least one wire of value V. If P hasn't since cut a V-wire, P still has one.

### Unknown Pool Computation

`compute_unknown_pool()` determines which wires could be at hidden positions:

```
unknown_pool = wires_in_play
             - observer's own wires (all slots)
             - other players' cut wires
             - other players' info-revealed wires (where identity is known)
```

For blue info-revealed wires, the identity is reconstructed from the numeric info token (`Wire(BLUE, float(value))`). For yellow info tokens, the exact yellow wire cannot always be determined (we know it's yellow but not which yellow wire), so it's handled conservatively — the constraint solver assigns a yellow wire to that position.

### Position Constraints

`compute_position_constraints()` creates a `PositionConstraint` for each hidden slot on other players' stands. Each constraint specifies:

- `player_index`, `slot_index`: Which slot this constraint is for.
- `lower_bound`, `upper_bound`: The valid `sort_value` range for a wire at this position, derived from scanning left and right for the nearest publicly visible wire (CUT or INFO_REVEALED with known identity).
- `required_color`: For yellow info-revealed slots in calculator mode (where the wire is None but we know it's yellow), this restricts the position to only yellow wires.

Sort-value bounds use only publicly visible information. Hidden wires on other players' stands are never used for bounds computation — this was the subject of a critical bug fix where hidden wire identities were leaking into bounds, causing incorrect probability calculations.

### Solver Context Setup

`_setup_solver()` assembles the complete solver context:

1. Extracts known info and computes the unknown pool.
2. Computes position constraints for all hidden/unknown slots.
3. Handles uncertain wire groups (see [Uncertain Wire Groups](#uncertain-x-of-y-wire-groups)).
4. Groups constraints by player and sorts by slot index.
5. Orders players by average bound width (widest first — see [Player Ordering](#player-ordering)).
6. Counts distinct wire types and their multiplicities in the pool.

Returns a `SolverContext` containing `positions_by_player`, `player_order`, `distinct_wires`, `initial_pool` (tuple of counts), and `must_have` constraints.

### Backward Solve (Memo Builder)

`_solve_backward()` builds the complete memo via recursive composition-based enumeration. This is the computationally expensive phase.

**State**: `(player_level_index, remaining_pool_tuple)` — which player we're assigning wires to, and how many of each wire type remain in the pool.

**For each player level**, the solver enumerates compositions: how many copies of each wire type to place on this player's positions. A composition `(k_1, k_2, ..., k_D)` assigns `k_1` copies of wire type 1 to the first `k_1` positions, then `k_2` copies of type 2 to the next `k_2` positions, and so on. This works because both positions and wire types are sorted by `sort_value`, so the ascending order constraint is automatically satisfied.

**Memo entry** (`MemoEntry`): A tuple of three values:
- `total_ways`: Total weighted count of valid distributions for this player and all downstream players.
- `slot_counts`: `dict[int, Counter[Wire]]` — for each position index within this player, the weighted count of each wire type across all valid distributions.
- `transitions`: `dict[tuple[int,...], int]` — for each resulting `remaining_pool` after this player, the total composition weight that produces that pool state. Used by forward passes to propagate weights without backtracking.

**Recursion**: After choosing a composition for the current player, update the remaining pool and recurse to the next player level. The base case is `pli == num_players`, which returns `(1, {}, {})`.

### Forward Passes

Forward passes extract specific probabilities from the prebuilt memo with no backtracking — just dictionary lookups and multiplications.

**`_forward_pass_positions()`**: Computes per-position wire probability distributions for all hidden slots. Maintains a `frontier` mapping `remaining_pool -> forward_weight`, initialized to `{initial_pool: 1}`. For each player level, looks up the memo entry for each frontier state, accumulates weighted slot counts into the result, and advances the frontier using transitions.

**`_forward_pass_dd()`**: Computes Double Detector probability P(at least one of two target slots has guessed value). Advances the frontier to the target player's level, then enumerates the target player's wire sequences via per-position backtracking (using `_enumerate_target_player()`), checking whether either target position has the guessed value. Uses downstream `total_ways` from the memo to weight each sequence.

**`_forward_pass_red_dd()`**: Same structure as DD, but checks whether both target positions are red wires.

## Key Optimizations

### Composition-Based Enumeration

The original solver used per-position backtracking: for each hidden position, try every wire that fits. With ~40 wires in the pool and ~10 positions per player, this creates a massive search tree.

The composition approach asks instead: "How many of each wire *type* do I place on this player?" Since both positions and wire types are sorted by `sort_value`, a composition `(k_1, k_2, ..., k_D)` uniquely determines which wire goes at which position. The first `k_1` positions get wire type 1, the next `k_2` positions get wire type 2, and so on.

This reduces the branching factor from (pool_size) per position to (max_count_per_type ≤ 4) per wire type. For a typical mid-game with ~15 distinct wire types and ~9 positions per player, the composition tree is vastly smaller than the per-position tree.

### Combinatorial Weights

A composition `(k_1, k_2, ..., k_D)` drawn from a pool with counts `(c_1, c_2, ..., c_D)` does not have weight 1 — it has weight `∏ C(c_d, k_d)`, the product of binomial coefficients. This is because there are `C(c_d, k_d)` ways to choose which `k_d` copies of wire type `d` are placed on this player's stand (the remaining `c_d - k_d` copies go to other players).

This weighting implements the multivariate hypergeometric distribution — the correct probability model for dealing without replacement from a finite pool. Without these weights, all compositions would be treated as equally likely, producing incorrect probabilities. This was a significant bug that was fixed after the initial composition-based solver was written.

### Memoization at Player Boundaries

The solver processes players sequentially. After assigning wires to one player, the state is fully characterized by `(player_level_index, remaining_pool_tuple)`. Different composition paths that leave the same remaining pool converge to the same subproblem.

The memo caches `(pli, remaining) -> MemoEntry`, so each subproblem is solved exactly once. This is highly effective because many compositions for earlier players leave the same pool state — especially when the pool has few distinct wire types.

### Max-Run Pruning

For each wire type `d` and starting position `pi`, `max_run[d][pi]` stores the maximum number of consecutive positions starting at `pi` that wire type `d` can legally fill (i.e., `wire_fits()` returns True for all positions in the run).

This caps the number of copies of type `d` that can be placed starting at position `pi`, avoiding enumeration of compositions that would violate sort-value bounds partway through. The precomputation is O(D × N) per player level and is performed once before the composition loop.

### Player Ordering

Players are processed in order of decreasing average bound width. The rationale: the most unconstrained players (widest bounds) have the most valid compositions. Placing them first in the processing order means they run with fewer frontier states (ideally just 1 at the root). More constrained players (narrow bounds, fewer valid compositions) run later when there are more frontier states, but each frontier state has fewer valid compositions to enumerate.

This ordering minimizes the total number of composition-evaluation calls across all levels.

### Build Once, Query Many

The `build_solver()` function returns a `(SolverContext, MemoDict)` pair that can be passed to any number of forward-pass functions:

- `compute_position_probabilities(ctx=ctx, memo=memo)` — all per-position distributions
- `probability_of_double_detector(ctx=ctx, memo=memo)` — DD joint probability
- `probability_of_red_wire_dd(ctx=ctx, memo=memo)` — DD red wire joint probability

The backward solve is by far the most expensive operation (seconds to minutes). Forward passes complete in milliseconds. This architecture avoids redundant backward solves when multiple probability queries are needed for the same game state — which is always the case, since `rank_all_moves()` and `print_probability_analysis()` query multiple positions and values.

## Uncertain (X of Y) Wire Groups

When a mission uses "X of Y" colored wire setup (e.g., draw 3 yellow wires, keep 2), the solver doesn't know which subset of candidates is in the game. This is handled via **discard slots** — virtual positions that absorb candidate wires NOT in the game.

**Mechanism:**

1. All unresolved candidate wires are added to the unknown pool.
2. For each wire that must be discarded (`candidates - count_in_play`), a `PositionConstraint` is created with `player_index = _DISCARD_PLAYER_INDEX` (sentinel value -1), bounds `(0.0, 13.0)`, and `required_color` matching the wire color.
3. The solver treats the discard "player" like any other player, assigning wires to its positions.
4. `_forward_pass_positions()` filters out discard player entries from the result, so callers only see real player positions.

This approach elegantly handles the combinatorics: the solver naturally enumerates which candidates are discarded vs. distributed to real player positions, producing correct joint probabilities in a single solve.

## Constraints and Deductions

The solver incorporates the following information:

| Source | How it's used |
|--------|--------------|
| Observer's own wires | Removed from unknown pool |
| Cut wires (all players) | Removed from unknown pool |
| Info tokens (blue) | Removed from unknown pool; creates tight sort-value bounds for neighbors |
| Info tokens (yellow) | Position constrained to yellow wires via `required_color` |
| Validation tokens | Implicitly handled — wires of validated values are all cut, so removed from pool |
| Sort order | `PositionConstraint` bounds from nearest known neighbors |
| Failed dual cuts | `must_have` constraints — compositions must include the required values for that player |

**What the solver does NOT explicitly deduce:**

- **Cross-player position interactions**: The solver doesn't reason about "if player A has wire X, then player B can't." This is handled implicitly through the pool — once a wire is assigned to player A, it's removed from the pool available to player B. The composition enumeration and memoization capture these interactions correctly.
- **Negative deductions from successful dual cuts**: A successful dual cut reveals that the target had value V, but does not directly narrow down other positions. The solver handles this by removing the cut wire from the pool.

## Limitations

**Time complexity**: The solver's runtime depends heavily on the number of hidden positions and the diversity of the wire pool. With ~36 hidden positions (early game, 5 players, blue only), the solver can take minutes. The composition-based approach and memoization help enormously, but the state space grows combinatorially.

- **~20 hidden positions**: Under 1 second
- **~24 hidden positions**: ~2 seconds
- **~28+ hidden positions**: 60+ seconds, potentially much longer
- **~36 hidden positions (full early game)**: Intractable

The `MC_POSITION_THRESHOLD` (default 22) is the cutoff above which `print_probability_analysis()` automatically switches to the Monte Carlo solver.

**Memory**: The memo can grow large for complex game states, since it stores per-position `Counter` objects for every `(pli, remaining_pool)` state. In practice this hasn't been a bottleneck, but it scales with the number of distinct pool states.

**Single-observer perspective**: The solver computes probabilities from one player's perspective. Different observers see different information (their own wires), so a separate solver run is needed per observer.

## Ideas for Further Optimization

### Symmetry Reduction

When multiple players have identical constraint profiles (same number of hidden positions, same sort-value bounds), their subproblems are equivalent. The solver could detect this and solve the subproblem once, reusing results for symmetric players. This would be most impactful in early-game blue-only scenarios where all non-observer players have very similar stands.

### Incremental Updates

After a single action (one wire cut, one info token placed), most of the memo remains valid. An incremental solver could invalidate only the affected entries and recompute them, rather than rebuilding the entire memo from scratch. This would be valuable for interactive analysis where the user explores "what if" scenarios.

### Tighter Bound Propagation

Currently, bounds are computed only from the nearest known left/right neighbors. More aggressive constraint propagation could use pool counting: if there are only 3 blue-7 wires remaining and 4 positions that could hold blue-7, at least one of those positions must hold something else. This kind of reasoning could tighten bounds and reduce the search space, though it adds complexity to the constraint setup phase.

### Parallel Composition Enumeration

The composition loop within each player level is inherently sequential (it modifies `rem` in place). However, the outer loop over frontier states is embarrassingly parallel — each `(remaining_pool, forward_weight)` pair is independent. A multiprocessing approach could partition the frontier across cores, with each core running the composition enumeration for its subset of frontier states. The results would be merged to form the next frontier.

### Profile-Guided Wire Type Ordering

Within the composition enumeration, wire types are processed in sort-value order. Reordering to process the rarest wire types first (those with count 1 in the pool) could enable earlier pruning, since rare types have fewer valid placement options and are more likely to produce dead-end compositions that can be pruned via `max_run`.
