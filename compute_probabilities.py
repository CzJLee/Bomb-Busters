"""Probability engine for Bomb Busters.

Computes the probability of success for different cut actions based on
the observable game state from a specific player's perspective. Supports
deduction from sort-order constraints, info tokens, validation tokens,
and turn history.

Architecture:
    SolverContext + _solve_backward() build a reusable memo once per
    game-state + observer. Forward-pass functions then compute specific
    probabilities (dual cut, Double Detector, red wire risk, etc.)
    instantly from the shared memo.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import random

import bomb_busters

try:
    import tqdm as _tqdm_module
except ImportError:  # pragma: no cover
    _tqdm_module = None  # type: ignore[assignment]

_C = bomb_busters._Colors

# Sentinel player index for discard slots (slack variables) used by the
# constraint solver when uncertain (X of Y) wire groups are present.
# Discard positions absorb candidate wires that are NOT in the game.
_DISCARD_PLAYER_INDEX = -1


# =============================================================================
# Known Information
# =============================================================================

@dataclasses.dataclass
class KnownInfo:
    """All information known to the observing player.

    Aggregates public information (visible to all) and the observer's
    private hand knowledge into a single structure for probability
    calculations.

    Attributes:
        active_player_index: Index of the observing player.
        observer_wires: Exact wires in the observer's hand (all slots).
        cut_wires: All wires that have been cut across all players.
        info_revealed: List of (player_index, slot_index, revealed_value)
            for wires with info tokens placed on them.
        validation_tokens: Set of blue values (1-12) where all 4 are cut.
        player_must_have: Dict mapping player_index to a set of gameplay
            values that player is known to still have (from failed dual cuts).
        markers: Board markers indicating which red/yellow values are in play.
        wires_in_play: All wire objects that were shuffled into the game.
    """
    active_player_index: int
    observer_wires: list[bomb_busters.Wire]
    cut_wires: list[bomb_busters.Wire]
    info_revealed: list[tuple[int, int, int | str]]
    validation_tokens: set[int]
    player_must_have: dict[int, set[int | str]]
    markers: list[bomb_busters.Marker]
    wires_in_play: list[bomb_busters.Wire]


def extract_known_info(
    game: bomb_busters.GameState, active_player_index: int
) -> KnownInfo:
    """Extract all information visible to the observing player.

    Collects the observer's own hand, all publicly visible information
    (cut wires, info tokens, validation tokens, markers), and deductions
    from turn history.

    Args:
        game: The current game state.
        active_player_index: Index of the player whose perspective to use.

    Returns:
        A KnownInfo object with all observable information.
    """
    observer = game.players[active_player_index]

    # The observer must know all their own wires for probability
    # calculations to work. Unknown wires (None) indicate incomplete
    # tile stand information.
    unknown_indices = [
        i for i, s in enumerate(observer.tile_stand.slots)
        if s.wire is None
    ]
    if unknown_indices:
        letters = ", ".join(chr(ord("A") + i) for i in unknown_indices)
        raise ValueError(
            f"Cannot compute probabilities: player "
            f"{active_player_index} ({observer.name}) has "
            f"incomplete tile stand info at slot(s) {letters}. "
            f"The active player's tile stand must have all wire "
            f"identities known."
        )

    # Observer's own wires (all slots, including cut ones)
    observer_wires = [
        s.wire for s in observer.tile_stand.slots
        if s.wire is not None
    ]

    # All cut wires across all players
    cut_wires = game.get_all_cut_wires()

    # Info-revealed wires on all stands
    info_revealed: list[tuple[int, int, int | str]] = []
    for p_idx, player in enumerate(game.players):
        for s_idx, slot in enumerate(player.tile_stand.slots):
            if slot.is_info_revealed and slot.info_token is not None:
                info_revealed.append((p_idx, s_idx, slot.info_token))

    # Deductions from turn history
    player_must_have = _compute_must_have(game, active_player_index)

    return KnownInfo(
        active_player_index=active_player_index,
        observer_wires=observer_wires,
        cut_wires=cut_wires,
        info_revealed=info_revealed,
        validation_tokens=game.validation_tokens,
        player_must_have=player_must_have,
        markers=game.markers,
        wires_in_play=list(game.wires_in_play),
    )


def _compute_must_have(
    game: bomb_busters.GameState, active_player_index: int
) -> dict[int, set[int | str]]:
    """Determine which values each player must still have from failed dual cuts.

    A failed dual cut by player P for value V means P had at least one wire
    of value V. If P has not since cut any wire of value V, P still has it.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.

    Returns:
        Dict mapping player_index to set of gameplay values they must have.
    """
    must_have: dict[int, set[int | str]] = {}
    for action in game.history.actions:
        if (
            isinstance(action, bomb_busters.DualCutAction)
            and action.result == bomb_busters.ActionResult.FAIL_BLUE_YELLOW
        ):
            actor = action.actor_index
            value = action.guessed_value
            # Check if actor has since cut a wire of this value
            actor_stand = game.players[actor].tile_stand
            has_cut_since = any(
                s.is_cut
                and s.wire is not None
                and s.wire.gameplay_value == value
                for s in actor_stand.slots
            )
            if not has_cut_since:
                must_have.setdefault(actor, set()).add(value)
    return must_have


# =============================================================================
# Unknown Pool Computation
# =============================================================================

def compute_unknown_pool(
    known: KnownInfo, game: bomb_busters.GameState
) -> list[bomb_busters.Wire]:
    """Compute the pool of wires whose locations are unknown to the observer.

    Unknown pool = all wires in play - observer's wires - cut wires
    - info-revealed wires (their exact identity is known from the token
    and position constraints).

    Args:
        known: The observer's known information.
        game: The current game state.

    Returns:
        List of Wire objects in the unknown pool.
    """
    # Start with all wires in play
    remaining = list(known.wires_in_play)

    # Remove observer's own wires (includes both hidden and cut)
    for wire in known.observer_wires:
        if wire in remaining:
            remaining.remove(wire)

    # Remove cut wires from OTHER players only
    # (observer's cut wires are already removed above)
    observer_cut_wires: list[bomb_busters.Wire] = []
    observer = game.players[known.active_player_index]
    for slot in observer.tile_stand.slots:
        if slot.is_cut and slot.wire is not None:
            observer_cut_wires.append(slot.wire)

    other_cut_wires = list(known.cut_wires)
    for wire in observer_cut_wires:
        if wire in other_cut_wires:
            other_cut_wires.remove(wire)

    for wire in other_cut_wires:
        if wire in remaining:
            remaining.remove(wire)

    # Remove info-revealed wires from OTHER players only
    # (observer's info-revealed wires are already removed above
    # as part of observer_wires)
    for p_idx, s_idx, revealed_value in known.info_revealed:
        if p_idx == known.active_player_index:
            continue
        wire = _identify_info_revealed_wire(game, p_idx, s_idx, revealed_value)
        if wire is not None and wire in remaining:
            remaining.remove(wire)

    return remaining


def _identify_info_revealed_wire(
    game: bomb_busters.GameState,
    player_index: int,
    slot_index: int,
    revealed_value: int | str,
) -> bomb_busters.Wire | None:
    """Identify the exact Wire at an info-revealed slot.

    In simulation mode, the wire is directly available. In calculator mode,
    we may need to infer it from the revealed value and sort constraints.

    Args:
        game: The current game state.
        player_index: Player whose stand contains the slot.
        slot_index: The slot index.
        revealed_value: The value shown by the info token.

    Returns:
        The Wire object, or None if it cannot be determined.
    """
    slot = game.players[player_index].tile_stand.slots[slot_index]
    if slot.wire is not None:
        return slot.wire

    # Calculator mode: wire is None, use revealed_value to reconstruct
    if isinstance(revealed_value, int):
        # Blue wire with this value
        return bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(revealed_value))
    elif revealed_value == "YELLOW":
        # Yellow wire — we know it's yellow but not the exact sort_value.
        # Use sort bounds to narrow down possibilities.
        # For pool removal, we can't definitively identify which yellow wire
        # it is without more info. Return None to be conservative.
        # The constraint solver will handle this via position constraints.
        return None
    return None


# =============================================================================
# Position Constraints
# =============================================================================

@dataclasses.dataclass
class PositionConstraint:
    """Sort-value constraints on a hidden or info-revealed slot.

    Defines the valid range of sort_values for a wire at a specific
    position on another player's stand, based on known neighboring wires.

    Attributes:
        player_index: Index of the player whose stand this slot is on.
        slot_index: The slot index on that player's stand.
        lower_bound: Sort value must be >= this (allows duplicates).
        upper_bound: Sort value must be <= this (allows duplicates).
        required_color: If set, only wires of this color can occupy the
            position (e.g., for yellow info-revealed slots where the
            exact wire is unknown but the color is known).
    """
    player_index: int
    slot_index: int
    lower_bound: float
    upper_bound: float
    required_color: bomb_busters.WireColor | None = None

    def wire_fits(self, wire: bomb_busters.Wire) -> bool:
        """Check if a wire could legally occupy this position.

        Args:
            wire: The wire to check.

        Returns:
            True if the wire's sort_value is within bounds and matches
            the required color (if any).
        """
        if self.required_color is not None and wire.color != self.required_color:
            return False
        return self.lower_bound <= wire.sort_value <= self.upper_bound


def compute_position_constraints(
    game: bomb_busters.GameState, active_player_index: int
) -> list[PositionConstraint]:
    """Compute sort-value constraints for unknown slots on other players' stands.

    For each hidden slot, scans left and right to find the nearest known
    wire (from CUT or INFO_REVEALED slots) to establish valid sort-value
    bounds. Also includes INFO_REVEALED slots where the wire identity is
    unknown (e.g., yellow info tokens in calculator mode).

    Args:
        game: The current game state.
        active_player_index: The observing player's index.

    Returns:
        List of PositionConstraint objects for all constrained slots on
        other players' stands.
    """
    constraints: list[PositionConstraint] = []
    for p_idx, player in enumerate(game.players):
        if p_idx == active_player_index:
            continue
        stand = player.tile_stand
        for s_idx, slot in enumerate(stand.slots):
            if slot.state == bomb_busters.SlotState.HIDDEN:
                lower, upper = bomb_busters.get_sort_value_bounds(
                    stand.slots, s_idx,
                )
                constraints.append(PositionConstraint(
                    player_index=p_idx,
                    slot_index=s_idx,
                    lower_bound=lower,
                    upper_bound=upper,
                ))
            elif (
                slot.state == bomb_busters.SlotState.INFO_REVEALED
                and slot.wire is None
                and slot.info_token == "YELLOW"
            ):
                # Yellow info token in calculator mode: we know the color
                # but not which yellow wire. Include as a constrained
                # position so the solver assigns a yellow wire here.
                lower, upper = bomb_busters.get_sort_value_bounds(
                    stand.slots, s_idx,
                )
                constraints.append(PositionConstraint(
                    player_index=p_idx,
                    slot_index=s_idx,
                    lower_bound=lower,
                    upper_bound=upper,
                    required_color=bomb_busters.WireColor.YELLOW,
                ))
    return constraints


# =============================================================================
# Solver Context & Memo
# =============================================================================

# A memo entry stores (total_ways, slot_counts, transitions):
#   total_ways: int — total valid distributions for players[pli:]
#   slot_counts: dict[int, Counter[Wire, int]] — for each position
#       index within this player, count of (seq × sub_ways) per wire
#   transitions: dict[tuple[int,...], int] — for each new_remaining
#       pool state, # of valid sequences that produce that state
MemoEntry = tuple[
    int,
    dict[int, collections.Counter[bomb_busters.Wire]],
    dict[tuple[int, ...], int],
]

# Complete memo: maps (player_level_index, remaining_pool_tuple) -> MemoEntry
MemoDict = dict[tuple[int, tuple[int, ...]], MemoEntry]


@dataclasses.dataclass
class MCSamples:
    """Raw per-sample wire assignments from Monte Carlo sampling.

    Stores individual sample assignments and their importance weights,
    enabling post-hoc joint probability queries (e.g., Double Detector,
    joint red wire risk for DD).

    Attributes:
        samples: List of per-sample assignment dicts. Each dict maps
            (player_index, slot_index) to the Wire assigned in that
            sample. Discard player entries are excluded.
        weights: Importance weight for each sample (parallel to
            samples). The weight is the product of per-player
            normalization constants from backward-guided sampling.
    """
    samples: list[dict[tuple[int, int], bomb_busters.Wire]]
    weights: list[int]


@dataclasses.dataclass
class SolverContext:
    """Immutable context for the constraint solver.

    Built once per game-state + observer combination via _setup_solver().
    Reusable across multiple forward passes (dual cut, Double Detector,
    Triple Detector, etc.) when paired with a MemoDict from
    _solve_backward().

    Attributes:
        positions_by_player: Position constraints grouped by player index,
            sorted by slot_index within each player.
        player_order: Player indices ordered for solver processing
            (widest average bounds first for optimal memoization).
        distinct_wires: Sorted list of unique Wire objects in the
            unknown pool.
        initial_pool: Tuple of counts for each distinct wire type.
        must_have: Dict mapping player_index to set of gameplay values
            they must have (from failed dual cut deductions).
    """
    positions_by_player: dict[int, list[PositionConstraint]]
    player_order: list[int]
    distinct_wires: list[bomb_busters.Wire]
    initial_pool: tuple[int, ...]
    must_have: dict[int, set[int | str]]


def _setup_solver(
    game: bomb_busters.GameState,
    active_player_index: int,
) -> SolverContext | None:
    """Set up solver context from game state.

    Extracts known information, computes the unknown pool, derives
    position constraints, and orders players for optimal solving.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.

    Returns:
        A SolverContext, or None if there are no hidden positions to solve.
    """
    known = extract_known_info(game, active_player_index)
    unknown_pool = compute_unknown_pool(known, game)
    constraints = compute_position_constraints(game, active_player_index)

    # Handle uncertain wire groups: add unresolved candidates to the
    # unknown pool and create discard positions (slack variables) for
    # candidate wires that are NOT in the game.
    discard_slot_idx = 0
    for group in game.uncertain_wire_groups:
        # Determine which candidates are already accounted for
        # (on observer's stand, cut elsewhere, or in wires_in_play).
        accounted_count = 0
        unresolved: list[bomb_busters.Wire] = []
        for wire in group.candidates:
            is_in_pool = wire in unknown_pool
            is_accounted = (
                wire in known.observer_wires
                or wire in known.cut_wires
            )
            if is_in_pool or is_accounted:
                accounted_count += 1
            else:
                unresolved.append(wire)

        # How many unresolved candidates are actually in the game
        remaining_in_play = max(
            0, group.count_in_play - accounted_count,
        )
        discard_count = len(unresolved) - remaining_in_play

        # Add unresolved candidates to the unknown pool
        unknown_pool.extend(unresolved)

        # Create discard positions for wires not in the game
        for _ in range(discard_count):
            constraints.append(PositionConstraint(
                player_index=_DISCARD_PLAYER_INDEX,
                slot_index=discard_slot_idx,
                lower_bound=0.0,
                upper_bound=13.0,
                required_color=group.color,
            ))
            discard_slot_idx += 1

    if not constraints:
        return None

    positions_by_player: dict[int, list[PositionConstraint]] = {}
    for c in constraints:
        positions_by_player.setdefault(c.player_index, []).append(c)
    for p_idx in positions_by_player:
        positions_by_player[p_idx].sort(key=lambda c: c.slot_index)

    # Order players so that the most unconstrained (widest average bounds)
    # are processed first. This minimizes total leaf visits in the memoized
    # solver because the expensive wide-bound players run with fewer
    # frontier states (ideally just 1 at the root).
    def _avg_bound_width(p: int) -> float:
        positions = positions_by_player[p]
        return sum(pc.upper_bound - pc.lower_bound for pc in positions) / len(positions)

    player_order = sorted(
        positions_by_player.keys(),
        key=lambda p: _avg_bound_width(p),
        reverse=True,
    )

    pool_counter = collections.Counter(unknown_pool)
    distinct_wires = sorted(pool_counter.keys(), key=lambda w: w.sort_value)
    initial_pool = tuple(pool_counter[w] for w in distinct_wires)

    return SolverContext(
        positions_by_player=positions_by_player,
        player_order=player_order,
        distinct_wires=distinct_wires,
        initial_pool=initial_pool,
        must_have=known.player_must_have,
    )


# =============================================================================
# Backward Solve (Memo Builder)
# =============================================================================

def _count_root_compositions(ctx: SolverContext) -> int:
    """Count wire-type compositions for the first player level.

    Fast dry-run of the composition enumeration for the root player
    without recursing into downstream player levels. Used to estimate
    total work for the progress bar.

    Args:
        ctx: Solver context with constraints and pool data.

    Returns:
        Number of valid compositions at pli=0.
    """
    if not ctx.player_order:
        return 0

    p = ctx.player_order[0]
    positions = ctx.positions_by_player[p]
    num_pos = len(positions)
    distinct_wires = ctx.distinct_wires
    num_types = len(distinct_wires)
    rem = list(ctx.initial_pool)

    max_run: list[list[int]] = []
    for d, w in enumerate(distinct_wires):
        runs = [0] * (num_pos + 1)
        for pi in range(num_pos - 1, -1, -1):
            if positions[pi].wire_fits(w):
                runs[pi] = runs[pi + 1] + 1
            else:
                runs[pi] = 0
        max_run.append(runs)

    count = 0

    def _compose(d: int, pi: int) -> None:
        nonlocal count
        if pi == num_pos:
            count += 1
            return
        if d == num_types:
            return
        run_limit = max_run[d][pi]
        max_k = min(rem[d], num_pos - pi, run_limit)
        saved = rem[d]
        for k in range(max_k + 1):
            rem[d] = saved - k
            _compose(d + 1, pi + k)
        rem[d] = saved

    _compose(0, 0)
    return count


def _solve_backward(
    ctx: SolverContext,
    show_progress: bool = False,
) -> MemoDict:
    """Build the complete backward-solve memo.

    Uses composition-based enumeration: for each player, decides how many
    copies of each wire type to place rather than choosing per-position.
    Since positions are ascending and wire types are sorted, the assignment
    is deterministic from the composition. This reduces branching
    dramatically compared to per-position backtracking.

    Memoized at player boundaries by (player_level_index, remaining_pool).

    Args:
        ctx: Solver context with constraints and pool data.
        show_progress: If True, display a tqdm progress bar. Uses
            root-level compositions as the progress metric — each
            composition for the first player represents a roughly
            comparable chunk of downstream work.

    Returns:
        Complete memo dict mapping (pli, remaining) -> MemoEntry.
    """
    memo: MemoDict = {}
    num_players = len(ctx.player_order)
    distinct_wires = ctx.distinct_wires
    positions_by_player = ctx.positions_by_player
    player_order = ctx.player_order
    must_have = ctx.must_have

    pbar = None
    if show_progress and _tqdm_module is not None:
        total_compositions = _count_root_compositions(ctx)
        pbar = _tqdm_module.tqdm(
            total=total_compositions,
            desc="Solving",
            unit=" compositions",
            dynamic_ncols=True,
        )

    def _solve(pli: int, remaining: tuple[int, ...]) -> MemoEntry:
        key = (pli, remaining)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if pli == num_players:
            entry: MemoEntry = (1, {}, {})
            memo[key] = entry
            return entry

        p = player_order[pli]
        positions = positions_by_player[p]
        num_pos = len(positions)
        required = must_have.get(p, set())
        rem = list(remaining)
        total = [0]
        slot_counts: dict[int, collections.Counter[bomb_busters.Wire]] = {}
        transitions: dict[tuple[int, ...], int] = {}

        # Precompute: for each wire type, the max consecutive positions
        # it can fill starting from each position index.
        # max_run[d][pi] = max k such that positions pi..pi+k-1 all fit
        # wire type d.
        num_types = len(distinct_wires)
        max_run: list[list[int]] = []
        for d, w in enumerate(distinct_wires):
            runs = [0] * (num_pos + 1)
            for pi in range(num_pos - 1, -1, -1):
                if positions[pi].wire_fits(w):
                    runs[pi] = runs[pi + 1] + 1
                else:
                    runs[pi] = 0
            max_run.append(runs)

        def _compose(
            d: int,
            pi: int,
            seen: set[int | str],
            weight: int = 1,
        ) -> None:
            if pi == num_pos:
                # All positions filled. Skip remaining wire types.
                if pbar is not None and pli == 0:
                    pbar.update(1)
                if required and not required.issubset(seen):
                    return
                new_remaining = tuple(rem)
                sub_total = _solve(pli + 1, new_remaining)[0]
                if sub_total == 0:
                    return
                contrib = weight * sub_total
                total[0] += contrib
                # Accumulate per-position wire counts. Walk the
                # composition encoded in (remaining vs current rem).
                pos = 0
                for di in range(num_types):
                    used = remaining[di] - rem[di]
                    if used > 0:
                        w_di = distinct_wires[di]
                        for _ in range(used):
                            counter = slot_counts.get(pos)
                            if counter is None:
                                counter = collections.Counter()
                                slot_counts[pos] = counter
                            counter[w_di] += contrib
                            pos += 1
                transitions[new_remaining] = (
                    transitions.get(new_remaining, 0) + weight
                )
                return

            if d == num_types:
                # No more wire types but positions remain — invalid.
                return

            w = distinct_wires[d]
            run_limit = max_run[d][pi]
            max_k = min(rem[d], num_pos - pi, run_limit)

            saved = rem[d]
            for k in range(max_k + 1):
                rem[d] = saved - k
                new_seen = (
                    seen | {w.gameplay_value}
                    if k > 0 and required
                    else seen
                )
                _compose(d + 1, pi + k, new_seen,
                         weight * math.comb(saved, k))
            rem[d] = saved

        _compose(0, 0, set())

        entry = (total[0], slot_counts, transitions)
        memo[key] = entry
        return entry

    _solve(0, ctx.initial_pool)

    if pbar is not None:
        pbar.close()

    return memo


# =============================================================================
# Public Solver API
# =============================================================================

def build_solver(
    game: bomb_busters.GameState,
    active_player_index: int,
    show_progress: bool = True,
) -> tuple[SolverContext, MemoDict] | None:
    """Build solver context and memo for a game state + observer.

    Call this once, then pass ctx/memo to any number of forward-pass
    functions (compute_position_probabilities, probability_of_double_detector,
    probability_of_red_wire_dd, etc.) for instant results.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        show_progress: If True (default), display a tqdm progress bar
            during the backward solve.

    Returns:
        A (SolverContext, MemoDict) tuple, or None if there are no
        hidden positions to solve.
    """
    ctx = _setup_solver(game, active_player_index)
    if ctx is None:
        return None
    memo = _solve_backward(ctx, show_progress=show_progress)
    return ctx, memo


# =============================================================================
# Monte Carlo Sampling
# =============================================================================

# Default threshold: when hidden positions exceed this, prefer Monte Carlo
# over the exact solver. Based on timing: 24 positions ≈ 2s exact,
# 28+ positions > 60s exact.
MC_POSITION_THRESHOLD = 22


def count_hidden_positions(
    game: bomb_busters.GameState,
    active_player_index: int,
) -> int:
    """Count hidden positions on other players' stands.

    Fast estimation of solver complexity without building the full
    SolverContext. Used to decide between exact solver and Monte Carlo.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.

    Returns:
        Total number of hidden and unknown-info-revealed positions
        on other players' stands, plus any discard positions from
        uncertain wire groups.
    """
    count = 0
    for p_idx, player in enumerate(game.players):
        if p_idx == active_player_index:
            continue
        for slot in player.tile_stand.slots:
            if slot.state == bomb_busters.SlotState.HIDDEN:
                count += 1
            elif (
                slot.state == bomb_busters.SlotState.INFO_REVEALED
                and slot.wire is None
                and slot.info_token == "YELLOW"
            ):
                count += 1
    # Add discard positions from uncertain wire groups
    known = extract_known_info(game, active_player_index)
    for group in game.uncertain_wire_groups:
        accounted_count = 0
        for wire in group.candidates:
            if wire in known.observer_wires or wire in known.cut_wires:
                accounted_count += 1
            else:
                # Check if wire is in the unknown pool (not yet accounted)
                is_on_other_stand = False
                for p_idx, player in enumerate(game.players):
                    if p_idx == active_player_index:
                        continue
                    for slot in player.tile_stand.slots:
                        if (
                            slot.wire == wire
                            and slot.state != bomb_busters.SlotState.HIDDEN
                        ):
                            is_on_other_stand = True
                            break
                    if is_on_other_stand:
                        break
                if is_on_other_stand:
                    accounted_count += 1
        remaining_in_play = max(0, group.count_in_play - accounted_count)
        unresolved = len(group.candidates) - accounted_count
        discard = unresolved - remaining_in_play
        count += max(0, discard)
    return count


def _guided_mc_sample(
    ctx: SolverContext,
    num_samples: int = 1_000,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> tuple[
    dict[tuple[int, int], collections.Counter[bomb_busters.Wire]],
    MCSamples,
] | None:
    """Approximate position probabilities via backward-guided MC sampling.

    For each sample, processes players sequentially. For each player,
    builds a lightweight backward table (like the exact solver, but
    for a single player with the current pool) and samples a composition
    from it. This guarantees valid ascending sequences with no dead ends.

    The per-player sampling doesn't account for downstream feasibility:
    some compositions leave pools that are harder for later players to
    fill. To correct for this, each sample is weighted by the product of
    per-player normalization constants (f[0][0] values), implementing
    self-normalized importance sampling. This produces unbiased
    probability estimates.

    The only source of rejection is must-have constraints from failed
    dual cuts, which are checked after each player's composition is
    sampled. Dead ends from must-have violations are rare.

    Args:
        ctx: Solver context from _setup_solver().
        num_samples: Target number of valid samples to collect.
        seed: Optional random seed for reproducibility.
        max_attempts: Maximum total attempts (including must-have
            rejections) before stopping. Defaults to
            ``num_samples * 5``.

    Returns:
        A tuple of (aggregated_probs, mc_samples), or None if zero
        valid samples were found. aggregated_probs maps
        (player_index, slot_index) to Counter of {Wire: weighted_count}.
        mc_samples holds raw per-sample assignments for joint queries.
    """
    if max_attempts is None:
        max_attempts = num_samples * 5

    rng = random.Random(seed)

    distinct_wires = ctx.distinct_wires
    num_types = len(distinct_wires)

    # Pre-compute per-player data
    player_data: list[tuple[int, list[PositionConstraint]]] = []
    for p in ctx.player_order:
        positions = ctx.positions_by_player[p]
        player_data.append((p, positions))

    # Validate pool size matches total positions
    total_positions = sum(len(positions) for _, positions in player_data)
    pool_size = sum(ctx.initial_pool)
    if pool_size != total_positions:
        return None

    must_have = ctx.must_have

    result: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]] = {}
    raw_samples: list[dict[tuple[int, int], bomb_busters.Wire]] = []
    raw_weights: list[int] = []
    valid_count = 0

    for _ in range(max_attempts):
        if valid_count >= num_samples:
            break

        # Reset remaining pool counts
        rem = list(ctx.initial_pool)

        dead_end = False
        sample_weight = 1
        assignments: list[tuple[int, int, bomb_busters.Wire]] = []

        for p, positions in player_data:
            num_pos = len(positions)

            # Precompute max_run for this player with current pool.
            # max_run[d][pi] = max consecutive positions starting at pi
            # that wire type d can fill.
            max_run: list[list[int]] = []
            for d in range(num_types):
                w = distinct_wires[d]
                runs = [0] * (num_pos + 1)
                for pi in range(num_pos - 1, -1, -1):
                    if positions[pi].wire_fits(w):
                        runs[pi] = runs[pi + 1] + 1
                    else:
                        runs[pi] = 0
                max_run.append(runs)

            # Backward pass: f[d][pi] = total combinatorial weight of
            # valid compositions that fill positions pi..N-1 using wire
            # types d..D-1 from the current pool.
            f = [[0] * (num_pos + 1) for _ in range(num_types + 1)]
            f[num_types][num_pos] = 1

            for d in range(num_types - 1, -1, -1):
                for pi in range(num_pos, -1, -1):
                    total = 0
                    max_k = min(rem[d], num_pos - pi, max_run[d][pi])
                    for k in range(max_k + 1):
                        total += math.comb(rem[d], k) * f[d + 1][pi + k]
                    f[d][pi] = total

            if f[0][0] == 0:
                dead_end = True
                break

            # Accumulate importance weight: product of per-player
            # normalization constants corrects for the sequential
            # sampler not accounting for downstream feasibility.
            sample_weight *= f[0][0]

            # Forward sampling: at each wire type d, sample how many
            # copies k to place, proportional to C(c_d, k) * f[d+1][pi+k].
            composition = [0] * num_types
            pi = 0
            for d in range(num_types):
                max_k = min(rem[d], num_pos - pi, max_run[d][pi])
                # Build weighted options
                total_w = 0
                options: list[tuple[int, int]] = []
                for k in range(max_k + 1):
                    w = math.comb(rem[d], k) * f[d + 1][pi + k]
                    if w > 0:
                        options.append((k, w))
                        total_w += w

                # Sample from options
                r = rng.randrange(total_w)
                cumulative = 0
                chosen_k = 0
                for k, w in options:
                    cumulative += w
                    if r < cumulative:
                        chosen_k = k
                        break

                composition[d] = chosen_k
                pi += chosen_k

            # Check must-have constraints
            required = must_have.get(p)
            if required:
                seen_values: set[int | str] = set()
                for d in range(num_types):
                    if composition[d] > 0:
                        seen_values.add(distinct_wires[d].gameplay_value)
                if not required.issubset(seen_values):
                    dead_end = True
                    break

            # Map composition to ascending sequence and record
            pi = 0
            for d in range(num_types):
                wire = distinct_wires[d]
                for _ in range(composition[d]):
                    if p != _DISCARD_PLAYER_INDEX:
                        assignments.append(
                            (p, positions[pi].slot_index, wire),
                        )
                    pi += 1

            # Update pool
            for d in range(num_types):
                rem[d] -= composition[d]

        if dead_end:
            continue

        valid_count += 1

        # Store raw per-sample assignments (excluding discard slots)
        sample_dict: dict[tuple[int, int], bomb_busters.Wire] = {}
        for p_idx, s_idx, wire in assignments:
            sample_dict[(p_idx, s_idx)] = wire
        raw_samples.append(sample_dict)
        raw_weights.append(sample_weight)

        # Aggregate into marginal per-position counters
        for p_idx, s_idx, wire in assignments:
            key = (p_idx, s_idx)
            counter = result.get(key)
            if counter is None:
                counter = collections.Counter()
                result[key] = counter
            counter[wire] += sample_weight

    if valid_count == 0:
        return None

    return result, MCSamples(samples=raw_samples, weights=raw_weights)


def monte_carlo_probabilities(
    game: bomb_busters.GameState,
    active_player_index: int,
    num_samples: int = 1_000,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]:
    """Approximate position probabilities via backward-guided MC sampling.

    Alternative to compute_position_probabilities() for game states
    where the exact solver is too slow (typically >22 hidden positions).
    Uses backward-guided composition sampling: for each player in each
    sample, builds a lightweight backward table (single-player DP) from
    the current pool and samples a valid composition from it. This
    guarantees valid ascending sequences with no dead ends. Samples are
    weighted by the product of per-player normalization constants
    (self-normalized importance sampling) to correct for the sequential
    sampler not accounting for downstream feasibility.

    Returns the same format as compute_position_probabilities(), so all
    downstream functions (rank_all_moves, print_probability_analysis,
    etc.) work identically with either result.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        num_samples: Target number of valid samples (default 1,000).
        seed: Optional random seed for reproducibility.
        max_attempts: Maximum total attempts (including must-have
            rejections). Defaults to ``num_samples * 5``.

    Returns:
        Dict mapping (player_index, slot_index) to Counter of
        {Wire: count}. Empty dict if no valid samples found or
        no hidden positions to solve.
    """
    ctx = _setup_solver(game, active_player_index)
    if ctx is None:
        return {}

    result = _guided_mc_sample(
        ctx,
        num_samples=num_samples,
        seed=seed,
        max_attempts=max_attempts if max_attempts is not None else None,
    )
    return result[0] if result is not None else {}


def monte_carlo_analysis(
    game: bomb_busters.GameState,
    active_player_index: int,
    num_samples: int = 1_000,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> tuple[
    dict[tuple[int, int], collections.Counter[bomb_busters.Wire]],
    MCSamples | None,
]:
    """Run Monte Carlo sampling returning both marginals and raw samples.

    Like ``monte_carlo_probabilities()``, but also returns the raw
    per-sample assignments needed for joint probability queries
    (Double Detector, joint red wire risk). Use ``mc_dd_probability()``
    and ``mc_red_dd_probability()`` with the returned MCSamples.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        num_samples: Target number of valid samples (default 1,000).
        seed: Optional random seed for reproducibility.
        max_attempts: Maximum total attempts (including must-have
            rejections). Defaults to ``num_samples * 5``.

    Returns:
        A tuple of (marginal_probs, mc_samples). marginal_probs is
        the same dict as ``monte_carlo_probabilities()``. mc_samples
        is an MCSamples object for joint queries, or None if no valid
        samples were found.
    """
    ctx = _setup_solver(game, active_player_index)
    if ctx is None:
        return {}, None

    result = _guided_mc_sample(
        ctx,
        num_samples=num_samples,
        seed=seed,
        max_attempts=max_attempts if max_attempts is not None else None,
    )
    if result is None:
        return {}, None
    return result[0], result[1]


def mc_dd_probability(
    mc_samples: MCSamples,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
    guessed_value: int | str,
) -> float:
    """Compute Double Detector probability from MC samples.

    Calculates P(at least one of the two target slots has the guessed
    value) using weighted self-normalized importance sampling over the
    raw per-sample assignments.

    Args:
        mc_samples: Raw MC samples from ``monte_carlo_analysis()``.
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.
        guessed_value: The value being guessed.

    Returns:
        Probability of success as a float between 0.0 and 1.0.
    """
    key1 = (target_player_index, slot_index_1)
    key2 = (target_player_index, slot_index_2)
    total_weight = 0
    match_weight = 0
    for sample, weight in zip(mc_samples.samples, mc_samples.weights):
        total_weight += weight
        wire1 = sample.get(key1)
        wire2 = sample.get(key2)
        if (
            (wire1 is not None
             and wire1.gameplay_value == guessed_value)
            or (wire2 is not None
                and wire2.gameplay_value == guessed_value)
        ):
            match_weight += weight
    if total_weight == 0:
        return 0.0
    return match_weight / total_weight


def mc_red_dd_probability(
    mc_samples: MCSamples,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
) -> float:
    """Compute joint red-wire probability for DD from MC samples.

    Calculates P(both target slots are red wires) using weighted
    self-normalized importance sampling.

    Args:
        mc_samples: Raw MC samples from ``monte_carlo_analysis()``.
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.

    Returns:
        Probability that both slots are red wires (0.0 to 1.0).
    """
    key1 = (target_player_index, slot_index_1)
    key2 = (target_player_index, slot_index_2)
    total_weight = 0
    match_weight = 0
    for sample, weight in zip(mc_samples.samples, mc_samples.weights):
        total_weight += weight
        wire1 = sample.get(key1)
        wire2 = sample.get(key2)
        if (
            wire1 is not None
            and wire1.color == bomb_busters.WireColor.RED
            and wire2 is not None
            and wire2.color == bomb_busters.WireColor.RED
        ):
            match_weight += weight
    if total_weight == 0:
        return 0.0
    return match_weight / total_weight


# =============================================================================
# Forward Passes
# =============================================================================

def _forward_pass_positions(
    ctx: SolverContext,
    memo: MemoDict,
) -> dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]:
    """Compute per-position wire probability distributions from a prebuilt memo.

    Iterates through players using memoized data, accumulating weighted
    per-position wire counts. No backtracking needed — just dict lookups
    and multiplications.

    Args:
        ctx: Solver context.
        memo: Prebuilt memo from _solve_backward().

    Returns:
        Dict mapping (player_index, slot_index) to a Counter of
        {Wire: count_of_valid_distributions}.
    """
    num_players = len(ctx.player_order)
    result: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]] = {}
    frontier: dict[tuple[int, ...], int] = {ctx.initial_pool: 1}

    for pli in range(num_players):
        p = ctx.player_order[pli]
        positions = ctx.positions_by_player[p]
        next_frontier: dict[tuple[int, ...], int] = {}

        for remaining, fwd_weight in frontier.items():
            entry = memo[(pli, remaining)]
            _, slot_counts, transitions = entry

            # Accumulate per-position counts for this player
            for pos_idx, counter in slot_counts.items():
                slot_idx = positions[pos_idx].slot_index
                rkey = (p, slot_idx)
                rcounter = result.get(rkey)
                if rcounter is None:
                    rcounter = collections.Counter()
                    result[rkey] = rcounter
                for wire, local_count in counter.items():
                    rcounter[wire] += fwd_weight * local_count

            # Advance frontier using transitions
            for new_remaining, seq_count in transitions.items():
                next_frontier[new_remaining] = (
                    next_frontier.get(new_remaining, 0)
                    + fwd_weight * seq_count
                )

        frontier = next_frontier

    # Filter out discard player entries (slack variables from uncertain
    # wire groups) — callers only care about real player positions.
    return {
        key: counter for key, counter in result.items()
        if key[0] != _DISCARD_PLAYER_INDEX
    }


def _find_target_positions(
    ctx: SolverContext,
    target_player_index: int,
    target_slot_indices: list[int],
) -> tuple[int, list[int]] | None:
    """Find a target player's level index and position indices in the solver.

    Args:
        ctx: Solver context.
        target_player_index: The player index to find.
        target_slot_indices: Slot indices to locate within that player's
            position constraints.

    Returns:
        A tuple of (player_level_index, position_indices), or None if
        the target player is not in the solver (e.g., the observer).
    """
    for pli, p in enumerate(ctx.player_order):
        if p == target_player_index:
            pos_indices: list[int] = []
            for pi, pc in enumerate(ctx.positions_by_player[p]):
                if pc.slot_index in target_slot_indices:
                    pos_indices.append(pi)
            return pli, pos_indices
    return None


def _advance_frontier(
    ctx: SolverContext,
    memo: MemoDict,
    stop_pli: int,
) -> dict[tuple[int, ...], int]:
    """Advance the frontier from the root to a specific player level.

    Uses memo transitions to propagate forward weights without any
    backtracking.

    Args:
        ctx: Solver context.
        memo: Prebuilt memo.
        stop_pli: Player level index to stop at (exclusive).

    Returns:
        Frontier dict mapping remaining_pool_tuple -> forward_weight
        at the stop_pli boundary.
    """
    frontier: dict[tuple[int, ...], int] = {ctx.initial_pool: 1}
    for pli in range(stop_pli):
        next_frontier: dict[tuple[int, ...], int] = {}
        for remaining, fwd_weight in frontier.items():
            transitions = memo[(pli, remaining)][2]
            for new_rem, seq_count in transitions.items():
                next_frontier[new_rem] = (
                    next_frontier.get(new_rem, 0)
                    + fwd_weight * seq_count
                )
        frontier = next_frontier
    return frontier


def _enumerate_target_player(
    ctx: SolverContext,
    memo: MemoDict,
    target_pli: int,
    frontier: dict[tuple[int, ...], int],
    check_fn: collections.abc.Callable[[list[bomb_busters.Wire], int], bool],
) -> tuple[int, int]:
    """Enumerate a target player's wire sequences and check a condition.

    Advances through the target player's positions using per-position
    backtracking, checking a condition on each complete sequence. Uses
    the memo for downstream total_ways (players after the target).

    Args:
        ctx: Solver context.
        memo: Prebuilt memo.
        target_pli: The target player's level index in player_order.
        frontier: Frontier at the target player boundary.
        check_fn: Called with (sequence, weight) for each valid complete
            sequence. Returns True if the sequence matches the desired
            condition (e.g., DD match, both-red). The weight accounts
            for both the forward weight and downstream completions.

    Returns:
        A tuple of (total_count, match_count) where total_count is the
        sum of weights for all valid sequences, and match_count is the
        sum of weights where check_fn returned True.
    """
    p = ctx.player_order[target_pli]
    positions = ctx.positions_by_player[p]
    distinct_wires = ctx.distinct_wires
    required = ctx.must_have.get(p, set())

    total_count = 0
    match_count = 0

    def _bt(
        pi: int, min_sv: float,
        seq: list[bomb_busters.Wire], seen: set[int | str],
        rem: list[int], fwd_weight: int,
        initial_rem: tuple[int, ...],
    ) -> None:
        nonlocal total_count, match_count
        if pi == len(positions):
            if required and not required.issubset(seen):
                return
            # Composition weight: ∏_d C(initial_count_d, k_d)
            comp_weight = 1
            for i in range(len(distinct_wires)):
                k = initial_rem[i] - rem[i]
                if k > 0:
                    comp_weight *= math.comb(initial_rem[i], k)
            sub_ways = memo[(target_pli + 1, tuple(rem))][0]
            if sub_ways == 0:
                return
            weight = fwd_weight * comp_weight * sub_ways
            total_count += weight
            if check_fn(seq, weight):
                match_count += weight
            return

        pc = positions[pi]
        lo = max(pc.lower_bound, min_sv)
        for i, w in enumerate(distinct_wires):
            if rem[i] <= 0:
                continue
            if w.sort_value < lo:
                continue
            if w.sort_value > pc.upper_bound:
                break
            rem[i] -= 1
            seq.append(w)
            new_seen = (
                seen | {w.gameplay_value} if required else seen
            )
            _bt(pi + 1, w.sort_value, seq, new_seen, rem,
                fwd_weight, initial_rem)
            seq.pop()
            rem[i] += 1

    for remaining, fwd_weight in frontier.items():
        _bt(0, 0.0, [], set(), list(remaining), fwd_weight,
            remaining)

    return total_count, match_count


def _forward_pass_dd(
    ctx: SolverContext,
    memo: MemoDict,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
    guessed_value: int | str,
) -> float:
    """Compute Double Detector probability from a prebuilt memo.

    Calculates P(at least one of the two target slots has the guessed value)
    using joint enumeration through the target player's sequences.

    Args:
        ctx: Solver context.
        memo: Prebuilt memo from _solve_backward().
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.
        guessed_value: The value being guessed.

    Returns:
        Probability of success as a float between 0.0 and 1.0.
    """
    result = _find_target_positions(
        ctx, target_player_index, [slot_index_1, slot_index_2],
    )
    if result is None:
        return 0.0
    target_pli, target_pos_indices = result

    # Check that root has valid distributions
    root_entry = memo.get((0, ctx.initial_pool))
    if root_entry is None or root_entry[0] == 0:
        return 0.0

    frontier = _advance_frontier(ctx, memo, target_pli)

    def check_dd(seq: list[bomb_busters.Wire], weight: int) -> bool:
        return any(
            seq[tpi].gameplay_value == guessed_value
            for tpi in target_pos_indices
        )

    total_count, match_count = _enumerate_target_player(
        ctx, memo, target_pli, frontier, check_dd,
    )

    if total_count == 0:
        return 0.0
    return match_count / total_count


def _forward_pass_red_dd(
    ctx: SolverContext,
    memo: MemoDict,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
) -> float:
    """Compute joint red-wire probability for Double Detector from a prebuilt memo.

    Calculates P(both target slots are red wires) using joint enumeration.

    Args:
        ctx: Solver context.
        memo: Prebuilt memo from _solve_backward().
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.

    Returns:
        Probability that both slots are red wires (0.0 to 1.0).
    """
    result = _find_target_positions(
        ctx, target_player_index, [slot_index_1, slot_index_2],
    )
    if result is None or len(result[1]) != 2:
        return 0.0
    target_pli, target_pos_indices = result

    # Check that root has valid distributions
    root_entry = memo.get((0, ctx.initial_pool))
    if root_entry is None or root_entry[0] == 0:
        return 0.0

    frontier = _advance_frontier(ctx, memo, target_pli)

    def check_both_red(seq: list[bomb_busters.Wire], weight: int) -> bool:
        return all(
            seq[tpi].color == bomb_busters.WireColor.RED
            for tpi in target_pos_indices
        )

    total_count, match_count = _enumerate_target_player(
        ctx, memo, target_pli, frontier, check_both_red,
    )

    if total_count == 0:
        return 0.0
    return match_count / total_count


# =============================================================================
# High-Level Probability API
# =============================================================================

def compute_position_probabilities(
    game: bomb_busters.GameState,
    active_player_index: int,
    ctx: SolverContext | None = None,
    memo: MemoDict | None = None,
    show_progress: bool = False,
) -> dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]:
    """Compute the probability distribution for each hidden slot.

    For each hidden position on other players' stands, counts how many
    valid wire distributions place each possible wire at that position.

    When ctx and memo are provided, skips the expensive backward solve
    and runs only the fast forward pass.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        ctx: Pre-built solver context. If None, will be computed.
        memo: Pre-built solver memo. If None, will be computed.
        show_progress: If True and memo needs building, show a tqdm
            progress bar.

    Returns:
        Dict mapping (player_index, slot_index) to a Counter of
        {Wire: count_of_valid_distributions}. The probability of a wire
        at a position is count / sum(counter.values()).
    """
    if ctx is None or memo is None:
        solver = build_solver(game, active_player_index, show_progress=show_progress)
        if solver is None:
            return {}
        ctx, memo = solver

    root_entry = memo.get((0, ctx.initial_pool))
    if root_entry is None or root_entry[0] == 0:
        return {}

    return _forward_pass_positions(ctx, memo)


def probability_of_dual_cut(
    game: bomb_busters.GameState,
    active_player_index: int,
    target_player_index: int,
    target_slot_index: int,
    guessed_value: int | str,
    ctx: SolverContext | None = None,
    memo: MemoDict | None = None,
    show_progress: bool = False,
) -> float:
    """Compute the probability that a dual cut succeeds.

    Calculates P(target slot has the guessed value) from the observer's
    perspective, considering all known information and constraints.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        target_player_index: Index of the target player.
        target_slot_index: Slot index on the target's stand.
        guessed_value: The value being guessed (int 1-12 or 'YELLOW').
        ctx: Pre-built solver context. If None, will be computed.
        memo: Pre-built solver memo. If None, will be computed.
        show_progress: If True and memo needs building, show progress.

    Returns:
        Probability of success as a float between 0.0 and 1.0.
    """
    probs = compute_position_probabilities(
        game, active_player_index, ctx=ctx, memo=memo, show_progress=show_progress,
    )
    key = (target_player_index, target_slot_index)
    if key not in probs:
        return 0.0

    counter = probs[key]
    total = sum(counter.values())
    if total == 0:
        return 0.0

    matching = sum(
        count for wire, count in counter.items()
        if wire.gameplay_value == guessed_value
    )
    return matching / total


def probability_of_double_detector(
    game: bomb_busters.GameState,
    active_player_index: int,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
    guessed_value: int | str,
    ctx: SolverContext | None = None,
    memo: MemoDict | None = None,
    show_progress: bool = False,
) -> float:
    """Compute the probability that a Double Detector succeeds.

    Calculates P(at least one of the two target slots has the guessed value).
    Uses joint probability from enumerated distributions, NOT the naive
    P(A) + P(B) - P(A)*P(B) formula (since slots share the same pool and
    are not independent).

    When ctx and memo are provided, skips the expensive backward solve
    and runs only the fast forward pass through the target player.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.
        guessed_value: The value being guessed (int 1-12 only, not YELLOW).
        ctx: Pre-built solver context. If None, will be computed.
        memo: Pre-built solver memo. If None, will be computed.
        show_progress: If True and memo needs building, show progress.

    Returns:
        Probability of success as a float between 0.0 and 1.0.
    """
    if ctx is None or memo is None:
        solver = build_solver(game, active_player_index, show_progress=show_progress)
        if solver is None:
            return 0.0
        ctx, memo = solver

    return _forward_pass_dd(
        ctx, memo, target_player_index,
        slot_index_1, slot_index_2, guessed_value,
    )


def probability_of_red_wire(
    game: bomb_busters.GameState,
    active_player_index: int,
    target_player_index: int,
    target_slot_index: int,
    probs: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]
    | None = None,
) -> float:
    """Compute the probability that a specific slot contains a red wire.

    Calculates P(target slot is a red wire) from the observer's perspective,
    considering all known information and constraints.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        target_player_index: Index of the target player.
        target_slot_index: Slot index on the target's stand.
        probs: Pre-computed position probabilities. If None, will be
            computed internally.

    Returns:
        Probability of the slot containing a red wire (0.0 to 1.0).
    """
    if probs is None:
        probs = compute_position_probabilities(game, active_player_index)
    key = (target_player_index, target_slot_index)
    if key not in probs:
        return 0.0

    counter = probs[key]
    total = sum(counter.values())
    if total == 0:
        return 0.0

    red_count = sum(
        count for wire, count in counter.items()
        if wire.color == bomb_busters.WireColor.RED
    )
    return red_count / total


def probability_of_red_wire_dd(
    game: bomb_busters.GameState,
    active_player_index: int,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
    ctx: SolverContext | None = None,
    memo: MemoDict | None = None,
    show_progress: bool = False,
) -> float:
    """Compute the probability that a Double Detector fails by hitting red.

    With the Double Detector, the bomb only explodes if BOTH targeted slots
    contain red wires. If only one is red and the other is not, the bomb
    does not explode (an info token is placed on the non-red wire instead).

    This computes P(both slots are red) using joint enumeration from the
    constraint solver, not naive independence assumptions.

    When ctx and memo are provided, skips the expensive backward solve
    and runs only the fast forward pass.

    Args:
        game: The current game state.
        active_player_index: The observing player's index.
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.
        ctx: Pre-built solver context. If None, will be computed.
        memo: Pre-built solver memo. If None, will be computed.
        show_progress: If True and memo needs building, show progress.

    Returns:
        Probability that both slots are red wires (0.0 to 1.0).
    """
    if ctx is None or memo is None:
        solver = build_solver(game, active_player_index, show_progress=show_progress)
        if solver is None:
            return 0.0
        ctx, memo = solver

    return _forward_pass_red_dd(
        ctx, memo, target_player_index,
        slot_index_1, slot_index_2,
    )


def guaranteed_actions(
    game: bomb_busters.GameState,
    active_player_index: int,
    probs: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]
    | None = None,
) -> dict[str, list | bool]:
    """Find all actions guaranteed to succeed.

    Identifies solo cuts, dual cuts with 100% probability, and whether
    reveal-red-wires is available.

    Args:
        game: The current game state.
        active_player_index: The observing player's index (the active player).
        probs: Pre-computed position probabilities. If None, will be
            computed internally.

    Returns:
        Dict with keys:
        - 'solo_cuts': list of (value, [slot_indices]) tuples
        - 'dual_cuts': list of (target_player, target_slot, value) tuples
        - 'reveal_red': bool indicating if reveal red action is available
    """
    player = game.players[active_player_index]
    result: dict[str, list | bool] = {
        "solo_cuts": [],
        "dual_cuts": [],
        "reveal_red": False,
    }

    # Solo cuts
    for value in game.available_solo_cuts(active_player_index):
        slots = [
            i for i, s in enumerate(player.tile_stand.slots)
            if s.is_hidden
            and s.wire is not None
            and s.wire.gameplay_value == value
        ]
        result["solo_cuts"].append((value, slots))  # type: ignore[union-attr]

    # Reveal red: all remaining hidden wires are red
    hidden = player.tile_stand.hidden_slots
    if hidden:
        all_red = all(
            s.wire is not None and s.wire.color == bomb_busters.WireColor.RED
            for _, s in hidden
        )
        result["reveal_red"] = all_red

    # 100% probability dual cuts from solver
    if probs is None:
        probs = compute_position_probabilities(game, active_player_index)
    for (p_idx, s_idx), counter in probs.items():
        total = sum(counter.values())
        if total == 0:
            continue
        for wire, count in counter.items():
            if count == total:
                # This position is 100% certain to be this wire
                gv = wire.gameplay_value
                # Check if observer has a matching wire to cut
                observer_has = any(
                    s.wire is not None
                    and s.wire.gameplay_value == gv
                    and s.is_hidden
                    for s in player.tile_stand.slots
                )
                if observer_has:
                    result["dual_cuts"].append((p_idx, s_idx, gv))  # type: ignore[union-attr]

    # Info-revealed guaranteed dual cuts: slots with info tokens have
    # publicly known values. If the observer has a matching wire, the
    # dual cut is guaranteed to succeed. These slots are not in the
    # solver (their identity is already determined) so they must be
    # checked separately.
    found_dual_cuts: set[tuple[int, int, int | str]] = {
        (p, s, v) for p, s, v in result["dual_cuts"]  # type: ignore[union-attr]
    }
    for p_idx, other_player in enumerate(game.players):
        if p_idx == active_player_index:
            continue
        for s_idx, slot in enumerate(other_player.tile_stand.slots):
            if not slot.is_info_revealed:
                continue
            if slot.info_token is None:
                continue
            gv = slot.info_token
            if (p_idx, s_idx, gv) in found_dual_cuts:
                continue
            observer_has = any(
                s.wire is not None
                and s.wire.gameplay_value == gv
                and s.is_hidden
                for s in player.tile_stand.slots
            )
            if observer_has:
                result["dual_cuts"].append((p_idx, s_idx, gv))  # type: ignore[union-attr]

    return result


@dataclasses.dataclass
class RankedMove:
    """A possible move ranked by probability of success.

    Attributes:
        action_type: 'dual_cut', 'solo_cut', 'double_detector', or 'reveal_red'.
        target_player: Target player index (for dual cuts).
        target_slot: Target slot index (for dual cuts).
        second_slot: Second slot index (for double detector).
        guessed_value: The value being guessed.
        probability: Probability of success (0.0 to 1.0).
        red_probability: Probability that the target slot is a red wire
            (0.0 to 1.0). For Double Detector, this is the probability
            that both target slots are red (the only way DD fails with red).
    """
    action_type: str
    target_player: int | None = None
    target_slot: int | None = None
    second_slot: int | None = None
    guessed_value: int | str | None = None
    probability: float = 0.0
    red_probability: float = 0.0

    def __str__(self) -> str:
        red = f" [RED {self.red_probability:.1%}]" if self.red_probability > 0 else ""
        if self.action_type == "solo_cut":
            return f"Solo Cut {self.guessed_value} (100%)"
        elif self.action_type == "reveal_red":
            return f"Reveal Red Wires (100%)"
        elif self.action_type == "double_detector":
            return (
                f"Double Detector P{self.target_player}"
                f"[{self.target_slot},{self.second_slot}]"
                f" = {self.guessed_value}"
                f" ({self.probability:.1%}){red}"
            )
        else:
            return (
                f"Dual Cut P{self.target_player}"
                f"[{self.target_slot}]"
                f" = {self.guessed_value}"
                f" ({self.probability:.1%}){red}"
            )


def rank_all_moves(
    game: bomb_busters.GameState,
    active_player_index: int,
    probs: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]
    | None = None,
    ctx: SolverContext | None = None,
    memo: MemoDict | None = None,
    include_dd: bool = False,
    mc_samples: MCSamples | None = None,
) -> list[RankedMove]:
    """Rank all possible moves by probability of success.

    Considers solo cuts (always 100%), reveal red (always 100% if available),
    and all possible dual cut targets with their probabilities. Optionally
    includes Double Detector moves for all valid slot pairs.

    Args:
        game: The current game state.
        active_player_index: The observing player's index (the active player).
        probs: Pre-computed position probabilities. If None, will be
            computed internally.
        ctx: Pre-built solver context (needed for exact DD moves).
        memo: Pre-built solver memo (needed for exact DD moves).
        include_dd: If True, also enumerate Double Detector moves for
            all pairs of hidden slots on each target player. Uses
            mc_samples if provided, otherwise ctx/memo.
        mc_samples: Raw MC samples for joint DD probability computation.
            When provided with ``include_dd=True``, DD probabilities are
            computed from the MC samples instead of the exact solver.

    Returns:
        List of RankedMove objects sorted by probability descending.
    """
    moves: list[RankedMove] = []
    player = game.players[active_player_index]

    # Solo cuts (guaranteed 100%)
    for value in game.available_solo_cuts(active_player_index):
        moves.append(RankedMove(
            action_type="solo_cut",
            guessed_value=value,
            probability=1.0,
        ))

    # Reveal red (guaranteed 100%)
    hidden = player.tile_stand.hidden_slots
    if hidden:
        all_red = all(
            s.wire is not None
            and s.wire.color == bomb_busters.WireColor.RED
            for _, s in hidden
        )
        if all_red:
            moves.append(RankedMove(
                action_type="reveal_red",
                probability=1.0,
            ))

    # Dual cuts: compute probabilities for all possible targets
    if probs is None:
        probs = compute_position_probabilities(game, active_player_index)

    # Determine which values the observer can cut (has in hand)
    observer_values: set[int | str] = set()
    for slot in player.tile_stand.slots:
        if slot.is_hidden and slot.wire is not None:
            observer_values.add(slot.wire.gameplay_value)

    # Blue values the observer holds (DD only works with blue)
    observer_blue_values: set[int] = {
        v for v in observer_values if isinstance(v, int)
    }

    for (p_idx, s_idx), counter in probs.items():
        total = sum(counter.values())
        if total == 0:
            continue

        # Compute red wire probability for this slot
        red_count = sum(
            count for wire, count in counter.items()
            if wire.color == bomb_busters.WireColor.RED
        )
        slot_red_prob = red_count / total if total > 0 else 0.0

        # Group by gameplay_value
        value_counts: collections.Counter[int | str] = collections.Counter()
        for wire, count in counter.items():
            value_counts[wire.gameplay_value] += count

        for value, match_count in value_counts.items():
            if value not in observer_values:
                continue
            prob = match_count / total
            if prob > 0:
                moves.append(RankedMove(
                    action_type="dual_cut",
                    target_player=p_idx,
                    target_slot=s_idx,
                    guessed_value=value,
                    probability=prob,
                    red_probability=slot_red_prob,
                ))

    # Info-revealed dual cuts: slots with info tokens have publicly known
    # values. These are guaranteed (100%) dual cuts if the observer has a
    # matching wire. They may not appear in probs because the solver
    # excludes slots whose identity is already determined.
    for p_idx, other_player in enumerate(game.players):
        if p_idx == active_player_index:
            continue
        for s_idx, slot in enumerate(other_player.tile_stand.slots):
            if not slot.is_info_revealed:
                continue
            if slot.info_token is None:
                continue
            gv = slot.info_token
            if gv not in observer_values:
                continue
            # Skip if already covered by probs-based dual cuts
            if (p_idx, s_idx) in probs:
                continue
            moves.append(RankedMove(
                action_type="dual_cut",
                target_player=p_idx,
                target_slot=s_idx,
                guessed_value=gv,
                probability=1.0,
                red_probability=0.0,
            ))

    # Double Detector moves
    if include_dd and observer_blue_values:
        dd_available = (
            player.character_card is not None
            and player.character_card.name == "Double Detector"
            and not player.character_card.used
        )
        if dd_available:
            if mc_samples is not None:
                # MC path: compute DD from raw per-sample assignments
                for p_idx in range(len(game.players)):
                    if p_idx == active_player_index:
                        continue
                    target_hidden = (
                        game.players[p_idx].tile_stand.hidden_slots
                    )
                    hidden_indices = [i for i, _ in target_hidden]
                    for a in range(len(hidden_indices)):
                        for b in range(a + 1, len(hidden_indices)):
                            s1 = hidden_indices[a]
                            s2 = hidden_indices[b]
                            for value in observer_blue_values:
                                dd_prob = mc_dd_probability(
                                    mc_samples, p_idx, s1, s2,
                                    value,
                                )
                                if dd_prob > 0:
                                    dd_red = mc_red_dd_probability(
                                        mc_samples, p_idx, s1, s2,
                                    )
                                    moves.append(RankedMove(
                                        action_type="double_detector",
                                        target_player=p_idx,
                                        target_slot=s1,
                                        second_slot=s2,
                                        guessed_value=value,
                                        probability=dd_prob,
                                        red_probability=dd_red,
                                    ))
            else:
                # Exact solver path
                if ctx is None or memo is None:
                    solver = build_solver(game, active_player_index)
                    if solver is not None:
                        ctx, memo = solver
                if ctx is not None and memo is not None:
                    for p_idx in range(len(game.players)):
                        if p_idx == active_player_index:
                            continue
                        target_hidden = (
                            game.players[p_idx].tile_stand.hidden_slots
                        )
                        hidden_indices = [i for i, _ in target_hidden]
                        for a in range(len(hidden_indices)):
                            for b in range(a + 1, len(hidden_indices)):
                                s1 = hidden_indices[a]
                                s2 = hidden_indices[b]
                                for value in observer_blue_values:
                                    dd_prob = _forward_pass_dd(
                                        ctx, memo, p_idx, s1, s2,
                                        value,
                                    )
                                    if dd_prob > 0:
                                        dd_red = _forward_pass_red_dd(
                                            ctx, memo, p_idx, s1, s2,
                                        )
                                        moves.append(RankedMove(
                                            action_type="double_detector",
                                            target_player=p_idx,
                                            target_slot=s1,
                                            second_slot=s2,
                                            guessed_value=value,
                                            probability=dd_prob,
                                            red_probability=dd_red,
                                        ))

    # Sort by probability descending
    moves.sort(key=lambda m: m.probability, reverse=True)
    return moves


# =============================================================================
# Indication Analysis
# =============================================================================

@dataclasses.dataclass
class IndicationChoice:
    """A ranked wire indication choice with quality metrics.

    Attributes:
        slot_index: Which slot on the stand to indicate.
        wire: The wire at that slot.
        information_gain: Entropy reduction in bits from indicating
            this wire. Higher means more information revealed.
        uncertainty_resolved: Fraction of baseline entropy resolved
            (information_gain / baseline_entropy), between 0.0 and 1.0.
        remaining_entropy: Total per-position entropy (bits) remaining
            on the stand after this indication.
    """
    slot_index: int
    wire: bomb_busters.Wire
    information_gain: float
    uncertainty_resolved: float
    remaining_entropy: float

    def __str__(self) -> str:
        letter = chr(ord("A") + self.slot_index)
        return (
            f"Indicate {self.wire.base_number} at [{letter}]: "
            f"{self.information_gain:.2f} bits "
            f"({self.uncertainty_resolved:.1%} resolved)"
        )


def _build_indication_constraints(
    slots: list[bomb_busters.Slot],
    indicate_slot: int | None = None,
) -> list[PositionConstraint]:
    """Build position constraints for hidden slots on a stand.

    When ``indicate_slot`` is specified, that slot is treated as
    INFO_REVEALED (with its wire's gameplay_value as the info token),
    and sort-value bounds for neighboring hidden slots are computed
    from the modified state.

    Args:
        slots: The stand's slot list (indicating player's own stand).
        indicate_slot: If set, the slot index to treat as indicated.

    Returns:
        List of PositionConstraint for each hidden slot (excluding
        the indicated slot if any).
    """
    # Create a working copy with the indication applied
    if indicate_slot is not None:
        work_slots: list[bomb_busters.Slot] = []
        for i, s in enumerate(slots):
            if i == indicate_slot:
                # Simulate the info token placement
                work_slots.append(bomb_busters.Slot(
                    wire=s.wire,
                    state=bomb_busters.SlotState.INFO_REVEALED,
                    info_token=(
                        s.wire.gameplay_value if s.wire is not None else None
                    ),
                ))
            else:
                work_slots.append(s)
    else:
        work_slots = slots

    constraints: list[PositionConstraint] = []
    for i, slot in enumerate(work_slots):
        if slot.state != bomb_busters.SlotState.HIDDEN:
            continue
        lower, upper = bomb_busters.get_sort_value_bounds(work_slots, i)
        constraints.append(PositionConstraint(
            player_index=0,
            slot_index=i,
            lower_bound=lower,
            upper_bound=upper,
        ))
    return constraints


def _enumerate_stand_distributions(
    constraints: list[PositionConstraint],
    distinct_wires: list[bomb_busters.Wire],
    pool_counts: tuple[int, ...],
) -> tuple[dict[int, collections.Counter[bomb_busters.Wire]], int]:
    """Compute per-position wire distributions on a single stand.

    Uses a two-pass dynamic programming approach over (wire_type,
    position) states:

    1. **Backward pass**: Computes ``f[d][pi]`` — the total
       combinatorial weight of valid compositions that fill positions
       ``pi..N-1`` using wire types ``d..D-1``.

    2. **Forward pass**: Propagates ``g[d][pi]`` — the accumulated
       weight reaching state ``(d, pi)`` from the root — and
       distributes per-position wire counts using both ``g`` and
       ``f`` values.

    Complexity: O(D * N * max_k) where D = distinct wire types,
    N = positions, max_k = max pool count per type (typically 4).

    Args:
        constraints: Position constraints for hidden slots, sorted by
            slot_index.
        distinct_wires: Sorted list of unique wire types in the pool.
        pool_counts: Tuple of available counts for each distinct wire
            type (parallel to distinct_wires).

    Returns:
        A tuple of (distributions, total_weight) where distributions
        maps position index (within constraints) to a Counter of
        {Wire: weighted_count}, and total_weight is the sum of all
        composition weights.
    """
    num_pos = len(constraints)
    num_types = len(distinct_wires)

    if num_pos == 0:
        return {}, 0

    # Precompute max_run: for each wire type, the max consecutive
    # positions it can fill starting from each position index.
    max_run: list[list[int]] = []
    for w in distinct_wires:
        runs = [0] * (num_pos + 1)
        for pi in range(num_pos - 1, -1, -1):
            if constraints[pi].wire_fits(w):
                runs[pi] = runs[pi + 1] + 1
            else:
                runs[pi] = 0
        max_run.append(runs)

    # ── Backward pass ──────────────────────────────────────────
    # f[d][pi] = total weight of valid completions for positions
    # pi..N-1 using wire types d..D-1.
    # Transition: f[d][pi] = sum_{k=0}^{max_k} C(c_d, k) * f[d+1][pi+k]
    f = [[0] * (num_pos + 1) for _ in range(num_types + 1)]
    f[num_types][num_pos] = 1  # base: all types used, all positions filled

    for d in range(num_types - 1, -1, -1):
        for pi in range(num_pos, -1, -1):
            total = 0
            max_k = min(pool_counts[d], num_pos - pi, max_run[d][pi])
            for k in range(max_k + 1):
                total += math.comb(pool_counts[d], k) * f[d + 1][pi + k]
            f[d][pi] = total

    total_weight = f[0][0]
    if total_weight == 0:
        return {}, 0

    # ── Forward pass ───────────────────────────────────────────
    # g[d][pi] = accumulated weight reaching state (d, pi) from root.
    # For each transition (d, pi) --k--> (d+1, pi+k):
    #   contribution = g[d][pi] * C(c_d, k) * f[d+1][pi+k]
    #   positions pi..pi+k-1 each get wire type d with this weight.
    g = [[0] * (num_pos + 1) for _ in range(num_types + 1)]
    g[0][0] = 1

    slot_counts: dict[int, collections.Counter[bomb_busters.Wire]] = {}

    for d in range(num_types):
        w = distinct_wires[d]
        for pi in range(num_pos + 1):
            if g[d][pi] == 0:
                continue
            max_k = min(pool_counts[d], num_pos - pi, max_run[d][pi])
            for k in range(max_k + 1):
                coeff = math.comb(pool_counts[d], k)
                g[d + 1][pi + k] += g[d][pi] * coeff
                if k > 0:
                    contribution = g[d][pi] * coeff * f[d + 1][pi + k]
                    for j in range(pi, pi + k):
                        counter = slot_counts.get(j)
                        if counter is None:
                            counter = collections.Counter()
                            slot_counts[j] = counter
                        counter[w] += contribution

    return slot_counts, total_weight


def _compute_entropy(
    distributions: dict[int, collections.Counter[bomb_busters.Wire]],
    total_weight: int,
) -> float:
    """Compute total Shannon entropy from per-position distributions.

    Args:
        distributions: Per-position Counter of {Wire: weighted_count}.
        total_weight: Sum of all composition weights (normalization).

    Returns:
        Total entropy in bits (sum of per-position entropies).
        Returns 0.0 if total_weight is zero.
    """
    if total_weight == 0:
        return 0.0

    total_entropy = 0.0
    for counter in distributions.values():
        pos_entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total_weight
                pos_entropy -= p * math.log2(p)
        total_entropy += pos_entropy
    return total_entropy


def rank_indications(
    game: bomb_busters.GameState,
    player_index: int,
) -> list[IndicationChoice]:
    """Rank all possible indication choices for a player.

    For each blue wire on the player's stand, computes the information
    gain (in bits) of indicating it. The information gain measures
    how much the indication reduces teammates' uncertainty about the
    remaining hidden wires on this stand.

    The metric is based on Shannon entropy reduction: baseline entropy
    (all hidden) minus remaining entropy (after indicating). See
    ``docs/INDICATION_QUALITY.md`` for the full mathematical
    formulation.

    Args:
        game: The current game state. The specified player must have
            all wire identities known on their stand.
        player_index: Index of the player choosing which wire to
            indicate.

    Returns:
        List of IndicationChoice objects sorted by information_gain
        descending (best indication first).

    Raises:
        ValueError: If the player's stand has unknown wires.
    """
    player = game.players[player_index]
    stand = player.tile_stand

    # Validate all wires are known
    unknown = [
        i for i, s in enumerate(stand.slots) if s.wire is None
    ]
    if unknown:
        letters = ", ".join(chr(ord("A") + i) for i in unknown)
        raise ValueError(
            f"Cannot compute indication quality: player "
            f"{player_index} ({player.name}) has unknown wires at "
            f"slot(s) {letters}. All wire identities must be known."
        )

    # Build the wire pool: all wires in play represent what a generic
    # observer considers possible for this stand. We do NOT remove the
    # indicating player's own wires — the observer doesn't know them.
    # We only remove publicly known wires from other stands.
    pool = list(game.wires_in_play)

    # Add uncertain wire group candidates to the pool
    for group in game.uncertain_wire_groups:
        pool.extend(group.candidates)

    # Remove publicly known wires from other players' stands
    for p_idx, other_player in enumerate(game.players):
        if p_idx == player_index:
            continue
        for slot in other_player.tile_stand.slots:
            if slot.state == bomb_busters.SlotState.CUT and slot.wire is not None:
                if slot.wire in pool:
                    pool.remove(slot.wire)
            elif (
                slot.state == bomb_busters.SlotState.INFO_REVEALED
                and slot.info_token is not None
            ):
                # Info-revealed: the wire identity is publicly known
                wire = _identify_info_revealed_wire(
                    game, p_idx,
                    other_player.tile_stand.slots.index(slot),
                    slot.info_token,
                )
                if wire is not None and wire in pool:
                    pool.remove(wire)

    # Build pool statistics
    pool_counter = collections.Counter(pool)
    distinct = sorted(pool_counter.keys(), key=lambda w: w.sort_value)
    pool_counts = tuple(pool_counter[w] for w in distinct)

    # Compute baseline entropy (all positions hidden)
    baseline_constraints = _build_indication_constraints(stand.slots)
    baseline_dist, baseline_weight = _enumerate_stand_distributions(
        baseline_constraints, distinct, pool_counts,
    )
    baseline_entropy = _compute_entropy(baseline_dist, baseline_weight)

    # Evaluate each possible indication
    choices: list[IndicationChoice] = []
    seen_wires: set[tuple[int, float]] = set()

    for s_idx, slot in enumerate(stand.slots):
        if slot.state != bomb_busters.SlotState.HIDDEN:
            continue
        if slot.wire is None:
            continue
        # Only blue wires can be indicated
        if slot.wire.color != bomb_busters.WireColor.BLUE:
            continue

        # Deduplicate: if we've already evaluated the same wire value
        # at this position, skip. (Different slots with the same wire
        # value adjacent to each other will have different constraints
        # depending on their position, so we DO evaluate the same value
        # at different positions — we only skip exact duplicates at the
        # exact same slot.)
        wire_key = (s_idx, slot.wire.sort_value)
        if wire_key in seen_wires:
            continue
        seen_wires.add(wire_key)

        # Build pool with this wire removed
        after_pool_counter = collections.Counter(pool_counter)
        wire_pool_key = slot.wire
        if after_pool_counter[wire_pool_key] > 0:
            after_pool_counter[wire_pool_key] -= 1
            if after_pool_counter[wire_pool_key] == 0:
                del after_pool_counter[wire_pool_key]

        after_distinct = sorted(
            after_pool_counter.keys(), key=lambda w: w.sort_value,
        )
        after_pool_counts = tuple(
            after_pool_counter[w] for w in after_distinct
        )

        # Build constraints with this slot indicated
        after_constraints = _build_indication_constraints(
            stand.slots, indicate_slot=s_idx,
        )
        after_dist, after_weight = _enumerate_stand_distributions(
            after_constraints, after_distinct, after_pool_counts,
        )
        after_entropy = _compute_entropy(after_dist, after_weight)

        ig = baseline_entropy - after_entropy
        resolved = ig / baseline_entropy if baseline_entropy > 0 else 0.0

        choices.append(IndicationChoice(
            slot_index=s_idx,
            wire=slot.wire,
            information_gain=ig,
            uncertainty_resolved=resolved,
            remaining_entropy=after_entropy,
        ))

    # Sort by information gain descending
    choices.sort(key=lambda c: c.information_gain, reverse=True)
    return choices


# =============================================================================
# Terminal Display
# =============================================================================

def _player_name(game: bomb_busters.GameState, index: int) -> str:
    """Return a colored player name string."""
    return f"{_C.BOLD}{game.players[index].name}{_C.RESET}"


def _value_label(value: int | str) -> str:
    """Return a colored label for a gameplay value."""
    if isinstance(value, int):
        return f"{_C.BLUE}{value}{_C.RESET}"
    elif value == "YELLOW":
        return f"{_C.YELLOW}Y{_C.RESET}"
    elif value == "RED":
        return f"{_C.RED}R{_C.RESET}"
    return str(value)


def _slot_letter(index: int) -> str:
    """Return the letter label for a slot index."""
    return chr(ord("A") + index)


def _prob_colored(probability: float) -> str:
    """Return a probability string colored by confidence level."""
    if probability >= 1.0:
        return f"{_C.GREEN}{_C.BOLD} 100%{_C.RESET}"
    elif probability >= 0.75:
        return f"{_C.GREEN}{probability:>5.1%}{_C.RESET}"
    elif probability >= 0.50:
        return f"{_C.BLUE}{probability:>5.1%}{_C.RESET}"
    elif probability >= 0.25:
        return f"{_C.YELLOW}{probability:>5.1%}{_C.RESET}"
    else:
        return f"{_C.RED}{probability:>5.1%}{_C.RESET}"


def _red_warning(red_prob: float, show_zero: bool = False) -> str:
    """Return a colored red wire warning string.

    Args:
        red_prob: Probability that the slot contains a red wire.
        show_zero: If True, display even when probability is 0%.

    Returns:
        Formatted warning string, or empty if zero and not show_zero.
    """
    if red_prob <= 0:
        if show_zero:
            return f"  {_C.RED}RED 0.0%{_C.RESET}"
        return ""
    return f"  {_C.RED}⚠ RED {red_prob:.1%}{_C.RESET}"


def _format_move(
    game: bomb_busters.GameState, move: RankedMove, rank: int,
    has_red_wires: bool = False,
) -> str:
    """Format a single ranked move as a colored terminal line.

    Args:
        game: The current game state.
        move: The ranked move to format.
        rank: The display rank number.
        has_red_wires: Whether the game includes red wires in play.
            When True, red probability is shown even at 0% for non-100%
            dual cuts.
    """
    num = f"{rank:>2}."
    prob = _prob_colored(move.probability)
    val = _value_label(move.guessed_value) if move.guessed_value is not None else "?"
    # Show red probability at 0% when red wires are in play,
    # except for guaranteed (100%) actions.
    show_zero_red = has_red_wires and move.probability < 1.0
    red = _red_warning(move.red_probability, show_zero=show_zero_red)

    if move.action_type == "solo_cut":
        label = f"{_C.GREEN}⚡ Solo Cut{_C.RESET}"
        return f"  {num} {prob}  {label} {val}"
    elif move.action_type == "reveal_red":
        label = f"{_C.RED}◆ Reveal Red Wires{_C.RESET}"
        return f"  {num} {prob}  {label}"
    elif move.action_type == "dual_cut" and move.target_player is not None:
        target = _player_name(game, move.target_player)
        slot = _slot_letter(move.target_slot) if move.target_slot is not None else "?"
        label = f"{_C.BLUE}✂ Dual Cut{_C.RESET}"
        return (
            f"  {num} {prob}  {label} → {target}"
            f" [{_C.BOLD}{slot}{_C.RESET}] = {val}{red}"
        )
    elif (
        move.action_type == "double_detector"
        and move.target_player is not None
    ):
        target = _player_name(game, move.target_player)
        s1 = _slot_letter(move.target_slot) if move.target_slot is not None else "?"
        s2 = _slot_letter(move.second_slot) if move.second_slot is not None else "?"
        label = f"{_C.YELLOW}⚙ Double Detector{_C.RESET}"
        return (
            f"  {num} {prob}  {label} → {target}"
            f" [{_C.BOLD}{s1},{s2}{_C.RESET}] = {val}{red}"
        )
    return f"  {num} {prob}  {move}"


def print_probability_analysis(
    game: bomb_busters.GameState,
    active_player_index: int,
    max_moves: int = 10,
    show_progress: bool = True,
    include_dd: bool = False,
    mc_threshold: int | None = None,
    mc_num_samples: int = 10_000,
) -> None:
    """Print a probability analysis for the active player.

    Shows guaranteed actions, then ranks the top moves by success
    probability with colored output. Automatically uses Monte Carlo
    sampling when the number of hidden positions exceeds the threshold.

    Args:
        game: The current game state.
        active_player_index: The player whose perspective to analyze.
        max_moves: Maximum number of ranked moves to display.
        show_progress: If True (default), show a tqdm progress bar
            during the backward solve.
        include_dd: If True, include Double Detector moves in ranking.
        mc_threshold: Hidden position count above which Monte Carlo
            is used instead of the exact solver. Defaults to
            ``MC_POSITION_THRESHOLD``. Set to 0 to always use Monte
            Carlo, or a very large number to always use the exact
            solver.
        mc_num_samples: Number of Monte Carlo samples when MC is used.
    """
    if mc_threshold is None:
        mc_threshold = MC_POSITION_THRESHOLD
    player = game.players[active_player_index]
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print(
        f"{_C.BOLD}Probability Analysis for "
        f"{player.name} (Player {active_player_index}){_C.RESET}"
    )
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print()

    # Decide: exact solver or Monte Carlo
    position_count = count_hidden_positions(game, active_player_index)
    use_mc = position_count > mc_threshold

    ctx: SolverContext | None = None
    memo: MemoDict | None = None
    mc_samples_data: MCSamples | None = None

    if use_mc:
        probs, mc_samples_data = monte_carlo_analysis(
            game, active_player_index,
            num_samples=mc_num_samples,
        )
        print(
            f"  {_C.DIM}(Monte Carlo: {position_count} unknown wires,"
            f" {mc_num_samples:,} samples){_C.RESET}"
        )
        print()
    else:
        solver = build_solver(
            game, active_player_index, show_progress=show_progress,
        )
        if show_progress:
            print()  # Extra spacing after progress bar
        if solver is not None:
            ctx, memo = solver
            probs = _forward_pass_positions(ctx, memo)
        else:
            probs = {}

    # Guaranteed actions
    ga = guaranteed_actions(game, active_player_index, probs=probs)
    solo_cuts: list[tuple[int | str, list[int]]] = ga["solo_cuts"]  # type: ignore[assignment]
    guaranteed_duals: list[tuple[int, int, int | str]] = ga["dual_cuts"]  # type: ignore[assignment]
    reveal_red: bool = ga["reveal_red"]  # type: ignore[assignment]

    has_guaranteed = solo_cuts or guaranteed_duals or reveal_red
    if has_guaranteed:
        print(f"  {_C.GREEN}{_C.BOLD}Guaranteed actions:{_C.RESET}")
        for value, slots in solo_cuts:
            slot_letters = ", ".join(_slot_letter(s) for s in slots)
            print(
                f"    • Solo Cut {_value_label(value)}"
                f" (slots {slot_letters})"
            )
        for p_idx, s_idx, value in guaranteed_duals:
            target = _player_name(game, p_idx)
            print(
                f"    • Dual Cut → {target}"
                f" [{_C.BOLD}{_slot_letter(s_idx)}{_C.RESET}]"
                f" = {_value_label(value)}"
            )
        if reveal_red:
            print("    • Reveal Red Wires")
        print()

    # Ranked moves
    moves = rank_all_moves(
        game, active_player_index, probs=probs,
        ctx=ctx, memo=memo, include_dd=include_dd,
        mc_samples=mc_samples_data,
    )

    if not moves:
        print(f"  {_C.DIM}No available moves.{_C.RESET}")
        return

    # Deduplicate DD moves: keep only the best pair per (player, value).
    # Without this, a single high-probability slot generates many DD
    # entries pairing it with every other slot on the same stand.
    seen_dd: set[tuple[int | None, int | str | None]] = set()
    deduped: list[RankedMove] = []
    for move in moves:
        if move.action_type == "double_detector":
            dd_key = (move.target_player, move.guessed_value)
            if dd_key in seen_dd:
                continue
            seen_dd.add(dd_key)
        deduped.append(move)
    moves = deduped

    has_red_wires = any(
        w.color == bomb_busters.WireColor.RED for w in game.wires_in_play
    )

    shown = moves[:max_moves]
    remaining = len(moves) - len(shown)

    print(f"  {_C.BOLD}Top moves by probability:{_C.RESET}")
    print()
    for i, move in enumerate(shown, 1):
        print(_format_move(game, move, i, has_red_wires=has_red_wires))
    print()

    if remaining > 0:
        print(f"  {_C.DIM}... and {remaining} more moves{_C.RESET}")
        print()


def print_indication_analysis(
    game: bomb_busters.GameState,
    player_index: int,
) -> None:
    """Print an indication quality analysis for a player.

    Shows each possible indication choice ranked by information gain,
    with the player's stand for context. Used at the start of a mission
    to help decide which wire to indicate.

    Args:
        game: The current game state. The specified player must have
            all wire identities known on their stand.
        player_index: Index of the player choosing which wire to
            indicate.
    """
    player = game.players[player_index]
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print(
        f"{_C.BOLD}Indication Analysis for "
        f"{player.name} (Player {player_index}){_C.RESET}"
    )
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print()

    # Show the player's stand for context
    status_line, values_line, letters_line = player.tile_stand.stand_lines()
    prefix = "    "
    print(f"{prefix}{status_line}")
    print(f"{prefix}{values_line}")
    print(f"{prefix}{letters_line}")
    print()

    choices = rank_indications(game, player_index)

    if not choices:
        print(f"  {_C.DIM}No blue wires available to indicate.{_C.RESET}")
        return

    # Show baseline entropy
    baseline = choices[0].remaining_entropy + choices[0].information_gain
    print(
        f"  {_C.DIM}Baseline uncertainty: "
        f"{baseline:.2f} bits{_C.RESET}"
    )
    print()

    print(f"  {_C.BOLD}Ranked indication choices:{_C.RESET}")
    print()
    for i, choice in enumerate(choices, 1):
        letter = _slot_letter(choice.slot_index)
        val = _value_label(choice.wire.gameplay_value)

        # Color the information gain by indication quality.
        # Thresholds use uncertainty_resolved (%) rather than raw
        # bits because % is normalized by baseline entropy and stable
        # across different game sizes (blue 1-12 vs 1-8 etc.).
        # Derived from 500 random 5-player games (blue 1-12, 0-4
        # yellow, 0-3 red): mean ≈ 20%, std ≈ 7%.
        #   Green:  ≥ 25%  (above mean + 1σ — excellent)
        #   Blue:   ≥ 20%  (above mean — good)
        #   Yellow: ≥ 13%  (above mean - 1σ — moderate)
        #   Red:    < 13%  (below mean - 1σ — poor)
        ig = choice.information_gain
        pct = choice.uncertainty_resolved
        if pct >= 0.25:
            ig_str = f"{_C.GREEN}{_C.BOLD}{ig:>5.2f}{_C.RESET}"
        elif pct >= 0.20:
            ig_str = f"{_C.BLUE}{ig:>5.2f}{_C.RESET}"
        elif pct >= 0.13:
            ig_str = f"{_C.YELLOW}{ig:>5.2f}{_C.RESET}"
        else:
            ig_str = f"{_C.RED}{ig:>5.2f}{_C.RESET}"

        pct = choice.uncertainty_resolved
        pct_str = f"{pct:.1%}"

        print(
            f"  {i:>2}. {ig_str} bits  "
            f"[{_C.BOLD}{letter}{_C.RESET}] = {val}"
            f"  {_C.DIM}({pct_str} resolved){_C.RESET}"
        )
    print()
