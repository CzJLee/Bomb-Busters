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

    # Remove info-revealed wires
    for p_idx, s_idx, revealed_value in known.info_revealed:
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
                total[0] += sub_total
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
                            counter[w_di] += sub_total
                            pos += 1
                transitions[new_remaining] = (
                    transitions.get(new_remaining, 0) + 1
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
                _compose(d + 1, pi + k, new_seen)
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
    ) -> None:
        nonlocal total_count, match_count
        if pi == len(positions):
            if required and not required.issubset(seen):
                return
            sub_ways = memo[(target_pli + 1, tuple(rem))][0]
            if sub_ways == 0:
                return
            weight = fwd_weight * sub_ways
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
            _bt(pi + 1, w.sort_value, seq, new_seen, rem, fwd_weight)
            seq.pop()
            rem[i] += 1

    for remaining, fwd_weight in frontier.items():
        _bt(0, 0.0, [], set(), list(remaining), fwd_weight)

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

    # 100% probability dual cuts
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
        ctx: Pre-built solver context (needed for DD moves).
        memo: Pre-built solver memo (needed for DD moves).
        include_dd: If True, also enumerate Double Detector moves for
            all pairs of hidden slots on each target player. Requires
            ctx and memo to be provided (or they will be built).

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

    # Double Detector moves
    if include_dd and observer_blue_values:
        if ctx is None or memo is None:
            solver = build_solver(game, active_player_index)
            if solver is not None:
                ctx, memo = solver
        if ctx is not None and memo is not None:
            # Check DD availability
            dd_available = (
                player.character_card is not None
                and player.character_card.name == "Double Detector"
                and not player.character_card.used
            )
            if dd_available:
                for p_idx in range(len(game.players)):
                    if p_idx == active_player_index:
                        continue
                    target_hidden = game.players[p_idx].tile_stand.hidden_slots
                    hidden_indices = [i for i, _ in target_hidden]
                    # Enumerate all pairs of hidden slots
                    for a in range(len(hidden_indices)):
                        for b in range(a + 1, len(hidden_indices)):
                            s1 = hidden_indices[a]
                            s2 = hidden_indices[b]
                            for value in observer_blue_values:
                                dd_prob = _forward_pass_dd(
                                    ctx, memo, p_idx, s1, s2, value,
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
) -> None:
    """Print a probability analysis for the active player.

    Shows guaranteed actions, then ranks the top moves by success
    probability with colored output.

    Args:
        game: The current game state.
        active_player_index: The player whose perspective to analyze.
        max_moves: Maximum number of ranked moves to display.
        show_progress: If True (default), show a tqdm progress bar
            during the backward solve.
        include_dd: If True, include Double Detector moves in ranking.
    """
    player = game.players[active_player_index]
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print(
        f"{_C.BOLD}Probability Analysis for "
        f"{player.name} (Player {active_player_index}){_C.RESET}"
    )
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print()

    # Build solver once and share across all calls
    solver = build_solver(game, active_player_index, show_progress=show_progress)
    if show_progress:
        print()  # Extra spacing after progress bar
    ctx: SolverContext | None = None
    memo: MemoDict | None = None
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
