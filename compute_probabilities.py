"""Probability engine for Bomb Busters.

Computes the probability of success for different cut actions based on
the observable game state from a specific player's perspective. Supports
deduction from sort-order constraints, info tokens, validation tokens,
and turn history.
"""

from __future__ import annotations

import collections
import dataclasses
import bomb_busters

_C = bomb_busters._Colors


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
        observer_index: Index of the observing player.
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
    observer_index: int
    observer_wires: list[bomb_busters.Wire]
    cut_wires: list[bomb_busters.Wire]
    info_revealed: list[tuple[int, int, int | str]]
    validation_tokens: set[int]
    player_must_have: dict[int, set[int | str]]
    markers: list[bomb_busters.Marker]
    wires_in_play: list[bomb_busters.Wire]


def extract_known_info(
    game: bomb_busters.GameState, observer_index: int
) -> KnownInfo:
    """Extract all information visible to the observing player.

    Collects the observer's own hand, all publicly visible information
    (cut wires, info tokens, validation tokens, markers), and deductions
    from turn history.

    Args:
        game: The current game state.
        observer_index: Index of the player whose perspective to use.

    Returns:
        A KnownInfo object with all observable information.
    """
    observer = game.players[observer_index]

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
    player_must_have = _compute_must_have(game, observer_index)

    return KnownInfo(
        observer_index=observer_index,
        observer_wires=observer_wires,
        cut_wires=cut_wires,
        info_revealed=info_revealed,
        validation_tokens=game.validation_tokens,
        player_must_have=player_must_have,
        markers=game.markers,
        wires_in_play=list(game.wires_in_play),
    )


def _compute_must_have(
    game: bomb_busters.GameState, observer_index: int
) -> dict[int, set[int | str]]:
    """Determine which values each player must still have from failed dual cuts.

    A failed dual cut by player P for value V means P had at least one wire
    of value V. If P has not since cut any wire of value V, P still has it.

    Args:
        game: The current game state.
        observer_index: The observing player's index.

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
    observer = game.players[known.observer_index]
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
    """Sort-value constraints on a hidden slot.

    Defines the valid range of sort_values for a wire at a specific
    position on another player's stand, based on known neighboring wires.

    Attributes:
        player_index: Index of the player whose stand this slot is on.
        slot_index: The slot index on that player's stand.
        lower_bound: Sort value must be >= this (allows duplicates).
        upper_bound: Sort value must be <= this (allows duplicates).
    """
    player_index: int
    slot_index: int
    lower_bound: float
    upper_bound: float

    def wire_fits(self, wire: bomb_busters.Wire) -> bool:
        """Check if a wire could legally occupy this position.

        Args:
            wire: The wire to check.

        Returns:
            True if the wire's sort_value is within bounds.
        """
        return self.lower_bound <= wire.sort_value <= self.upper_bound


def compute_position_constraints(
    game: bomb_busters.GameState, observer_index: int
) -> list[PositionConstraint]:
    """Compute sort-value constraints for all hidden slots on other players' stands.

    For each hidden slot, scans left and right to find the nearest known
    wire (from CUT or INFO_REVEALED slots) to establish valid sort-value bounds.

    Args:
        game: The current game state.
        observer_index: The observing player's index.

    Returns:
        List of PositionConstraint objects for all hidden slots on
        other players' stands.
    """
    constraints: list[PositionConstraint] = []
    for p_idx, player in enumerate(game.players):
        if p_idx == observer_index:
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
    return constraints


# =============================================================================
# Enumeration Engine (Backtracking Solver)
# =============================================================================

def _setup_solver(
    game: bomb_busters.GameState,
    observer_index: int,
) -> tuple[
    dict[int, list[PositionConstraint]],
    list[int],
    list[bomb_busters.Wire],
    dict[bomb_busters.Wire, int],
    tuple[int, ...],
    dict[int, set[int | str]],
] | None:
    """Set up common data structures for the constraint solver.

    Returns None if there are no hidden positions to solve.

    Returns:
        Tuple of (positions_by_player, player_order, distinct_wires,
        pool_counter, initial_pool_tuple, must_have), or None.
    """
    known = extract_known_info(game, observer_index)
    unknown_pool = compute_unknown_pool(known, game)
    constraints = compute_position_constraints(game, observer_index)

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

    return (
        positions_by_player,
        player_order,
        distinct_wires,
        pool_counter,
        initial_pool,
        known.player_must_have,
    )


def _check_must_have(
    assignment: dict[tuple[int, int], bomb_busters.Wire],
    positions_by_player: dict[int, list[PositionConstraint]],
    must_have: dict[int, set[int | str]],
) -> bool:
    """Verify that a complete assignment satisfies must-have constraints.

    Args:
        assignment: Mapping from (player_index, slot_index) to Wire.
        positions_by_player: Position constraints grouped by player.
        must_have: Required values per player.

    Returns:
        True if all must-have constraints are satisfied.
    """
    if not must_have:
        return True

    for player_idx, required_values in must_have.items():
        if player_idx not in positions_by_player:
            # Skip the observer or players with no hidden slots —
            # their hands are already known and not part of the assignment.
            continue
        player_values: set[int | str] = set()
        for (p_idx, _s_idx), wire in assignment.items():
            if p_idx == player_idx:
                player_values.add(wire.gameplay_value)
        if not required_values.issubset(player_values):
            return False
    return True


def compute_position_probabilities(
    game: bomb_busters.GameState, observer_index: int
) -> dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]:
    """Compute the probability distribution for each hidden slot.

    For each hidden position on other players' stands, counts how many
    valid wire distributions place each possible wire at that position.

    Uses a two-phase DP approach for efficiency:
    1. Memoized total_ways: counts valid distributions for players[pli:]
       given a remaining pool state, with memoization at player boundaries.
    2. Forward pass: iterates through players, enumerating per-player
       sequences weighted by downstream completion counts.

    Args:
        game: The current game state.
        observer_index: The observing player's index.

    Returns:
        Dict mapping (player_index, slot_index) to a Counter of
        {Wire: count_of_valid_distributions}. The probability of a wire
        at a position is count / sum(counter.values()).
    """
    setup = _setup_solver(game, observer_index)
    if setup is None:
        return {}

    (
        positions_by_player,
        player_order,
        distinct_wires,
        pool_counter,
        initial_pool,
        must_have,
    ) = setup
    num_players = len(player_order)

    # Memo entry: (total_ways, slot_counts, transitions)
    #   total_ways: int — total valid distributions for players[pli:]
    #   slot_counts: dict[int, Counter[Wire, int]] — for each position
    #       index within this player, count of (seq × sub_ways) per wire
    #   transitions: dict[tuple[int,...], int] — for each new_remaining
    #       pool state, # of valid sequences that produce that state
    _MemoEntry = tuple[
        int,
        dict[int, collections.Counter[bomb_busters.Wire]],
        dict[tuple[int, ...], int],
    ]
    memo: dict[tuple[int, tuple[int, ...]], _MemoEntry] = {}

    def _solve(pli: int, remaining: tuple[int, ...]) -> _MemoEntry:
        """Compute and cache total_ways, slot_counts, and transitions.

        Uses composition-based enumeration: instead of deciding which wire
        goes at each position, decides how many copies of each wire type
        to use. Since positions are ascending and wire types are sorted,
        the assignment is deterministic from the composition. This reduces
        branching dramatically compared to per-position backtracking.
        """
        key = (pli, remaining)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if pli == num_players:
            entry: _MemoEntry = (1, {}, {})
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
            """Enumerate wire-type compositions via type-by-type DFS.

            Instead of choosing a wire for each position, chooses how
            many copies of each wire type to use. Since positions are
            ascending and types are sorted, the assignment is deterministic:
            the first k positions get k copies of the current type.

            Args:
                d: Current wire type index (0..num_types-1).
                pi: Next position to fill (0..num_pos).
                seen: Gameplay values seen so far (for must-have check).
            """
            if pi == num_pos:
                # All positions filled. Skip remaining wire types.
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

    # Seed the memo by calling _solve from the root.
    root_total = _solve(0, initial_pool)[0]
    if root_total == 0:
        return {}

    # Forward pass: iterate through players using memoized data.
    # No backtracking needed — just dict lookups and multiplications.
    result: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]] = {}
    frontier: dict[tuple[int, ...], int] = {initial_pool: 1}

    for pli in range(num_players):
        p = player_order[pli]
        positions = positions_by_player[p]
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

    return result


# =============================================================================
# High-Level Probability API
# =============================================================================

def probability_of_dual_cut(
    game: bomb_busters.GameState,
    observer_index: int,
    target_player_index: int,
    target_slot_index: int,
    guessed_value: int | str,
) -> float:
    """Compute the probability that a dual cut succeeds.

    Calculates P(target slot has the guessed value) from the observer's
    perspective, considering all known information and constraints.

    Args:
        game: The current game state.
        observer_index: The observing player's index.
        target_player_index: Index of the target player.
        target_slot_index: Slot index on the target's stand.
        guessed_value: The value being guessed (int 1-12 or 'YELLOW').

    Returns:
        Probability of success as a float between 0.0 and 1.0.
    """
    probs = compute_position_probabilities(game, observer_index)
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
    observer_index: int,
    target_player_index: int,
    slot_index_1: int,
    slot_index_2: int,
    guessed_value: int | str,
) -> float:
    """Compute the probability that a Double Detector succeeds.

    Calculates P(at least one of the two target slots has the guessed value).
    Uses joint probability from enumerated distributions, NOT the naive
    P(A) + P(B) - P(A)*P(B) formula (since slots share the same pool and
    are not independent).

    Uses the same memoized DP approach as compute_position_probabilities.
    Builds a forward frontier to the target player, then enumerates that
    player's sequences to check the joint condition weighted by downstream
    completion counts.

    Args:
        game: The current game state.
        observer_index: The observing player's index.
        target_player_index: Index of the target player.
        slot_index_1: First slot index on the target's stand.
        slot_index_2: Second slot index on the target's stand.
        guessed_value: The value being guessed (int 1-12 only, not YELLOW).

    Returns:
        Probability of success as a float between 0.0 and 1.0.
    """
    setup = _setup_solver(game, observer_index)
    if setup is None:
        return 0.0

    (
        positions_by_player,
        player_order,
        distinct_wires,
        _pool_counter,
        initial_pool,
        must_have,
    ) = setup
    num_players = len(player_order)

    # Find the target player's index in player_order and
    # which position indices correspond to the two target slots.
    target_pli: int | None = None
    target_pos_indices: list[int] = []
    for pli, p in enumerate(player_order):
        if p == target_player_index:
            target_pli = pli
            for pi, pc in enumerate(positions_by_player[p]):
                if pc.slot_index in (slot_index_1, slot_index_2):
                    target_pos_indices.append(pi)
            break

    if target_pli is None:
        return 0.0

    # Build memo using the same _solve structure.
    memo: dict[
        tuple[int, tuple[int, ...]],
        tuple[int, dict[int, collections.Counter[bomb_busters.Wire]],
              dict[tuple[int, ...], int]],
    ] = {}

    def _solve(
        pli: int, remaining: tuple[int, ...],
    ) -> tuple[int, dict, dict]:
        key = (pli, remaining)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if pli == num_players:
            entry = (1, {}, {})
            memo[key] = entry
            return entry

        p = player_order[pli]
        positions = positions_by_player[p]
        required = must_have.get(p, set())
        rem = list(remaining)
        total = [0]
        slot_counts: dict[int, collections.Counter[bomb_busters.Wire]] = {}
        transitions: dict[tuple[int, ...], int] = {}

        def _bt(
            pi: int, min_sv: float,
            seq: list[bomb_busters.Wire], seen: set[int | str],
        ) -> None:
            if pi == len(positions):
                if required and not required.issubset(seen):
                    return
                new_remaining = tuple(rem)
                sub_total = _solve(pli + 1, new_remaining)[0]
                if sub_total == 0:
                    return
                total[0] += sub_total
                for idx, wire in enumerate(seq):
                    counter = slot_counts.get(idx)
                    if counter is None:
                        counter = collections.Counter()
                        slot_counts[idx] = counter
                    counter[wire] += sub_total
                transitions[new_remaining] = (
                    transitions.get(new_remaining, 0) + 1
                )
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
                _bt(pi + 1, w.sort_value, seq, new_seen)
                seq.pop()
                rem[i] += 1

        _bt(0, 0.0, [], set())
        entry = (total[0], slot_counts, transitions)
        memo[key] = entry
        return entry

    root_total = _solve(0, initial_pool)[0]
    if root_total == 0:
        return 0.0

    # Forward pass: advance frontier to the target player using
    # memoized transitions, then enumerate target player's sequences
    # to compute the DD match count.
    frontier: dict[tuple[int, ...], int] = {initial_pool: 1}

    for pli in range(num_players):
        if pli == target_pli:
            break
        next_frontier: dict[tuple[int, ...], int] = {}
        for remaining, fwd_weight in frontier.items():
            transitions = memo[(pli, remaining)][2]
            for new_rem, seq_count in transitions.items():
                next_frontier[new_rem] = (
                    next_frontier.get(new_rem, 0)
                    + fwd_weight * seq_count
                )
        frontier = next_frontier

    # At target_pli: enumerate target player's sequences, checking
    # whether either target slot matches the guessed value.
    total_count = 0
    match_count = 0

    p = player_order[target_pli]
    positions = positions_by_player[p]
    required = must_have.get(p, set())

    for remaining, fwd_weight in frontier.items():
        rem = list(remaining)

        def _bt_dd(
            pi: int, min_sv: float,
            seq: list[bomb_busters.Wire], seen: set[int | str],
        ) -> None:
            nonlocal total_count, match_count
            if pi == len(positions):
                if required and not required.issubset(seen):
                    return
                sub_ways = _solve(target_pli + 1, tuple(rem))[0]
                if sub_ways == 0:
                    return
                weight = fwd_weight * sub_ways
                total_count += weight
                for tpi in target_pos_indices:
                    if seq[tpi].gameplay_value == guessed_value:
                        match_count += weight
                        break
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
                _bt_dd(pi + 1, w.sort_value, seq, new_seen)
                seq.pop()
                rem[i] += 1

        _bt_dd(0, 0.0, [], set())

    if total_count == 0:
        return 0.0
    return match_count / total_count


def guaranteed_actions(
    game: bomb_busters.GameState,
    observer_index: int,
    probs: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]
    | None = None,
) -> dict[str, list | bool]:
    """Find all actions guaranteed to succeed.

    Identifies solo cuts, dual cuts with 100% probability, and whether
    reveal-red-wires is available.

    Args:
        game: The current game state.
        observer_index: The observing player's index (the active player).
        probs: Pre-computed position probabilities. If None, will be
            computed internally.

    Returns:
        Dict with keys:
        - 'solo_cuts': list of (value, [slot_indices]) tuples
        - 'dual_cuts': list of (target_player, target_slot, value) tuples
        - 'reveal_red': bool indicating if reveal red action is available
    """
    player = game.players[observer_index]
    result: dict[str, list | bool] = {
        "solo_cuts": [],
        "dual_cuts": [],
        "reveal_red": False,
    }

    # Solo cuts
    for value in game.available_solo_cuts(observer_index):
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
        probs = compute_position_probabilities(game, observer_index)
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
    """
    action_type: str
    target_player: int | None = None
    target_slot: int | None = None
    second_slot: int | None = None
    guessed_value: int | str | None = None
    probability: float = 0.0

    def __str__(self) -> str:
        if self.action_type == "solo_cut":
            return f"Solo Cut {self.guessed_value} (100%)"
        elif self.action_type == "reveal_red":
            return f"Reveal Red Wires (100%)"
        elif self.action_type == "double_detector":
            return (
                f"DD P{self.target_player}"
                f"[{self.target_slot},{self.second_slot}]"
                f" = {self.guessed_value}"
                f" ({self.probability:.1%})"
            )
        else:
            return (
                f"Dual Cut P{self.target_player}"
                f"[{self.target_slot}]"
                f" = {self.guessed_value}"
                f" ({self.probability:.1%})"
            )


def rank_all_moves(
    game: bomb_busters.GameState,
    observer_index: int,
    probs: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]
    | None = None,
) -> list[RankedMove]:
    """Rank all possible moves by probability of success.

    Considers solo cuts (always 100%), reveal red (always 100% if available),
    and all possible dual cut targets with their probabilities.

    Args:
        game: The current game state.
        observer_index: The observing player's index (the active player).
        probs: Pre-computed position probabilities. If None, will be
            computed internally.

    Returns:
        List of RankedMove objects sorted by probability descending.
    """
    moves: list[RankedMove] = []
    player = game.players[observer_index]

    # Solo cuts (guaranteed 100%)
    for value in game.available_solo_cuts(observer_index):
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
        probs = compute_position_probabilities(game, observer_index)

    # Determine which values the observer can cut (has in hand)
    observer_values: set[int | str] = set()
    for slot in player.tile_stand.slots:
        if slot.is_hidden and slot.wire is not None:
            observer_values.add(slot.wire.gameplay_value)

    for (p_idx, s_idx), counter in probs.items():
        total = sum(counter.values())
        if total == 0:
            continue

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


def _format_move(
    game: bomb_busters.GameState, move: RankedMove, rank: int
) -> str:
    """Format a single ranked move as a colored terminal line."""
    num = f"{rank:>2}."
    prob = _prob_colored(move.probability)
    val = _value_label(move.guessed_value) if move.guessed_value is not None else "?"

    if move.action_type == "solo_cut":
        return f"  {num} {prob}  Solo Cut {val}"
    elif move.action_type == "reveal_red":
        return f"  {num} {prob}  Reveal Red Wires"
    elif move.action_type == "dual_cut" and move.target_player is not None:
        target = _player_name(game, move.target_player)
        slot = _slot_letter(move.target_slot) if move.target_slot is not None else "?"
        return (
            f"  {num} {prob}  Dual Cut → {target}"
            f" [{_C.BOLD}{slot}{_C.RESET}] = {val}"
        )
    elif (
        move.action_type == "double_detector"
        and move.target_player is not None
    ):
        target = _player_name(game, move.target_player)
        s1 = _slot_letter(move.target_slot) if move.target_slot is not None else "?"
        s2 = _slot_letter(move.second_slot) if move.second_slot is not None else "?"
        return (
            f"  {num} {prob}  DD → {target}"
            f" [{_C.BOLD}{s1},{s2}{_C.RESET}] = {val}"
        )
    return f"  {num} {prob}  {move}"


def print_probability_analysis(
    game: bomb_busters.GameState,
    observer_index: int,
    max_moves: int = 10,
) -> None:
    """Print a probability analysis for the active player.

    Shows guaranteed actions, then ranks the top moves by success
    probability with colored output.

    Args:
        game: The current game state.
        observer_index: The player whose perspective to analyze.
        max_moves: Maximum number of ranked moves to display.
    """
    player = game.players[observer_index]
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print(
        f"{_C.BOLD}Probability Analysis for "
        f"{player.name} (Player {observer_index}){_C.RESET}"
    )
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print()

    # Compute probabilities once and share across both calls
    probs = compute_position_probabilities(game, observer_index)

    # Guaranteed actions
    ga = guaranteed_actions(game, observer_index, probs=probs)
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
            print(f"    • Reveal Red Wires")
        print()

    # Ranked moves
    moves = rank_all_moves(game, observer_index, probs=probs)

    if not moves:
        print(f"  {_C.DIM}No available moves.{_C.RESET}")
        return

    shown = moves[:max_moves]
    remaining = len(moves) - len(shown)

    print(f"  {_C.BOLD}Top moves by probability:{_C.RESET}")
    print()
    for i, move in enumerate(shown, 1):
        print(_format_move(game, move, i))
    print()

    if remaining > 0:
        print(f"  {_C.DIM}... and {remaining} more moves{_C.RESET}")
        print()
