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

def compute_position_probabilities(
    game: bomb_busters.GameState, observer_index: int
) -> dict[tuple[int, int], collections.Counter[bomb_busters.Wire]]:
    """Compute the probability distribution for each hidden slot.

    For each hidden position on other players' stands, counts how many
    valid wire distributions place each possible wire at that position.

    Uses backtracking with identical-wire grouping for efficiency.

    Args:
        game: The current game state.
        observer_index: The observing player's index.

    Returns:
        Dict mapping (player_index, slot_index) to a Counter of
        {Wire: count_of_valid_distributions}. The probability of a wire
        at a position is count / sum(counter.values()).
    """
    known = extract_known_info(game, observer_index)
    unknown_pool = compute_unknown_pool(known, game)
    constraints = compute_position_constraints(game, observer_index)
    must_have = known.player_must_have

    if not constraints:
        return {}

    # Group constraints by player, sorted by slot index
    positions_by_player: dict[int, list[PositionConstraint]] = {}
    for c in constraints:
        positions_by_player.setdefault(c.player_index, []).append(c)
    for p_idx in positions_by_player:
        positions_by_player[p_idx].sort(key=lambda c: c.slot_index)

    # Build a flat ordered list of all positions to fill
    all_positions: list[PositionConstraint] = []
    player_order = sorted(positions_by_player.keys())
    for p_idx in player_order:
        all_positions.extend(positions_by_player[p_idx])

    # Get distinct wire types sorted by sort_value
    pool_counter = collections.Counter(unknown_pool)
    distinct_wires = sorted(pool_counter.keys(), key=lambda w: w.sort_value)

    # Result accumulators: (player, slot) -> Counter of wire -> valid count
    result: dict[tuple[int, int], collections.Counter[bomb_busters.Wire]] = {
        (c.player_index, c.slot_index): collections.Counter()
        for c in all_positions
    }
    total_valid = [0]  # Use list for mutability in closure

    def backtrack(
        pos_idx: int,
        remaining: dict[bomb_busters.Wire, int],
        last_sort_by_player: dict[int, float],
        assignment: dict[int, bomb_busters.Wire],
    ) -> None:
        """Recursively assign wires to positions.

        Args:
            pos_idx: Current position index in all_positions.
            remaining: Counter of available wires in the pool.
            last_sort_by_player: Last assigned sort_value per player
                (for ascending order enforcement).
            assignment: Current partial assignment {pos_idx: Wire}.
        """
        if pos_idx == len(all_positions):
            # Check must-have constraints
            if _check_must_have(assignment, all_positions, must_have):
                total_valid[0] += 1
                for idx, wire in assignment.items():
                    pc = all_positions[idx]
                    result[(pc.player_index, pc.slot_index)][wire] += 1
            return

        pc = all_positions[pos_idx]
        player = pc.player_index

        # Effective lower bound: max of position constraint and ascending
        # order within the player's stand
        min_sort = last_sort_by_player.get(player, 0.0)
        effective_lower = max(pc.lower_bound, min_sort)

        for wire in distinct_wires:
            if remaining.get(wire, 0) <= 0:
                continue
            if wire.sort_value < effective_lower:
                continue
            if wire.sort_value > pc.upper_bound:
                break  # Sorted, no more valid wires
            if not pc.wire_fits(wire):
                continue

            # Assign this wire
            remaining[wire] -= 1
            old_last = last_sort_by_player.get(player)
            last_sort_by_player[player] = wire.sort_value
            assignment[pos_idx] = wire

            backtrack(pos_idx + 1, remaining, last_sort_by_player, assignment)

            # Undo
            remaining[wire] += 1
            if old_last is not None:
                last_sort_by_player[player] = old_last
            else:
                del last_sort_by_player[player]
            del assignment[pos_idx]

    backtrack(0, dict(pool_counter), {}, {})

    return result


def _check_must_have(
    assignment: dict[int, bomb_busters.Wire],
    positions: list[PositionConstraint],
    must_have: dict[int, set[int | str]],
) -> bool:
    """Verify that a complete assignment satisfies must-have constraints.

    Args:
        assignment: Mapping from position index to assigned Wire.
        positions: The ordered list of PositionConstraint objects.
        must_have: Required values per player.

    Returns:
        True if all must-have constraints are satisfied.
    """
    if not must_have:
        return True

    for player_idx, required_values in must_have.items():
        player_values: set[int | str] = set()
        for pos_idx, wire in assignment.items():
            if positions[pos_idx].player_index == player_idx:
                player_values.add(wire.gameplay_value)
        if not required_values.issubset(player_values):
            return False
    return True


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
    # We need joint probabilities, so we must enumerate distributions
    # and check both positions together
    known = extract_known_info(game, observer_index)
    unknown_pool = compute_unknown_pool(known, game)
    constraints = compute_position_constraints(game, observer_index)

    if not constraints:
        return 0.0

    positions_by_player: dict[int, list[PositionConstraint]] = {}
    for c in constraints:
        positions_by_player.setdefault(c.player_index, []).append(c)
    for p_idx in positions_by_player:
        positions_by_player[p_idx].sort(key=lambda c: c.slot_index)

    all_positions: list[PositionConstraint] = []
    for p_idx in sorted(positions_by_player.keys()):
        all_positions.extend(positions_by_player[p_idx])

    pool_counter = collections.Counter(unknown_pool)
    distinct_wires = sorted(pool_counter.keys(), key=lambda w: w.sort_value)
    must_have = known.player_must_have

    # Find position indices for our two target slots
    target_pos_indices: list[int] = []
    for i, pc in enumerate(all_positions):
        if pc.player_index == target_player_index and pc.slot_index in (
            slot_index_1, slot_index_2
        ):
            target_pos_indices.append(i)

    total_valid = [0]
    matching_valid = [0]

    def backtrack(
        pos_idx: int,
        remaining: dict[bomb_busters.Wire, int],
        last_sort_by_player: dict[int, float],
        assignment: dict[int, bomb_busters.Wire],
    ) -> None:
        if pos_idx == len(all_positions):
            if _check_must_have(assignment, all_positions, must_have):
                total_valid[0] += 1
                # Check if either target slot has the guessed value
                for tpi in target_pos_indices:
                    if tpi in assignment:
                        if assignment[tpi].gameplay_value == guessed_value:
                            matching_valid[0] += 1
                            return
            return

        pc = all_positions[pos_idx]
        player = pc.player_index
        min_sort = last_sort_by_player.get(player, 0.0)
        effective_lower = max(pc.lower_bound, min_sort)

        for wire in distinct_wires:
            if remaining.get(wire, 0) <= 0:
                continue
            if wire.sort_value < effective_lower:
                continue
            if wire.sort_value > pc.upper_bound:
                break
            if not pc.wire_fits(wire):
                continue

            remaining[wire] -= 1
            old_last = last_sort_by_player.get(player)
            last_sort_by_player[player] = wire.sort_value
            assignment[pos_idx] = wire

            backtrack(pos_idx + 1, remaining, last_sort_by_player, assignment)

            remaining[wire] += 1
            if old_last is not None:
                last_sort_by_player[player] = old_last
            else:
                del last_sort_by_player[player]
            del assignment[pos_idx]

    backtrack(0, dict(pool_counter), {}, {})

    if total_valid[0] == 0:
        return 0.0
    return matching_valid[0] / total_valid[0]


def guaranteed_actions(
    game: bomb_busters.GameState, observer_index: int
) -> dict[str, list | bool]:
    """Find all actions guaranteed to succeed.

    Identifies solo cuts, dual cuts with 100% probability, and whether
    reveal-red-wires is available.

    Args:
        game: The current game state.
        observer_index: The observing player's index (the active player).

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
    game: bomb_busters.GameState, observer_index: int
) -> list[RankedMove]:
    """Rank all possible moves by probability of success.

    Considers solo cuts (always 100%), reveal red (always 100% if available),
    and all possible dual cut targets with their probabilities.

    Args:
        game: The current game state.
        observer_index: The observing player's index (the active player).

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

    # Guaranteed actions
    ga = guaranteed_actions(game, observer_index)
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
    moves = rank_all_moves(game, observer_index)

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
