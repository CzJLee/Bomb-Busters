"""Simulate a full Bomb Busters mission from start to finish.

Creates a 5-player game with blue wires, 2-of-3 yellow wires, and
1-of-2 red wires. Runs the indication phase (computing optimal
indications via Shannon entropy), then plays through the cutting phase
using probability analysis to select the best move each turn.

Automatically switches between the exact constraint solver and Monte
Carlo sampling based on game complexity (hidden position count).

Tracks per-player performance metrics and timing throughout.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities

_C = bomb_busters._Colors


# =============================================================================
# Metrics
# =============================================================================

class PlayerMetrics:
    """Track per-player action metrics during a simulation."""

    def __init__(self, player_name: str) -> None:
        self.player_name = player_name
        self.probabilities: list[float] = []
        self.successes: list[bool] = []
        self.action_types: list[str] = []

    def record(
        self, probability: float, success: bool, action_type: str,
    ) -> None:
        """Record an action taken by this player."""
        self.probabilities.append(probability)
        self.successes.append(success)
        self.action_types.append(action_type)

    @property
    def average_probability(self) -> float:
        """Average probability of actions taken."""
        if not self.probabilities:
            return 0.0
        return sum(self.probabilities) / len(self.probabilities)

    @property
    def success_rate(self) -> float:
        """Fraction of actions that succeeded."""
        if not self.successes:
            return 0.0
        return sum(self.successes) / len(self.successes)

    @property
    def total_actions(self) -> int:
        return len(self.probabilities)

    @property
    def success_count(self) -> int:
        return sum(self.successes)


# =============================================================================
# Move Selection
# =============================================================================

def _is_info_revealed_target(
    game: bomb_busters.GameState, move: compute_probabilities.RankedMove,
) -> bool:
    """Check if a dual cut targets an info-revealed slot."""
    if move.target_player is None or move.target_slot is None:
        return False
    slot = game.players[move.target_player].tile_stand.slots[move.target_slot]
    return slot.is_info_revealed


def _tiebreak_key(
    game: bomb_busters.GameState, move: compute_probabilities.RankedMove,
) -> tuple[int, int]:
    """Sort key for tiebreaking (lower is better).

    Priority: solo cut > reveal red > dual cut (non-indicated)
              > dual cut (indicated) > DD.
    """
    if move.action_type == "solo_cut":
        return (0, 0)
    if move.action_type == "reveal_red":
        return (1, 0)
    if move.action_type == "dual_cut":
        if _is_info_revealed_target(game, move):
            return (3, 0)
        return (2, 0)
    return (4, 0)


def select_best_move(
    game: bomb_busters.GameState,
    moves: list[compute_probabilities.RankedMove],
) -> compute_probabilities.RankedMove:
    """Select the best move with tiebreaking and DD decision logic.

    Tiebreaking among equal probabilities:
        solo cut > reveal red > dual cut (non-indicated)
        > dual cut (indicated)

    DD decision: use the Double Detector only when no 100% option
    exists, the best regular dual cut is < 80%, and a DD move offers
    higher probability.
    """
    if not moves:
        raise ValueError("No moves available")

    non_dd = [m for m in moves if m.action_type != "double_detector"]
    dd_only = [m for m in moves if m.action_type == "double_detector"]

    # Check for guaranteed (100%) options among non-DD moves
    guaranteed = [m for m in non_dd if m.probability >= 1.0 - 1e-9]
    if guaranteed:
        return min(guaranteed, key=lambda m: _tiebreak_key(game, m))

    # No guaranteed options — check DD decision
    dual_cuts_only = [m for m in non_dd if m.action_type == "dual_cut"]
    best_dual = (
        max(dual_cuts_only, key=lambda m: m.probability)
        if dual_cuts_only else None
    )
    best_dd = (
        max(dd_only, key=lambda m: m.probability)
        if dd_only else None
    )

    if (
        best_dual is not None
        and best_dual.probability < 0.80
        and best_dd is not None
        and best_dd.probability > best_dual.probability
    ):
        return best_dd

    # Pick best non-DD move with tiebreaking
    if non_dd:
        best_prob = max(m.probability for m in non_dd)
        tied = [m for m in non_dd if abs(m.probability - best_prob) < 1e-9]
        return min(tied, key=lambda m: _tiebreak_key(game, m))

    # Fallback: best DD move
    if dd_only:
        return max(dd_only, key=lambda m: m.probability)

    raise ValueError("No moves available")


# =============================================================================
# Move Execution
# =============================================================================

def execute_move(
    game: bomb_busters.GameState,
    move: compute_probabilities.RankedMove,
) -> tuple[bool, str]:
    """Execute a ranked move on the game state.

    Args:
        game: The game state (will be mutated).
        move: The move to execute.

    Returns:
        A tuple of (success, description).
    """
    pi = game.current_player_index
    player = game.players[pi]

    if move.action_type == "solo_cut":
        value = move.guessed_value
        slots = [
            i for i, s in enumerate(player.tile_stand.slots)
            if s.is_hidden
            and s.wire is not None
            and s.wire.gameplay_value == value
        ]
        game.execute_solo_cut(value, slots)
        letters = ", ".join(chr(ord("A") + i) for i in slots)
        return True, f"Solo Cut {value} (slots {letters})"

    if move.action_type == "reveal_red":
        game.execute_reveal_red()
        return True, "Reveal Red Wires"

    if move.action_type == "dual_cut":
        target_name = game.players[move.target_player].name
        letter = chr(ord("A") + move.target_slot)
        action = game.execute_dual_cut(
            move.target_player, move.target_slot, move.guessed_value,
        )
        success = action.result == bomb_busters.ActionResult.SUCCESS
        actual = action.actual_wire
        actual_label = ""
        if actual is not None and not success:
            if actual.color == bomb_busters.WireColor.RED:
                actual_label = f" (was RED-{actual.base_number})"
            elif actual.color == bomb_busters.WireColor.YELLOW:
                actual_label = f" (was YELLOW-{actual.base_number})"
            else:
                actual_label = f" (was blue-{actual.base_number})"
        return success, (
            f"Dual Cut \u2192 {target_name} [{letter}]"
            f" = {move.guessed_value}{actual_label}"
        )

    if move.action_type == "double_detector":
        target_name = game.players[move.target_player].name
        s1 = chr(ord("A") + move.target_slot)
        s2 = chr(ord("A") + move.second_slot)
        action = game.execute_dual_cut(
            move.target_player,
            move.target_slot,
            move.guessed_value,
            is_double_detector=True,
            second_target_slot_index=move.second_slot,
        )
        success = action.result == bomb_busters.ActionResult.SUCCESS
        return success, (
            f"Double Detector \u2192 {target_name} [{s1},{s2}]"
            f" = {move.guessed_value}"
        )

    raise ValueError(f"Unknown action type: {move.action_type}")


# =============================================================================
# Display Helpers
# =============================================================================

def _dedup_dd_moves(
    moves: list[compute_probabilities.RankedMove],
) -> list[compute_probabilities.RankedMove]:
    """Keep only the best DD pair per (target_player, guessed_value)."""
    seen: set[tuple[int | None, int | str | None]] = set()
    result: list[compute_probabilities.RankedMove] = []
    for move in moves:
        if move.action_type == "double_detector":
            key = (move.target_player, move.guessed_value)
            if key in seen:
                continue
            seen.add(key)
        result.append(move)
    return result


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Simulate a full Bomb Busters mission."""
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Full Mission Simulation{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print(
        "  Mission: 48 blue wires + 2-of-3 yellow (UNCERTAIN)"
        " + 1-of-2 red (UNCERTAIN)"
    )
    print("  Players: Alice (captain), Bob, Charlie, Diana, Eve")
    print()

    # ── Create game ────────────────────────────────────────────
    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        yellow_wires=(2, 3),
        red_wires=3,
        seed=2,
    )

    # Show initial game state (god mode)
    game.active_player_index = None
    print(game)

    # Initialize metrics
    metrics = {
        i: PlayerMetrics(game.players[i].name)
        for i in range(len(game.players))
    }
    total_indication_time = 0.0
    total_action_time = 0.0

    # ── Indication phase ───────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}INDICATION PHASE{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()

    for player_index in range(len(game.players)):
        print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
        print()

        t0 = time.perf_counter()
        choices = compute_probabilities.rank_indications(
            game, player_index,
        )
        elapsed = time.perf_counter() - t0
        total_indication_time += elapsed

        compute_probabilities.print_indication_analysis(
            game, player_index,
        )

        print(f"  {_C.DIM}Computed in {elapsed:.3f}s{_C.RESET}")
        print()

        if choices:
            best = choices[0]
            stand = game.players[player_index].tile_stand
            stand.place_info_token(best.slot_index, best.wire.gameplay_value)
            letter = chr(ord("A") + best.slot_index)
            print(
                f"  {_C.GREEN}{_C.BOLD}\u2192 Indicated"
                f" {best.wire.base_number} at"
                f" [{letter}]{_C.RESET}"
            )
        else:
            print(
                f"  {_C.DIM}No blue wires to indicate.{_C.RESET}"
            )
        print()

    # Show state after indications
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Game State After Indications{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    game.active_player_index = None
    print(game)

    # ── Cutting phase ──────────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}CUTTING PHASE{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()

    has_red_wires = any(
        w.color == bomb_busters.WireColor.RED
        for w in game.wires_in_play
    )

    turn_number = 0
    while not game.game_over:
        turn_number += 1
        player_index = game.current_player_index
        player = game.players[player_index]

        print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
        print(
            f"{_C.BOLD}Turn {turn_number}: "
            f"{player.name} (Player {player_index}){_C.RESET}"
        )
        print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
        print()

        t0 = time.perf_counter()

        # Decide: exact solver or Monte Carlo based on complexity
        position_count = compute_probabilities.count_hidden_positions(
            game, player_index,
        )
        use_mc = position_count > compute_probabilities.MC_POSITION_THRESHOLD

        include_equipment: set[compute_probabilities.EquipmentType] | None = None
        if (
            player.character_card is not None
            and player.character_card.name == "Double Detector"
            and not player.character_card.used
        ):
            include_equipment = {
                compute_probabilities.EquipmentType.DOUBLE_DETECTOR,
            }

        if use_mc:
            # Monte Carlo path
            mc_num_samples = 10_000
            probs, mc_samples = (
                compute_probabilities.monte_carlo_analysis(
                    game, player_index,
                    num_samples=mc_num_samples,
                    show_progress=True,
                )
            )
            moves = compute_probabilities.rank_all_moves(
                game, player_index, probs=probs,
                include_equipment=include_equipment,
                mc_samples=mc_samples,
            )
            elapsed = time.perf_counter() - t0
            total_action_time += elapsed

            display_moves = _dedup_dd_moves(moves)[:5]
            method_label = (
                f"Monte Carlo, {position_count} unknown wires,"
                f" {mc_num_samples:,} samples"
            )

        else:
            # Exact solver path
            solver = compute_probabilities.build_solver(
                game, player_index, show_progress=False,
            )
            ctx = memo = None
            probs = {}
            if solver is not None:
                ctx, memo = solver
                probs = compute_probabilities._forward_pass_positions(
                    ctx, memo,
                )
            moves = compute_probabilities.rank_all_moves(
                game, player_index, probs=probs,
                ctx=ctx, memo=memo, include_equipment=include_equipment,
            )
            elapsed = time.perf_counter() - t0
            total_action_time += elapsed

            display_moves = _dedup_dd_moves(moves)[:5]
            method_label = (
                f"exact solver, {position_count} unknown wires"
            )

        # Display top moves
        print(f"  {_C.BOLD}Top moves ({method_label}):{_C.RESET}")
        print()
        for i, move in enumerate(display_moves, 1):
            print(compute_probabilities._format_move(
                game, move, i, has_red_wires=has_red_wires,
            ))
        print()
        print(f"  {_C.DIM}Computed in {elapsed:.3f}s{_C.RESET}")
        print()

        # Select and execute
        best_move = select_best_move(game, moves)
        move_prob = best_move.probability

        success, description = execute_move(game, best_move)

        # Record metrics
        metrics[player_index].record(
            move_prob, success, best_move.action_type,
        )

        # Print result
        if success:
            result_str = f"{_C.GREEN}\u2713 SUCCESS{_C.RESET}"
        elif game.game_over and not game.game_won:
            if any(
                isinstance(a, bomb_busters.DualCutAction)
                and a.result == bomb_busters.ActionResult.FAIL_RED
                for a in game.history.actions[-1:]
            ):
                result_str = (
                    f"{_C.RED}\u2717 FAIL (RED WIRE \u2014 "
                    f"BOMB EXPLODED!){_C.RESET}"
                )
            else:
                result_str = (
                    f"{_C.RED}\u2717 FAIL (DETONATOR "
                    f"EXPLODED!){_C.RESET}"
                )
        else:
            result_str = f"{_C.RED}\u2717 FAIL{_C.RESET}"

        print(f"  {_C.BOLD}Action:{_C.RESET} {description}")
        print(f"  {_C.BOLD}Probability:{_C.RESET} {move_prob:.1%}")
        print(f"  {_C.BOLD}Result:{_C.RESET} {result_str}")

        if not success and not game.game_over:
            print(
                f"  {_C.BOLD}Mistakes remaining:{_C.RESET} "
                f"{game.detonator.remaining_failures}"
            )
        print()

    # ── Final state ────────────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    if game.game_won:
        print(f"{_C.GREEN}{_C.BOLD}MISSION SUCCESS!{_C.RESET}")
    else:
        print(f"{_C.RED}{_C.BOLD}MISSION FAILED!{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    game.active_player_index = None
    print(game)

    # ── Metrics summary ───────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Performance Metrics{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()

    print(f"  {_C.BOLD}Per-player action metrics:{_C.RESET}")
    print()
    for i in range(len(game.players)):
        m = metrics[i]
        if m.total_actions > 0:
            avg = m.average_probability
            sr = m.success_rate
            print(
                f"    {m.player_name:>10}: "
                f"avg probability {avg:>5.1%} | "
                f"success rate {sr:>5.1%} "
                f"({m.success_count}/{m.total_actions} actions)"
            )
        else:
            print(f"    {m.player_name:>10}: no actions taken")
    print()

    print(f"  {_C.BOLD}Timing:{_C.RESET}")
    print(f"    Indication analysis: {total_indication_time:.3f}s")
    print(f"    Action analysis:     {total_action_time:.3f}s")
    total_time = total_indication_time + total_action_time
    print(f"    Total:               {total_time:.3f}s")
    print()

    total_successes = sum(m.success_count for m in metrics.values())
    total_actions = sum(m.total_actions for m in metrics.values())
    total_failures = total_actions - total_successes
    print(f"  {_C.BOLD}Game summary:{_C.RESET}")
    print(f"    Total turns: {turn_number}")
    print(f"    Successful actions: {total_successes}")
    print(f"    Failed actions: {total_failures}")
    print(
        f"    Mistakes remaining: "
        f"{game.detonator.remaining_failures}"
    )
    print()


if __name__ == "__main__":
    main()
