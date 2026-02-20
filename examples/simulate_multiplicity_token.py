"""Simulate the indication phase using x1/x2/x3 multiplicity tokens.

Demonstrates using ``rank_indications_multiplicity()`` and
``print_indication_analysis_multiplicity()`` to determine the optimal
wire to indicate when the mission uses multiplicity tokens instead of
standard info tokens.

Multiplicity tokens can be used with yellow wires (unlike even/odd
tokens). This simulation uses blue wires only (1-12) for simplicity.

Each player's multiplicity indication analysis is timed and compared
against what the standard info token analysis would recommend.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities

_C = bomb_busters._Colors


def main() -> None:
    """Simulate the multiplicity indication phase for all 5 players.

    Creates a seeded game with blue wires only. For each player
    (starting with the captain, clockwise):
    1. Runs and prints the multiplicity indication analysis with timing.
    2. Compares against standard indication for reference.
    3. Applies the top-ranked multiplicity indication to the game state.
    """
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Multiplicity Token Indication Simulation{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print("  Mission: 48 blue wires (1-12)")
    print("  Indicator type: x1/x2/x3 multiplicity tokens")
    print("  Players: Alice (captain), Bob, Charlie, Diana, Eve")
    print()

    # ── Create game ────────────────────────────────────────────
    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        seed=42,
    )

    # Show initial game state (god mode — all wires visible)
    game.active_player_index = None
    print(game)

    # ── Indication phase ───────────────────────────────────────
    total_mult_time = 0.0
    total_standard_time = 0.0

    for player_index in range(len(game.players)):
        print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
        print()

        # Time the multiplicity analysis
        t0 = time.perf_counter()
        mult_choices = compute_probabilities.rank_indications_multiplicity(
            game, player_index,
        )
        mult_elapsed = time.perf_counter() - t0
        total_mult_time += mult_elapsed

        # Time the standard analysis for comparison
        t0 = time.perf_counter()
        standard_choices = compute_probabilities.rank_indications(
            game, player_index,
        )
        standard_elapsed = time.perf_counter() - t0
        total_standard_time += standard_elapsed

        # Print the multiplicity analysis
        compute_probabilities.print_indication_analysis_multiplicity(
            game, player_index,
        )

        # Show comparison with standard
        if mult_choices and standard_choices:
            mul_best = mult_choices[0]
            std_best = standard_choices[0]
            mul_val = mul_best.wire.gameplay_value
            assert isinstance(mul_val, int)
            total_on_stand = sum(
                1 for s in game.players[player_index].tile_stand.slots
                if s.wire is not None
                and s.wire.gameplay_value == mul_val
            )
            mul_letter = chr(ord("A") + mul_best.slot_index)
            std_letter = chr(ord("A") + std_best.slot_index)

            print(
                f"  {_C.DIM}Comparison with standard info tokens:{_C.RESET}"
            )
            print(
                f"    Multiplicity best: [{mul_letter}] x{total_on_stand}"
                f" ({mul_best.information_gain:.2f} bits,"
                f" {mul_best.uncertainty_resolved:.1%} resolved)"
            )
            print(
                f"    Standard best:     [{std_letter}]"
                f" = {std_best.wire.gameplay_value}"
                f" ({std_best.information_gain:.2f} bits,"
                f" {std_best.uncertainty_resolved:.1%} resolved)"
            )
            ratio = (
                mul_best.information_gain / std_best.information_gain
                if std_best.information_gain > 0 else 0.0
            )
            print(
                f"    Multiplicity/Standard ratio: {ratio:.1%}"
            )

        print(
            f"  {_C.DIM}Computed in {mult_elapsed:.3f}s"
            f" (standard: {standard_elapsed:.3f}s){_C.RESET}"
        )
        print()

        # Apply the top-ranked multiplicity indication.
        # With multiplicity tokens, the slot gets a multiplicity
        # indicator — we simulate this by marking the slot as
        # INFO_REVEALED with the value. (In a real game, only xN is
        # visible; we use the full value here for the simulation to
        # work with the standard indication engine.)
        if mult_choices:
            best = mult_choices[0]
            stand = game.players[player_index].tile_stand
            stand.place_info_token(best.slot_index, best.wire.gameplay_value)
            letter = chr(ord("A") + best.slot_index)
            val = best.wire.gameplay_value
            assert isinstance(val, int)
            total_on_stand = sum(
                1 for s in stand.slots
                if s.wire is not None
                and s.wire.gameplay_value == val
            )
            print(
                f"  {_C.GREEN}{_C.BOLD}→ Indicated"
                f" x{total_on_stand} at"
                f" [{letter}] (actual: {val}){_C.RESET}"
            )
        else:
            print(
                f"  {_C.DIM}No blue wires to indicate.{_C.RESET}"
            )
        print()

    # ── Final state ────────────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(
        f"{_C.BOLD}Game State After All Multiplicity Indications{_C.RESET}"
    )
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print(game)

    # ── Timing summary ─────────────────────────────────────────
    print(
        f"  {_C.DIM}Total multiplicity analysis time: "
        f"{total_mult_time:.3f}s{_C.RESET}"
    )
    print(
        f"  {_C.DIM}Total standard analysis time: "
        f"{total_standard_time:.3f}s (for comparison){_C.RESET}"
    )
    print()


if __name__ == "__main__":
    main()
