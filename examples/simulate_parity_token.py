"""Simulate the indication phase using even/odd parity tokens.

Demonstrates using ``rank_indications_parity()`` and
``print_indication_analysis_parity()`` to determine the optimal wire
to indicate when the mission uses even/odd tokens instead of standard
info tokens.

Even/odd tokens are never used in missions with yellow wires. This
simulation uses blue wires only (1-12) with 1-2 red wires.

Each player's parity indication analysis is timed and compared against
what the standard info token analysis would recommend.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities

_C = bomb_busters._Colors


def main() -> None:
    """Simulate the parity indication phase for all 5 players.

    Creates a seeded game with blue wires + red wires (no yellow —
    even/odd tokens are never used with yellow wires). For each player
    (starting with the captain, clockwise):
    1. Runs and prints the parity indication analysis with timing.
    2. Compares against standard indication for reference.
    3. Applies the top-ranked parity indication to the game state.
    """
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Parity Token Indication Simulation{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print(
        "  Mission: 48 blue wires + 2 red (KNOWN)"
    )
    print("  Indicator type: Even/Odd parity tokens")
    print("  Players: Alice (captain), Bob, Charlie, Diana, Eve")
    print()

    # ── Create game ────────────────────────────────────────────
    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        red_wires=2,
        seed=42,
    )

    # Show initial game state (god mode — all wires visible)
    game.active_player_index = None
    print(game)

    # ── Indication phase ───────────────────────────────────────
    total_parity_time = 0.0
    total_standard_time = 0.0

    for player_index in range(len(game.players)):
        print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
        print()

        # Time the parity analysis
        t0 = time.perf_counter()
        parity_choices = compute_probabilities.rank_indications_parity(
            game, player_index,
        )
        parity_elapsed = time.perf_counter() - t0
        total_parity_time += parity_elapsed

        # Time the standard analysis for comparison
        t0 = time.perf_counter()
        standard_choices = compute_probabilities.rank_indications(
            game, player_index,
        )
        standard_elapsed = time.perf_counter() - t0
        total_standard_time += standard_elapsed

        # Print the parity analysis
        compute_probabilities.print_indication_analysis_parity(
            game, player_index,
        )

        # Show comparison with standard
        if parity_choices and standard_choices:
            par_best = parity_choices[0]
            std_best = standard_choices[0]
            par_val = par_best.wire.gameplay_value
            assert isinstance(par_val, int)
            par_parity = "E" if par_val % 2 == 0 else "O"
            par_letter = chr(ord("A") + par_best.slot_index)
            std_letter = chr(ord("A") + std_best.slot_index)

            print(
                f"  {_C.DIM}Comparison with standard info tokens:{_C.RESET}"
            )
            print(
                f"    Parity best:   [{par_letter}] {par_parity}"
                f" ({par_best.information_gain:.2f} bits,"
                f" {par_best.uncertainty_resolved:.1%} resolved)"
            )
            print(
                f"    Standard best: [{std_letter}]"
                f" = {std_best.wire.gameplay_value}"
                f" ({std_best.information_gain:.2f} bits,"
                f" {std_best.uncertainty_resolved:.1%} resolved)"
            )
            ratio = (
                par_best.information_gain / std_best.information_gain
                if std_best.information_gain > 0 else 0.0
            )
            print(
                f"    Parity/Standard ratio: {ratio:.1%}"
            )

        print(
            f"  {_C.DIM}Computed in {parity_elapsed:.3f}s"
            f" (standard: {standard_elapsed:.3f}s){_C.RESET}"
        )
        print()

        # Apply the top-ranked parity indication.
        # With parity tokens, the slot gets a parity indicator — we
        # simulate this by marking the slot as INFO_REVEALED with the
        # value. (In a real game, only E/O is visible; we use the full
        # value here for the simulation to work with the standard
        # indication engine.)
        if parity_choices:
            best = parity_choices[0]
            stand = game.players[player_index].tile_stand
            stand.place_info_token(best.slot_index, best.wire.gameplay_value)
            letter = chr(ord("A") + best.slot_index)
            val = best.wire.gameplay_value
            assert isinstance(val, int)
            parity_label = "EVEN" if val % 2 == 0 else "ODD"
            print(
                f"  {_C.GREEN}{_C.BOLD}→ Indicated"
                f" {parity_label} at"
                f" [{letter}] (actual: {val}){_C.RESET}"
            )
        else:
            print(
                f"  {_C.DIM}No blue wires to indicate.{_C.RESET}"
            )
        print()

    # ── Final state ────────────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Game State After All Parity Indications{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print(game)

    # ── Timing summary ─────────────────────────────────────────
    print(
        f"  {_C.DIM}Total parity analysis time: "
        f"{total_parity_time:.3f}s{_C.RESET}"
    )
    print(
        f"  {_C.DIM}Total standard analysis time: "
        f"{total_standard_time:.3f}s (for comparison){_C.RESET}"
    )
    print()


if __name__ == "__main__":
    main()
