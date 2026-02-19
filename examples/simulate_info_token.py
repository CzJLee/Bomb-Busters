"""Simulate the indication phase of a Bomb Busters mission.

Demonstrates using ``rank_indications()`` and ``print_indication_analysis()``
to determine the optimal wire to indicate for each player. Creates a 5-player
game with 2-of-3 yellow wires and 1 known red wire, then simulates the
clockwise indication phase starting with the captain.

Each player's indication analysis is timed to verify performance.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities

_C = bomb_busters._Colors


def main() -> None:
    """Simulate the indication phase for all 5 players.

    Creates a seeded game with colored wires, then for each player
    (starting with the captain, clockwise):
    1. Runs and prints the indication analysis with timing.
    2. Applies the top-ranked indication to the game state.
    3. Advances to the next player.
    """
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Indication Phase Simulation{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print(
        "  Mission: 48 blue wires + 2-of-3 yellow (UNCERTAIN)"
        " + 1 red (KNOWN)"
    )
    print("  Players: Alice (captain), Bob, Charlie, Diana, Eve")
    print()

    # ── Create game ────────────────────────────────────────────
    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        yellow_wires=(2, 3),
        red_wires=1,
        seed=42,
    )

    # Show initial game state (god mode — all wires visible)
    game.active_player_index = None
    print(game)

    # ── Indication phase ───────────────────────────────────────
    total_time = 0.0

    for player_index in range(len(game.players)):
        print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
        print()

        # Time the analysis
        t0 = time.perf_counter()
        choices = compute_probabilities.rank_indications(
            game, player_index,
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Print the analysis
        compute_probabilities.print_indication_analysis(
            game, player_index,
        )

        print(f"  {_C.DIM}Computed in {elapsed:.3f}s{_C.RESET}")
        print()

        # Apply the top-ranked indication
        if choices:
            best = choices[0]
            stand = game.players[player_index].tile_stand
            stand.place_info_token(best.slot_index, best.wire.gameplay_value)
            letter = chr(ord("A") + best.slot_index)
            print(
                f"  {_C.GREEN}{_C.BOLD}→ Indicated"
                f" {best.wire.base_number} at"
                f" [{letter}]{_C.RESET}"
            )
        else:
            print(
                f"  {_C.DIM}No blue wires to indicate.{_C.RESET}"
            )
        print()

    # ── Final state ────────────────────────────────────────────
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Game State After All Indications{_C.RESET}")
    print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
    print()
    print(game)

    # ── Timing summary ─────────────────────────────────────────
    print(
        f"  {_C.DIM}Total indication analysis time: "
        f"{total_time:.3f}s{_C.RESET}"
    )
    print()


if __name__ == "__main__":
    main()
