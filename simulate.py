"""Quick simulation script for Bomb Busters."""

import bomb_busters
import compute_probabilities


# =============================================================================
# Examples
# =============================================================================


def example_full_game() -> None:
    """Example 1: Blue-only game in god mode."""
    print("=" * 60)
    print("EXAMPLE 1: Blue-only game (god mode — all wires visible)")
    print("=" * 60)
    print()

    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        seed=42,
    )
    print(game)


def example_colored_god_mode() -> None:
    """Example 2: Colored wires game in god mode.

    Includes 2 yellow and 1 red wire so all wire types are visible.
    """
    print("=" * 60)
    print("EXAMPLE 2: Colored wires (god mode — all wires visible)")
    print("=" * 60)
    print()

    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        wire_configs=[
            bomb_busters.WireConfig(color=bomb_busters.WireColor.YELLOW, count=2),
            bomb_busters.WireConfig(color=bomb_busters.WireColor.RED, count=1),
        ],
        seed=1,
    )
    print(game)


def example_colored_observer() -> None:
    """Example 3: Same colored wires game from Bob's perspective.

    Bob (Player 1) has a red wire at B and a yellow wire at J.
    His own wires are visible; everyone else's hidden wires show as '?'.
    """
    print("=" * 60)
    print("EXAMPLE 3: Same game — Bob's perspective (observer_index=1)")
    print("=" * 60)
    print()

    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        wire_configs=[
            bomb_busters.WireConfig(color=bomb_busters.WireColor.YELLOW, count=2),
            bomb_busters.WireConfig(color=bomb_busters.WireColor.RED, count=1),
        ],
        seed=1,
    )
    game.observer_index = 1
    print(game)


def example_mid_game_probabilities() -> None:
    """Example 4: Mid-game probability analysis from Eve's perspective.

    A 5-player game with 2 yellow and 1 red wire, progressed through
    9 turns. ~1/3 of wires have been cut (16 of 51), with 3 validations
    completed (1, 2, 3), one failed dual cut revealing info, and 1
    detonator failure.

    Eve (Player 4) is the active player. The analysis shows her
    guaranteed actions and top moves ranked by probability.
    """
    print("=" * 60)
    print("EXAMPLE 4: Mid-game probability analysis (Eve's perspective)")
    print("=" * 60)
    print()

    game = bomb_busters.GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        wire_configs=[
            bomb_busters.WireConfig(color=bomb_busters.WireColor.YELLOW, count=2),
            bomb_busters.WireConfig(color=bomb_busters.WireColor.RED, count=1),
        ],
        seed=3,
    )

    # --- Replay 9 turns of game history ---

    # Turn 1: Alice (P0) dual cuts blue-1 on Bob (P1) — SUCCESS
    game.current_player_index = 0
    game.execute_dual_cut(
        target_player_index=1,
        target_slot_index=0,
        guessed_value=1,
    )

    # Turn 2: Bob (P1) dual cuts blue-2 on Alice (P0) — SUCCESS
    game.execute_dual_cut(
        target_player_index=0,
        target_slot_index=1,
        guessed_value=2,
    )

    # Turn 3: Charlie (P2) dual cuts blue-1 on Eve (P4) — SUCCESS
    game.execute_dual_cut(
        target_player_index=4,
        target_slot_index=0,
        guessed_value=1,
    )

    # Turn 4: Diana (P3) dual cuts blue-3 on Eve (P4) — SUCCESS
    game.execute_dual_cut(
        target_player_index=4,
        target_slot_index=2,
        guessed_value=3,
    )

    # Turn 5: Eve (P4) dual cuts blue-2 on Charlie (P2) — SUCCESS
    game.execute_dual_cut(
        target_player_index=2,
        target_slot_index=1,
        guessed_value=2,
    )

    # Turn 6: Alice (P0) dual cuts blue-3 on Bob (P1) — SUCCESS
    game.execute_dual_cut(
        target_player_index=1,
        target_slot_index=2,
        guessed_value=3,
    )

    # Turn 7: Bob (P1) guesses blue-5 on Charlie (P2) slot E — FAIL
    #   Actual wire is blue-7; info token placed, detonator advances.
    game.execute_dual_cut(
        target_player_index=2,
        target_slot_index=4,
        guessed_value=5,
    )

    # Turn 8: Charlie (P2) dual cuts blue-6 on Alice (P0) — SUCCESS
    game.execute_dual_cut(
        target_player_index=0,
        target_slot_index=5,
        guessed_value=6,
    )

    # Turn 9: Diana (P3) dual cuts blue-7 on Alice (P0) — SUCCESS
    game.execute_dual_cut(
        target_player_index=0,
        target_slot_index=6,
        guessed_value=7,
    )

    # --- Display from Eve's perspective ---
    observer = game.current_player_index  # Eve (P4)
    game.observer_index = observer
    print(game)

    # --- Probability analysis ---
    compute_probabilities.print_probability_analysis(game, observer)


if __name__ == "__main__":
    example_full_game()
    print()
    example_colored_god_mode()
    print()
    example_colored_observer()
    print()
    example_mid_game_probabilities()
