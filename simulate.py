"""Quick simulation script for Bomb Busters."""

from bomb_busters import (
    GameState,
    WireColor,
    WireConfig,
)


def example_full_game() -> None:
    """Example 1: Blue-only game in god mode."""
    print("=" * 60)
    print("EXAMPLE 1: Blue-only game (god mode — all wires visible)")
    print("=" * 60)
    print()

    game = GameState.create_game(
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

    game = GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        wire_configs=[
            WireConfig(color=WireColor.YELLOW, count=2),
            WireConfig(color=WireColor.RED, count=1),
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

    game = GameState.create_game(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        wire_configs=[
            WireConfig(color=WireColor.YELLOW, count=2),
            WireConfig(color=WireColor.RED, count=1),
        ],
        seed=1,
    )
    game.observer_index = 1
    print(game)


if __name__ == "__main__":
    example_full_game()
    print()
    example_colored_god_mode()
    print()
    example_colored_observer()
