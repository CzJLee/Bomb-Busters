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


def example_red_wire_probability() -> None:
    """Example 5: Red wire probability analysis.

    Demonstrates computing the probability that a target slot contains a
    red wire, which would cause an instant mission failure if cut.

    Scenario (4 players, small hands for fast computation):
    - Alice (P0, observer): [blue-5, blue-7, blue-10]
    - Bob (P1): [blue-3, red-3.5, blue-5] — red wire hidden at slot B
    - Charlie (P2): [blue-6, red-6.5, blue-7]  — red wire hidden at slot B
    - Diana (P3): [blue-1, blue-2, blue-10]

    Bob's slot A (blue-3) and Charlie's slot A (blue-6) have already been
    cut. From Alice's perspective, she can see the remaining hidden wires
    and wants to know both the probability of a successful dual cut AND
    the risk of hitting a red wire.
    """
    print("=" * 60)
    print("EXAMPLE 5: Red wire probability analysis")
    print("=" * 60)
    print()

    hands = [
        # Alice (P0, observer)
        [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
         bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
         bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        # Bob (P1) — red wire at slot B (sort_value 3.5)
        [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
         bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
         bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
        # Charlie (P2) — red wire at slot B (sort_value 6.5)
        [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
         bomb_busters.Wire(bomb_busters.WireColor.RED, 6.5),
         bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
        # Diana (P3)
        [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
         bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
         bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
    ]

    players = [
        bomb_busters.Player(
            name=name,
            tile_stand=bomb_busters.TileStand.from_wires(hand),
            character_card=bomb_busters.create_double_detector(),
        )
        for name, hand in zip(
            ["Alice", "Bob", "Charlie", "Diana"], hands
        )
    ]
    all_wires = [w for hand in hands for w in hand]
    game = bomb_busters.GameState(
        players=players,
        detonator=bomb_busters.Detonator(failures=0, max_failures=3),
        info_token_pool=bomb_busters.InfoTokenPool.create_full(),
        validation_tokens=set(),
        markers=[
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 6.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ],
        equipment=[],
        history=bomb_busters.TurnHistory(),
        wires_in_play=all_wires,
    )

    # Simulate: Bob's slot A (blue-3) has been cut (e.g. from a prior turn)
    game.players[1].tile_stand.cut_wire_at(0)
    # Charlie's slot A (blue-6) has been cut
    game.players[2].tile_stand.cut_wire_at(0)

    # Alice is the observer and active player
    observer = 0
    game.observer_index = observer
    game.current_player_index = observer
    print(game)

    # Probability analysis with red wire warnings
    compute_probabilities.print_probability_analysis(game, observer)

    # Detailed red wire analysis
    _C = bomb_busters._Colors
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print(f"{_C.BOLD}Red Wire Risk Analysis{_C.RESET}")
    print(f"{_C.BOLD}{'─' * 60}{_C.RESET}")
    print()

    probs = compute_probabilities.compute_position_probabilities(
        game, observer,
    )

    for p_idx in [1, 2, 3]:
        player = game.players[p_idx]
        for s_idx, slot in enumerate(player.tile_stand.slots):
            if not slot.is_hidden:
                continue
            red_p = compute_probabilities.probability_of_red_wire(
                game, observer, p_idx, s_idx, probs=probs,
            )
            letter = chr(ord("A") + s_idx)
            name = f"{_C.BOLD}{player.name}{_C.RESET}"
            if red_p > 0:
                print(
                    f"  {name} [{_C.BOLD}{letter}{_C.RESET}]:"
                    f" {_C.RED}⚠ {red_p:.1%} chance of RED{_C.RESET}"
                )
            else:
                print(
                    f"  {name} [{_C.BOLD}{letter}{_C.RESET}]:"
                    f" {_C.GREEN}0% red — safe{_C.RESET}"
                )
    print()

    # Highlight a specific risky action
    target_p, target_s = 1, 1  # Bob slot B
    dual_prob = compute_probabilities.probability_of_dual_cut(
        game, observer, target_p, target_s, 5,
    )
    red_prob = compute_probabilities.probability_of_red_wire(
        game, observer, target_p, target_s, probs=probs,
    )
    print(
        f"  Targeting {_C.BOLD}Bob [B]{_C.RESET} with guess "
        f"{_C.BLUE}5{_C.RESET}:"
    )
    print(f"    Success probability: {dual_prob:.1%}")
    print(f"    Red wire risk:       {_C.RED}{red_prob:.1%}{_C.RESET}")
    print(
        f"    Other failure:       "
        f"{1.0 - dual_prob - red_prob:.1%}"
    )
    print()


if __name__ == "__main__":
    example_full_game()
    print()
    example_colored_god_mode()
    print()
    example_colored_observer()
    print()
    example_mid_game_probabilities()
    print()
    example_red_wire_probability()
