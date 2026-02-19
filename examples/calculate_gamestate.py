"""Simulation script for Bomb Busters probability analysis.

Demonstrates using ``GameState.from_partial_state()`` to enter a
mid-game scenario and run probability analysis from the active
player's perspective. Includes dual cut, solo cut, Double Detector,
and red wire risk calculations.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities

# Equipment types to include in probability analysis.
INCLUDE_EQUIPMENT = {compute_probabilities.EquipmentType.DOUBLE_DETECTOR}


# ── Main ────────────────────────────────────────────────────

def main() -> None:
    """Mid-game probability analysis from Alice's perspective.

    A 5-player game with all 48 blue wires (values 1-12), 2 yellow
    wires (Y4 and Y7), and 1 red wire (R4). The game is about 45%
    resolved (22 wires cut, 1 info token). Blue-3, blue-9, and
    blue-12 are fully validated.

    Alice (Player 0) is the active player and observer. She has:
    - A yellow wire (Y4) for yellow cut potential
    - Two blue-8s with all other blue-8s cut (solo cut available)
    - Blue-4, blue-6, and blue-7 for dual cut targets
    - No red wire (red wire risk is visible on other players' stands)

    Diana's slot E has an info token showing blue-6 from a prior
    failed dual cut. The detonator has advanced once.

    Wire distribution (51 total = 48 blue + Y4.1 + Y7.1 + R4.5):

        Alice  (11): B1  B2  B4  Y4  B6  B7  B8  B8  B9 B11 B12
        Bob    (10): B1  B3  B4  B5  B7  B8  B9 B10 B11 B12
        Charlie(10): B2  B3  B4  B6  B7  B8  B9 B10 B11 B12
        Diana  (10): B2  B3  R4  B5  B6  B7  Y7  B9 B10 B11
        Eve    (10): B1  B1  B2  B3  B4  B5  B5  B6 B10 B12
    """
    print("=" * 60)
    print("Mid-game probability analysis (Alice's perspective)")
    print("=" * 60)
    print()

    # ── Turn history ──────────────────────────────────────────
    # Diana's slot E has an info token from a prior failed dual cut
    # (entered directly via "i6" in her stand notation below). The
    # detonator advanced once. No history record is needed — the
    # info token and detonator state are set directly in calculator
    # mode.
    history = bomb_busters.TurnHistory()

    # ── Character cards ─────────────────────────────────────
    character_cards: list[bomb_busters.CharacterCard | None] = [
        bomb_busters.create_double_detector() for _ in range(5)
    ]

    # ── Tile stands ─────────────────────────────────────────
    #
    # Alice (P0) — Observer. All wires known to her.
    # Hidden: B4, Y4, B6, B7, B8×2  |  Cut: B1, B2, B9, B11, B12
    # Solo cut available: blue-8 (all other B8s are cut)
    # Dual cut values: 4, YELLOW, 6, 7
    alice_stand = bomb_busters.TileStand.from_string(
        "1 2 ?4 iY4 ?6 ?7 ?8 ?8 9 11 12", num_tiles=11,
    )

    # Bob (P1) — 5 cut, 5 hidden (unknown to Alice)
    bob_stand = bomb_busters.TileStand.from_string(
        "1 3 ? ? ? 8 9 ? ? 12", num_tiles=10,
    )

    # Charlie (P2) — 5 cut, 5 hidden (unknown to Alice)
    charlie_stand = bomb_busters.TileStand.from_string(
        "2 3 ? ? i7 8 9 ? ? 12", num_tiles=10,
    )

    # Diana (P3) — 4 cut, 1 info-revealed, 5 hidden
    # Slot E has an info token showing blue-6 (from Eve's
    # failed dual cut guessing blue-7).
    # The red wire R5.5 lurks in slot D (unknown to Alice).
    diana_stand = bomb_busters.TileStand.from_string(
        "2 3 ? ? i6 ? ? 9 ? 11", num_tiles=10,
    )

    # Eve (P4) — 3 cut, 7 hidden (unknown to Alice)
    eve_stand = bomb_busters.TileStand.from_string(
        "1 ? ? 3 ? ? ? ? ? 12", num_tiles=10,
    )

    # ── Create game state ───────────────────────────────────
    game = bomb_busters.GameState.from_partial_state(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        stands=[
            alice_stand, bob_stand, charlie_stand,
            diana_stand, eve_stand,
        ],
        mistakes_remaining=3,
        character_cards=character_cards,
        history=history,
        active_player_index=0,
        captain=0,
        yellow_wires=[4, 7],
        red_wires=[4],
    )

    # ── Display game state ──────────────────────────────────
    print(game)

    # ── Probability analysis ────────────────────────────────
    compute_probabilities.print_probability_analysis(
        game, active_player_index=0, max_moves=20,
        include_equipment=INCLUDE_EQUIPMENT,
    )


if __name__ == "__main__":
    main()
