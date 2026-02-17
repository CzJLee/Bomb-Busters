"""Simulation script for Bomb Busters probability analysis.

Demonstrates using ``GameState.from_partial_state()`` to enter a
mid-game scenario and run probability analysis from the active
player's perspective. Includes dual cut, solo cut, Double Detector,
and red wire risk calculations.
"""

import bomb_busters
import compute_probabilities

# Toggle to include Double Detector moves in probability analysis.
INCLUDE_DOUBLE_DETECTOR = True


# ── Wire helpers ────────────────────────────────────────────

def _blue(n: int) -> bomb_busters.Wire:
    """Create a blue wire with value n."""
    return bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(n))


def _yellow(n: int) -> bomb_busters.Wire:
    """Create a yellow wire with sort value n.1."""
    return bomb_busters.Wire(bomb_busters.WireColor.YELLOW, n + 0.1)


def _red(n: int) -> bomb_busters.Wire:
    """Create a red wire with sort value n.5."""
    return bomb_busters.Wire(bomb_busters.WireColor.RED, n + 0.5)


# ── Slot helpers ────────────────────────────────────────────

def _cut(wire: bomb_busters.Wire) -> bomb_busters.Slot:
    """A cut (face-up) slot with a known wire."""
    return bomb_busters.Slot(
        wire=wire, state=bomb_busters.SlotState.CUT,
    )


def _hidden(wire: bomb_busters.Wire | None = None) -> bomb_busters.Slot:
    """A hidden (face-down) slot. Wire is None for unknown slots."""
    return bomb_busters.Slot(
        wire=wire, state=bomb_busters.SlotState.HIDDEN,
    )


def _info(
    token: int | str, wire: bomb_busters.Wire | None = None,
) -> bomb_busters.Slot:
    """An info-revealed slot from a failed dual cut."""
    return bomb_busters.Slot(
        wire=wire, state=bomb_busters.SlotState.INFO_REVEALED,
        info_token=token,
    )


# ── Main ────────────────────────────────────────────────────

def main() -> None:
    """Mid-game probability analysis from Alice's perspective.

    A 5-player game with all 48 blue wires (values 1-12), 2 yellow
    wires (Y4 and Y7), and 1 red wire (R5). The game is about 45%
    resolved (22 wires cut, 1 info token). Blue-3, blue-9, and
    blue-12 are fully validated.

    Alice (Player 0) is the active player and observer. She has:
    - A yellow wire (Y4) for yellow cut potential
    - Two blue-8s with all other blue-8s cut (solo cut available)
    - Blue-4, blue-6, and blue-7 for dual cut targets
    - No red wire (red wire risk is visible on other players' stands)

    Diana's slot E has an info token showing blue-6 from a prior
    failed dual cut. The detonator has advanced once.

    Wire distribution (51 total = 48 blue + Y4.1 + Y7.1 + R5.5):

        Alice  (11): B1  B2  B4  Y4  B6  B7  B8  B8  B9 B11 B12
        Bob    (10): B1  B3  B4  B5  B7  B8  B9 B10 B11 B12
        Charlie(10): B2  B3  B4  B6  B7  B8  B9 B10 B11 B12
        Diana  (10): B2  B3  B5  R5  B6  B7  Y7  B9 B10 B11
        Eve    (10): B1  B1  B2  B3  B4  B5  B5  B6 B10 B12
    """
    print("=" * 60)
    print("Mid-game probability analysis (Alice's perspective)")
    print("=" * 60)
    print()

    # ── Build wires_in_play ─────────────────────────────────
    wires_in_play = (
        bomb_busters.create_all_blue_wires()
        + [_yellow(4), _yellow(7)]
        + [_red(5)]
    )

    # ── Board markers ───────────────────────────────────────
    markers = [
        bomb_busters.Marker(
            bomb_busters.WireColor.YELLOW, 4.1,
            bomb_busters.MarkerState.KNOWN,
        ),
        bomb_busters.Marker(
            bomb_busters.WireColor.YELLOW, 7.1,
            bomb_busters.MarkerState.KNOWN,
        ),
        bomb_busters.Marker(
            bomb_busters.WireColor.RED, 5.5,
            bomb_busters.MarkerState.KNOWN,
        ),
    ]

    # ── Turn history ──────────────────────────────────────────
    # Diana's slot E has an info token from a prior failed dual cut
    # (entered directly via _info() above). The detonator advanced
    # once. No history record is needed — the info token and
    # detonator state are set directly in calculator mode.
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
    alice_stand = [
        _cut(_blue(1)),        # A
        _cut(_blue(2)),        # B
        _hidden(_blue(4)),     # C
        _hidden(_yellow(4)),   # D — yellow wire
        _hidden(_blue(6)),     # E
        _hidden(_blue(7)),     # F
        _hidden(_blue(8)),     # G — solo cut pair
        _hidden(_blue(8)),     # H — solo cut pair
        _cut(_blue(9)),        # I
        _cut(_blue(11)),       # J
        _cut(_blue(12)),       # K
    ]

    # Bob (P1) — 5 cut, 5 hidden (unknown to Alice)
    bob_stand = [
        _cut(_blue(1)),   # A
        _cut(_blue(3)),   # B
        _hidden(),        # C — B4
        _hidden(),        # D — B5
        _hidden(),        # E — B7
        _cut(_blue(8)),   # F
        _cut(_blue(9)),   # G
        _hidden(),        # H — B10
        _hidden(),        # I — B11
        _cut(_blue(12)),  # J
    ]

    # Charlie (P2) — 5 cut, 5 hidden (unknown to Alice)
    charlie_stand = [
        _cut(_blue(2)),   # A
        _cut(_blue(3)),   # B
        _hidden(),        # C — B4
        _hidden(),        # D — B6
        _hidden(),        # E — B7
        _cut(_blue(8)),   # F
        _cut(_blue(9)),   # G
        _hidden(),        # H — B10
        _hidden(),        # I — B11
        _cut(_blue(12)),  # J
    ]

    # Diana (P3) — 4 cut, 1 info-revealed, 5 hidden
    # Slot E has an info token showing blue-6 (from Eve's
    # failed dual cut guessing blue-7).
    # The red wire R5.5 lurks in slot D (unknown to Alice).
    diana_stand = [
        _cut(_blue(2)),        # A
        _cut(_blue(3)),        # B
        _hidden(),             # C — B5
        _hidden(),             # D — R5.5 (red wire!)
        _info(6, _blue(6)),    # E — info: blue-6
        _hidden(),             # F — B7
        _hidden(),             # G — Y7.1
        _cut(_blue(9)),        # H
        _hidden(),             # I — B10
        _cut(_blue(11)),       # J
    ]

    # Eve (P4) — 3 cut, 7 hidden (unknown to Alice)
    eve_stand = [
        _cut(_blue(1)),   # A
        _hidden(),        # B — B1
        _hidden(),        # C — B2
        _cut(_blue(3)),   # D
        _hidden(),        # E — B4
        _hidden(),        # F — B5
        _hidden(),        # G — B5
        _hidden(),        # H — B6
        _hidden(),        # I — B10
        _cut(_blue(12)),  # J
    ]

    # ── Create game state ───────────────────────────────────
    game = bomb_busters.GameState.from_partial_state(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        stands=[
            alice_stand, bob_stand, charlie_stand,
            diana_stand, eve_stand,
        ],
        detonator_failures=1,
        validation_tokens={3, 9, 12},
        markers=markers,
        wires_in_play=wires_in_play,
        character_cards=character_cards,
        history=history,
    )
    game.current_player_index = 0
    game.observer_index = 0

    # ── Display game state ──────────────────────────────────
    print(game)

    # ── Probability analysis ────────────────────────────────
    compute_probabilities.print_probability_analysis(
        game, observer_index=0, max_moves=10,
        include_dd=INCLUDE_DOUBLE_DETECTOR,
    )


if __name__ == "__main__":
    main()
