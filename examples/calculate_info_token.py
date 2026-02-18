"""Quick indication calculator for a live game session.

Edit the tile stands and wires_in_play below, then run:
    python examples/calculate_info_token.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities


def main() -> None:
    # ── Wires in play ──────────────────────────────────────────
    wires_in_play = bomb_busters.create_all_blue_wires()
    # + [bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 4.1)]
    # + [bomb_busters.Wire(bomb_busters.WireColor.RED, 4.5)]

    # ── Board markers ──────────────────────────────────────────
    markers: list[bomb_busters.Marker] = [
        # bomb_busters.Marker(bomb_busters.WireColor.YELLOW, 4.1, bomb_busters.MarkerState.KNOWN),
    ]

    # ── Uncertain wire groups (X of Y) ─────────────────────────
    uncertain_wire_groups: list[bomb_busters.UncertainWireGroup] = [
        # bomb_busters.UncertainWireGroup.yellow([2, 3, 9], count=2),
    ]

    # ── Tile stands ────────────────────────────────────────────
    # Captain (P0) — your stand, all wires known. Use ?N for hidden.
    captain = bomb_busters.TileStand.from_string(
        "?1 ?2 ?4 ?5 ?6 ?7 ?8 ?9 ?10 ?12",
    )

    # Other players — all hidden at game start
    p1 = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ? ?")
    p2 = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ? ?")
    p3 = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")
    p4 = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")

    # ── Create game state ──────────────────────────────────────
    game = bomb_busters.GameState.from_partial_state(
        player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
        stands=[captain, p1, p2, p3, p4],
        wires_in_play=wires_in_play,
        markers=markers,
        uncertain_wire_groups=uncertain_wire_groups or None,
        active_player_index=0,
        captain=0,
    )

    # ── Display & analysis ─────────────────────────────────────
    print(game)
    compute_probabilities.print_indication_analysis(game, player_index=0)


if __name__ == "__main__":
    main()
