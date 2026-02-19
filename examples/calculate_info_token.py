"""Quick indication calculator for a live game session.

Edit the tile stands and wire parameters below, then run:
    python examples/calculate_info_token.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bomb_busters
import compute_probabilities


def main() -> None:
    # ── Wire configuration ────────────────────────────────────
    # Colored wires: uncomment and adjust as needed.
    # Known yellow/red (list form):
    #   yellow_wires = [4, 7]    # Y4, Y7 definitely in play
    #   red_wires = [4]          # R4 definitely in play
    # Uncertain X-of-Y (tuple form):
    #   yellow_wires = ([2, 3, 9], 2)  # drew 3, kept 2
    #   red_wires = ([3, 7], 1)        # drew 2, kept 1

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
        active_player_index=0,
        captain=0,
        # blue_wires=None,       # default: all blue 1-12 (48 wires)
        # yellow_wires=[4, 7],   # uncomment for known yellow wires
        # red_wires=[4],         # uncomment for known red wires
    )

    # ── Display & analysis ─────────────────────────────────────
    print(game)
    compute_probabilities.print_indication_analysis(game, player_index=0)


if __name__ == "__main__":
    main()
