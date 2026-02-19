# Bomb Busters

A calculator for the board game [Bomb Busters](https://boardgamegeek.com/boardgame/413246/bomb-busters) to compute the best moves, or chance of success, in different scenarios.

## Overview

Bomb Busters is a co-op game for 4-5 players where each player has a tile stand of sorted wire tiles visible only to them. The goal is to cut all the wires to defuse the bomb. This project provides a Python probability engine that computes optimal cut actions from any player's perspective.

**What the calculator computes:**

- **Dual cut probability** — "What is the % chance this wire is a 5?"
- **Guaranteed actions** — solo cuts, dual cuts, and reveals that are 100% certain to succeed.
- **Best move ranking** — all possible moves ranked by probability of success.
- **Double Detector probability** — joint probability for the two-slot character ability.
- **Red wire risk** — probability a target slot contains a red wire (instant game-over).
- **Indication quality** — which wire to indicate at game start for maximum information gain.

## Quick Start

```python
import bomb_busters
import compute_probabilities

# Enter a mid-game state from your perspective (player 0)
alice   = bomb_busters.TileStand.from_string("?2 3 ?5 ?7 ?9")       # your hand
bob     = bomb_busters.TileStand.from_string("? 4 ? i8 ?")          # partial knowledge
charlie = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")   # all unknown
diana   = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")
eve     = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")

game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    mistakes_remaining=3,
)

# Print ranked moves with probabilities
compute_probabilities.print_probability_analysis(game, active_player_index=0)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Rules](docs/RULES.md) | Game rules, setup, actions, and gameplay tips |
| [Documentation](docs/DOCUMENTATION.md) | Code architecture, API reference, and usage examples |
| [Exact Solver](docs/EXACT_SOLVER.md) | Algorithm details for the exact constraint solver |
| [Monte Carlo Solver](docs/MONTE_CARLO_SOLVER.md) | Algorithm details for the MC fallback solver |
| [Indication Quality](docs/INDICATION_QUALITY.md) | Information-theoretic foundation for indication analysis |
| [Web UI Roadmap](docs/WEB_UI_ROADMAP.md) | Planned browser-based UI via Pyodide |
| [Official Rulebook](docs/Bomb%20Busters%20Rulebook.pdf) | Published PDF rulebook |
| [Official FAQ](docs/Bomb%20Busters%20FAQ.pdf) | Published PDF FAQ |

## Repository Layout

```
bomb_busters.py              # Game model: enums, dataclasses, game state, actions
compute_probabilities.py     # Probability engine: constraint solver and API
simulate.py                  # Example mid-game probability analysis
simulate_info_token.py       # Example indication phase simulation
simulate_game.py             # Full mission simulation (indication + cutting phases)
docs/                        # Documentation (see table above)
tests/                       # Unit tests (unittest)
web/                         # Future: browser-based UI
```

## Environment Setup

Requires Python 3.14. No external dependencies (stdlib only).

```bash
# Using pyenv
pyenv virtualenv 3.14.0a3 bomb-busters
pyenv activate bomb-busters

# Run tests
python -m unittest discover tests/
```
