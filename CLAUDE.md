# Claude instructions for Bomb Busters repo

See @README.md for information about Bomb Busters and the rules of the game.

See @Bomb Busters Rulebook.pdf for the official published PDF rulebook of the game. Refer to this for any unclear or ambiguous rules that the README does not cover. The @Bomb Busters FAQ.pdf file contains some additional clarifications. 

## Conventions

- Always assume a 5-player count in examples unless otherwise specified.
- Use Google Python Style Guide imports: `import module` only, never `from module import Class`. This applies to both project modules and standard library (e.g., `import dataclasses` not `from dataclasses import dataclass`).

## Project Overview

This Bomb Busters project builds a calculator for the game in Python to compute the probability of success of different cut actions. 

### Python game simulator

This repo should create a Python class structure for different game components to allow for easy probability calculations. @bomb_busters.py is the main script. 

### Probability calculations

The probability engine in @compute_probabilities.py computes the following from any player's perspective:

- **Specific cut probability**: "What is the % probability that I successfully dual cut this '2' on this player's tile stand?"
- **Guaranteed actions**: "What solo cuts, dual cuts, or reveal-red actions are 100% guaranteed to succeed?"
- **Best move ranking**: "What is the highest probability cut action I have?" — ranks all possible moves by probability descending.
- **Double Detector probability**: Joint probability that at least one of two target slots matches the guessed value (computed from enumerated distributions, not naive independence).
- **Red wire risk**: Per-slot probability that a target contains a red wire (instant game-over if cut). For Double Detector, computes joint probability that both target slots are red. Integrated into `RankedMove` as `red_probability` and displayed as a warning in terminal output.
- **Equipment modifications**: The engine accepts the full game state including equipment, so future equipment that modifies cut rules can be integrated.

## Architecture

- @bomb_busters.py — Game model: enums (`WireColor`, `SlotState`, `ActionType`, `ActionResult`, `MarkerState`), dataclasses (`Wire`, `Slot`, `TileStand`, `Player`, `CharacterCard`, `Detonator`, `Marker`, `Equipment`, `WireConfig`), action records (`DualCutAction`, `SoloCutAction`, `RevealRedAction`, `TurnHistory`), and `GameState` with two factory methods (`create_game` for simulation, `from_partial_state` for calculator mode).
- @compute_probabilities.py — Probability engine: `KnownInfo` extraction, unknown pool computation, `PositionConstraint` sort-value bounds, backtracking constraint solver with identical-wire grouping, and high-level API (`probability_of_dual_cut`, `probability_of_double_detector`, `probability_of_red_wire`, `probability_of_red_wire_dd`, `guaranteed_actions`, `rank_all_moves`).

## Environment setup

Use `pyenv virtualenv` to manage the python environment for this repo. The `virtualenv` for this repo is `bomb-busters`. It uses Python 3.14. 