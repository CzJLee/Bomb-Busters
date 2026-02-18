# Claude instructions for Bomb Busters repo

See @README.md for information about Bomb Busters and the rules of the game.

See @docs/Bomb Busters Rulebook.pdf for the official published PDF rulebook of the game. Refer to this for any unclear or ambiguous rules that the README does not cover. The @docs/Bomb Busters FAQ.pdf file contains some additional clarifications.

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
- **Monte Carlo fallback**: When the exact solver is too slow (>22 hidden positions), `monte_carlo_probabilities()` uses backward-guided composition sampling with importance weighting. For each sample, it processes players sequentially, building a lightweight single-player backward DP table per player and sampling a valid composition from it — guaranteeing valid ascending sequences with no dead ends. Returns the same format as the exact solver for seamless downstream use. Auto-switching is built into `print_probability_analysis()` via `MC_POSITION_THRESHOLD`. Double Detector probabilities are supported in MC mode via `monte_carlo_analysis()`, which returns raw per-sample assignments alongside marginal distributions — enabling joint probability queries through `mc_dd_probability()` and `mc_red_dd_probability()`.
- **Indication quality analysis**: At game start, each player indicates one blue wire. `rank_indications()` computes which wire to indicate for maximum information gain using Shannon entropy reduction. See @docs/INDICATION_QUALITY.md for the information-theoretic foundation.

## Architecture

### Repository layout

- Python source files stay flat at the repo root for trivial imports and Pyodide compatibility. Do not nest them into a package unless the file count clearly warrants it.
- `docs/` — Documentation (PDFs, roadmap). Reference with `@docs/` prefix.
- `tests/` — Unit tests (unittest).
- `web/` — Future browser-based UI (see @docs/WEB_UI_ROADMAP.md).

### Key modules

- @bomb_busters.py — Game model: enums (`WireColor`, `SlotState`, `ActionType`, `ActionResult`, `MarkerState`), dataclasses (`Wire`, `Slot`, `TileStand`, `Player`, `CharacterCard`, `Detonator`, `Marker`, `Equipment`, `WireConfig`, `UncertainWireGroup`), action records (`DualCutAction`, `SoloCutAction`, `RevealRedAction`, `TurnHistory`), and `GameState` with two factory methods (`create_game` for simulation, `from_partial_state` for calculator mode).
- @compute_probabilities.py — Probability engine: `KnownInfo` extraction, unknown pool computation, `PositionConstraint` sort-value bounds, backtracking constraint solver with identical-wire grouping and discard slots for uncertain (X of Y) wires, Monte Carlo fallback (`monte_carlo_probabilities`, `monte_carlo_analysis`, `MCSamples`, `mc_dd_probability`, `mc_red_dd_probability`, `count_hidden_positions`, `MC_POSITION_THRESHOLD`), indication quality analysis (`rank_indications`, `print_indication_analysis`), and high-level API (`probability_of_dual_cut`, `probability_of_double_detector`, `probability_of_red_wire`, `probability_of_red_wire_dd`, `guaranteed_actions`, `rank_all_moves`).

## Environment setup

Use `pyenv virtualenv` to manage the python environment for this repo. The `virtualenv` for this repo is `bomb-busters`. It uses Python 3.14.

## Web UI Constraints

A browser-based UI is planned (see @docs/WEB_UI_ROADMAP.md for full details). The Python engine will run in the browser via Pyodide (CPython compiled to WebAssembly). Keep these constraints in mind during development:

- **Keep stdlib-only.** No dependencies that Pyodide cannot load. Optional imports (like `tqdm`) must use `try/except ImportError`.
- **Structured data over formatted strings.** New analysis features should always provide a data-returning function separate from any print/display function. The web UI consumes `RankedMove`, `Counter`, and dict structures — not terminal output.
- **`from_partial_state()` is the API contract.** The web UI constructs game state exclusively through this factory method. Changes to its signature must be backward-compatible (new optional parameters only).
- **Serialization readiness.** Dataclass fields should use simple types (int, float, str, bool, None, lists, dicts). Avoid fields with functions, generators, or circular references.
- **Separate ANSI formatting from logic.** Keep data extraction (what to show) separate from formatting (how to show it). The `value_label()` and `stand_lines()` patterns are good — maintain this separation.