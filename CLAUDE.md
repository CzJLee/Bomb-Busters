# Claude instructions for Bomb Busters repo

See @docs/RULES.md for the game rules and gameplay tips. See @docs/DOCUMENTATION.md for code architecture, API reference, and usage examples.

See @docs/Bomb Busters Rulebook.pdf for the official published PDF rulebook of the game. Refer to this for any unclear or ambiguous rules that the docs do not cover. The @docs/Bomb Busters FAQ.pdf file contains some additional clarifications.

## Conventions

- Always assume a 5-player count in examples unless otherwise specified.
- Use Google Python Style Guide imports: `import module` only, never `from module import Class`. This applies to both project modules and standard library (e.g., `import dataclasses` not `from dataclasses import dataclass`).
- **LaTeX in markdown**: When writing or updating markdown docs with LaTeX math (`$...$` or `$$...$$`), never use escaped underscores (`\_`) inside `\text{}` blocks — this causes GitHub's KaTeX renderer to show `'_' allowed only in math mode` errors. Instead, use proper LaTeX subscript syntax (e.g., `k_{\max}` not `\text{max\_k}`) or move variable names with underscores out of math mode into backtick code spans.

## Project Overview

This Bomb Busters project builds a calculator for the game in Python to compute the probability of success of different cut actions. 

### Python game simulator

This repo should create a Python class structure for different game components to allow for easy probability calculations. @bomb_busters.py is the main script. 

### Probability calculations

The probability engine in @compute_probabilities.py computes the following from any player's perspective:

- **Specific cut probability**: "What is the % probability that I successfully dual cut this '2' on this player's tile stand?"
- **Guaranteed actions**: "What solo cuts, dual cuts, or reveal-red actions are 100% guaranteed to succeed?"
- **Best move ranking**: "What is the highest probability cut action I have?" — ranks all possible moves by probability descending.
- **Multi-slot detector probabilities**: Joint probability computations for equipment that targets multiple slots — Double Detector (2 slots), Triple Detector (3 slots), Super Detector (all hidden slots). All use joint enumeration from the constraint solver, not naive independence. X or Y Ray uses marginal probabilities (1 slot, 2 values — mutually exclusive events).
- **Red wire risk**: Per-slot probability that a target contains a red wire (instant game-over if cut). For multi-slot detectors, computes joint probability that all target slots are red. Integrated into `RankedMove` as `red_probability` and displayed as a warning in terminal output.
- **Equipment support**: `rank_all_moves()` and `guaranteed_actions()` accept an `include_equipment: set[EquipmentType]` parameter to enumerate equipment-based moves. Supported equipment: Double Detector, Triple Detector, Super Detector, X or Y Ray, Fast Pass. Constraint-based equipment (Label !=, Label =, General Radar) is handled via `SlotConstraint` subclasses on `GameState.slot_constraints`. Game state modification methods (`place_info_token`, `adjust_detonator`, `reactivate_character_cards`, `set_current_player`, `cut_all_of_value`) support Post-It, Rewinder, Emergency Batteries, Coffee Mug, and Disintegrator.
- **Monte Carlo fallback**: When the exact solver is too slow (>22 hidden positions), `monte_carlo_probabilities()` uses backward-guided composition sampling with importance weighting. For each sample, it processes players sequentially, building a lightweight single-player backward DP table per player and sampling a valid composition from it — guaranteeing valid ascending sequences with no dead ends. Returns the same format as the exact solver for seamless downstream use. Auto-switching is built into `print_probability_analysis()` via `MC_POSITION_THRESHOLD`. Double Detector probabilities are supported in MC mode via `monte_carlo_analysis()`, which returns raw per-sample assignments alongside marginal distributions — enabling joint probability queries through `mc_dd_probability()` and `mc_red_dd_probability()`. See @docs/EXACT_SOLVER.md and @docs/MONTE_CARLO_SOLVER.md for detailed algorithm documentation.
- **Indication quality analysis**: At game start, each player indicates one blue wire. `rank_indications()` computes which wire to indicate for maximum information gain using Shannon entropy reduction. See @docs/INDICATION_QUALITY.md for the information-theoretic foundation.

## Architecture

### Repository layout

- Python source files stay flat at the repo root for trivial imports and Pyodide compatibility. Do not nest them into a package unless the file count clearly warrants it.
- `docs/` — Documentation (PDFs, algorithm docs, roadmap). Reference with `@docs/` prefix. Key docs: @docs/RULES.md (game rules), @docs/DOCUMENTATION.md (code architecture and API), @docs/EXACT_SOLVER.md (exact constraint solver), @docs/MONTE_CARLO_SOLVER.md (MC fallback solver), @docs/INDICATION_QUALITY.md (indication quality metric).
- `tests/` — Unit tests (unittest).
- `web/` — Future browser-based UI (see @docs/WEB_UI_ROADMAP.md).

### Key modules

- @bomb_busters.py — Game model: enums (`WireColor`, `SlotState`, `ActionType`, `ActionResult`, `MarkerState`, `Parity`, `Multiplicity`), dataclasses (`Wire`, `Slot`, `TileStand`, `Player`, `CharacterCard`, `Detonator`, `Marker`, `Equipment`, `UncertainWireGroup`), action records (`DualCutAction`, `SoloCutAction`, `RevealRedAction`, `TurnHistory`), constraint classes (`SlotConstraint` base, `AdjacentNotEqual`, `AdjacentEqual`, `MustHaveValue`, `MustNotHaveValue`, `SlotParity`, `ValueMultiplicity`, `UnsortedSlot`, `SlotExcludedValue`), and `GameState` with two factory methods (`create_game` for simulation, `from_partial_state` for calculator mode) plus equipment-supporting methods (`place_info_token`, `adjust_detonator`, `set_detonator`, `reactivate_character_cards`, `set_current_player`, `cut_all_of_value`, constraint adders). Extended token parser supports indicator token notation: `E`/`O` (parity), `1x`/`2x`/`3x` (multiplicity), `X` (unsorted), `!N` (false info), `b`/`y`/`r` (color prefix). `Slot` dataclass has metadata fields (`parity`, `multiplicity`, `is_unsorted`, `excluded_value`, `required_color`) populated by the parser; `from_partial_state()` auto-generates solver constraints from these fields.
- @missions.py — Mission definitions: enums (`IndicatorTokenType`, `IndicationRule`), `EquipmentCard` frozen dataclass, `EQUIPMENT_CATALOG` (18 entries: standard 1-12, yellow, double-numbered), `Mission` frozen dataclass with validation and `available_equipment()` / `wire_config()` methods, mission registry (`MISSIONS` dict, `get_mission()`, `all_missions()`) with definitions for missions 1-30.
- @compute_probabilities.py — Probability engine: `EquipmentType` enum, `KnownInfo` extraction, unknown pool computation, `PositionConstraint` sort-value bounds, backtracking constraint solver with identical-wire grouping, discard slots for uncertain (X of Y) wires, adjacent constraint and must-not-have enforcement, Monte Carlo fallback (`monte_carlo_probabilities`, `monte_carlo_analysis`, `MCSamples`, `mc_dd_probability`, `mc_red_dd_probability`, `mc_td_probability`, `mc_sd_probability`, `mc_red_multi_probability`, `count_hidden_positions`, `MC_POSITION_THRESHOLD`), indication quality analysis (`rank_indications`, `print_indication_analysis`), and high-level API (`probability_of_dual_cut`, `probability_of_double_detector`, `probability_of_triple_detector`, `probability_of_super_detector`, `probability_of_x_or_y_ray`, `probability_of_red_wire`, `probability_of_red_wire_dd`, `probability_of_red_wire_multi`, `guaranteed_actions`, `rank_all_moves`).

## Environment setup

Use `pyenv virtualenv` to manage the python environment for this repo. The `virtualenv` for this repo is `bomb-busters`. It uses Python 3.14.

## Web UI Constraints

A browser-based UI is planned (see @docs/WEB_UI_ROADMAP.md for full details). The Python engine will run in the browser via Pyodide (CPython compiled to WebAssembly). Keep these constraints in mind during development:

- **Keep stdlib-only.** No dependencies that Pyodide cannot load. Optional imports (like `tqdm`) must use `try/except ImportError`.
- **Structured data over formatted strings.** New analysis features should always provide a data-returning function separate from any print/display function. The web UI consumes `RankedMove`, `Counter`, and dict structures — not terminal output.
- **`from_partial_state()` is the API contract.** The web UI constructs game state exclusively through this factory method.
- **Serialization readiness.** Dataclass fields should use simple types (int, float, str, bool, None, lists, dicts). Avoid fields with functions, generators, or circular references.
- **Separate ANSI formatting from logic.** Keep data extraction (what to show) separate from formatting (how to show it). The `value_label()` and `stand_lines()` patterns are good — maintain this separation.