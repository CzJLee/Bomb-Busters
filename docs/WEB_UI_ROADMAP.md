# Web UI Roadmap

The long-term goal is to build a static web application that provides a graphical interface for the Bomb Busters probability calculator. The app must run entirely in the browser with no compute backend — suitable for hosting on GitHub Pages or any static file server.

## Technical Approach: Pyodide + Web Worker

The Python engine runs in the browser via [Pyodide](https://pyodide.org/), which compiles CPython to WebAssembly. This is viable because:

- The codebase is **stdlib-only** (no C extensions, no numpy) — the #1 requirement for Pyodide compatibility.
- The probability engine is algorithmically dense (backtracking solver with memoization, composition-based enumeration, forward-pass DP). Rewriting in JavaScript would risk subtle correctness bugs in the constraint solver, sort-order pruning, and must-have deductions. Keeping Python as the single source of truth avoids maintaining two divergent implementations.
- The ~10MB Pyodide runtime is a one-time download (cached by the browser). For a calculator tool used during a board game session, a brief initial load is acceptable.

The solver **must** run in a [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Worker_API) so that the constraint solver's backtracking does not block the UI thread. The worker loads Pyodide, receives game state from the main thread as JSON, constructs a `GameState` via `from_partial_state()`, runs the probability engine, and posts structured results back.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  Browser Main Thread                            │
│  ┌───────────────┐    ┌──────────────────────┐  │
│  │  HTML/CSS/JS  │◄──►│  State Manager (JS)  │  │
│  │  Game State   │    │  - Serialize to JSON  │  │
│  │  Entry Form   │    │  - Post to Worker     │  │
│  └───────────────┘    │  - Receive results    │  │
│                       └──────────┬───────────┘  │
│  ┌───────────────┐               │              │
│  │  Results       │◄─────────────┘              │
│  │  Display (HTML)│                             │
│  └───────────────┘                              │
├─────────────────────────────────────────────────┤
│  Web Worker Thread                              │
│  ┌─────────────┐    ┌────────────────────────┐  │
│  │   Pyodide    │───►│  bomb_busters.py       │  │
│  │   (WASM)     │    │  compute_probabilities │  │
│  │              │    │  missions.py            │  │
│  │              │    │  bridge.py (new)        │  │
│  └─────────────┘    └────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Key components:**

- **Game State Entry Form (JS/HTML)**: Visual representation of tile stands where users click to set slot states, enter wire values, configure markers, etc. This replaces the `TileStand.from_string()` shorthand with a GUI. The shorthand notation should remain available as a power-user input mode.
- **State Manager (JS)**: Serializes the UI state to a JSON structure, posts it to the Web Worker, and processes the structured result JSON. Also handles URL-based state sharing (encode game state in URL parameters so players can share a link).
- **Bridge module (`bridge.py`, new)**: A thin Python module loaded in the Web Worker alongside the existing modules. Accepts a JSON dict, constructs a `GameState` via `from_partial_state()`, runs the probability engine, and returns results as a JSON-serializable dict. This keeps the boundary between JS and Python clean — JS never manipulates Python objects directly. See [Bridge Module Specification](#bridge-module-specification) for the full API.
- **Results Display (HTML/CSS)**: Renders the ranked moves, guaranteed actions, red wire warnings, per-position probability distributions, and indication analysis. Replaces the ANSI terminal output with styled HTML.

## Bridge Module Specification

The bridge module (`bridge.py`) is the only Python interface the Web Worker calls. It accepts plain JSON dicts and returns plain JSON dicts — no Python objects cross the boundary. The module exposes two entry points:

### `analyze(config: dict) -> dict`

The primary analysis function. Accepts a game state configuration and returns probability analysis results.

**Input schema** (maps directly to `from_partial_state()` parameters):

```python
{
    # Required
    "player_names": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "stands": ["?2 3 ?5 ?7 ?9", "? 4 ? i8 ?", ...],  # TileStand.from_string() notation
    "active_player_index": 0,

    # Optional — wire configuration
    "mission": 8,                          # int mission number (inherits blue_wire_range)
    "blue_wires": [1, 8],                  # [low, high] tuple, or null for default 1-12
    "yellow_wires": [4, 7],                # list of known, or {"candidates": [2,3,9], "count": 2}
    "red_wires": [3],                      # list of known, or {"candidates": [3,7], "count": 1}

    # Optional — game state
    "mistakes_remaining": 3,
    "captain": 0,

    # Optional — constraints
    "constraints": [
        {"type": "must_have", "player": 1, "value": 5, "source": "failed dual cut"},
        {"type": "must_not_have", "player": 2, "value": 3, "source": "General Radar"},
        {"type": "adjacent_not_equal", "player": 1, "slot_left": 2, "slot_right": 3},
        {"type": "adjacent_equal", "player": 1, "slot_left": 4, "slot_right": 5},
    ],

    # Optional — character cards
    "character_cards": [
        {"name": "Double Detector", "used": false},  # per player, or null
        null,
        ...
    ],

    # Optional — turn history (for must-have deductions)
    "failed_dual_cuts": [
        {"actor": 1, "value": 5},   # player 1 failed guessing 5
    ],

    # Optional — analysis options
    "include_equipment": ["double_detector", "triple_detector"],  # EquipmentType values
    "max_moves": 10,
    "mc_threshold": 22,       # override MC_POSITION_THRESHOLD
    "mc_num_samples": 10000,  # override sample count
}
```

**Output schema:**

```python
{
    "guaranteed_actions": {
        "solo_cuts": [{"value": 5, "slots": [2, 7]}],
        "fast_pass_solo_cuts": [],
        "dual_cuts": [{"target_player": 1, "target_slot": 3, "value": 8}],
        "reveal_red": false,
    },
    "ranked_moves": [
        {
            "action_type": "dual_cut",        # or "solo_cut", "reveal_red",
                                               # "double_detector", "triple_detector",
                                               # "super_detector", "x_or_y_ray",
                                               # "fast_pass_solo"
            "target_player": 1,
            "target_slot": 3,                  # slot index (0-based)
            "target_slot_letter": "D",         # letter label
            "second_slot": null,               # DD/TD second slot
            "third_slot": null,                # TD third slot
            "guessed_value": 8,
            "second_value": null,              # X or Y Ray
            "probability": 0.75,
            "red_probability": 0.05,
        },
        ...
    ],
    "position_probabilities": {
        "1:3": {                               # "player_index:slot_index"
            "total": 1000,
            "wires": [
                {"value": 7, "color": "blue", "count": 400},
                {"value": 8, "color": "blue", "count": 350},
                {"value": "YELLOW", "color": "yellow", "count": 250},
            ]
        },
        ...
    },
    "solver_mode": "exact",                    # or "monte_carlo"
    "hidden_positions": 18,                    # count_hidden_positions() result
    "mc_samples": 10000,                       # null if exact
    "error": null,                             # error message string, or null
}
```

### `analyze_indication(config: dict) -> dict`

Indication quality analysis for the setup phase. Called separately from the main analysis since it applies to a different game phase (pre-game indication vs. mid-game probability).

**Input schema:**

```python
{
    "player_names": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "stands": ["?1 ?1 ?2 ?2 ?3 ?4 ?8 ?10 ?10 ?12", ...],  # all wires known
    "player_index": 0,           # which player is choosing an indication
    "indicator_type": "info",    # "info", "even_odd", or "multiplicity"

    # Optional — wire configuration (same as analyze())
    "mission": null,
    "blue_wires": null,
    "yellow_wires": null,
    "red_wires": null,
    "captain": 0,
}
```

**Output schema:**

```python
{
    "baseline_entropy": 25.3,
    "choices": [
        {
            "slot_index": 6,
            "slot_letter": "G",
            "wire_value": 8,
            "wire_color": "blue",
            "information_gain": 5.2,
            "uncertainty_resolved": 0.205,
            "remaining_entropy": 20.1,
            # Parity-specific fields (only for even_odd):
            "parity": "even",           # or "odd", null for other types
            # Multiplicity-specific fields (only for multiplicity):
            "multiplicity": null,       # e.g., 2 for "x2", null for other types
        },
        ...
    ],
    "error": null,
}
```

### `get_missions() -> dict`

Returns available mission definitions for the mission selector UI.

```python
{
    "missions": [
        {
            "number": 1,
            "name": "TRAINING, Day 1",
            "blue_wire_range": [1, 6],
            "yellow_wires": null,
            "red_wires": null,
            "indicator_type": "info",
            "x_tokens": false,
            "double_detector_disabled": false,
            "equipment_ids": [],
            "notes": "",
        },
        ...
    ]
}
```

### Wire Serialization

Wire objects appear as Counter keys in position probability results. The bridge serializes them as:

```python
{"value": 8, "color": "blue", "sort_value": 8.0}     # Blue wire
{"value": "YELLOW", "color": "yellow", "sort_value": 4.1}  # Yellow wire
{"value": "RED", "color": "red", "sort_value": 7.5}   # Red wire
```

The `value` field uses the gameplay value (int 1-12, `"YELLOW"`, or `"RED"`). The `sort_value` field is included for UI display of colored wires (e.g., showing "Y4" for yellow at sort position 4.1).

### Solver Mode Auto-Selection

The bridge automatically selects exact vs. Monte Carlo solving based on `count_hidden_positions()`:

- If hidden positions <= `mc_threshold` (default 22): exact solver via `build_solver()` + `_forward_pass_positions()`.
- If hidden positions > `mc_threshold`: Monte Carlo via `monte_carlo_analysis()` with `mc_num_samples` samples.

The response includes `solver_mode`, `hidden_positions`, and `mc_samples` so the UI can display which solver was used.

## Phased Work Plan

### Phase 0: Python API Hardening (Pre-Web)

Prepare the Python codebase for web consumption without changing any architecture. These changes benefit the CLI tool as well.

- [ ] **Add a `bridge.py` module** implementing the [Bridge Module Specification](#bridge-module-specification). This is the only module the Web Worker calls. It translates JSON dicts to `from_partial_state()` calls, runs the probability engine, and serializes results back to JSON. Handles: mission lookup, constraint construction, character card setup, equipment type mapping, indication analysis dispatch, and error wrapping.
- [ ] **Add result serialization helpers** to convert probability engine outputs (`RankedMove`, `Counter[Wire]`, `IndicationChoice`) into JSON-serializable dicts. These are internal to `bridge.py` — no changes needed to the core dataclasses since input goes through `from_partial_state()` parameters (already simple types) and output is serialized by the bridge.
- [ ] **Add a callback-based progress API** to `build_solver()` and `_guided_mc_sample()` as an alternative to `tqdm`. Accept an optional `on_progress(current, total)` callable. The Web Worker will use this to post progress messages to the main thread.
- [ ] **Unit tests** for the bridge module: JSON input -> analysis results round-trips, error handling, mission lookup, constraint parsing, indication analysis for all three token types.

### Phase 1: Web Scaffold

Set up the static web project structure alongside the existing Python code.

- [ ] **Create `web/` directory** with `index.html`, `style.css`, `app.js`, `worker.js`.
- [ ] **Pyodide loader**: `worker.js` loads Pyodide, fetches `bomb_busters.py`, `compute_probabilities.py`, `missions.py`, and `bridge.py` as Python files, and exposes a `postMessage` API.
- [ ] **Basic game config form**: Player count (4-5), player names, mission selector dropdown (populated from `get_missions()`), wire setup (blue-only, or add red/yellow with count and optional X-of-Y pool size).
- [ ] **Minimal end-to-end test**: Hard-coded game state -> Worker -> Python -> results displayed as raw JSON in the page.

### Phase 2: Game State Entry UI

The core UX challenge — building an intuitive interface for entering a mid-game state.

- [ ] **Tile stand editor**: For each player, render a row of slots. Each slot is clickable and cycles through states (hidden/cut/info-revealed). For the active player, all hidden wires require a value (they know their own hand). For other players, hidden slots are unknown by default but can optionally have a known value (observer's private knowledge).
- [ ] **Wire value entry**: Dropdowns or number inputs for wire values, filtered by what's still possible (exclude fully validated values, respect sort ordering). Color-coded: blue values as numbers, yellow as "Y", red as "R".
- [ ] **Indicator token entry**: When a slot has an indicator token, allow selecting the token type based on the mission's `indicator_type`:
  - **Standard info tokens**: Numeric value (1-12) or yellow token — auto-set when a slot is marked as INFO_REVEALED.
  - **Even/odd parity tokens**: Toggle even (E) or odd (O) on a hidden slot.
  - **x1/x2/x3 multiplicity tokens**: Select multiplicity (1, 2, or 3) on a hidden slot.
  - **X (unsorted) tokens**: Mark a slot as unsorted (not in ascending order with rest of stand).
  - **False info tokens**: Mark "not value N" on a hidden slot.
- [ ] **Slot constraints panel**: UI for entering equipment-derived constraints:
  - **Label != (#1)**: Select two adjacent slots on a player's stand as "different values".
  - **Label = (#12)**: Select two adjacent slots as "same value".
  - **Must-have / Must-not-have**: From General Radar (#8) — per-player value assertions.
- [ ] **Markers panel**: Auto-populated from wire config, with KNOWN/UNCERTAIN state. Read-only when derived from wire config input.
- [ ] **Character card toggles**: Per-player Double Detector available/used status.
- [ ] **Detonator display**: Visual mistakes-remaining counter with editable input.
- [ ] **Equipment type checkboxes**: Select which equipment-based moves to include in analysis: Double Detector, Triple Detector, Super Detector, X or Y Ray, Fast Pass.
- [ ] **`TileStand.from_string()` shorthand input**: A text input per player that accepts the existing notation (e.g., `"1 3 ? ? i6 8 9 ? ? 12"`) as a power-user alternative to clicking through the GUI. Supports the full extended token syntax including indicator tokens (`E`, `O`, `1x`, `2x`, `3x`, `X`, `!N`, `b?`, `y?`, `r?`). Parses on blur/enter and populates the visual editor.
- [ ] **Turn history entry**: Record past failed dual cuts (actor player index, guessed value) to enable must-have deductions in the solver.

### Phase 3: Results Display

Replace ANSI terminal output with styled HTML.

- [ ] **Solver mode indicator**: Display whether exact or Monte Carlo solver was used, hidden position count, and sample count (for MC). Mirrors the `(Monte Carlo: N unknown wires, M samples)` note in terminal output.
- [ ] **Guaranteed actions section**: Styled cards/badges for solo cuts (including fast pass solo), guaranteed dual cuts, reveal red.
- [ ] **Ranked moves table**: Sortable table with probability bars, color-coded by confidence level (green >= 75%, blue >= 50%, yellow >= 25%, red < 25% — matching the terminal thresholds). Red wire risk shown as a warning badge. Supports all action types: dual cut, solo cut, reveal red, double detector, triple detector, super detector, x or y ray, fast pass solo.
- [ ] **Per-position probability breakdown**: Expandable detail showing the full wire distribution for each hidden slot (which wire values are possible and with what probability). Color-coded by wire type (blue/yellow/red).
- [ ] **Equipment move results**: Deduplicated display for multi-slot detectors (best slot combination per player-value pair), matching the `_DEDUP_TYPES` logic in `print_probability_analysis()`. Includes:
  - **Double Detector**: Best pair per (target player, value).
  - **Triple Detector**: Best triplet per (target player, value).
  - **Super Detector**: Per (target player, value).
  - **X or Y Ray**: Per (target player, slot, value pair).

### Phase 4: Indication Analysis UI

The indication phase occurs at game start — each player chooses one wire to indicate. This is a separate analysis mode from mid-game probability calculations.

- [ ] **Indication mode toggle**: Switch between "mid-game analysis" and "indication analysis" modes in the UI.
- [ ] **Indication stand entry**: All wires on the indicating player's stand must be known (full hand entry). Other players' cut/info-revealed wires are entered for pool computation.
- [ ] **Indication type selector**: Based on the mission's `indicator_type`, select which ranking function to use: `rank_indications()` (standard), `rank_indications_parity()` (even/odd), or `rank_indications_multiplicity()` (x1/x2/x3).
- [ ] **Indication results display**: Table of ranked indication choices showing:
  - Slot letter and wire value (or parity label / multiplicity label).
  - Information gain in bits, color-coded by quality thresholds (green/blue/yellow/red matching the terminal thresholds — different per token type).
  - Uncertainty resolved percentage.
  - Baseline entropy context.

### Phase 5: Polish and Features

- [ ] **Progress bar**: Worker posts progress messages during `build_solver()` and `_guided_mc_sample()` -> main thread renders a progress indicator. Use the `_count_root_compositions()` estimate for total work (exact solver) or sample count (MC solver).
- [ ] **URL state sharing**: Encode the full game state in URL query parameters (compressed/base64). Users can share a link that reconstructs the exact game state.
- [ ] **Error handling**: Validation errors from Python (e.g., pool/position mismatch, incomplete active player stand, mission wire count mismatch) displayed as user-friendly messages in the UI. The bridge module wraps all errors in the `error` field of the response.
- [ ] **Mobile-friendly layout**: Tile stands are compact — a phone-friendly layout with horizontal scrolling per stand. This is a real use case (checking probabilities at the game table on a phone).
- [ ] **Mission auto-configuration**: When a mission is selected, auto-populate wire config fields (blue range, yellow/red counts), available equipment, indicator token type, and special rules (X tokens, DD disabled, etc.).
- [ ] **GitHub Pages deployment**: GitHub Actions workflow to build and deploy the `web/` directory.

## Future Consideration: JavaScript Port

If Pyodide performance proves insufficient for complex game states (many hidden slots, uncertain wire groups), a JavaScript/TypeScript port of the probability engine could be considered. The Python test suite (`test_compute_probabilities.py`, ~2,800 lines) would serve as the validation harness — port the tests first, then implement until they pass. However, this should only be pursued if Pyodide is demonstrably too slow in real usage, not preemptively.
