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
│  │              │    │  bridge.py (new)        │  │
│  └─────────────┘    └────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Key components:**

- **Game State Entry Form (JS/HTML)**: Visual representation of tile stands where users click to set slot states, enter wire values, configure markers, etc. This replaces the `TileStand.from_string()` shorthand with a GUI. The shorthand notation should remain available as a power-user input mode.
- **State Manager (JS)**: Serializes the UI state to a JSON structure, posts it to the Web Worker, and processes the structured result JSON. Also handles URL-based state sharing (encode game state in URL parameters so players can share a link).
- **Bridge module (`bridge.py`, new)**: A thin Python module loaded in the Web Worker alongside the existing modules. Accepts a JSON dict, constructs a `GameState` via `from_partial_state()`, runs `rank_all_moves()` / `guaranteed_actions()`, and returns results as a JSON-serializable dict. This keeps the boundary between JS and Python clean — JS never manipulates Python objects directly.
- **Results Display (HTML/CSS)**: Renders the ranked moves, guaranteed actions, red wire warnings, and per-position probability distributions. Replaces the ANSI terminal output with styled HTML.

## Phased Work Plan

### Phase 0: Python API Hardening (Pre-Web)

Prepare the Python codebase for web consumption without changing any architecture. These changes benefit the CLI tool as well.

- [ ] **Add a `bridge.py` module** that provides a JSON-in/JSON-out interface: accepts a game state dict, returns analysis results as a plain dict. This is the only module the Web Worker calls. It insulates the web layer from internal API changes.
- [ ] **Add `to_dict()` / `from_dict()` round-trip methods** to core dataclasses (`Wire`, `Slot`, `TileStand`, `Marker`, `WireConfig`, `UncertainWireGroup`, `GameState`). These enable JSON serialization without depending on any web framework.
- [ ] **Add a callback-based progress API** to `build_solver()` as an alternative to `tqdm`. Accept an optional `on_progress(current, total)` callable. The Web Worker will use this to post progress messages.
- [ ] **Unit tests** for serialization round-trips and the bridge module.

### Phase 1: Web Scaffold

Set up the static web project structure alongside the existing Python code.

- [ ] **Create `web/` directory** with `index.html`, `style.css`, `app.js`, `worker.js`.
- [ ] **Pyodide loader**: `worker.js` loads Pyodide, fetches `bomb_busters.py`, `compute_probabilities.py`, and `bridge.py` as Python files, and exposes a `postMessage` API.
- [ ] **Basic game config form**: Player count (4-5), player names, mission wire setup (blue-only, or add red/yellow with count and optional X-of-Y pool size).
- [ ] **Minimal end-to-end test**: Hard-coded game state -> Worker -> Python -> results displayed as raw JSON in the page.

### Phase 2: Game State Entry UI

The core UX challenge — building an intuitive interface for entering a mid-game state.

- [ ] **Tile stand editor**: For each player, render a row of slots. Each slot is clickable and cycles through states (hidden/cut/info-revealed). For the active player, all hidden wires require a value (they know their own hand). For other players, hidden slots are unknown by default but can optionally have a known value (observer's private knowledge).
- [ ] **Wire value entry**: Dropdowns or number inputs for wire values, filtered by what's still possible (exclude fully validated values, respect sort ordering). Color-coded: blue values as numbers, yellow as "Y", red as "R".
- [ ] **Markers panel**: Auto-populated from wire config, with KNOWN/UNCERTAIN state toggleable.
- [ ] **Character card toggles**: Per-player Double Detector available/used status.
- [ ] **Detonator display**: Visual mistakes-remaining counter.
- [ ] **`TileStand.from_string()` shorthand input**: A text input per player that accepts the existing notation (e.g., `"1 3 ? ? i6 8 9 ? ? 12"`) as a power-user alternative to clicking through the GUI. Parses on blur/enter and populates the visual editor.

### Phase 3: Results Display

Replace ANSI terminal output with styled HTML.

- [ ] **Guaranteed actions section**: Styled cards/badges for solo cuts, guaranteed dual cuts, reveal red.
- [ ] **Ranked moves table**: Sortable table with probability bars, color-coded by confidence level (green/blue/yellow/red matching the terminal thresholds). Red wire risk shown as a warning badge.
- [ ] **Per-position probability breakdown**: Expandable detail showing the full wire distribution for each hidden slot (which wire values are possible and with what probability).
- [ ] **Double Detector results**: Separate section or toggle for DD moves, with the best-pair-per-target deduplication already present in `print_probability_analysis`.

### Phase 4: Polish and Features

- [ ] **Progress bar**: Worker posts progress messages during `build_solver()` -> main thread renders a progress indicator. Use the `_count_root_compositions()` estimate for total work.
- [ ] **URL state sharing**: Encode the full game state in URL query parameters (compressed/base64). Users can share a link that reconstructs the exact game state.
- [ ] **Error handling**: Validation errors from Python (e.g., pool/position mismatch, incomplete active player stand) displayed as user-friendly messages in the UI.
- [ ] **Mobile-friendly layout**: Tile stands are compact — a phone-friendly layout with horizontal scrolling per stand. This is a real use case (checking probabilities at the game table on a phone).
- [ ] **Turn history entry**: Optional UI for recording past failed dual cuts, enabling must-have deductions.
- [ ] **GitHub Pages deployment**: GitHub Actions workflow to build and deploy the `web/` directory.

## Future Consideration: JavaScript Port

If Pyodide performance proves insufficient for complex game states (many hidden slots, uncertain wire groups), a JavaScript/TypeScript port of the probability engine could be considered. The Python test suite (`test_compute_probabilities.py`, ~2,800 lines) would serve as the validation harness — port the tests first, then implement until they pass. However, this should only be pursued if Pyodide is demonstrably too slow in real usage, not preemptively.
