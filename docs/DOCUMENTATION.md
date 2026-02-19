# Bomb Busters Documentation

Developer documentation for the Bomb Busters probability calculator. This covers the game model, how to create game states, and the probability engine API.

For the game rules, see [RULES.md](RULES.md). For algorithm internals, see [EXACT_SOLVER.md](EXACT_SOLVER.md), [MONTE_CARLO_SOLVER.md](MONTE_CARLO_SOLVER.md), and [INDICATION_QUALITY.md](INDICATION_QUALITY.md).

## Table of Contents

- [Repository Layout](#repository-layout)
- [Terminal Display Guide](#terminal-display-guide)
  - [Header](#header)
  - [Validated](#validated)
  - [Markers](#markers)
  - [Players](#players)
  - [Tile Stand](#tile-stand)
  - [Game Over](#game-over)
- [Game Model (`bomb_busters.py`)](#game-model-bomb_busterspy)
  - [Enums](#enums)
  - [Core Classes](#core-classes)
  - [Action Records](#action-records)
  - [GameState](#gamestate)
- [Creating a GameState](#creating-a-gamestate)
  - [`GameState.create_game()`](#gamestatecreat_game)
  - [`GameState.from_partial_state()`](#gamestatefrom_partial_state)
  - [`TileStand.from_string()`](#tilestandfrom_string)
  - [Uncertain (X of Y) Wires](#uncertain-x-of-y-wires)
- [Missions (`missions.py`)](#missions-missionspy)
  - [Enums](#enums-1)
  - [Equipment Catalog](#equipment-catalog)
  - [Mission Dataclass](#mission-dataclass)
  - [Mission Registry](#mission-registry)
- [Probability Engine (`compute_probabilities.py`)](#probability-engine-compute_probabilitiespy)
  - [Knowledge Extraction](#knowledge-extraction)
  - [Constraint Solver](#constraint-solver)
  - [Monte Carlo Fallback](#monte-carlo-fallback)
  - [High-Level API](#high-level-api)
  - [Indication Quality Analysis](#indication-quality-analysis)

---

## Repository Layout

```
bomb_busters.py              # Game model: enums, dataclasses, game state, actions
compute_probabilities.py     # Probability engine: constraint solver and API
missions.py                  # Mission definitions: equipment catalog, mission 1-30
simulate.py                  # Example mid-game probability analysis
simulate_info_token.py       # Example indication phase simulation
simulate_game.py             # Full mission simulation (indication + cutting phases)
docs/
  RULES.md                   # Game rules and gameplay tips
  DOCUMENTATION.md           # This file
  EXACT_SOLVER.md            # Exact constraint solver algorithm documentation
  MONTE_CARLO_SOLVER.md      # Monte Carlo solver algorithm documentation
  INDICATION_QUALITY.md      # Indication quality metric documentation
  WEB_UI_ROADMAP.md          # Web UI development roadmap
  Bomb Busters Rulebook.pdf  # Official rulebook
  Bomb Busters FAQ.pdf       # Official FAQ
tests/
  __init__.py
  test_bomb_busters.py       # Game model tests
  test_compute_probabilities.py  # Probability engine tests
  test_missions.py           # Mission definitions tests
web/                         # Future: browser-based UI (see WEB_UI_ROADMAP.md)
CLAUDE.md                    # Project instructions for Claude
README.md                    # Project overview
```

---

## Terminal Display Guide

The game state is printed to the terminal with ANSI color coding.

### Header

```
=== Bomb Busters ===
Mistakes remaining: 3
```

**Mistakes remaining** shows how many more failed dual cuts the team can survive before the bomb explodes. Starts at N-1 for N players (e.g., 4 for a 5-player game). When it reaches 0, one more mistake ends the game.

### Validated

```
Validated: 3, 7
```

Lists blue wire values (1-12) where all 4 copies have been cut. Displays `(none)` if no values have been completed yet.

### Markers

```
Markers: 3(✓) 7(?)
```

Shown only when red or yellow wires are in play. Each marker indicates a colored wire value included in the mission:

- `✓` — **Known**: this wire is definitely in the game.
- `?` — **Uncertain**: this wire might be in the game ("X of Y" random selection mode).

### Players

Each player is displayed with a header line followed by their tile stand.

The active player (whose turn it is) is marked with a bold `>>>` prefix. All other players are indented with spaces:

```
>>> Player 0: Alice | Card: Double Detector (available)
```

Character card status is shown after the player's name:

- **(available)** in green — the Double Detector has not been used yet.
- **(used)** in dim text — the Double Detector has already been used this mission.

### Tile Stand

Each player's stand is displayed as three lines: a status indicator row, a values row, and a position letters row.

```
      ✓        i     ✓  ✓
      1  ?  ?  6  ?  5  8 11
      A  B  C  D  E  F  G  H
```

**Position letters** — Each slot is labeled with a letter starting from **A** (leftmost). Wires are always sorted in ascending order and stay in their position even after being cut.

**Wire values:**

| Display | Meaning |
|---------|---------|
| Blue number (e.g., `5`, `12`) | A blue wire with that value. Dimmed when hidden, green when cut. |
| `Y` | A yellow wire. All yellow wires share the gameplay value "YELLOW". |
| `R` | A red wire. |
| `?` | An unknown wire (another player's hidden wire in observer mode). |

**Status indicators** (displayed above each wire value):

| Symbol | Meaning |
|--------|---------|
| *(blank)* | **Hidden** — the wire is face-down and has not been acted on. |
| **✓** (green) | **Cut** — the wire has been successfully cut. It remains in position but is face-up. |
| **i** (bold) | **Info revealed** — a failed dual cut placed an info token on this wire. The wire is still uncut but its identity is now public. |

### Game Over

When the game ends, a status line appears at the bottom:

- **MISSION SUCCESS!** in green — all players' stands are empty.
- **MISSION FAILED!** in red — a red wire was cut or the detonator reached the skull.

---

## Game Model (`bomb_busters.py`)

### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `WireColor` | BLUE, RED, YELLOW | Color of a wire tile |
| `SlotState` | HIDDEN, CUT, INFO_REVEALED | State of a tile stand slot |
| `ActionType` | DUAL_CUT, SOLO_CUT, REVEAL_RED | Player action types |
| `ActionResult` | SUCCESS, FAIL_BLUE_YELLOW, FAIL_RED | Dual cut outcomes |
| `MarkerState` | KNOWN, UNCERTAIN | Board marker state (certain vs "X of Y" mode) |
| `Parity` | EVEN, ODD | Parity of a wire value for even/odd indicator tokens |
| `Multiplicity` | SINGLE (1), DOUBLE (2), TRIPLE (3) | Wire value count for x1/x2/x3 tokens |

### Core Classes

**`Wire`** (frozen dataclass) — A physical wire tile. Uses a `sort_value` encoding for natural sort order: blue N = `N.0`, yellow = `N.1`, red = `N.5`. Properties: `base_number` (integer part), `gameplay_value` (int 1-12 for blue, `"YELLOW"`, or `"RED"`).

**`Slot`** — A single position on a tile stand. Holds a `Wire` (or `None` in calculator mode), a `SlotState`, and an optional `info_token` value. Additional indicator token metadata fields: `parity` (Parity enum for even/odd tokens), `multiplicity` (Multiplicity enum for x1/x2/x3 tokens), `is_unsorted` (bool for X tokens), `excluded_value` (int for false info tokens), `required_color` (WireColor for color-constrained slots).

**`TileStand`** — A player's wire rack. Slots are always sorted ascending by `sort_value`. Wires stay in position even after being cut. Factory methods: `from_wires()` for simulation mode, `from_string()` for quick entry from shorthand notation. Properties: `hidden_slots`, `cut_slots`, `is_empty`, `remaining_count`.

**`Player`** — A bomb disposal expert with a `TileStand` and optional `CharacterCard`.

**`CharacterCard`** — A one-use personal ability (e.g., Double Detector). Tracks `used` status.

**`Detonator`** — The bomb's failure counter. With N players, N-1 failures are tolerated. Tracks `failures`, `max_failures`, `is_exploded`, `remaining_failures`.

**`Marker`** — Board marker for red/yellow wires in play. State is `KNOWN` (direct inclusion) or `UNCERTAIN` ("X of Y" selection mode).

**`Equipment`** — Extensible equipment card with `unlock_value` (unlocked when 2 wires of that value are cut) and `used` tracking.

**`WireConfig`** — Mission setup for colored wires. Specifies `count` wires in play, with optional `pool_size` for "X of Y" random selection.

**`UncertainWireGroup`** — Represents colored wire candidates with uncertain inclusion from X-of-Y setup. Holds `candidates` (all drawn wires) and `count_in_play` (how many were kept). Factory methods: `yellow(numbers, count)`, `red(numbers, count)`. Auto-derived internally by `from_partial_state` from the `yellow_wires` and `red_wires` tuple parameters.

### Slot Constraints

Constraint classes encode information from equipment cards (Label !=, Label =, General Radar) and other game effects that restrict which wires can occupy which positions. All are frozen dataclasses subclassing `SlotConstraint`.

| Constraint | Description | Source |
|-----------|-------------|--------|
| `AdjacentNotEqual` | Two adjacent slots must have different gameplay values | Label != (#1) |
| `AdjacentEqual` | Two adjacent slots must have the same gameplay value | Label = (#12) |
| `MustHaveValue` | Player must have at least one uncut wire of a specific value | General Radar "yes" (#8), failed dual cuts |
| `MustNotHaveValue` | Player must NOT have any uncut wire of a specific value | General Radar "no" (#8) |
| `SlotParity` | A slot is known to be even or odd | Even/odd indicator tokens |
| `ValueMultiplicity` | Wire value appears exactly N times on this stand | x1/x2/x3 indicator tokens |
| `UnsortedSlot` | A slot is not sorted with the rest of the stand | X indicator tokens |
| `SlotExcludedValue` | A slot is known to NOT be a specific value | False info tokens |

Constraints are stored on `GameState.slot_constraints` and added via convenience methods: `add_adjacent_equal()`, `add_adjacent_not_equal()`, `add_must_have()`, `add_must_not_have()`. The solver (`compute_probabilities.py`) enforces all constraint types during both exact and Monte Carlo solving. Indicator token constraints (`SlotParity`, `ValueMultiplicity`, `UnsortedSlot`, `SlotExcludedValue`) are auto-generated from `Slot` metadata fields when using `from_partial_state()` with extended token notation.

### Action Records

**`DualCutAction`** — Records actor, target, guessed value, result, and Double Detector details.

**`SoloCutAction`** — Records actor, value, slot indices, and wire count (2 or 4).

**`RevealRedAction`** — Records actor and revealed slot indices.

**`TurnHistory`** — Chronological list of all actions. Supports deduction queries like `failed_dual_cuts_by_player()`.

### GameState

The central class managing the full game. Two factory methods are provided: `create_game()` for full simulation mode and `from_partial_state()` for calculator/mid-game mode. See [Creating a GameState](#creating-a-gamestate) for detailed usage and examples.

Action execution methods: `execute_dual_cut()`, `execute_solo_cut()`, `execute_reveal_red()`. Each validates the action, resolves outcomes, updates the detonator, places info tokens, checks validation tokens, and records to history.

Equipment-supporting methods (named for actions, not equipment — reusable across game effects):

| Method | Description | Equipment |
|--------|-------------|-----------|
| `place_info_token(player_index, slot_index)` | Mark hidden blue wire as INFO_REVEALED | Post-It (#4) |
| `adjust_detonator(delta)` | Change detonator failures by delta (negative = rewind) | Rewinder (#6) |
| `set_detonator(mistakes_remaining)` | Set detonator to specific value | — |
| `reactivate_character_cards(player_indices)` | Reset character cards to available | Emergency Batteries (#7) |
| `set_current_player(player_index)` | Change active player | Coffee Mug (#11) |
| `cut_all_of_value(value)` | Cut all remaining wires of a value | Disintegrator (#10.10) |
| `add_must_have(player_index, value)` | Add MustHaveValue constraint | General Radar "yes" (#8) |
| `add_must_not_have(player_index, value)` | Add MustNotHaveValue constraint | General Radar "no" (#8) |
| `add_adjacent_equal(player_index, slot_left, slot_right)` | Add AdjacentEqual constraint | Label = (#12) |
| `add_adjacent_not_equal(player_index, slot_left, slot_right)` | Add AdjacentNotEqual constraint | Label != (#1) |

Solo cut methods accept `fast_pass: bool = False` parameter. When True, skips the "all remaining must be in hand" check (Fast Pass #9.9).

---

## Creating a GameState

### `GameState.create_game()`

Create a new game in full simulation mode. Shuffles all wires, deals them evenly to players, sorts each stand in ascending order, and sets up board markers. All wire identities are known (god mode).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `player_names` | `list[str]` | Names of the players. Must be 4-5 players. |
| `seed` | `int \| None` | Optional random seed for reproducible shuffles. |
| `captain` | `int` | Player index of the captain. Defaults to `0`. |
| `yellow_wires` | `int \| tuple[int, int] \| None` | Yellow wire specification. `None` (default) = no yellow wires. `2` = include 2 randomly selected yellow wires (KNOWN markers). `(2, 3)` = draw 3 random yellow wires, keep 2 (UNCERTAIN markers on all 3 drawn). |
| `red_wires` | `int \| tuple[int, int] \| None` | Red wire specification. Same semantics as `yellow_wires`. |

**Returns:** A fully initialized `GameState` with all wires dealt and sorted.

**Raises:** `ValueError` if player count is not 4-5 or captain index is out of range.

**Wire dealing:** Wires are dealt starting with the captain, clockwise. If the total wire count doesn't divide evenly, earlier players receive one extra wire. For example, 48 blue wires among 5 players gives 10-10-10-9-9. With 51 wires (48 blue + 3 colored), it's 11-11-10-10-10.

**Examples:**

```python
import bomb_busters

# Blue-only game (simplest setup)
game = bomb_busters.GameState.create_game(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    seed=42,
)

# Game with 2 yellow wires and 1 red wire (direct inclusion, markers KNOWN)
game = bomb_busters.GameState.create_game(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    yellow_wires=2,
    red_wires=1,
    seed=1,
)

# Game with "X of Y" mode: draw 6 yellow wires, keep 3 (markers UNCERTAIN)
game = bomb_busters.GameState.create_game(
    player_names=["Alice", "Bob", "Charlie", "Diana"],
    yellow_wires=(3, 6),
)

# Set active player perspective (Bob sees his own wires, others' hidden wires
# are masked as '?' in display output)
game.active_player_index = 1
print(game)
```

### `GameState.from_partial_state()`

Create a game state from partial mid-game information. Use this to enter an in-progress game for probability calculations without needing to replay all turns. Other players' hidden wires are set to `None` to represent unknown information.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `player_names` | `list[str]` | Names of all players. |
| `stands` | `list[TileStand]` | List of `TileStand` objects, one per player. Use `TileStand.from_string()` for quick entry or build manually with `TileStand(slots=[...])`. |
| `mistakes_remaining` | `int \| None` | How many more mistakes the team can survive. Defaults to `player_count - 1` (a fresh mission). |
| `equipment` | `list[Equipment] \| None` | Equipment cards in play. |
| `character_cards` | `list[CharacterCard \| None] \| None` | Character card for each player (or `None` per player). |
| `history` | `TurnHistory \| None` | Optional turn history for deduction (e.g., failed dual cuts reveal the actor holds that value). |
| `active_player_index` | `int` | Index of the player whose turn it is. Display output is rendered from this player's perspective. Defaults to `0`. |
| `captain` | `int` | Player index of the captain. Defaults to `0`. |
| `blue_wires` | `list[Wire] \| tuple[int, int] \| None` | Blue wire pool. `None` (default) = all blue 1-12 (48 wires). `(low, high)` tuple = blue wires for values low through high (4 copies each). `list[Wire]` = custom wire list. |
| `yellow_wires` | `list[int] \| tuple[list[int], int] \| None` | Yellow wire specification. `None` (default) = no yellow wires. `[4, 7]` = Y4 and Y7 definitely in play (KNOWN markers). `([2, 3, 9], 2)` = 2-of-3 uncertain (UNCERTAIN markers). |
| `red_wires` | `list[int] \| tuple[list[int], int] \| None` | Red wire specification. Same semantics as `yellow_wires`. `[4]` = R4 definitely in play. `([3, 7], 1)` = 1-of-2 uncertain. |

Markers and uncertain wire groups are auto-derived internally from the `blue_wires`, `yellow_wires`, and `red_wires` parameters. There is no need to construct `Marker` or `UncertainWireGroup` objects manually.

**Returns:** A `GameState` initialized from the provided partial information.

**Raises:** `ValueError` if the number of stands doesn't match the number of players.

**Examples:**

```python
import bomb_busters
import compute_probabilities

# 5 players, active player is player 0.
# Use TileStand.from_string() for quick entry (see below).
# Player 0 (active player) knows their own wires: ?N for hidden, N for cut.
# Other players' hidden wires are ? (unknown to the active player).

alice = bomb_busters.TileStand.from_string("?2 3 ?5 ?7 ?9")       # active player
bob   = bomb_busters.TileStand.from_string("? 4 ? i8 ?")          # partial knowledge
charlie = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")  # all unknown
diana   = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")
eve     = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")

# Blue-only game (default: all blue 1-12, 48 wires)
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    mistakes_remaining=3,
)

# Use the probability engine
moves = compute_probabilities.rank_all_moves(game, active_player_index=0)
for move in moves[:5]:
    print(move)

# Game with restricted blue range (only blue 1-8)
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    blue_wires=(1, 8),
)

# Game with known yellow and red wires
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    yellow_wires=[4, 7],       # Y4, Y7 definitely in play (KNOWN markers)
    red_wires=[3],             # R3 definitely in play (KNOWN marker)
)

# Game with uncertain (X of Y) colored wires
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    yellow_wires=([2, 3, 9], 2),  # 2-of-3 uncertain (UNCERTAIN markers)
    red_wires=([3, 7], 1),        # 1-of-2 uncertain (UNCERTAIN markers)
)
```

### `TileStand.from_string()`

Create a tile stand from shorthand string notation for quick mid-game entry. Each token in the string describes one slot on the stand. This is the fastest way to enter a game state for probability calculations.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `notation` | `str` | The shorthand string describing the tile stand. |
| `sep` | `str` | Token separator (default `" "`). |
| `num_tiles` | `int \| None` | If provided, validates that the parsed tile count matches exactly. |

**Returns:** A `TileStand` with slots in the order given.

**Raises:** `ValueError` if a token cannot be parsed, the notation is empty, or `num_tiles` doesn't match.

**Token reference:**

| Token | State | Meaning | Example |
|-------|-------|---------|---------|
| `N` | CUT | Blue wire with value N | `5` → cut blue-5 |
| `YN` | CUT | Yellow wire at sort position N | `Y4` → cut yellow-4 |
| `RN` | CUT | Red wire at sort position N | `R5` → cut red-5 |
| `?` | HIDDEN | Unknown wire (other player's hidden) | `?` → unknown |
| `?N` | HIDDEN | Blue wire known to observer | `?4` → hidden blue-4 |
| `?YN` | HIDDEN | Yellow wire known to observer | `?Y4` → hidden yellow-4 |
| `?RN` | HIDDEN | Red wire known to observer | `?R5` → hidden red-5 |
| `iN` | INFO_REVEALED | Blue info token from failed dual cut | `i5` → info blue-5 |
| `iY` | INFO_REVEALED | Yellow info token from failed dual cut | `iY` → info yellow |
| `iYN` | INFO_REVEALED | Yellow info token, observer knows exact wire | `iY4` → info yellow-4 |
| `!N` | HIDDEN | False info: slot is NOT value N | `!3` → hidden, not blue-3 |
| `!N?M` | HIDDEN | False info with god-mode wire | `!3?5` → hidden blue-5, not blue-3 |
| `E` | HIDDEN | Even parity token (value unknown) | `E` → hidden, even wire |
| `E?N` | HIDDEN | Even parity with god-mode wire | `E?4` → hidden blue-4, even |
| `O` | HIDDEN | Odd parity token (value unknown) | `O` → hidden, odd wire |
| `O?N` | HIDDEN | Odd parity with god-mode wire | `O?5` → hidden blue-5, odd |
| `1x` | HIDDEN | x1 multiplicity token (value unknown) | `1x` → hidden, unique value |
| `2xN` | CUT | x2 multiplicity with cut wire | `2x7` → cut blue-7, appears twice |
| `2x?N` | HIDDEN | x2 multiplicity with god-mode wire | `2x?7` → hidden blue-7, appears twice |
| `2xYN` | CUT | x2 multiplicity with cut yellow wire | `2xY4` → cut yellow-4, appears twice |
| `X` | HIDDEN | Unsorted X token (value unknown) | `X` → hidden, unsorted |
| `XN` | CUT | Unsorted with cut wire | `X5` → cut blue-5, unsorted |
| `X?N` | HIDDEN | Unsorted with god-mode wire | `X?5` → hidden blue-5, unsorted |
| `b?` | HIDDEN | Color-constrained blue (value unknown) | `b?` → hidden, known blue |
| `y?` | HIDDEN | Color-constrained yellow (value unknown) | `y?` → hidden, known yellow |
| `r?` | HIDDEN | Color-constrained red (value unknown) | `r?` → hidden, known red |

All prefixes (`Y`, `R`, `i`, `?`) are case-insensitive. Color prefixes (`b`, `y`, `r`) can be combined with indicator tokens (e.g., `bE?4`, `rX?5`). Even/odd tokens cannot be combined with yellow or red wires. Multiplicity tokens cannot be combined with red wires.

**Examples:**

```python
import bomb_busters

# Observer's own stand — all wires known (use ?N for hidden, N for cut)
alice = bomb_busters.TileStand.from_string(
    "1 2 ?4 ?Y4 ?6 ?7 ?8 ?8 9 11 12", num_tiles=11,
)

# Another player's stand — hidden wires unknown to observer
bob = bomb_busters.TileStand.from_string("1 3 ? ? ? 8 9 ? ? 12")

# Stand with an info token from a failed dual cut
diana = bomb_busters.TileStand.from_string("2 3 ? ? i6 ? ? 9 ? 11")

# Use with from_partial_state (default: all blue 1-12)
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, ...],
)

# Custom separator
stand = bomb_busters.TileStand.from_string("1,2,?3,4", sep=",")
```

### Uncertain (X of Y) Wires

When a mission uses X-of-Y colored wire setup (e.g., "draw 3 yellow wires, keep 2"), players see UNCERTAIN markers for all drawn candidates but don't know which subset was actually kept. Pass a tuple `(candidates, count)` to `yellow_wires` or `red_wires` in `from_partial_state` to represent this uncertainty. The probability engine's constraint solver handles the combinatorics automatically using discard slots (slack variables) — no need to enumerate subsets manually.

**How it works:** All candidate wires are added to the solver's pool. Virtual "discard slots" absorb the wires that are NOT in the game. The solver naturally enumerates which candidates are discarded vs. distributed to real player positions, producing correct joint probabilities in a single solve.

**Example:**

```python
import bomb_busters
import compute_probabilities

# Mission setup: drew 3 yellow wires (Y2, Y3, Y9), keeping 2.
# We see 3 UNCERTAIN markers but don't know which 2 are in the game.

alice   = bomb_busters.TileStand.from_string("?2 ?5 ?7 ?9 ?11")
bob     = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")
charlie = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")
diana   = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")
eve     = bomb_busters.TileStand.from_string("? ? ? ? ? ? ? ? ?")

game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    # 2-of-3 uncertain yellow wires (UNCERTAIN markers auto-generated)
    yellow_wires=([2, 3, 9], 2),
)

# UNCERTAIN markers are auto-generated — no need to create them manually.
# The probability engine accounts for the uncertainty automatically.
compute_probabilities.print_probability_analysis(game, active_player_index=0)

# You can also combine yellow and red uncertain groups:
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    yellow_wires=([2, 3, 9], 2),   # 2-of-3 uncertain
    red_wires=([3, 7], 1),         # 1-of-2 uncertain
)
```

**Notes:**

- Blue wires default to all 1-12 (48 wires). You only need `blue_wires` if the mission uses a restricted range.
- If the observer already has a candidate wire on their stand (e.g., Y3 is visible to them), the solver automatically accounts for it — no special handling needed.
- UNCERTAIN markers are auto-generated from the tuple form. KNOWN markers are auto-generated from the list form. There is no need to construct `Marker` or `UncertainWireGroup` objects manually.

---

## Missions (`missions.py`)

Static mission definitions for missions 1-30. Each mission specifies wire configuration, equipment availability, indicator token system, and special rules. Missions are reference data — they describe game setup but do not manage game state.

### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `IndicatorTokenType` | INFO, EVEN_ODD, MULTIPLICITY | Which indicator token system a mission uses. INFO is standard numbered 1-12. |
| `IndicationRule` | STANDARD, RANDOM_TOKEN, CAPTAIN_FAKE, SKIP, NEGATIVE | How the indication phase works during setup. |

### Equipment Catalog

**`EquipmentCard`** (frozen dataclass) — Static definition of an equipment card. Fields: `card_id` (str), `name`, `description`, `unlock_value` (int), `is_double` (bool). This is reference data, not a game-state object.

**`EQUIPMENT_CATALOG`** — Dict mapping card ID strings to `EquipmentCard` objects. Contains 18 entries: standard `"1"`-`"12"`, yellow `"Y"`, and double-numbered `"2.2"`, `"3.3"`, `"9.9"`, `"10.10"`, `"11.11"`.

**`get_equipment(card_id)`** — Look up an equipment card by ID. Raises `ValueError` if not found.

### Mission Dataclass

**`Mission`** (frozen dataclass) — A mission configuration with sensible defaults. Key fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `number` | `int` | *(required)* | Mission number (1-66) |
| `name` | `str` | *(required)* | Mission name |
| `blue_wire_range` | `tuple[int, int]` | `(1, 12)` | Blue wire values in play (4 copies each) |
| `yellow_wires` | `int \| tuple[int, int] \| None` | `None` | Yellow wire specification (matches `create_game()` signature) |
| `red_wires` | `int \| tuple[int, int] \| None` | `None` | Red wire specification |
| `equipment_forbidden` | `tuple[str, ...]` | `()` | Equipment IDs removed from the default pool |
| `equipment_override` | `tuple[str, ...] \| None` | `None` | If set, replaces the default pool entirely |
| `indicator_type` | `IndicatorTokenType` | `INFO` | Which indicator token system is used |
| `indication_rule` | `IndicationRule` | `STANDARD` | How the indication phase works |
| `x_tokens` | `bool` | `False` | Whether X tokens mark unsorted wires |
| `double_detector_disabled` | `bool` | `False` | Disable all players' Double Detectors |
| `number_cards` | `bool` | `False` | Whether number cards are used |
| `timer_minutes` | `int \| None` | `None` | Optional time limit |
| `notes` | `str` | `""` | Free-text special rules |

**`available_equipment()`** — Computes equipment card IDs available for this mission. Default pool is inferred from mission number: missions 1-2 get no equipment, mission 3 gets 1-10, missions 4+ get 1-12, missions 9+ add yellow equipment (if yellow wires are present), missions 55+ add double-numbered equipment. `equipment_forbidden` removes IDs from the default; `equipment_override` replaces the default entirely.

**`wire_config()`** — Returns wire configuration as a dict with keys matching `GameState.from_partial_state()` parameter names (`blue_wires`, `yellow_wires`, `red_wires`). Only includes non-default values.

### Mission Registry

**`MISSIONS`** — Dict mapping mission number (int) to `Mission` objects. Contains missions 1-30.

| Function | Description |
|----------|-------------|
| `get_mission(number)` | Look up a mission by number. Raises `ValueError` if not defined. |
| `all_missions()` | Return all defined missions sorted by number. |

---

## Probability Engine (`compute_probabilities.py`)

### Knowledge Extraction

**`KnownInfo`** — Aggregates all information visible to the active player: their own wires, all cut wires, info tokens, validation tokens, markers, and must-have deductions from turn history.

**`extract_known_info(game, active_player_index)`** — Collects public + private knowledge into a `KnownInfo` object.

**`compute_unknown_pool(known, game)`** — Computes the set of wires whose locations are unknown: all wires in play minus the active player's wires, minus other players' cut/revealed wires.

### Constraint Solver

The exact solver uses composition-based enumeration with memoization at player boundaries to compute precise probability distributions. See [EXACT_SOLVER.md](EXACT_SOLVER.md) for the full algorithm documentation.

**`PositionConstraint`** — Defines valid `sort_value` bounds for a hidden slot based on known neighboring wires (from cut or info-revealed positions). Supports additional constraint fields: `required_color` (wire color filter, e.g., yellow info tokens), `required_parity` (even/odd filter from parity tokens), `excluded_values` (frozenset of values the wire cannot be, from false info tokens), `is_unsorted` (X token positions get wide bounds and don't constrain neighbors).

**`SolverContext`** — Immutable context built once per game-state + active player via `build_solver()`. Contains position constraints grouped by player, player processing order (widest bounds first for optimal memoization), distinct wire types, and must-have deductions.

**`build_solver(game, active_player_index)`** — Builds a `SolverContext` and `MemoDict` pair. The backward solve uses composition-based enumeration (how many of each wire type per player, not per-position) weighted by combinatorial coefficients `∏ C(c_d, k_d)` to correctly account for the multivariate hypergeometric distribution, with memoization at player boundaries. Optionally displays a tqdm progress bar. Call once, then pass the result to any number of forward-pass functions for instant results.

**`compute_position_probabilities(game, active_player_index, ctx, memo)`** — Forward pass that computes per-position wire probability distributions from a prebuilt memo. Returns `{(player_index, slot_index): Counter({Wire: count})}`. When `ctx`/`memo` are omitted, builds them internally.

### Monte Carlo Fallback

The exact solver becomes intractable for early-game states with many hidden positions (>22). A Monte Carlo sampler provides approximate probabilities when the exact solver would be too slow. See [MONTE_CARLO_SOLVER.md](MONTE_CARLO_SOLVER.md) for the full algorithm documentation.

**`MC_POSITION_THRESHOLD`** — Module-level constant (default `22`). When `count_hidden_positions()` exceeds this, `print_probability_analysis()` automatically uses Monte Carlo instead of the exact solver.

**`count_hidden_positions(game, active_player_index)`** — Counts hidden + uncertain-info-revealed positions on other players' stands, plus discard positions from uncertain wire groups. Lightweight — no solver setup needed. Used by callers to decide between exact and Monte Carlo.

**`monte_carlo_probabilities(game, active_player_index, num_samples, seed, max_attempts)`** — Backward-guided composition sampler with importance weighting. For each sample, processes players sequentially: builds a lightweight single-player backward DP table per player and samples a valid composition proportional to `C(c_d, k) * f[d+1][pi+k]`, guaranteeing valid ascending sequences with no dead ends. Samples are weighted by the product of per-player normalization constants (self-normalized importance sampling). Returns the same `{(player_index, slot_index): Counter({Wire: count})}` format as `compute_position_probabilities()`, so callers can switch seamlessly. Default `num_samples=1_000`. Typical throughput: 1,000-8,000 valid samples/sec depending on game stage.

**`monte_carlo_analysis(game, active_player_index, num_samples, seed, max_attempts)`** — Like `monte_carlo_probabilities()` but also returns an `MCSamples` object containing raw per-sample wire assignments. This enables joint probability queries needed for Double Detector support in MC mode. Returns `(marginal_probs_dict, MCSamples | None)`.

**`MCSamples`** — Dataclass holding raw per-sample wire assignments and importance weights from MC sampling. Fields: `samples` (list of `{(player_index, slot_index): Wire}` dicts, one per sample) and `weights` (importance weights parallel to samples). Used with `mc_dd_probability()` and `mc_red_dd_probability()` for joint probability queries.

**`mc_dd_probability(mc_samples, target_player, slot1, slot2, guessed_value)`** — Compute Double Detector probability from MC samples: P(at least one target slot has the guessed value). Uses weighted self-normalized importance sampling over raw per-sample assignments.

**`mc_red_dd_probability(mc_samples, target_player, slot1, slot2)`** — Compute joint red wire probability from MC samples: P(both target slots are red). Used to assess DD red-wire risk in MC mode.

### High-Level API

| Function | Description |
|----------|-------------|
| `probability_of_dual_cut(game, active_player, target_player, target_slot, value)` | Probability that a specific dual cut succeeds (0.0 to 1.0) |
| `probability_of_double_detector(game, active_player, target_player, slot1, slot2, value)` | Joint probability for Double Detector (2 slots, not naive independence) |
| `probability_of_triple_detector(game, active_player, target_player, s1, s2, s3, value)` | Joint probability for Triple Detector (3 slots) |
| `probability_of_super_detector(game, active_player, target_player, value)` | Joint probability for Super Detector (all hidden slots) |
| `probability_of_x_or_y_ray(game, active_player, target_player, slot, value1, value2)` | P(slot matches value1 or value2) — uses marginals (mutually exclusive) |
| `probability_of_red_wire(game, active_player, target_player, target_slot, probs)` | Probability that a specific slot contains a red wire (0.0 to 1.0) |
| `probability_of_red_wire_dd(game, active_player, target_player, slot1, slot2)` | Joint probability that both DD target slots are red (instant game-over) |
| `probability_of_red_wire_multi(game, active_player, target_player, slots)` | Joint probability that all specified slots are red |
| `guaranteed_actions(game, active_player, include_equipment)` | Find all 100% success actions (solo cuts, fast pass solo cuts, guaranteed dual cuts, reveal red) |
| `rank_all_moves(game, active_player, include_equipment, mc_samples)` | Rank all possible moves by probability, sorted descending |
| `print_probability_analysis(game, active_player, max_moves, include_equipment)` | Print a colored terminal report with guaranteed actions and ranked moves |
| `monte_carlo_analysis(game, active_player, num_samples)` | MC sampling returning both marginal probs and raw `MCSamples` for joint queries |
| `mc_dd_probability(mc_samples, target_player, slot1, slot2, value)` | Double Detector probability from MC samples |
| `mc_td_probability(mc_samples, target_player, s1, s2, s3, value)` | Triple Detector probability from MC samples |
| `mc_sd_probability(mc_samples, target_player, slots, value)` | Super Detector probability from MC samples |
| `mc_red_dd_probability(mc_samples, target_player, slot1, slot2)` | Joint red-wire probability for DD from MC samples |
| `mc_red_multi_probability(mc_samples, target_player, slots)` | Joint red-wire probability for any slot set from MC samples |

All high-level functions accept optional `ctx`/`memo` parameters to reuse a prebuilt solver, avoiding redundant computation when calling multiple functions on the same game state. `rank_all_moves` also accepts `mc_samples` for joint probability computation in Monte Carlo mode.

**`EquipmentType`** — Enum of equipment types that affect probability calculations: `DOUBLE_DETECTOR`, `TRIPLE_DETECTOR`, `SUPER_DETECTOR`, `X_OR_Y_RAY`, `FAST_PASS`. Used with `include_equipment` parameter on `rank_all_moves`, `guaranteed_actions`, and `print_probability_analysis`.

**`RankedMove`** — Dataclass for ranked results with `action_type` (one of `"dual_cut"`, `"solo_cut"`, `"reveal_red"`, `"double_detector"`, `"triple_detector"`, `"super_detector"`, `"x_or_y_ray"`, `"fast_pass_solo"`), target details, `guessed_value`, `second_value` (for X or Y Ray), `third_slot` (for Triple Detector), `probability`, and `red_probability` (risk of hitting a red wire).

### Indication Quality Analysis

At game start, each player indicates one blue wire by placing an info token on it. This publicly reveals the wire's value and position, constraining what the remaining hidden wires could be via sort order. The indication quality analysis computes which wire to indicate for maximum information gain.

**`IndicationChoice`** — Dataclass with `slot_index`, `wire`, `information_gain` (bits), `uncertainty_resolved` (fraction 0-1), and `remaining_entropy` (bits).

- **Information gain (bits)**: Each bit corresponds to halving the number of equally-likely wire arrangements. An indication gaining 6 bits reduces possibilities by ~64x ($2^6$). The scale is logarithmic: 8 bits resolves 16x more uncertainty than 4 bits, not 2x.
- **Uncertainty resolved (%)**: The fraction of total baseline uncertainty that this indication eliminates. Linear and intuitive — 30% resolved is twice as informative as 15% resolved. Use this to compare indication choices at a glance.

| Function | Description |
|----------|-------------|
| `rank_indications(game, player_index)` | Rank all blue hidden wires by information gain (Shannon entropy reduction), sorted descending. The top choice is the recommended indication. |
| `print_indication_analysis(game, player_index)` | Print a colored terminal report showing the player's stand, baseline entropy, and ranked indication choices with information gain in bits and percentage of uncertainty resolved. |
| `rank_indications_parity(game, player_index)` | Rank indication choices when using even/odd parity tokens. Reveals only whether the wire is even or odd — not its exact value. Information gains are lower than standard indication. |
| `print_indication_analysis_parity(game, player_index)` | Print a colored terminal report for parity indication choices, showing parity label (E/O) and actual value. |

The metric uses a dedicated single-stand two-pass DP solver. For each candidate indication, it simulates revealing the wire (tightening sort-value bounds on neighbors), then measures how much the total per-position entropy decreases. See [INDICATION_QUALITY.md](INDICATION_QUALITY.md) for the full information-theoretic foundation.

**Parity indication semantics**: With even/odd tokens, the indicated slot stays HIDDEN (only parity is revealed, not the exact value). The wire is NOT removed from the pool. Neighbors do NOT get tighter sort-value bounds. The only new information is a `required_parity` constraint on the indicated position, which filters out wires of the wrong parity. Since parity reveals strictly less information than standard indication, information gains are always lower.
