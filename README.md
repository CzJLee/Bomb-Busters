# Bomb Busters

A calculator for the board game [Bomb Busters](https://boardgamegeek.com/boardgame/413246/bomb-busters) to compute the best moves, or chance of success, in different scenarios.

## Table of Contents

- [Rules](#rules)
  - [Game Components](#game-components)
  - [Setup](#setup)
  - [Game Play](#game-play)
  - [End of the Game](#end-of-the-game)
- [Gameplay Tips](#gameplay-tips)
- [Terminal Display Guide](#terminal-display-guide)
  - [Header](#header)
  - [Validated](#validated)
  - [Markers](#markers)
  - [Players](#players)
  - [Tile Stand](#tile-stand)
  - [Game Over](#game-over)
- [Code Architecture](#code-architecture)
  - [Repository Layout](#repository-layout)
  - [Game Model (`bomb_busters.py`)](#game-model-bomb_busterspy)
  - [Creating a GameState](#creating-a-gamestate)
  - [Probability Engine (`compute_probabilities.py`)](#probability-engine-compute_probabilitiespy)

## Rules

Bomb Busters is a co-op game best played with 4-5 players. Each player is given a tile stand which contains sorted wire tiles that is only visible to the player the stand belongs to. The goal of the game is to successfully cut all the wires, thus defusing the bomb and passing the mission.

### Game Components

- At least for now, only the starting components are used. New items in surprise boxes may be added later but are not supported at the moment.
- There are 5 tile stands, for up to 5 players. In this project, I am assuming a player count of 4 or 5 players where each player only plays and has information on one tile stand.
- There are 70 total wire tiles: 48 blue wires (4 of each number 1-12), 11 red wires numbered 1.5-11.5, 11 yellow wires numbered 1.1-11.1.
- 4 yellow markers and 3 red markers.
- 26 info tokens: 2 for each number 1-12 and 2 yellow tokens.
- 12 validation tokens which indicate when all wires of that number have been cut.
- An `=` token and a `!=` token to be used with equipment cards 12 and 1 respectively.
- 5 character cards with a personal item. The personal item is the Double Detector.
- 12 equipment cards.
- Multiple mission cards that provide different scenarios and challenges.

### Setup

1. Select a mission card to play.
2. Randomly shuffle all 48 blue tiles in addition to any red and/or yellow tiles as required by the mission card.
3. Designate a captain.
4. Deal the shuffled wires evenly, starting with the captain and going clockwise. For example, with 5 players and 52 total wires, the captain and the player to the captain's left will draw 11 wires, and all other players will draw 10 wires.
5. All players will sort all tiles on their rack in ascending order with no gaps between any tiles.
6. The captain will indicate first with an info token for any of their blue wires. Players indicate one blue wire in clockwise direction until all players have indicated once.
7. Play begins with the captain after all players have indicated. Each player must take one action per turn: dual cut, solo cut, or using equipment followed by a cut action.

### Game Play

Starting with the captain and going clockwise, each bomb disposal expert takes a turn. On their turn, the active bomb disposal expert must do one of the following 3 actions: Dual Cut, Solo Cut, or Reveal Your Red Wires.

#### Dual Cut

The active bomb disposal expert must cut 2 identical wires: 1 of their own and 1 of their teammate's. They clearly point to a specific teammate's wire and guess what it is, stating its value (e.g., "This wire is a 9").

- If correct, the action succeeds:
  - The teammate places that wire faceup in front of their tile stand, without changing its position.
  - The active bomb disposal expert places their identical wire (or one of them if they have several) faceup in front of their tile stand.
- If wrong, the action fails:
  - If the wire is red, the bomb explodes and the mission ends in failure.
  - If the wire is blue or yellow, the detonator dial advances 1 space (the bomb explodes if the dial reaches the skull), and the teammate places an info token in front of the wire to show its real value.

#### Solo Cut

If the last of identical wires still in the game appear only in the active bomb disposal expert's hand, they can cut those identical wires in pairs (either 2 or 4). This can be done on their own, without involving another bomb disposal expert.

- If they have a full set of 4, they can cut all 4 wires at once.
- If a pair of wires of a given value have already been cut, they can cut the remaining 2 matching wires in their hand.

Cut wires are placed faceup on the table in front of the tile stand.

#### Reveal Your Red Wires

This action can occur only if the active bomb disposal expert's remaining uncut wires are all red. They reveal them, placing them faceup on the table in front of their tile stand.

#### Validation Tokens

As soon as all 4 wires of the same value have been cut, place 1 validation token on the matching number on the board.

#### Yellow Wires

Yellow wires are cut the same way as blue wires (Dual or Solo Cut), but the numeric value is used only when sorting the tiles on the stand in ascending order during setup. During the game, all yellow wires are considered to have the same value: "YELLOW". To cut a yellow wire, the active bomb disposal expert must have one in their hand, point to a teammate's wire, and say "This wire is yellow." If correct, the 2 wires are cut. If incorrect, an info token that reveals the actual value of the identified wire is placed, and the detonator dial advances 1 space.

- If a yellow wire is pointed at incorrectly, a yellow info token is used.
- A Solo Cut using yellow wires can occur only if the bomb disposal expert has all the remaining yellow wires in their hand.

#### Character Cards (Double Detector)

Each bomb disposal expert can use the personal equipment on their character card once per mission. To show that it has been used, flip it facedown.

**Double Detector:** During a Dual Cut action, the active bomb disposal expert states a value and points to 2 wires on a teammate's stand (instead of only 1).

- If either of these 2 wires matches the stated value, the action succeeds.
  - If both wires match, the teammate simply chooses which of the 2 chosen wires to cut.
- If neither of the 2 wires matches the stated value, the action fails.
  - The detonator dial advances 1 space, and the teammate places 1 info token in front of 1 of the 2 chosen wires (their choice).
  - If only 1 of the 2 chosen wires is red, the bomb does not explode. The teammate places an info token in front of the non-red wire without sharing any details.

### End of the Game

The mission ends in success when all bomb disposal experts have empty tile stands.

If the mission ends in failure (red wire cut or detonator dial reaches the skull), change which player is the captain and restart the mission.

## Gameplay Tips

- If a player attempts and fails a dual cut, we know they have at least one wire of the guessed value.
- You can narrow down possibilities on someone's tile stand by referencing which values are fully validated (all 4 cut). Those numbers cannot exist on anyone's stand.

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

## Code Architecture

### Repository Layout

```
bomb_busters.py              # Game model: enums, dataclasses, game state, actions
compute_probabilities.py     # Probability engine: constraint solver and API
simulate.py                  # Example mid-game probability analysis
docs/
  WEB_UI_ROADMAP.md          # Web UI development roadmap
  Bomb Busters Rulebook.pdf  # Official rulebook
  Bomb Busters FAQ.pdf       # Official FAQ
tests/
  __init__.py
  test_bomb_busters.py       # Game model tests
  test_compute_probabilities.py  # Probability engine tests
web/                         # Future: browser-based UI (see docs/WEB_UI_ROADMAP.md)
CLAUDE.md                    # Project instructions for Claude
README.md                    # This file
```

### Game Model (`bomb_busters.py`)

#### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `WireColor` | BLUE, RED, YELLOW | Color of a wire tile |
| `SlotState` | HIDDEN, CUT, INFO_REVEALED | State of a tile stand slot |
| `ActionType` | DUAL_CUT, SOLO_CUT, REVEAL_RED | Player action types |
| `ActionResult` | SUCCESS, FAIL_BLUE_YELLOW, FAIL_RED | Dual cut outcomes |
| `MarkerState` | KNOWN, UNCERTAIN | Board marker state (certain vs "X of Y" mode) |

#### Core Classes

**`Wire`** (frozen dataclass) — A physical wire tile. Uses a `sort_value` encoding for natural sort order: blue N = `N.0`, yellow = `N.1`, red = `N.5`. Properties: `base_number` (integer part), `gameplay_value` (int 1-12 for blue, `"YELLOW"`, or `"RED"`).

**`Slot`** — A single position on a tile stand. Holds a `Wire` (or `None` in calculator mode), a `SlotState`, and an optional `info_token` value.

**`TileStand`** — A player's wire rack. Slots are always sorted ascending by `sort_value`. Wires stay in position even after being cut. Factory methods: `from_wires()` for simulation mode, `from_string()` for quick entry from shorthand notation. Properties: `hidden_slots`, `cut_slots`, `is_empty`, `remaining_count`.

**`Player`** — A bomb disposal expert with a `TileStand` and optional `CharacterCard`.

**`CharacterCard`** — A one-use personal ability (e.g., Double Detector). Tracks `used` status.

**`Detonator`** — The bomb's failure counter. With N players, N-1 failures are tolerated. Tracks `failures`, `max_failures`, `is_exploded`, `remaining_failures`.

**`Marker`** — Board marker for red/yellow wires in play. State is `KNOWN` (direct inclusion) or `UNCERTAIN` ("X of Y" selection mode).

**`Equipment`** — Extensible equipment card with `unlock_value` (unlocked when 2 wires of that value are cut) and `used` tracking.

**`WireConfig`** — Mission setup for colored wires. Specifies `count` wires in play, with optional `pool_size` for "X of Y" random selection.

**`UncertainWireGroup`** — Represents colored wire candidates with uncertain inclusion from X-of-Y setup. Holds `candidates` (all drawn wires) and `count_in_play` (how many were kept). Factory methods: `yellow(numbers, count)`, `red(numbers, count)`. Used with `from_partial_state` for calculator mode.

#### Action Records

**`DualCutAction`** — Records actor, target, guessed value, result, and Double Detector details.

**`SoloCutAction`** — Records actor, value, slot indices, and wire count (2 or 4).

**`RevealRedAction`** — Records actor and revealed slot indices.

**`TurnHistory`** — Chronological list of all actions. Supports deduction queries like `failed_dual_cuts_by_player()`.

#### GameState

The central class managing the full game. Two factory methods are provided: `create_game()` for full simulation mode and `from_partial_state()` for calculator/mid-game mode. See [Creating a GameState](#creating-a-gamestate) for detailed usage and examples.

Action execution methods: `execute_dual_cut()`, `execute_solo_cut()`, `execute_reveal_red()`. Each validates the action, resolves outcomes, updates the detonator, places info tokens, checks validation tokens, and records to history.

### Creating a GameState

#### `GameState.create_game(player_names, wire_configs, seed)`

Create a new game in full simulation mode. Shuffles all wires, deals them evenly to players, sorts each stand in ascending order, and sets up board markers. All wire identities are known (god mode). The captain is always player index 0.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `player_names` | `list[str]` | Names of the players. Must be 4-5 players. |
| `wire_configs` | `list[WireConfig] \| None` | Optional list of `WireConfig` objects for red/yellow wires. If `None`, only blue wires are used (48 wires total). |
| `seed` | `int \| None` | Optional random seed for reproducible shuffles. |

**Returns:** A fully initialized `GameState` with all wires dealt and sorted.

**Raises:** `ValueError` if player count is not 4-5.

**Wire dealing:** Wires are dealt starting with the captain (index 0). If the total wire count doesn't divide evenly, earlier players receive one extra wire. For example, 48 blue wires among 5 players gives 10-10-10-9-9. With 51 wires (48 blue + 3 colored), it's 11-11-10-10-10.

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
    wire_configs=[
        bomb_busters.WireConfig(color=bomb_busters.WireColor.YELLOW, count=2),
        bomb_busters.WireConfig(color=bomb_busters.WireColor.RED, count=1),
    ],
    seed=1,
)

# Game with "X of Y" mode: draw 6 yellow wires, keep 3 (markers UNCERTAIN)
game = bomb_busters.GameState.create_game(
    player_names=["Alice", "Bob", "Charlie", "Diana"],
    wire_configs=[
        bomb_busters.WireConfig(
            color=bomb_busters.WireColor.YELLOW, count=3, pool_size=6,
        ),
    ],
)

# Set active player perspective (Bob sees his own wires, others' hidden wires
# are masked as '?' in display output)
game.active_player_index = 1
print(game)
```

#### `GameState.from_partial_state(player_names, stands, ...)`

Create a game state from partial mid-game information. Use this to enter an in-progress game for probability calculations without needing to replay all turns. Other players' hidden wires are set to `None` to represent unknown information.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `player_names` | `list[str]` | Names of all players. |
| `stands` | `list[TileStand]` | List of `TileStand` objects, one per player. Use `TileStand.from_string()` for quick entry or build manually with `TileStand(slots=[...])`. |
| `mistakes_remaining` | `int \| None` | How many more mistakes the team can survive. Defaults to `player_count - 1` (a fresh mission). |
| `markers` | `list[Marker] \| None` | Board markers for red/yellow wires in play. |
| `equipment` | `list[Equipment] \| None` | Equipment cards in play. |
| `wires_in_play` | `list[Wire] \| None` | All wire objects that were included in this mission. Required for probability calculations. |
| `character_cards` | `list[CharacterCard \| None] \| None` | Character card for each player (or `None` per player). |
| `history` | `TurnHistory \| None` | Optional turn history for deduction (e.g., failed dual cuts reveal the actor holds that value). |
| `active_player_index` | `int` | Index of the player whose turn it is. Display output is rendered from this player's perspective. Defaults to `0`. |
| `uncertain_wire_groups` | `list[UncertainWireGroup] \| None` | Groups of colored wires with uncertain inclusion from X-of-Y mission setup. UNCERTAIN markers are auto-generated from the candidates. See [Uncertain (X of Y) Wires](#uncertain-x-of-y-wires) for details. |

**Returns:** A `GameState` initialized from the provided partial information.

**Raises:** `ValueError` if the number of stands doesn't match the number of players.

**Example:**

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

game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    mistakes_remaining=3,
    wires_in_play=bomb_busters.create_all_blue_wires(),
)

# Use the probability engine
moves = compute_probabilities.rank_all_moves(game, active_player_index=0)
for move in moves[:5]:
    print(move)
```

#### `TileStand.from_string(notation, sep, num_tiles)`

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

All prefixes (`Y`, `R`, `i`, `?`) are case-insensitive.

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

# Use with from_partial_state
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, ...],
    wires_in_play=bomb_busters.create_all_blue_wires(),
)

# Custom separator
stand = bomb_busters.TileStand.from_string("1,2,?3,4", sep=",")
```

#### Uncertain (X of Y) Wires

When a mission uses X-of-Y colored wire setup (e.g., "draw 3 yellow wires, keep 2"), players see UNCERTAIN markers for all drawn candidates but don't know which subset was actually kept. Use `UncertainWireGroup` with `from_partial_state` to represent this uncertainty. The probability engine's constraint solver handles the combinatorics automatically using discard slots (slack variables) — no need to enumerate subsets manually.

**`UncertainWireGroup`** — Describes a group of colored wires with uncertain inclusion. Factory methods `yellow(numbers, count)` and `red(numbers, count)` provide ergonomic construction.

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
    # Blue wires are definite — list them in wires_in_play as usual
    wires_in_play=bomb_busters.create_all_blue_wires(),
    # Yellow candidates go in uncertain_wire_groups (NOT wires_in_play)
    uncertain_wire_groups=[
        bomb_busters.UncertainWireGroup.yellow([2, 3, 9], count=2),
    ],
)

# UNCERTAIN markers are auto-generated — no need to create them manually.
# The probability engine accounts for the uncertainty automatically.
compute_probabilities.print_probability_analysis(game, active_player_index=0)

# You can also combine yellow and red uncertain groups:
game = bomb_busters.GameState.from_partial_state(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve"],
    stands=[alice, bob, charlie, diana, eve],
    wires_in_play=bomb_busters.create_all_blue_wires(),
    uncertain_wire_groups=[
        bomb_busters.UncertainWireGroup.yellow([2, 3, 9], count=2),
        bomb_busters.UncertainWireGroup.red([3, 7], count=1),
    ],
)
```

**Notes:**
- Pass only definite wires (e.g., all blue wires) in `wires_in_play`. Uncertain colored wires go exclusively in `uncertain_wire_groups`.
- If the observer already has a candidate wire on their stand (e.g., Y3 is visible to them), the solver automatically accounts for it — no special handling needed.
- UNCERTAIN markers are auto-generated from the groups and merged with any manually provided markers (no duplicates).

### Probability Engine (`compute_probabilities.py`)

#### Knowledge Extraction

**`KnownInfo`** — Aggregates all information visible to the active player: their own wires, all cut wires, info tokens, validation tokens, markers, and must-have deductions from turn history.

**`extract_known_info(game, active_player_index)`** — Collects public + private knowledge into a `KnownInfo` object.

**`compute_unknown_pool(known, game)`** — Computes the set of wires whose locations are unknown: all wires in play minus the active player's wires, minus other players' cut/revealed wires.

#### Constraint Solver

**`PositionConstraint`** — Defines valid `sort_value` bounds for a hidden slot based on known neighboring wires (from cut or info-revealed positions). Supports a `required_color` field for slots where the wire color is known but the exact identity is not (e.g., yellow info tokens in calculator mode).

**`SolverContext`** — Immutable context built once per game-state + active player via `build_solver()`. Contains position constraints grouped by player, player processing order (widest bounds first for optimal memoization), distinct wire types, and must-have deductions.

**`build_solver(game, active_player_index)`** — Builds a `SolverContext` and `MemoDict` pair. The backward solve uses composition-based enumeration (how many of each wire type per player, not per-position) with memoization at player boundaries. Optionally displays a tqdm progress bar. Call once, then pass the result to any number of forward-pass functions for instant results.

**`compute_position_probabilities(game, active_player_index, ctx, memo)`** — Forward pass that computes per-position wire probability distributions from a prebuilt memo. Returns `{(player_index, slot_index): Counter({Wire: count})}`. When `ctx`/`memo` are omitted, builds them internally.

#### High-Level API

| Function | Description |
|----------|-------------|
| `probability_of_dual_cut(game, active_player, target_player, target_slot, value)` | Probability that a specific dual cut succeeds (0.0 to 1.0) |
| `probability_of_double_detector(game, active_player, target_player, slot1, slot2, value)` | Joint probability for Double Detector (not naive independence) |
| `probability_of_red_wire(game, active_player, target_player, target_slot, probs)` | Probability that a specific slot contains a red wire (0.0 to 1.0) |
| `probability_of_red_wire_dd(game, active_player, target_player, slot1, slot2)` | Joint probability that both DD target slots are red (instant game-over) |
| `guaranteed_actions(game, active_player)` | Find all 100% success actions (solo cuts, guaranteed dual cuts, reveal red) |
| `rank_all_moves(game, active_player, include_dd)` | Rank all possible moves by probability, sorted descending |
| `print_probability_analysis(game, active_player, max_moves, include_dd)` | Print a colored terminal report with guaranteed actions and ranked moves |

All high-level functions accept optional `ctx`/`memo` parameters to reuse a prebuilt solver, avoiding redundant computation when calling multiple functions on the same game state.

**`RankedMove`** — Dataclass for ranked results with `action_type`, target details, `guessed_value`, `probability`, and `red_probability` (risk of hitting a red wire).
