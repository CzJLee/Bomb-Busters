# Bomb Busters

A calculator for the board game [Bomb Busters](https://boardgamegeek.com/boardgame/413246/bomb-busters) to compute the best moves, or chance of success, in different scenarios. 

## Rules

Bomb Busters is a co-op game best played with 4-5 players. Each player is given a `tile stand` which contains sorted wire tiles that is only visible to the player the stand belongs to. The goal of the game is to successfully cut all the wires, thus defusing the bomb and passing the mission. 

### Game component overview

- At least for now, only the starting components are used. New items in surprise boxes may be added later but are not supported at the moment. 
- There are 5 `tile stands`, for up to 5 players. In this project, I am assuming a player count of 4 or 5 players where each player only plays and has information on one tile stand. 
- There are 70 total `wire tiles`. 48 `blue` wires (4 of each number 1 to 12), 11 `red` wires numbered 1.5 to 11.5, 11 `yellow` wires numbered 1.1 to 11.1. 
- 4 `yellow markers` and 3 `red markers`.
- 26 `info tokens`, 2 for each number 1 to 12 and two yellow tokens. 
- 12 `validation tokens` which indicate when all wires of that number have been cut. 
- An `= token` and a `!= token` to be used with item card 12 and 1 respectively. 
- 5 `character cards` with a personal item. The personal item is the `Double Detector`. 
- 12 `equipment cards`. 
- Multiple `mission cards` that provide different scenarios and challenges.

### Setup

1. Select a `mission card` to play.
2. Randomly shuffle all 48 blue tiles in addition to any red and/or yellow tiles as required by the mission card. 
3. Designate a captain. 
4. Deal the shuffled wires evenly, starting with the captain and going clockwise. For example, with 5 players and 52 total wires, the captain and the player to the captain's left will draw 11 wires, and all other players will draw 10 wires. 
5. All players will sort all tiles on their rack in ascending order with no gaps between any tiles. 
6. The captain will indicate first with an `info token` for any of their blue wires. Players indicate one blue wire in clockwise direction until all players have indicated once. 
7. Play begins with the captain after all players have indicated. Each player must take one action per turn. Actions include `dual cut`, `solo cut`, or using an equipment followed by a cut action. 

### Game Play

Starting with the Captain and going clockwise, each bomb disposal expert takes a turn. On their turn, a bomb disposal expert (called the “active bomb disposal expert“) must do 1 of the following 3 actions: `Dual Cut` action, `Solo Cut` action, or Reveal Your Red Wires action.

#### Dual Cut action

The active bomb disposal expert must cut 2 identical wires: 1 of their own and 1 of their teammates’. They clearly point to a specific teammate’s wire and guess what it is, stating its value. For example “This wire is a 9”.

- If the active bomb disposal expert is correct, the action succeeds
  - Their teammate takes that wire and places it faceup in front of their tile stand, without changing its position.
  - Then the active bomb disposal expert takes their identical wire (or one of them if they have several) and places it faceup in front of their tile stand.
- But if they are wrong, the action fails:
  - If the wire in question is red red, the bomb explodes, and the mission ends in failure;
  - If the wire in question is blue blue or yellow yellow, the detonator dial advances 1 space (the bomb explodes if the dial reaches the skull and the mission fails), and their teammate places an Info token in front of the wire in question to show its real value.

#### Solo Cut action

If the last of identical wires still in the game appear only in the active bomb disposal expert’s hand, then they can cut those identical wires in pairs (either 2 or 4). This can be done on their own, without involving another bomb disposal expert. 
- If they are lucky enough to have a full set of 4, they can cut all 4 wires at once.
- If a pair of wires of a given value have already been cut, they can cut the remaining 2 matching wires in their hand.
These cut wires are placed faceup on the table in front of the tile stand.

#### Reveal your red wires action

This action can occur only if the active bomb disposal expert’s remaining uncut wires are all RED RED. They reveal them, placing them faceup on the table in front of their tile stand.

#### Validation tokens

As soon as all 4 wires of the same value have been cut, place 1 Validation token on the matching number on the board. 

#### Yellow wires

Yellow wires are cut the same way as blue blue wires (Dual or Solo Cut), but the numeric value is used only when sorting the tiles on the stand in ascending order during setup. During the game, all yellow wires are considered to have the same value: "YELLOW". To cut a yellow wire, the active bomb disposal expert must have one in their hand, point to a teammate’s wire, and say “This wire is yellow.” If they are correct, the 2 wires are cut. Otherwise if incorrect, just as with blue wires, an Info token that reveals the actual
value of the identified wire is placed, and the detonator dial advances 1 space.
- If a yellow wire is pointed at incorrectly, a yellow Info token is used, and the detonator dial advances 1 space.
- A Solo Cut action using yellow wires can occur only with a bomb disposal expert who has all the remaining yellow wires in their hand.

#### Character cards (Double Detector)

Each bomb disposal expert can use the personal equipment on their character card once per mission. To show that it has been used, flip it facedown. 

Double Detector: During a Dual Cut action, the active bomb disposal expert states a value and points to 2 wires
in a teammate’s stand (instead of only 1).
- If either of these 2 wires matches the stated value, the action succeeds.
  - If both wires are named correctly, the teammate does not share any details and simply chooses which of the 2 chosen wires to cut.
- If neither of the 2 wires matches the stated value, the action fails.
  - The detonator dial advances 1 space, and the teammate places 1 Info token on the table in front of 1 of the 2 chosen wires (their choice).
  - If only 1 of the 2 chosen wires is red, the bomb does not explode. The teammate does not share any details and simply places an Info token in front of the “not red” wire.

### End of the Game

The mission ends in success when all bomb disposal experts have empty tile stands!

If the mission ends in failure (red red wire cut or detonator dial advances to the space),
change which player is the Captain and restart the mission!

## Gameplay Tips

These are some useful tips / shortcuts that can be used in some of the calculations or can be used to make simplifications.

- If a player attempts and fails to cut a wire with a `dual cut` action, we know that they have at least one wire of that value.
- You can narrow down the possibilities of wires in someone's `tile stand` by referencing which wires are fully cut. You know that those numbers can not exist on someones stand.

## Code Architecture

### Repository Layout

```
bomb_busters.py              # Game model: enums, dataclasses, game state, actions
compute_probabilities.py     # Probability engine: constraint solver and API
tests/
  __init__.py
  test_bomb_busters.py       # 108 tests for the game model
  test_compute_probabilities.py  # 24 tests for the probability engine
CLAUDE.md                    # Project instructions for Claude
README.md                    # This file
Bomb Busters Rulebook.pdf    # Official rulebook
Bomb Busters FAQ.pdf         # Official FAQ
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

**`TileStand`** — A player's wire rack. Slots are always sorted ascending by `sort_value`. Wires stay in position even after being cut. Properties: `hidden_slots`, `cut_slots`, `is_empty`, `remaining_count`.

**`Player`** — A bomb disposal expert with a `TileStand` and optional `CharacterCard`.

**`CharacterCard`** — A one-use personal ability (e.g., Double Detector). Tracks `used` status.

**`Detonator`** — The bomb's failure counter. With N players, N-1 failures are tolerated. Tracks `failures`, `max_failures`, `is_exploded`, `remaining_failures`.

**`InfoTokenPool`** — Pool of 26 info tokens (2 per blue value 1-12, plus 2 yellow). Tokens are consumed on failed dual cuts.

**`Marker`** — Board marker for red/yellow wires in play. State is `KNOWN` (direct inclusion) or `UNCERTAIN` ("X of Y" selection mode).

**`Equipment`** — Extensible equipment card with `unlock_value` (unlocked when 2 wires of that value are cut) and `used` tracking.

**`WireConfig`** — Mission setup for colored wires. Specifies `count` wires in play, with optional `pool_size` for "X of Y" random selection.

#### Action Records

**`DualCutAction`** — Records actor, target, guessed value, result, and Double Detector details.

**`SoloCutAction`** — Records actor, value, slot indices, and wire count (2 or 4).

**`RevealRedAction`** — Records actor and revealed slot indices.

**`TurnHistory`** — Chronological list of all actions. Supports deduction queries like `failed_dual_cuts_by_player()`.

#### GameState

The central class managing the full game. Two factory methods:

- **`GameState.create_game(player_names, wire_configs, seed)`** — Simulation mode. Shuffles wires, deals evenly (captain + next player get extras if uneven), sorts stands, places markers. All wire identities are known.

- **`GameState.from_partial_state(...)`** — Calculator mode. Enter a mid-game state directly for probability calculations without replaying turns. Other players' hidden wires are set to `None`.

Action execution methods: `execute_dual_cut()`, `execute_solo_cut()`, `execute_reveal_red()`. Each validates the action, resolves outcomes, updates the detonator, places info tokens, checks validation tokens, and records to history.

### Probability Engine (`compute_probabilities.py`)

#### Knowledge Extraction

**`KnownInfo`** — Aggregates all information visible to the observing player: their own wires, all cut wires, info tokens, validation tokens, markers, and must-have deductions from turn history.

**`extract_known_info(game, observer_index)`** — Collects public + private knowledge into a `KnownInfo` object.

**`compute_unknown_pool(known, game)`** — Computes the set of wires whose locations are unknown: all wires in play minus observer's wires, minus other players' cut/revealed wires.

#### Constraint Solver

**`PositionConstraint`** — Defines valid `sort_value` bounds for a hidden slot based on known neighboring wires (from cut or info-revealed positions).

**`compute_position_probabilities(game, observer_index)`** — Backtracking solver that enumerates all valid wire-to-position assignments. Uses identical-wire grouping via `Counter` (e.g., four blue-5s are interchangeable) and sort-order pruning for efficiency. Returns `{(player_index, slot_index): Counter({Wire: count})}`.

#### High-Level API

| Function | Description |
|----------|-------------|
| `probability_of_dual_cut(game, observer, target_player, target_slot, value)` | Probability that a specific dual cut succeeds (0.0 to 1.0) |
| `probability_of_double_detector(game, observer, target_player, slot1, slot2, value)` | Joint probability for Double Detector (not naive independence) |
| `guaranteed_actions(game, observer)` | Find all 100% success actions (solo cuts, guaranteed dual cuts, reveal red) |
| `rank_all_moves(game, observer)` | Rank all possible moves by probability, sorted descending |

**`RankedMove`** — Dataclass for ranked results with `action_type`, target details, `guessed_value`, and `probability`. Supports `__str__` for display.