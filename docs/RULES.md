# Bomb Busters Rules

> See the [official rulebook](Bomb%20Busters%20Rulebook.pdf) and [FAQ](Bomb%20Busters%20FAQ.pdf) for the complete published rules.

Bomb Busters is a co-op game best played with 4-5 players. Each player is given a tile stand which contains sorted wire tiles that is only visible to the player the stand belongs to. The goal of the game is to successfully cut all the wires, thus defusing the bomb and passing the mission.

## Table of Contents

- [Game Components](#game-components)
- [Setup](#setup)
- [Game Play](#game-play)
  - [Dual Cut](#dual-cut)
  - [Solo Cut](#solo-cut)
  - [Reveal Your Red Wires](#reveal-your-red-wires)
  - [Validation Tokens](#validation-tokens)
  - [Yellow Wires](#yellow-wires)
  - [Character Cards (Double Detector)](#character-cards-double-detector)
- [End of the Game](#end-of-the-game)
- [Gameplay Tips](#gameplay-tips)

## Game Components

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

## Setup

1. Select a mission card to play.
2. Randomly shuffle all 48 blue tiles in addition to any red and/or yellow tiles as required by the mission card.
3. Designate a captain.
4. Deal the shuffled wires evenly, starting with the captain and going clockwise. For example, with 5 players and 52 total wires, the captain and the player to the captain's left will draw 11 wires, and all other players will draw 10 wires.
5. All players will sort all tiles on their rack in ascending order with no gaps between any tiles.
6. The captain will indicate first with an info token for any of their blue wires. Players indicate one blue wire in clockwise direction until all players have indicated once.
7. Play begins with the captain after all players have indicated. Each player must take one action per turn: dual cut, solo cut, or using equipment followed by a cut action.

## Game Play

Starting with the captain and going clockwise, each bomb disposal expert takes a turn. On their turn, the active bomb disposal expert must do one of the following 3 actions: Dual Cut, Solo Cut, or Reveal Your Red Wires.

### Dual Cut

The active bomb disposal expert must cut 2 identical wires: 1 of their own and 1 of their teammate's. They clearly point to a specific teammate's wire and guess what it is, stating its value (e.g., "This wire is a 9").

- If correct, the action succeeds:
  - The teammate places that wire faceup in front of their tile stand, without changing its position.
  - The active bomb disposal expert places their identical wire (or one of them if they have several) faceup in front of their tile stand.
- If wrong, the action fails:
  - If the wire is red, the bomb explodes and the mission ends in failure.
  - If the wire is blue or yellow, the detonator dial advances 1 space (the bomb explodes if the dial reaches the skull), and the teammate places an info token in front of the wire to show its real value.

### Solo Cut

If the last of identical wires still in the game appear only in the active bomb disposal expert's hand, they can cut those identical wires in pairs (either 2 or 4). This can be done on their own, without involving another bomb disposal expert.

- If they have a full set of 4, they can cut all 4 wires at once.
- If a pair of wires of a given value have already been cut, they can cut the remaining 2 matching wires in their hand.

Cut wires are placed faceup on the table in front of the tile stand.

### Reveal Your Red Wires

This action can occur only if the active bomb disposal expert's remaining uncut wires are all red. They reveal them, placing them faceup on the table in front of their tile stand.

### Validation Tokens

As soon as all 4 wires of the same value have been cut, place 1 validation token on the matching number on the board.

### Yellow Wires

Yellow wires are cut the same way as blue wires (Dual or Solo Cut), but the numeric value is used only when sorting the tiles on the stand in ascending order during setup. During the game, all yellow wires are considered to have the same value: "YELLOW". To cut a yellow wire, the active bomb disposal expert must have one in their hand, point to a teammate's wire, and say "This wire is yellow." If correct, the 2 wires are cut. If incorrect, an info token that reveals the actual value of the identified wire is placed, and the detonator dial advances 1 space.

- If a yellow wire is pointed at incorrectly, a yellow info token is used.
- A Solo Cut using yellow wires can occur only if the bomb disposal expert has all the remaining yellow wires in their hand.

### Character Cards (Double Detector)

Each bomb disposal expert can use the personal equipment on their character card once per mission. To show that it has been used, flip it facedown.

**Double Detector:** During a Dual Cut action, the active bomb disposal expert states a value and points to 2 wires on a teammate's stand (instead of only 1).

- If either of these 2 wires matches the stated value, the action succeeds.
  - If both wires match, the teammate simply chooses which of the 2 chosen wires to cut.
- If neither of the 2 wires matches the stated value, the action fails.
  - The detonator dial advances 1 space, and the teammate places 1 info token in front of 1 of the 2 chosen wires (their choice).
  - If only 1 of the 2 chosen wires is red, the bomb does not explode. The teammate places an info token in front of the non-red wire without sharing any details.

## End of the Game

The mission ends in success when all bomb disposal experts have empty tile stands.

If the mission ends in failure (red wire cut or detonator dial reaches the skull), change which player is the captain and restart the mission.

## Gameplay Tips

- If a player attempts and fails a dual cut, we know they have at least one wire of the guessed value. The probability engine uses this deduction automatically via the "must-have" constraint system.
- You can narrow down possibilities on someone's tile stand by referencing which values are fully validated (all 4 cut). Those numbers cannot exist on anyone's stand.
