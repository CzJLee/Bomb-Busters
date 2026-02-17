"""Bomb Busters game model.

Core classes representing the Bomb Busters board game components,
including wires, tile stands, players, and game state management.
Supports both full simulation mode and calculator/mid-game mode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import total_ordering
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# ANSI Color Constants
# =============================================================================

class _Colors:
    """ANSI escape codes for terminal coloring."""
    BLUE = "\033[94m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# =============================================================================
# Enums
# =============================================================================

class WireColor(Enum):
    """Color of a wire tile."""
    BLUE = auto()
    RED = auto()
    YELLOW = auto()

    def ansi(self) -> str:
        """Returns the ANSI color code for this wire color."""
        return {
            WireColor.BLUE: _Colors.BLUE,
            WireColor.RED: _Colors.RED,
            WireColor.YELLOW: _Colors.YELLOW,
        }[self]


class SlotState(Enum):
    """State of a slot on a tile stand."""
    HIDDEN = auto()
    CUT = auto()
    INFO_REVEALED = auto()


class ActionType(Enum):
    """Type of action a player can take on their turn."""
    DUAL_CUT = auto()
    SOLO_CUT = auto()
    REVEAL_RED = auto()


class ActionResult(Enum):
    """Result of a dual cut action."""
    SUCCESS = auto()
    FAIL_BLUE_YELLOW = auto()
    FAIL_RED = auto()


class MarkerState(Enum):
    """State of a board marker for red/yellow wires.

    KNOWN: blank side up — this wire value is definitely in play.
    UNCERTAIN: '?' side up — this wire value might be in play (X of Y mode).
    """
    KNOWN = auto()
    UNCERTAIN = auto()


# =============================================================================
# Wire
# =============================================================================

@total_ordering
@dataclass(frozen=True)
class Wire:
    """A physical wire tile in the game.

    Attributes:
        color: The color of the wire (BLUE, RED, or YELLOW).
        sort_value: The numeric value printed on the tile, used for sorting.
            Blue N -> N.0, Yellow N.1 -> N.1, Red N.5 -> N.5.
            This encoding naturally gives the correct ascending sort order.
    """
    color: WireColor
    sort_value: float

    @property
    def base_number(self) -> int:
        """The integer part of the sort value.

        Returns:
            The base number (e.g., 5 for blue-5, yellow-5.1, red-5.5).
        """
        return int(self.sort_value)

    @property
    def gameplay_value(self) -> int | str:
        """The value used during gameplay for cutting.

        Returns:
            int 1-12 for blue wires, 'YELLOW' for yellow, 'RED' for red.
        """
        if self.color == WireColor.BLUE:
            return self.base_number
        elif self.color == WireColor.YELLOW:
            return "YELLOW"
        else:
            return "RED"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Wire):
            return NotImplemented
        return self.sort_value < other.sort_value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Wire):
            return NotImplemented
        return self.color == other.color and self.sort_value == other.sort_value

    def __hash__(self) -> int:
        return hash((self.color, self.sort_value))

    def __str__(self) -> str:
        color = self.color.ansi()
        if self.color == WireColor.BLUE:
            label = str(self.base_number)
        elif self.color == WireColor.YELLOW:
            label = "Y"
        else:
            label = "R"
        return f"{color}{label}{_Colors.RESET}"

    def __repr__(self) -> str:
        return f"Wire({self.color.name}, {self.sort_value})"


# =============================================================================
# Wire Factory Functions
# =============================================================================

def create_all_blue_wires() -> list[Wire]:
    """Create all 48 blue wire tiles (4 copies each of values 1-12).

    Returns:
        List of 48 blue Wire objects.
    """
    wires = []
    for number in range(1, 13):
        for _ in range(4):
            wires.append(Wire(WireColor.BLUE, float(number)))
    return wires


def create_all_red_wires() -> list[Wire]:
    """Create all 11 red wire tiles (1.5 through 11.5).

    Returns:
        List of 11 red Wire objects.
    """
    return [Wire(WireColor.RED, n + 0.5) for n in range(1, 12)]


def create_all_yellow_wires() -> list[Wire]:
    """Create all 11 yellow wire tiles (1.1 through 11.1).

    Returns:
        List of 11 yellow Wire objects.
    """
    return [Wire(WireColor.YELLOW, n + 0.1) for n in range(1, 12)]


# =============================================================================
# Slot
# =============================================================================

@dataclass
class Slot:
    """A single position on a tile stand.

    Attributes:
        wire: The wire at this position, or None if unknown (calculator mode).
        state: Current state of the slot (HIDDEN, CUT, or INFO_REVEALED).
        info_token: Value shown by an info token after a failed dual cut.
            int 1-12 for blue wires, 'YELLOW' for yellow wires, None if no token.
    """
    wire: Wire | None
    state: SlotState = SlotState.HIDDEN
    info_token: int | str | None = None

    @property
    def is_cut(self) -> bool:
        """Whether this wire has been cut."""
        return self.state == SlotState.CUT

    @property
    def is_hidden(self) -> bool:
        """Whether this wire is still face-down (not yet cut or revealed)."""
        return self.state == SlotState.HIDDEN

    @property
    def is_info_revealed(self) -> bool:
        """Whether this wire has an info token placed on it."""
        return self.state == SlotState.INFO_REVEALED

    def __str__(self) -> str:
        if self.state == SlotState.HIDDEN:
            if self.wire is not None:
                # Simulation mode: show the actual wire dimmed
                return f"{_Colors.DIM}[{self.wire}]{_Colors.RESET}"
            return f"{_Colors.DIM}[?]{_Colors.RESET}"
        elif self.state == SlotState.CUT:
            if self.wire is not None:
                return f"{_Colors.GREEN}✓{self.wire}{_Colors.RESET}"
            return f"{_Colors.GREEN}✓?{_Colors.RESET}"
        else:  # INFO_REVEALED
            token_str = str(self.info_token) if self.info_token is not None else "?"
            if self.wire is not None:
                return f"{_Colors.BOLD}i:{token_str}{_Colors.RESET}"
            return f"{_Colors.BOLD}i:{token_str}{_Colors.RESET}"


# =============================================================================
# TileStand
# =============================================================================

@dataclass
class TileStand:
    """A player's tile stand containing sorted wire tiles.

    Wires are always sorted in ascending order by sort_value.
    Wires remain in their position even after being cut.

    Attributes:
        slots: List of Slot objects, sorted ascending by wire sort_value.
    """
    slots: list[Slot]

    @classmethod
    def from_wires(cls, wires: list[Wire]) -> TileStand:
        """Create a tile stand from a list of wires, sorted ascending.

        Args:
            wires: List of Wire objects to place on the stand.

        Returns:
            A new TileStand with slots sorted by sort_value.
        """
        sorted_wires = sorted(wires)
        slots = [Slot(wire=w) for w in sorted_wires]
        return cls(slots=slots)

    @property
    def hidden_slots(self) -> list[tuple[int, Slot]]:
        """All slots that are still face-down (not cut or info-revealed).

        Returns:
            List of (index, Slot) tuples for hidden slots.
        """
        return [(i, s) for i, s in enumerate(self.slots) if s.state == SlotState.HIDDEN]

    @property
    def cut_slots(self) -> list[tuple[int, Slot]]:
        """All slots that have been cut.

        Returns:
            List of (index, Slot) tuples for cut slots.
        """
        return [(i, s) for i, s in enumerate(self.slots) if s.state == SlotState.CUT]

    @property
    def info_revealed_slots(self) -> list[tuple[int, Slot]]:
        """All slots that have info tokens.

        Returns:
            List of (index, Slot) tuples for info-revealed slots.
        """
        return [
            (i, s) for i, s in enumerate(self.slots)
            if s.state == SlotState.INFO_REVEALED
        ]

    @property
    def is_empty(self) -> bool:
        """Whether all wires have been cut or revealed (stand is clear).

        Returns:
            True if no hidden slots remain.
        """
        return all(s.state != SlotState.HIDDEN for s in self.slots)

    @property
    def remaining_count(self) -> int:
        """Number of wires still hidden on this stand.

        Returns:
            Count of hidden slots.
        """
        return sum(1 for s in self.slots if s.state == SlotState.HIDDEN)

    def cut_wire_at(self, index: int) -> None:
        """Mark the wire at the given index as cut.

        The wire remains in position but is flipped face-up.

        Args:
            index: The slot index to mark as cut.

        Raises:
            IndexError: If index is out of range.
            ValueError: If the slot is already cut.
        """
        if index < 0 or index >= len(self.slots):
            raise IndexError(f"Slot index {index} out of range (0-{len(self.slots) - 1})")
        if self.slots[index].is_cut:
            raise ValueError(f"Slot {index} is already cut")
        self.slots[index].state = SlotState.CUT

    def place_info_token(self, index: int, value: int | str) -> None:
        """Place an info token on a slot after a failed dual cut.

        Args:
            index: The slot index to place the token on.
            value: The revealed value (int 1-12 for blue, 'YELLOW' for yellow).

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.slots):
            raise IndexError(f"Slot index {index} out of range (0-{len(self.slots) - 1})")
        self.slots[index].state = SlotState.INFO_REVEALED
        self.slots[index].info_token = value

    def __str__(self) -> str:
        parts = []
        for i, slot in enumerate(self.slots):
            parts.append(f"{i}:{slot}")
        return " ".join(parts)


# =============================================================================
# Helper: Sort Value Bounds
# =============================================================================

def get_sort_value_bounds(
    slots: list[Slot], slot_index: int
) -> tuple[float, float]:
    """Get the sort_value bounds for a hidden slot based on known neighbors.

    Scans left and right from the given slot index to find the nearest
    known wire (CUT or INFO_REVEALED with a known wire) to establish
    the valid range of sort_values for the hidden slot.

    Args:
        slots: The list of slots from a tile stand.
        slot_index: The index of the hidden slot to compute bounds for.

    Returns:
        A tuple (lower_bound, upper_bound) where the hidden wire's
        sort_value must be strictly between these values.
        Defaults to (0.0, 13.0) if no known neighbors are found.
    """
    lower = 0.0
    upper = 13.0

    # Scan left for nearest known wire
    for i in range(slot_index - 1, -1, -1):
        wire = slots[i].wire
        if wire is not None:
            lower = wire.sort_value
            break

    # Scan right for nearest known wire
    for i in range(slot_index + 1, len(slots)):
        wire = slots[i].wire
        if wire is not None:
            upper = wire.sort_value
            break

    return (lower, upper)


# =============================================================================
# CharacterCard
# =============================================================================

@dataclass
class CharacterCard:
    """A character card with personal equipment.

    Each player has one character card with a unique ability that can
    be used once per mission.

    Attributes:
        name: The name of the character card ability.
        description: A brief description of what the ability does.
        used: Whether the ability has been used this mission.
    """
    name: str
    description: str
    used: bool = False

    def use(self) -> None:
        """Mark this character card ability as used.

        Raises:
            ValueError: If the ability has already been used.
        """
        if self.used:
            raise ValueError(f"Character card '{self.name}' has already been used")
        self.used = True

    def __str__(self) -> str:
        status = f"{_Colors.DIM}(used){_Colors.RESET}" if self.used else f"{_Colors.GREEN}(available){_Colors.RESET}"
        return f"{self.name} {status}"


def create_double_detector() -> CharacterCard:
    """Create a Double Detector character card.

    Returns:
        A CharacterCard configured as a Double Detector.
    """
    return CharacterCard(
        name="Double Detector",
        description=(
            "During a Dual Cut, state a value and point to 2 wires "
            "on a teammate's stand. Succeeds if either matches."
        ),
    )


# =============================================================================
# Player
# =============================================================================

@dataclass
class Player:
    """A bomb disposal expert in the game.

    Attributes:
        name: The player's name.
        tile_stand: The player's tile stand with their wires.
        character_card: The player's character card, or None.
    """
    name: str
    tile_stand: TileStand
    character_card: CharacterCard | None = None

    @property
    def is_finished(self) -> bool:
        """Whether this player has no remaining hidden wires.

        Returns:
            True if the tile stand is empty (all wires cut/revealed).
        """
        return self.tile_stand.is_empty

    def __str__(self) -> str:
        card_str = f"  Card: {self.character_card}" if self.character_card else ""
        return f"{_Colors.BOLD}{self.name}{_Colors.RESET}{card_str}\n  Stand: {self.tile_stand}"


# =============================================================================
# Detonator
# =============================================================================

@dataclass
class Detonator:
    """The bomb's detonator dial.

    Tracks failures and determines when the bomb explodes.
    With N players, N-1 failures are tolerated before the bomb explodes.

    Attributes:
        failures: Current number of failures (starts at 0).
        max_failures: Maximum failures before explosion (player_count - 1).
    """
    failures: int = 0
    max_failures: int = 4

    @property
    def is_exploded(self) -> bool:
        """Whether the bomb has exploded from too many failures.

        Returns:
            True if failures have reached the maximum.
        """
        return self.failures >= self.max_failures

    def advance(self) -> bool:
        """Advance the detonator dial by one space.

        Returns:
            True if the bomb explodes (dial reached the skull).
        """
        self.failures += 1
        return self.is_exploded

    @property
    def remaining_failures(self) -> int:
        """Number of additional failures before the bomb explodes.

        Returns:
            Non-negative integer of remaining tolerable failures.
        """
        return max(0, self.max_failures - self.failures)

    def __str__(self) -> str:
        filled = "✕" * self.failures
        empty = "○" * self.remaining_failures
        skull = "💀" if self.is_exploded else ""
        return f"Detonator: [{filled}{empty}] {self.failures}/{self.max_failures} {skull}"


# =============================================================================
# InfoTokenPool
# =============================================================================

@dataclass
class InfoTokenPool:
    """Pool of available info tokens.

    There are 2 tokens per blue number (1-12) and 2 yellow tokens,
    totaling 26 tokens.

    Attributes:
        blue_tokens: Dict mapping blue values (1-12) to available count.
        yellow_tokens: Number of available yellow info tokens.
    """
    blue_tokens: dict[int, int] = field(default_factory=dict)
    yellow_tokens: int = 2

    @classmethod
    def create_full(cls) -> InfoTokenPool:
        """Create a full pool of 26 info tokens.

        Returns:
            An InfoTokenPool with 2 of each blue (1-12) and 2 yellow.
        """
        return cls(
            blue_tokens={n: 2 for n in range(1, 13)},
            yellow_tokens=2,
        )

    def use_blue_token(self, value: int) -> bool:
        """Use a blue info token of the given value.

        Args:
            value: The blue wire value (1-12).

        Returns:
            True if a token was available and used, False if none available.
        """
        if self.blue_tokens.get(value, 0) > 0:
            self.blue_tokens[value] -= 1
            return True
        return False

    def use_yellow_token(self) -> bool:
        """Use a yellow info token.

        Returns:
            True if a token was available and used, False if none available.
        """
        if self.yellow_tokens > 0:
            self.yellow_tokens -= 1
            return True
        return False

    def __str__(self) -> str:
        used_blue = sum(2 - v for v in self.blue_tokens.values())
        used_yellow = 2 - self.yellow_tokens
        return f"Info tokens: {used_blue}/24 blue used, {used_yellow}/2 yellow used"


# =============================================================================
# Marker
# =============================================================================

@dataclass
class Marker:
    """A board marker indicating a red or yellow wire value in play.

    Attributes:
        color: The wire color this marker represents (RED or YELLOW).
        sort_value: The sort_value of the wire this marker represents.
        state: KNOWN (definitely in play) or UNCERTAIN (might be in play).
    """
    color: WireColor
    sort_value: float
    state: MarkerState

    @property
    def base_number(self) -> int:
        """The integer part of the sort value."""
        return int(self.sort_value)

    def __str__(self) -> str:
        color = self.color.ansi()
        state_str = "?" if self.state == MarkerState.UNCERTAIN else "✓"
        return f"{color}{self.base_number}({state_str}){_Colors.RESET}"


# =============================================================================
# Equipment
# =============================================================================

@dataclass
class Equipment:
    """An equipment card that provides a special ability.

    Equipment cards become usable when 2 wires of the card's unlock_value
    have been cut. Each equipment card can only be used once per mission.

    Attributes:
        name: The name of the equipment.
        description: What the equipment does.
        unlock_value: The blue wire value (1-12) that unlocks this equipment
            (2 wires of this value must be cut).
        used: Whether this equipment has been used.
        unlocked: Whether this equipment is currently unlocked.
    """
    name: str
    description: str
    unlock_value: int
    used: bool = False
    unlocked: bool = False

    def unlock(self) -> None:
        """Unlock this equipment (called when 2 wires of unlock_value are cut)."""
        self.unlocked = True

    def use(self) -> None:
        """Use this equipment.

        Raises:
            ValueError: If the equipment is not unlocked or already used.
        """
        if not self.unlocked:
            raise ValueError(f"Equipment '{self.name}' is not yet unlocked")
        if self.used:
            raise ValueError(f"Equipment '{self.name}' has already been used")
        self.used = True

    def __str__(self) -> str:
        if self.used:
            status = f"{_Colors.DIM}(used){_Colors.RESET}"
        elif self.unlocked:
            status = f"{_Colors.GREEN}(ready){_Colors.RESET}"
        else:
            status = f"{_Colors.DIM}(locked: need 2×{self.unlock_value}){_Colors.RESET}"
        return f"{self.name} {status}"


# =============================================================================
# Action Records
# =============================================================================

@dataclass(frozen=True)
class DualCutAction:
    """Record of a dual cut action.

    Attributes:
        actor_index: Index of the active player who performed the cut.
        target_player_index: Index of the targeted teammate.
        target_slot_index: Slot index on the target's stand.
        guessed_value: The value guessed (int 1-12 or 'YELLOW').
        result: The outcome of the action.
        actual_wire: The wire at the target slot (revealed on resolution).
        actor_cut_slot_index: Which slot the actor cut from their own hand
            (only on success).
        is_double_detector: Whether the Double Detector was used.
        second_target_slot_index: Second slot if Double Detector was used.
    """
    actor_index: int
    target_player_index: int
    target_slot_index: int
    guessed_value: int | str
    result: ActionResult
    actual_wire: Wire | None = None
    actor_cut_slot_index: int | None = None
    is_double_detector: bool = False
    second_target_slot_index: int | None = None


@dataclass(frozen=True)
class SoloCutAction:
    """Record of a solo cut action.

    Attributes:
        actor_index: Index of the player who performed the solo cut.
        value: The wire value being cut (int 1-12 or 'YELLOW').
        slot_indices: Which slots on the actor's stand were cut.
        wire_count: Number of wires cut (2 or 4).
    """
    actor_index: int
    value: int | str
    slot_indices: tuple[int, ...]
    wire_count: int


@dataclass(frozen=True)
class RevealRedAction:
    """Record of a reveal red wires action.

    Attributes:
        actor_index: Index of the player who revealed their red wires.
        slot_indices: Which slots were revealed as red.
    """
    actor_index: int
    slot_indices: tuple[int, ...]


ActionRecord = DualCutAction | SoloCutAction | RevealRedAction


# =============================================================================
# TurnHistory
# =============================================================================

@dataclass
class TurnHistory:
    """Record of all actions taken during the game.

    Useful for deduction: e.g., a failed dual cut reveals that the actor
    has at least one wire of the guessed value.

    Attributes:
        actions: List of all action records in chronological order.
    """
    actions: list[ActionRecord] = field(default_factory=list)

    def record(self, action: ActionRecord) -> None:
        """Record a new action.

        Args:
            action: The action record to add to history.
        """
        self.actions.append(action)

    def failed_dual_cuts_by_player(self, player_index: int) -> list[DualCutAction]:
        """Get all failed dual cuts where this player was the actor.

        A failed dual cut implies the actor has at least one wire of the
        guessed value (since dual cut requires cutting your own matching wire).

        Args:
            player_index: The player index to query.

        Returns:
            List of DualCutAction records where this player failed.
        """
        return [
            a for a in self.actions
            if isinstance(a, DualCutAction)
            and a.actor_index == player_index
            and a.result == ActionResult.FAIL_BLUE_YELLOW
        ]

    def __str__(self) -> str:
        if not self.actions:
            return "No actions taken yet."
        lines = []
        for i, action in enumerate(self.actions):
            if isinstance(action, DualCutAction):
                dd = " (DD)" if action.is_double_detector else ""
                result_str = "✓" if action.result == ActionResult.SUCCESS else "✗"
                lines.append(
                    f"  {i + 1}. P{action.actor_index} → P{action.target_player_index}"
                    f"[{action.target_slot_index}] guess={action.guessed_value}"
                    f"{dd} {result_str}"
                )
            elif isinstance(action, SoloCutAction):
                lines.append(
                    f"  {i + 1}. P{action.actor_index} solo cut"
                    f" {action.value} ×{action.wire_count}"
                )
            elif isinstance(action, RevealRedAction):
                lines.append(
                    f"  {i + 1}. P{action.actor_index} revealed red wires"
                )
        return "Turn History:\n" + "\n".join(lines)


# =============================================================================
# WireConfig
# =============================================================================

@dataclass
class WireConfig:
    """Configuration for including colored wires in a mission.

    Attributes:
        color: The wire color (RED or YELLOW).
        count: Number of wires actually shuffled into play.
        pool_size: Total wires drawn before selection. None for direct
            inclusion (all markers KNOWN). If set, enables 'X of Y' mode
            where pool_size wires are drawn, count are kept, and the rest
            are set aside (markers are UNCERTAIN).
    """
    color: WireColor
    count: int
    pool_size: int | None = None

    def __post_init__(self) -> None:
        """Validate wire config constraints."""
        if self.color == WireColor.BLUE:
            raise ValueError("WireConfig is only for RED and YELLOW wires")
        if self.color == WireColor.YELLOW and not (0 <= self.count <= 4):
            raise ValueError(f"Yellow wire count must be 0-4, got {self.count}")
        if self.color == WireColor.RED and not (0 <= self.count <= 3):
            raise ValueError(f"Red wire count must be 0-3, got {self.count}")
        if self.pool_size is not None:
            if self.pool_size < self.count:
                raise ValueError(
                    f"Pool size ({self.pool_size}) must be >= count ({self.count})"
                )
            max_pool = 11  # Maximum available wires of any color
            if self.pool_size > max_pool:
                raise ValueError(
                    f"Pool size ({self.pool_size}) exceeds available wires ({max_pool})"
                )


# =============================================================================
# GameState
# =============================================================================

@dataclass
class GameState:
    """The complete state of a Bomb Busters game.

    Supports two modes:
    - Simulation mode (via create_game): all wires are known.
    - Calculator mode (via from_partial_state): only observable info is set,
      other players' hidden wires are None.

    Attributes:
        players: List of all players in the game.
        detonator: The detonator dial tracking failures.
        info_token_pool: Available info tokens.
        validation_tokens: Set of blue wire values (1-12) where all 4 are cut.
        markers: Board markers for red/yellow wires in play.
        equipment: List of equipment cards in the game.
        history: Record of all actions taken.
        current_player_index: Index of the player whose turn it is.
        game_over: Whether the game has ended.
        game_won: Whether the game was won (all stands empty).
        wires_in_play: All wire objects that were shuffled into the game.
    """
    players: list[Player]
    detonator: Detonator
    info_token_pool: InfoTokenPool
    validation_tokens: set[int]
    markers: list[Marker]
    equipment: list[Equipment]
    history: TurnHistory
    current_player_index: int = 0
    game_over: bool = False
    game_won: bool = False
    wires_in_play: list[Wire] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Factory: Full Simulation Mode
    # -----------------------------------------------------------------

    @classmethod
    def create_game(
        cls,
        player_names: list[str],
        wire_configs: list[WireConfig] | None = None,
        seed: int | None = None,
    ) -> GameState:
        """Create a new game in full simulation mode.

        Shuffles all wires, deals them evenly to players, and sets up
        the board. The captain is player index 0.

        Args:
            player_names: Names of the players (4-5 players).
            wire_configs: Optional list of WireConfig for red/yellow wires.
                If None, only blue wires are used.
            seed: Optional random seed for reproducibility.

        Returns:
            A fully initialized GameState.

        Raises:
            ValueError: If player count is not 4-5.
        """
        if seed is not None:
            random.seed(seed)

        player_count = len(player_names)
        if not (4 <= player_count <= 5):
            raise ValueError(f"Player count must be 4-5, got {player_count}")

        # Build wire pool
        pool: list[Wire] = create_all_blue_wires()
        markers: list[Marker] = []

        if wire_configs:
            for config in wire_configs:
                all_colored = (
                    create_all_red_wires()
                    if config.color == WireColor.RED
                    else create_all_yellow_wires()
                )

                if config.pool_size is not None:
                    # "X of Y" mode: draw pool_size, keep count, set aside rest
                    drawn = random.sample(all_colored, config.pool_size)
                    # All drawn wires get UNCERTAIN markers
                    for w in drawn:
                        markers.append(
                            Marker(config.color, w.sort_value, MarkerState.UNCERTAIN)
                        )
                    # Shuffle drawn wires, keep count, discard rest
                    random.shuffle(drawn)
                    kept = drawn[:config.count]
                    pool.extend(kept)
                else:
                    # Direct inclusion: draw exactly count wires
                    drawn = random.sample(all_colored, config.count)
                    for w in drawn:
                        markers.append(
                            Marker(config.color, w.sort_value, MarkerState.KNOWN)
                        )
                    pool.extend(drawn)

        # Record all wires in play before shuffling
        wires_in_play = list(pool)

        # Shuffle and deal
        random.shuffle(pool)
        total_wires = len(pool)
        base_count = total_wires // player_count
        extra = total_wires % player_count

        hands: list[list[Wire]] = []
        idx = 0
        for i in range(player_count):
            count = base_count + (1 if i < extra else 0)
            hands.append(pool[idx:idx + count])
            idx += count

        # Create players with Double Detector character cards
        players = [
            Player(
                name=player_names[i],
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(player_count)
        ]

        return cls(
            players=players,
            detonator=Detonator(max_failures=player_count - 1),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=markers,
            equipment=[],
            history=TurnHistory(),
            wires_in_play=wires_in_play,
        )

    # -----------------------------------------------------------------
    # Factory: Calculator / Mid-Game Mode
    # -----------------------------------------------------------------

    @classmethod
    def from_partial_state(
        cls,
        player_names: list[str],
        stands: list[list[Slot]],
        detonator_failures: int = 0,
        validation_tokens: set[int] | None = None,
        markers: list[Marker] | None = None,
        equipment: list[Equipment] | None = None,
        wires_in_play: list[Wire] | None = None,
        character_cards: list[CharacterCard | None] | None = None,
        history: TurnHistory | None = None,
    ) -> GameState:
        """Create a game state from partial mid-game information.

        Use this to enter a game state during a real game for probability
        calculations, without needing to replay all turns.

        Args:
            player_names: Names of all players.
            stands: List of slot lists, one per player. Each slot should
                have its state set (HIDDEN, CUT, or INFO_REVEALED) and
                wire set for known wires (the observer's own hand, cut
                wires, etc.) or None for unknown hidden wires.
            detonator_failures: Current number of detonator failures.
            validation_tokens: Set of fully-cut blue values (1-12).
            markers: Board markers for red/yellow wires.
            equipment: Equipment cards in play.
            wires_in_play: All wires that were included in this mission.
            character_cards: Character card for each player (or None).
            history: Optional turn history for deduction.

        Returns:
            A GameState initialized from the provided partial information.
        """
        player_count = len(player_names)
        if len(stands) != player_count:
            raise ValueError(
                f"Number of stands ({len(stands)}) must match "
                f"number of players ({player_count})"
            )

        cards = character_cards or [None] * player_count
        players = [
            Player(
                name=player_names[i],
                tile_stand=TileStand(slots=stands[i]),
                character_card=cards[i] if i < len(cards) else None,
            )
            for i in range(player_count)
        ]

        return cls(
            players=players,
            detonator=Detonator(
                failures=detonator_failures,
                max_failures=player_count - 1,
            ),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=validation_tokens or set(),
            markers=markers or [],
            equipment=equipment or [],
            history=history or TurnHistory(),
            wires_in_play=wires_in_play or [],
        )

    # -----------------------------------------------------------------
    # Action Execution
    # -----------------------------------------------------------------

    def execute_dual_cut(
        self,
        target_player_index: int,
        target_slot_index: int,
        guessed_value: int | str,
        is_double_detector: bool = False,
        second_target_slot_index: int | None = None,
    ) -> DualCutAction:
        """Execute a dual cut action by the current player.

        The active player guesses the value of a wire on a teammate's stand.
        If correct, both the target wire and one matching wire from the
        actor's hand are cut.

        Args:
            target_player_index: Index of the targeted teammate.
            target_slot_index: Slot index on the target's stand.
            guessed_value: The value being guessed (int 1-12 or 'YELLOW').
            is_double_detector: Whether the Double Detector is being used.
            second_target_slot_index: Second slot index if using Double Detector.

        Returns:
            A DualCutAction record describing the outcome.

        Raises:
            ValueError: If the action is invalid (wrong turn, no matching
                wire in hand, invalid target, etc.).
        """
        if self.game_over:
            raise ValueError("Game is already over")

        actor_index = self.current_player_index
        actor = self.players[actor_index]
        target = self.players[target_player_index]

        if target_player_index == actor_index:
            raise ValueError("Cannot dual cut your own wires")

        # Validate actor has a matching wire
        actor_matching_slots = [
            (i, s) for i, s in enumerate(actor.tile_stand.slots)
            if s.is_hidden and s.wire is not None
            and s.wire.gameplay_value == guessed_value
        ]
        if not actor_matching_slots:
            raise ValueError(
                f"Active player has no hidden wire with value {guessed_value}"
            )

        # Validate target slot
        target_slot = target.tile_stand.slots[target_slot_index]
        if not target_slot.is_hidden:
            raise ValueError(f"Target slot {target_slot_index} is not hidden")

        # Handle Double Detector
        if is_double_detector:
            if actor.character_card is None:
                raise ValueError("Player has no character card")
            if actor.character_card.name != "Double Detector":
                raise ValueError("Player's character card is not a Double Detector")
            if actor.character_card.used:
                raise ValueError("Double Detector has already been used")
            if second_target_slot_index is None:
                raise ValueError("Double Detector requires a second target slot")
            second_slot = target.tile_stand.slots[second_target_slot_index]
            if not second_slot.is_hidden:
                raise ValueError(
                    f"Second target slot {second_target_slot_index} is not hidden"
                )
            # Cannot use DD with YELLOW or RED guesses
            if isinstance(guessed_value, str):
                raise ValueError(
                    "Double Detector can only be used with blue values (1-12)"
                )
            actor.character_card.use()

        # Resolve the cut
        target_wire = target_slot.wire
        if target_wire is None:
            raise ValueError("Target wire is unknown (calculator mode)")

        # Check for match
        match_primary = target_wire.gameplay_value == guessed_value
        match_secondary = False
        if is_double_detector and second_target_slot_index is not None:
            second_wire = target.tile_stand.slots[second_target_slot_index].wire
            if second_wire is not None:
                match_secondary = second_wire.gameplay_value == guessed_value

        if match_primary or match_secondary:
            # SUCCESS
            # Determine which target slot to cut (primary preferred,
            # or secondary if only secondary matched)
            if match_primary:
                cut_target_index = target_slot_index
            else:
                assert second_target_slot_index is not None
                cut_target_index = second_target_slot_index

            target.tile_stand.cut_wire_at(cut_target_index)

            # Cut one matching wire from actor's hand
            actor_cut_index = actor_matching_slots[0][0]
            actor.tile_stand.cut_wire_at(actor_cut_index)

            # Check for validation tokens
            if isinstance(guessed_value, int):
                self._check_validation(guessed_value)

            action = DualCutAction(
                actor_index=actor_index,
                target_player_index=target_player_index,
                target_slot_index=target_slot_index,
                guessed_value=guessed_value,
                result=ActionResult.SUCCESS,
                actual_wire=target_wire,
                actor_cut_slot_index=actor_cut_index,
                is_double_detector=is_double_detector,
                second_target_slot_index=second_target_slot_index,
            )
        else:
            # FAILURE
            if target_wire.color == WireColor.RED:
                # Check Double Detector: if DD and one of the two is red
                # but the other is not, bomb doesn't explode
                if is_double_detector and second_target_slot_index is not None:
                    second_wire = target.tile_stand.slots[second_target_slot_index].wire
                    both_red = (
                        target_wire.color == WireColor.RED
                        and second_wire is not None
                        and second_wire.color == WireColor.RED
                    )
                    if both_red:
                        # Both red → bomb explodes
                        self.game_over = True
                        action = DualCutAction(
                            actor_index=actor_index,
                            target_player_index=target_player_index,
                            target_slot_index=target_slot_index,
                            guessed_value=guessed_value,
                            result=ActionResult.FAIL_RED,
                            actual_wire=target_wire,
                            is_double_detector=True,
                            second_target_slot_index=second_target_slot_index,
                        )
                        self.history.record(action)
                        return action
                    else:
                        # Only one red → info token on the non-red wire
                        exploded = self.detonator.advance()
                        # Place info token on the non-red wire
                        if target_wire.color != WireColor.RED:
                            non_red_index = target_slot_index
                            non_red_wire = target_wire
                        else:
                            non_red_index = second_target_slot_index
                            non_red_wire = second_wire

                        if non_red_wire is not None:
                            token_val = non_red_wire.gameplay_value
                            if isinstance(token_val, int):
                                self.info_token_pool.use_blue_token(token_val)
                            else:
                                self.info_token_pool.use_yellow_token()
                            target.tile_stand.place_info_token(
                                non_red_index, token_val
                            )

                        if exploded:
                            self.game_over = True

                        action = DualCutAction(
                            actor_index=actor_index,
                            target_player_index=target_player_index,
                            target_slot_index=target_slot_index,
                            guessed_value=guessed_value,
                            result=ActionResult.FAIL_BLUE_YELLOW,
                            actual_wire=target_wire,
                            is_double_detector=True,
                            second_target_slot_index=second_target_slot_index,
                        )
                        self.history.record(action)
                        self._advance_turn()
                        return action
                else:
                    # Normal dual cut hit red → bomb explodes
                    self.game_over = True
                    action = DualCutAction(
                        actor_index=actor_index,
                        target_player_index=target_player_index,
                        target_slot_index=target_slot_index,
                        guessed_value=guessed_value,
                        result=ActionResult.FAIL_RED,
                        actual_wire=target_wire,
                        is_double_detector=is_double_detector,
                        second_target_slot_index=second_target_slot_index,
                    )
                    self.history.record(action)
                    return action
            else:
                # Failed on blue or yellow wire
                exploded = self.detonator.advance()

                # Place info token
                token_val = target_wire.gameplay_value
                if isinstance(token_val, int):
                    self.info_token_pool.use_blue_token(token_val)
                else:
                    self.info_token_pool.use_yellow_token()
                target.tile_stand.place_info_token(
                    target_slot_index, token_val
                )

                if exploded:
                    self.game_over = True

                action = DualCutAction(
                    actor_index=actor_index,
                    target_player_index=target_player_index,
                    target_slot_index=target_slot_index,
                    guessed_value=guessed_value,
                    result=ActionResult.FAIL_BLUE_YELLOW,
                    actual_wire=target_wire,
                    is_double_detector=is_double_detector,
                    second_target_slot_index=second_target_slot_index,
                )

        self.history.record(action)
        if not self.game_over:
            self._check_win()
            self._advance_turn()
        return action

    def execute_solo_cut(
        self,
        value: int | str,
        slot_indices: list[int],
    ) -> SoloCutAction:
        """Execute a solo cut action by the current player.

        The active player cuts 2 or 4 identical wires from their own hand,
        provided they hold all remaining wires of that value.

        Args:
            value: The wire value to solo cut (int 1-12 or 'YELLOW').
            slot_indices: List of slot indices to cut (must be 2 or 4).

        Returns:
            A SoloCutAction record.

        Raises:
            ValueError: If the solo cut is invalid.
        """
        if self.game_over:
            raise ValueError("Game is already over")

        actor_index = self.current_player_index
        actor = self.players[actor_index]

        if len(slot_indices) not in (2, 4):
            raise ValueError(f"Solo cut must cut 2 or 4 wires, got {len(slot_indices)}")

        # Validate all specified slots have matching wires
        for idx in slot_indices:
            slot = actor.tile_stand.slots[idx]
            if not slot.is_hidden:
                raise ValueError(f"Slot {idx} is not hidden")
            if slot.wire is None:
                raise ValueError(f"Slot {idx} has unknown wire (calculator mode)")
            if slot.wire.gameplay_value != value:
                raise ValueError(
                    f"Slot {idx} has value {slot.wire.gameplay_value}, expected {value}"
                )

        # Validate that all remaining wires of this value are in the actor's hand
        if not self.can_solo_cut(actor_index, value):
            raise ValueError(
                f"Cannot solo cut {value}: not all remaining wires "
                f"of this value are in your hand"
            )

        # Execute the cuts
        for idx in slot_indices:
            actor.tile_stand.cut_wire_at(idx)

        # Check validation tokens (blue wires only)
        if isinstance(value, int):
            self._check_validation(value)

        action = SoloCutAction(
            actor_index=actor_index,
            value=value,
            slot_indices=tuple(slot_indices),
            wire_count=len(slot_indices),
        )
        self.history.record(action)
        self._check_win()
        if not self.game_over:
            self._advance_turn()
        return action

    def execute_reveal_red(self) -> RevealRedAction:
        """Execute a reveal red wires action by the current player.

        Can only be done when all remaining hidden wires are RED.

        Returns:
            A RevealRedAction record.

        Raises:
            ValueError: If the action is invalid (not all remaining wires are red).
        """
        if self.game_over:
            raise ValueError("Game is already over")

        actor_index = self.current_player_index
        actor = self.players[actor_index]
        hidden = actor.tile_stand.hidden_slots

        if not hidden:
            raise ValueError("No hidden wires to reveal")

        # Check all hidden wires are red
        for idx, slot in hidden:
            if slot.wire is None:
                raise ValueError("Unknown wire in hand (calculator mode)")
            if slot.wire.color != WireColor.RED:
                raise ValueError(
                    f"Slot {idx} is not a red wire "
                    f"(is {slot.wire.color.name})"
                )

        # Reveal all red wires (mark as CUT)
        slot_indices = []
        for idx, slot in hidden:
            actor.tile_stand.cut_wire_at(idx)
            slot_indices.append(idx)

        action = RevealRedAction(
            actor_index=actor_index,
            slot_indices=tuple(slot_indices),
        )
        self.history.record(action)
        self._check_win()
        if not self.game_over:
            self._advance_turn()
        return action

    # -----------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------

    def _advance_turn(self) -> None:
        """Advance to the next player who hasn't finished."""
        if self.game_over:
            return
        player_count = len(self.players)
        for _ in range(player_count):
            self.current_player_index = (
                (self.current_player_index + 1) % player_count
            )
            if not self.players[self.current_player_index].is_finished:
                return
        # All players finished
        self._check_win()

    def _check_win(self) -> None:
        """Check if all players' stands are empty (game won)."""
        if all(p.is_finished for p in self.players):
            self.game_over = True
            self.game_won = True

    def _check_validation(self, value: int) -> None:
        """Check if all 4 blue wires of a value are cut; add validation token.

        Also checks and unlocks any equipment cards tied to this value.

        Args:
            value: The blue wire value to check (1-12).
        """
        cut_count = 0
        for player in self.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_cut
                    and slot.wire is not None
                    and slot.wire.color == WireColor.BLUE
                    and slot.wire.gameplay_value == value
                ):
                    cut_count += 1

        if cut_count >= 4:
            self.validation_tokens.add(value)

        # Check equipment unlocking (needs 2 wires cut of unlock_value)
        pairs_cut = cut_count // 2
        if pairs_cut >= 1:
            for equip in self.equipment:
                if equip.unlock_value == value and not equip.unlocked:
                    equip.unlock()

    def get_all_cut_wires(self) -> list[Wire]:
        """Get all wires that have been cut across all players.

        Returns:
            List of Wire objects that are in CUT state.
        """
        wires = []
        for player in self.players:
            for slot in player.tile_stand.slots:
                if slot.is_cut and slot.wire is not None:
                    wires.append(slot.wire)
        return wires

    def get_cut_count_for_value(self, value: int | str) -> int:
        """Count how many wires of a specific gameplay value have been cut.

        Args:
            value: The gameplay value to count (int 1-12 or 'YELLOW').

        Returns:
            Number of cut wires with this gameplay value.
        """
        count = 0
        for player in self.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_cut
                    and slot.wire is not None
                    and slot.wire.gameplay_value == value
                ):
                    count += 1
        return count

    def can_solo_cut(self, player_index: int, value: int | str) -> bool:
        """Check if a player can perform a solo cut for a given value.

        Solo cut requires that ALL remaining uncut wires of this value
        are in the player's hand.

        Args:
            player_index: The player to check.
            value: The wire value (int 1-12 or 'YELLOW').

        Returns:
            True if the player can solo cut this value.
        """
        player = self.players[player_index]

        # Count uncut wires of this value in the player's hand
        player_uncut = sum(
            1 for s in player.tile_stand.slots
            if s.is_hidden
            and s.wire is not None
            and s.wire.gameplay_value == value
        )

        if player_uncut < 2:
            return False

        # Count uncut wires of this value on OTHER players' stands
        other_uncut = 0
        for i, p in enumerate(self.players):
            if i == player_index:
                continue
            for s in p.tile_stand.slots:
                if (
                    s.is_hidden
                    and s.wire is not None
                    and s.wire.gameplay_value == value
                ):
                    other_uncut += 1

        # Also count info-revealed wires (still on the stand, not cut)
        for i, p in enumerate(self.players):
            if i == player_index:
                continue
            for s in p.tile_stand.slots:
                if (
                    s.is_info_revealed
                    and s.wire is not None
                    and s.wire.gameplay_value == value
                ):
                    other_uncut += 1

        return other_uncut == 0 and player_uncut in (2, 4)

    def available_solo_cuts(self, player_index: int) -> list[int | str]:
        """List all values the player can solo cut.

        Args:
            player_index: The player to check.

        Returns:
            List of gameplay values that can be solo cut.
        """
        possible_values: set[int | str] = set()
        player = self.players[player_index]
        for slot in player.tile_stand.slots:
            if slot.is_hidden and slot.wire is not None:
                possible_values.add(slot.wire.gameplay_value)

        return [v for v in possible_values if self.can_solo_cut(player_index, v)]

    def __str__(self) -> str:
        lines = [
            f"{_Colors.BOLD}=== Bomb Busters ==={_Colors.RESET}",
            str(self.detonator),
            "",
        ]

        # Validation tokens
        if self.validation_tokens:
            validated = sorted(self.validation_tokens)
            lines.append(
                f"Validated: {', '.join(str(v) for v in validated)}"
            )
        else:
            lines.append("Validated: (none)")

        # Markers
        if self.markers:
            marker_strs = [str(m) for m in self.markers]
            lines.append(f"Markers: {' '.join(marker_strs)}")

        # Equipment
        if self.equipment:
            lines.append("Equipment:")
            for e in self.equipment:
                lines.append(f"  {e}")

        lines.append("")

        # Players
        for i, player in enumerate(self.players):
            turn_indicator = " ◄" if i == self.current_player_index else ""
            lines.append(f"Player {i}{turn_indicator}: {player}")

        lines.append("")

        # Game status
        if self.game_over:
            if self.game_won:
                lines.append(f"{_Colors.GREEN}{_Colors.BOLD}MISSION SUCCESS!{_Colors.RESET}")
            else:
                lines.append(f"{_Colors.RED}{_Colors.BOLD}MISSION FAILED!{_Colors.RESET}")

        return "\n".join(lines)
