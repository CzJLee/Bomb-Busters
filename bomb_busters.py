"""Bomb Busters game model.

Core classes representing the Bomb Busters board game components,
including wires, tile stands, players, and game state management.
Supports both full simulation mode and calculator/mid-game mode.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import random


# =============================================================================
# ANSI Color Constants
# =============================================================================

class _Colors:
    """ANSI escape codes for terminal coloring."""
    BLUE = "\033[94m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    ORANGE = "\033[38;5;208m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# =============================================================================
# Enums
# =============================================================================

class WireColor(enum.Enum):
    """Color of a wire tile."""
    BLUE = enum.auto()
    RED = enum.auto()
    YELLOW = enum.auto()

    def ansi(self) -> str:
        """Returns the ANSI color code for this wire color."""
        return {
            WireColor.BLUE: _Colors.BLUE,
            WireColor.RED: _Colors.RED,
            WireColor.YELLOW: _Colors.YELLOW,
        }[self]


class SlotState(enum.Enum):
    """State of a slot on a tile stand."""
    HIDDEN = enum.auto()
    CUT = enum.auto()
    INFO_REVEALED = enum.auto()


class ActionType(enum.Enum):
    """Type of action a player can take on their turn."""
    DUAL_CUT = enum.auto()
    SOLO_CUT = enum.auto()
    REVEAL_RED = enum.auto()


class ActionResult(enum.Enum):
    """Result of a dual cut action."""
    SUCCESS = enum.auto()
    FAIL_BLUE_YELLOW = enum.auto()
    FAIL_RED = enum.auto()


class MarkerState(enum.Enum):
    """State of a board marker for red/yellow wires.

    KNOWN: blank side up — this wire value is definitely in play.
    UNCERTAIN: '?' side up — this wire value might be in play (X of Y mode).
    """
    KNOWN = enum.auto()
    UNCERTAIN = enum.auto()


# =============================================================================
# Constraint Enums
# =============================================================================

class Parity(enum.Enum):
    """Parity of a wire value for even/odd indicator tokens.

    Used with ``SlotParity`` constraints when even/odd tokens are in play.
    """
    EVEN = enum.auto()  # 2, 4, 6, 8, 10, 12
    ODD = enum.auto()   # 1, 3, 5, 7, 9, 11


class Multiplicity(enum.Enum):
    """How many copies of a wire value exist on a tile stand.

    Used with ``ValueMultiplicity`` constraints when x1/x2/x3 tokens
    are in play.
    """
    SINGLE = 1   # x1 token
    DOUBLE = 2   # x2 token
    TRIPLE = 3   # x3 token


# =============================================================================
# Slot Constraints
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SlotConstraint:
    """Base class for all game state constraints.

    All constraints are frozen (immutable) for hashability and safety.
    Subclasses define specific constraint semantics that the probability
    solver can consume.

    Attributes:
        player_index: Index of the player this constraint applies to.
    """
    player_index: int

    def describe(self) -> str:
        """Human-readable description for display."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class AdjacentNotEqual(SlotConstraint):
    """Two adjacent slots must have different gameplay values.

    Used by Label != equipment (#1). Reminder: any two yellow wires
    or any two red wires are always considered identical, so they
    cannot be "different".

    Attributes:
        slot_index_left: Left slot index (must be slot_index_right - 1).
        slot_index_right: Right slot index.
    """
    slot_index_left: int
    slot_index_right: int

    def describe(self) -> str:
        left = chr(ord("A") + self.slot_index_left)
        right = chr(ord("A") + self.slot_index_right)
        return f"P{self.player_index} [{left}] != [{right}]"


@dataclasses.dataclass(frozen=True)
class AdjacentEqual(SlotConstraint):
    """Two adjacent slots must have the same gameplay value.

    Used by Label = equipment (#12). Yellow-yellow and red-red count
    as "same" by gameplay value.

    Attributes:
        slot_index_left: Left slot index (must be slot_index_right - 1).
        slot_index_right: Right slot index.
    """
    slot_index_left: int
    slot_index_right: int

    def describe(self) -> str:
        left = chr(ord("A") + self.slot_index_left)
        right = chr(ord("A") + self.slot_index_right)
        return f"P{self.player_index} [{left}] = [{right}]"


@dataclasses.dataclass(frozen=True)
class MustHaveValue(SlotConstraint):
    """Player must have at least one uncut wire of a specific value.

    Sources: General Radar "yes" (#8), failed dual cuts.

    Attributes:
        value: The gameplay value (int 1-12 or 'YELLOW').
        source: Optional description of how this was determined.
    """
    value: int | str
    source: str = ""

    def describe(self) -> str:
        src = f" ({self.source})" if self.source else ""
        return f"P{self.player_index} has {self.value}{src}"


@dataclasses.dataclass(frozen=True)
class MustNotHaveValue(SlotConstraint):
    """Player must NOT have any uncut wire of a specific value.

    Sources: General Radar "no" response (#8).

    Attributes:
        value: The gameplay value (int 1-12 or 'YELLOW').
        source: Optional description of how this was determined.
    """
    value: int | str
    source: str = ""

    def describe(self) -> str:
        src = f" ({self.source})" if self.source else ""
        return f"P{self.player_index} does not have {self.value}{src}"


@dataclasses.dataclass(frozen=True)
class SlotParity(SlotConstraint):
    """A slot is known to be even or odd from even/odd indicator tokens.

    Not enforced by the solver in this version — reserved for future
    indicator token system.

    Attributes:
        slot_index: The slot this constraint applies to.
        parity: Whether the slot is even or odd.
    """
    slot_index: int
    parity: Parity

    def describe(self) -> str:
        letter = chr(ord("A") + self.slot_index)
        return f"P{self.player_index} [{letter}] is {self.parity.name.lower()}"


@dataclasses.dataclass(frozen=True)
class ValueMultiplicity(SlotConstraint):
    """Wire value appears exactly N times on this stand (x1/x2/x3 tokens).

    Not enforced by the solver in this version — reserved for future
    indicator token system.

    Attributes:
        slot_index: The slot the token is placed on.
        multiplicity: How many copies of this value exist on the stand.
    """
    slot_index: int
    multiplicity: Multiplicity

    def describe(self) -> str:
        letter = chr(ord("A") + self.slot_index)
        return (
            f"P{self.player_index} [{letter}] value appears "
            f"x{self.multiplicity.value}"
        )


@dataclasses.dataclass(frozen=True)
class UnsortedSlot(SlotConstraint):
    """A slot is not sorted with the rest of the stand (X tokens).

    Not enforced by the solver in this version — reserved for future
    indicator token system.

    Attributes:
        slot_index: The unsorted slot index.
    """
    slot_index: int

    def describe(self) -> str:
        letter = chr(ord("A") + self.slot_index)
        return f"P{self.player_index} [{letter}] is unsorted"


# =============================================================================
# Wire
# =============================================================================

@functools.total_ordering
@dataclasses.dataclass(frozen=True)
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
        label = str(self.base_number)
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
    return create_blue_wires(1, 12)


def create_blue_wires(low: int, high: int) -> list[Wire]:
    """Create blue wire tiles (4 copies each) for a range of values.

    Args:
        low: Lowest blue wire value (inclusive).
        high: Highest blue wire value (inclusive).

    Returns:
        List of blue Wire objects (4 per value).

    Raises:
        ValueError: If low > high or values are outside 1-12.
    """
    if not (1 <= low <= 12 and 1 <= high <= 12 and low <= high):
        raise ValueError(
            f"Blue wire range must be within 1-12 with low <= high, "
            f"got {low}-{high}"
        )
    wires = []
    for number in range(low, high + 1):
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
# Wire Config Builder (for from_partial_state)
# =============================================================================

def _build_wire_config(
    blue_wires: list[Wire] | tuple[int, int] | None,
    yellow_wires: list[int] | tuple[list[int], int] | None,
    red_wires: list[int] | tuple[list[int], int] | None,
) -> tuple[list[Wire], list[Marker], list[UncertainWireGroup]]:
    """Derive wires_in_play, markers, and uncertain_wire_groups.

    Translates the convenience parameters into the three internal
    GameState attributes needed by the probability engine and display.

    Args:
        blue_wires: Blue wire pool. ``None`` = all blue 1-12 (48 wires).
            ``(low, high)`` tuple = ``create_blue_wires(low, high)``.
            ``list[Wire]`` = custom wire list (for tests).
        yellow_wires: Yellow wire specification. ``None`` = no yellow.
            ``[4, 7]`` = Y4, Y7 definitely in play (KNOWN markers).
            ``([2, 3, 9], 2)`` = 2-of-3 uncertain (UNCERTAIN markers).
        red_wires: Red wire specification. Same semantics as yellow.

    Returns:
        A tuple of (wires_in_play, markers, uncertain_wire_groups).
    """
    # ── Blue wires ────────────────────────────────────────────
    if blue_wires is None:
        pool = create_all_blue_wires()
    elif isinstance(blue_wires, tuple):
        low, high = blue_wires
        pool = create_blue_wires(low, high)
    else:
        pool = list(blue_wires)

    markers: list[Marker] = []
    uncertain_groups: list[UncertainWireGroup] = []

    # ── Yellow wires ──────────────────────────────────────────
    if yellow_wires is not None:
        if isinstance(yellow_wires, tuple):
            numbers, count = yellow_wires
            uncertain_groups.append(
                UncertainWireGroup.yellow(numbers, count=count),
            )
            for n in numbers:
                markers.append(
                    Marker(WireColor.YELLOW, n + 0.1, MarkerState.UNCERTAIN),
                )
        else:
            for n in yellow_wires:
                wire = Wire(WireColor.YELLOW, n + 0.1)
                pool.append(wire)
                markers.append(
                    Marker(WireColor.YELLOW, n + 0.1, MarkerState.KNOWN),
                )

    # ── Red wires ─────────────────────────────────────────────
    if red_wires is not None:
        if isinstance(red_wires, tuple):
            numbers, count = red_wires
            uncertain_groups.append(
                UncertainWireGroup.red(numbers, count=count),
            )
            for n in numbers:
                markers.append(
                    Marker(WireColor.RED, n + 0.5, MarkerState.UNCERTAIN),
                )
        else:
            for n in red_wires:
                wire = Wire(WireColor.RED, n + 0.5)
                pool.append(wire)
                markers.append(
                    Marker(WireColor.RED, n + 0.5, MarkerState.KNOWN),
                )

    return pool, markers, uncertain_groups


# =============================================================================
# Slot
# =============================================================================

@dataclasses.dataclass
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

    def value_label(self, mask_hidden: bool = False) -> tuple[str, str]:
        """Return the display label and ANSI color for this slot's value.

        Args:
            mask_hidden: If True, hidden wires with known identity are
                displayed as '?' (used for observer perspective mode).

        Returns:
            A tuple of (plain_text_label, ansi_colored_label).
            The plain label is used for width calculations; the colored
            label is used for display.
        """
        if self.state == SlotState.HIDDEN:
            if self.wire is not None and not mask_hidden:
                plain = str(self.wire.base_number)
                colored = f"{_Colors.DIM}{self.wire}{_Colors.RESET}"
                return plain, colored
            return "?", f"{_Colors.DIM}?{_Colors.RESET}"
        elif self.state == SlotState.CUT:
            if self.wire is not None:
                plain = str(self.wire.base_number)
                colored = f"{_Colors.GREEN}{self.wire}{_Colors.RESET}"
                return plain, colored
            return "?", f"{_Colors.GREEN}?{_Colors.RESET}"
        else:  # INFO_REVEALED
            if self.info_token == "YELLOW":
                # Yellow info token: we know it's yellow but not which one.
                # Display as "Y" in yellow text (not the full "YELLOW" string).
                plain = "Y"
                colored = f"{_Colors.BOLD}{_Colors.YELLOW}Y{_Colors.RESET}"
                return plain, colored
            if self.info_token is not None:
                token_str = str(self.info_token)
            else:
                token_str = "?"
            if self.wire is not None:
                colored = f"{_Colors.BOLD}{self.wire.color.ansi()}{token_str}{_Colors.RESET}"
            elif isinstance(self.info_token, int):
                # Calculator mode: numeric info token is always blue
                colored = f"{_Colors.BOLD}{_Colors.BLUE}{token_str}{_Colors.RESET}"
            else:
                colored = f"{_Colors.BOLD}{token_str}{_Colors.RESET}"
            return token_str, colored

    def __str__(self) -> str:
        _, colored = self.value_label()
        return colored


# =============================================================================
# Slot Parsing Helper
# =============================================================================

def _parse_wire_value(token: str) -> Wire:
    """Parse a wire value string into a Wire object.

    Handles blue (plain number), yellow (Y/y prefix), and red (R/r prefix).

    Args:
        token: A string like "5", "Y4", "R5" (case-insensitive prefix).

    Returns:
        The corresponding Wire object.

    Raises:
        ValueError: If the token cannot be parsed as a wire value.
    """
    upper = token.upper()
    if upper.startswith("Y"):
        num_str = token[1:]
        try:
            n = int(num_str)
        except ValueError:
            raise ValueError(f"Invalid yellow wire number: {token!r}")
        return Wire(WireColor.YELLOW, n + 0.1)
    elif upper.startswith("R"):
        num_str = token[1:]
        try:
            n = int(num_str)
        except ValueError:
            raise ValueError(f"Invalid red wire number: {token!r}")
        return Wire(WireColor.RED, n + 0.5)
    else:
        try:
            n = int(token)
        except ValueError:
            raise ValueError(f"Invalid blue wire number: {token!r}")
        return Wire(WireColor.BLUE, float(n))


def _parse_slot_token(token: str) -> Slot:
    """Parse a single shorthand token into a Slot object.

    Token formats:
        N       — CUT blue wire (e.g., "5", "12")
        YN      — CUT yellow wire (e.g., "Y4")
        RN      — CUT red wire (e.g., "R5")
        ?       — HIDDEN unknown wire
        ?N      — HIDDEN blue wire (e.g., "?4")
        ?YN     — HIDDEN yellow wire (e.g., "?Y4")
        ?RN     — HIDDEN red wire (e.g., "?R5")
        iN      — INFO_REVEALED blue info token (e.g., "i5")
        iY      — INFO_REVEALED yellow info token (e.g., "iY")
        iYN     — INFO_REVEALED yellow info token, observer knows
                  exact wire (e.g., "iY4")

    Red info tokens (``iR``, ``iRN``) are invalid — there is no such
    thing as a red info token in Bomb Busters.

    Prefixes are case-insensitive.

    Args:
        token: The shorthand string for a single slot.

    Returns:
        A Slot object with the appropriate state and wire/info_token.

    Raises:
        ValueError: If the token cannot be parsed.
    """
    if not token:
        raise ValueError("Empty slot token")

    # INFO_REVEALED: starts with 'i' or 'I'
    if token[0] in ("i", "I"):
        rest = token[1:]
        if not rest:
            raise ValueError(f"Info token missing value: {token!r}")
        upper_rest = rest.upper()
        # Red info tokens do not exist (cutting red = game over)
        if upper_rest.startswith("R"):
            raise ValueError(
                f"Red info tokens do not exist: {token!r}. "
                f"Use '?R{rest[1:]}' for a hidden red wire known "
                f"to the observer."
            )
        if upper_rest == "Y":
            # iY — yellow info token, exact wire unknown to observer
            return Slot(wire=None, state=SlotState.INFO_REVEALED,
                        info_token="YELLOW")
        if upper_rest.startswith("Y"):
            # iYN — yellow info token, observer knows exact wire
            num_str = rest[1:]
            try:
                n = int(num_str)
            except ValueError:
                raise ValueError(
                    f"Invalid yellow info token number: {token!r}"
                )
            return Slot(
                wire=Wire(WireColor.YELLOW, n + 0.1),
                state=SlotState.INFO_REVEALED,
                info_token="YELLOW",
            )
        # iN — blue info token (a numeric info token is always blue)
        try:
            value = int(rest)
        except ValueError:
            raise ValueError(f"Invalid info token value: {token!r}")
        return Slot(
            wire=Wire(WireColor.BLUE, float(value)),
            state=SlotState.INFO_REVEALED,
            info_token=value,
        )

    # HIDDEN: starts with '?'
    if token[0] == "?":
        rest = token[1:]
        if not rest:
            return Slot(wire=None, state=SlotState.HIDDEN)
        wire = _parse_wire_value(rest)
        return Slot(wire=wire, state=SlotState.HIDDEN)

    # CUT: plain wire value
    wire = _parse_wire_value(token)
    return Slot(wire=wire, state=SlotState.CUT)


# =============================================================================
# TileStand
# =============================================================================

@dataclasses.dataclass
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

    @classmethod
    def from_string(
        cls,
        notation: str,
        sep: str = " ",
        num_tiles: int | None = None,
    ) -> TileStand:
        """Create a tile stand from shorthand string notation.

        Supports quick entry of game states. Tokens are separated by
        ``sep`` (default space). Each token describes one slot:

            N       — CUT blue wire (e.g., ``5``, ``12``)
            YN      — CUT yellow wire (e.g., ``Y4``)
            RN      — CUT red wire (e.g., ``R5``)
            ?       — HIDDEN unknown wire
            ?N      — HIDDEN blue wire known to observer (e.g., ``?4``)
            ?YN     — HIDDEN yellow wire known to observer (e.g., ``?Y4``)
            ?RN     — HIDDEN red wire known to observer (e.g., ``?R5``)
            iN      — INFO_REVEALED with blue info token (e.g., ``i5``)
            iY      — INFO_REVEALED with yellow info token
            iYN     — INFO_REVEALED yellow, observer knows exact wire
                      (e.g., ``iY4``)

        All prefixes (``Y``, ``R``, ``i``, ``?``) are case-insensitive.

        Args:
            notation: The shorthand string describing the tile stand.
            sep: Token separator (default ``" "``).
            num_tiles: If provided, validates that the parsed tile count
                matches this value exactly.

        Returns:
            A new TileStand with slots in the order given.

        Raises:
            ValueError: If a token cannot be parsed, the notation is
                empty, or ``num_tiles`` doesn't match the parsed count.

        Examples:
            Observer's own stand (hidden wires known)::

                TileStand.from_string("1 2 ?4 ?Y4 ?6 ?7 ?8 ?8 9 11 12")

            Another player's stand (hidden wires unknown)::

                TileStand.from_string("1 3 ? ? ? 8 9 ? ? 12")

            Stand with an info token::

                TileStand.from_string("2 3 ? ? i6 ? ? 9 ? 11")
        """
        if not notation or not notation.strip():
            raise ValueError("Notation string is empty")
        tokens = notation.split(sep)
        tokens = [t for t in tokens if t]
        if not tokens:
            raise ValueError("Notation string contains no tokens")
        if num_tiles is not None and len(tokens) != num_tiles:
            raise ValueError(
                f"Expected {num_tiles} tiles, got {len(tokens)}"
            )
        slots = [_parse_slot_token(t) for t in tokens]
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

    def stand_lines(self, mask_hidden: bool = False) -> tuple[str, str, str]:
        """Return the three display lines for this stand: status, values, letters.

        The status line uses symbols above each value to indicate state:
        nothing for hidden, checkmark for cut, 'i' for info-revealed.
        Each cell is 3 characters wide for consistent alignment.

        Args:
            mask_hidden: If True, hidden wires are shown as '?' even if
                the wire identity is known (observer perspective mode).

        Returns:
            A tuple of (status_line, values_line, letters_line).
        """
        CELL = 3
        status_parts: list[str] = []
        value_parts: list[str] = []
        letter_parts: list[str] = []
        for i, slot in enumerate(self.slots):
            letter = chr(ord('A') + i)
            plain, colored = slot.value_label(mask_hidden=mask_hidden)
            plain_width = len(plain)
            pad = CELL - plain_width

            # Status indicator above the value
            if slot.state == SlotState.CUT:
                indicator = f"{_Colors.GREEN}✓{_Colors.RESET}"
                indicator_plain_width = 1
            elif slot.state == SlotState.INFO_REVEALED:
                indicator = f"{_Colors.BOLD}i{_Colors.RESET}"
                indicator_plain_width = 1
            else:
                indicator = " "
                indicator_plain_width = 1
            status_parts.append(" " * (CELL - indicator_plain_width) + indicator)

            value_parts.append(" " * pad + colored)
            letter_parts.append(" " * (CELL - 1) + letter)
        status_line = "".join(status_parts)
        values_line = "".join(value_parts)
        letters_line = "".join(letter_parts)
        return status_line, values_line, letters_line

    def __str__(self) -> str:
        _, values_line, letters_line = self.stand_lines()
        return f"{values_line}\n{letters_line}"


# =============================================================================
# Helper: Sort Value Bounds
# =============================================================================

def _slot_sort_value(slot: Slot) -> float | None:
    """Extract a sort_value from a publicly visible slot.

    Returns the sort_value for CUT or INFO_REVEALED slots that have
    a known wire identity. For INFO_REVEALED slots in calculator mode
    (wire is None), falls back to the info_token value when it's a
    blue numeric token (sort_value = float(info_token)).

    HIDDEN slots always return None regardless of wire identity,
    because the observer cannot see hidden wires on other players'
    stands.

    Args:
        slot: The slot to inspect.

    Returns:
        The inferred sort_value, or None if not determinable.
    """
    if slot.state == SlotState.HIDDEN:
        return None
    if slot.wire is not None:
        return slot.wire.sort_value
    # Calculator mode: wire is None but info_token may give the value.
    # A numeric info token is always blue (sort_value = N.0).
    if (
        slot.state == SlotState.INFO_REVEALED
        and isinstance(slot.info_token, int)
    ):
        return float(slot.info_token)
    return None


def get_sort_value_bounds(
    slots: list[Slot], slot_index: int
) -> tuple[float, float]:
    """Get the sort_value bounds for a hidden slot based on known neighbors.

    Scans left and right from the given slot index to find the nearest
    known wire (CUT or INFO_REVEALED with a known wire or blue info
    token) to establish the valid range of sort_values for the hidden
    slot.

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

    # Scan left for nearest publicly visible wire (CUT or INFO_REVEALED).
    # HIDDEN slots are skipped even if they contain a wire (simulation mode),
    # because the observer cannot see hidden wires on other players' stands.
    for i in range(slot_index - 1, -1, -1):
        sv = _slot_sort_value(slots[i])
        if sv is not None:
            lower = sv
            break

    # Scan right for nearest publicly visible wire
    for i in range(slot_index + 1, len(slots)):
        sv = _slot_sort_value(slots[i])
        if sv is not None:
            upper = sv
            break

    return (lower, upper)


# =============================================================================
# CharacterCard
# =============================================================================

@dataclasses.dataclass
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

@dataclasses.dataclass
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
        card_str = f" | Card: {self.character_card}" if self.character_card else ""
        return f"{_Colors.BOLD}{self.name}{_Colors.RESET}{card_str}"


# =============================================================================
# Detonator
# =============================================================================

@dataclasses.dataclass
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
        if self.is_exploded:
            return f"Mistakes remaining: {_Colors.RED}💀 BOOM!{_Colors.RESET}"
        remaining = self.remaining_failures
        if remaining >= self.max_failures:
            color = _Colors.GREEN
        elif remaining >= 2:
            color = _Colors.YELLOW
        elif remaining == 1:
            color = _Colors.ORANGE
        else:
            color = _Colors.RED
        return f"Mistakes remaining: {color}{remaining}{_Colors.RESET}"


# =============================================================================
# Marker
# =============================================================================

@dataclasses.dataclass
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

@dataclasses.dataclass
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

@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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

@dataclasses.dataclass
class TurnHistory:
    """Record of all actions taken during the game.

    Useful for deduction: e.g., a failed dual cut reveals that the actor
    has at least one wire of the guessed value.

    Attributes:
        actions: List of all action records in chronological order.
    """
    actions: list[ActionRecord] = dataclasses.field(default_factory=list)

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

@dataclasses.dataclass
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
# UncertainWireGroup
# =============================================================================

@dataclasses.dataclass
class UncertainWireGroup:
    """A group of colored wires with uncertain inclusion from X-of-Y setup.

    In X-of-Y mode during mission setup, Y candidate wires are drawn
    but only X are kept and shuffled into the game. Players see UNCERTAIN
    markers for all Y candidates but don't know which X are in play.

    Use this with ``from_partial_state`` to represent the uncertainty.
    The probability engine's constraint solver uses "discard slots"
    (slack variables) to enumerate which candidates are in the game.

    Attributes:
        candidates: All Wire objects that might be in play (the Y drawn).
        count_in_play: How many candidates are actually in the game
            (the X in "X of Y").
    """
    candidates: list[Wire]
    count_in_play: int

    def __post_init__(self) -> None:
        """Validate uncertain wire group constraints."""
        if not self.candidates:
            raise ValueError("Candidates list must not be empty")
        colors = {w.color for w in self.candidates}
        if len(colors) != 1:
            raise ValueError(
                "All candidates must be the same color"
            )
        color = next(iter(colors))
        if color == WireColor.BLUE:
            raise ValueError(
                "UncertainWireGroup is only for RED and YELLOW wires"
            )
        if color == WireColor.YELLOW and not (0 <= self.count_in_play <= 4):
            raise ValueError(
                f"Yellow wire count_in_play must be 0-4, "
                f"got {self.count_in_play}"
            )
        if color == WireColor.RED and not (0 <= self.count_in_play <= 3):
            raise ValueError(
                f"Red wire count_in_play must be 0-3, "
                f"got {self.count_in_play}"
            )
        if self.count_in_play > len(self.candidates):
            raise ValueError(
                f"count_in_play ({self.count_in_play}) cannot exceed "
                f"number of candidates ({len(self.candidates)})"
            )

    @property
    def color(self) -> WireColor:
        """The color of all candidate wires."""
        return self.candidates[0].color

    @property
    def discard_count(self) -> int:
        """Number of candidates NOT in the game."""
        return len(self.candidates) - self.count_in_play

    @classmethod
    def yellow(cls, numbers: list[int], count: int) -> UncertainWireGroup:
        """Create a yellow uncertain group from base numbers.

        Args:
            numbers: Base numbers of the yellow wires drawn
                (e.g., ``[2, 3, 9]``).
            count: How many are actually in the game.

        Returns:
            An UncertainWireGroup for yellow wires.
        """
        candidates = [Wire(WireColor.YELLOW, n + 0.1) for n in numbers]
        return cls(candidates=candidates, count_in_play=count)

    @classmethod
    def red(cls, numbers: list[int], count: int) -> UncertainWireGroup:
        """Create a red uncertain group from base numbers.

        Args:
            numbers: Base numbers of the red wires drawn
                (e.g., ``[3, 7]``).
            count: How many are actually in the game.

        Returns:
            An UncertainWireGroup for red wires.
        """
        candidates = [Wire(WireColor.RED, n + 0.5) for n in numbers]
        return cls(candidates=candidates, count_in_play=count)


# =============================================================================
# GameState
# =============================================================================

@dataclasses.dataclass
class GameState:
    """The complete state of a Bomb Busters game.

    Supports two modes:
    - Simulation mode (via create_game): all wires are known.
    - Calculator mode (via from_partial_state): only observable info is set,
      other players' hidden wires are None.

    Attributes:
        players: List of all players in the game.
        detonator: The detonator dial tracking failures.
        markers: Board markers for red/yellow wires in play.
        equipment: List of equipment cards in the game.
        history: Record of all actions taken.
        current_player_index: Index of the player whose turn it is.
        game_over: Whether the game has ended.
        game_won: Whether the game was won (all stands empty).
        wires_in_play: All wire objects that were shuffled into the game.
            For uncertain (X of Y) colored wires, include only the
            definite wires here; use ``uncertain_wire_groups`` for the
            candidates whose inclusion is unknown.
        uncertain_wire_groups: Groups of colored wires with uncertain
            inclusion from X-of-Y mission setup. Used in calculator mode
            to let the solver enumerate which candidates are in play.
        active_player_index: If set, display output is rendered from this
            player's perspective (their wires visible, others' hidden
            wires masked as '?'). If None, all wires are shown (god mode).
        captain_index: Index of the player who is the captain. The
            captain deals first, indicates first, and takes the first
            turn. Defaults to 0.
        slot_constraints: List of SlotConstraint objects representing
            game constraints from equipment cards, General Radar, etc.
            Used by the probability engine to prune invalid distributions.
    """
    players: list[Player]
    detonator: Detonator
    markers: list[Marker]
    equipment: list[Equipment]
    history: TurnHistory
    current_player_index: int = 0
    game_over: bool = False
    game_won: bool = False
    wires_in_play: list[Wire] = dataclasses.field(default_factory=list)
    uncertain_wire_groups: list[UncertainWireGroup] = dataclasses.field(
        default_factory=list,
    )
    active_player_index: int | None = None
    captain_index: int = 0
    slot_constraints: list[SlotConstraint] = dataclasses.field(
        default_factory=list,
    )

    @property
    def validation_tokens(self) -> set[int]:
        """Blue wire values (1-12) where all 4 copies have been cut.

        Computed from the current state of all players' stands.
        """
        counts: dict[int, int] = {}
        for player in self.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_cut
                    and slot.wire is not None
                    and slot.wire.color == WireColor.BLUE
                ):
                    v = slot.wire.gameplay_value
                    assert isinstance(v, int)
                    counts[v] = counts.get(v, 0) + 1
        return {v for v, c in counts.items() if c >= 4}

    def add_constraint(self, constraint: SlotConstraint) -> None:
        """Add a constraint to the game state.

        Args:
            constraint: The constraint to add.
        """
        self.slot_constraints.append(constraint)

    def get_constraints_for_player(
        self, player_index: int
    ) -> list[SlotConstraint]:
        """Get all constraints that apply to a specific player.

        Args:
            player_index: The player to query.

        Returns:
            List of constraints for that player.
        """
        return [
            c for c in self.slot_constraints
            if c.player_index == player_index
        ]

    # -----------------------------------------------------------------
    # Factory: Full Simulation Mode
    # -----------------------------------------------------------------

    @classmethod
    def create_game(
        cls,
        player_names: list[str],
        seed: int | None = None,
        captain: int = 0,
        yellow_wires: int | tuple[int, int] | None = None,
        red_wires: int | tuple[int, int] | None = None,
    ) -> GameState:
        """Create a new game in full simulation mode.

        Shuffles all wires, deals them evenly to players, and sets up
        the board. Always includes all 48 blue wires (1-12, 4 copies
        each). Optionally adds yellow and/or red wires.

        Args:
            player_names: Names of the players (4-5 players).
            seed: Optional random seed for reproducibility.
            captain: Player index of the captain. The captain is dealt
                first, indicates first, and takes the first turn.
                Defaults to 0.
            yellow_wires: Yellow wire specification. ``None`` (default)
                = no yellow wires. ``2`` = include 2 randomly selected
                yellow wires (KNOWN markers). ``(2, 3)`` = draw 3
                random yellow wires, keep 2 (UNCERTAIN markers on all
                3 drawn).
            red_wires: Red wire specification. Same semantics as
                ``yellow_wires``. ``1`` = include 1 random red wire.
                ``(1, 2)`` = draw 2 random red wires, keep 1.

        Returns:
            A fully initialized GameState.

        Raises:
            ValueError: If player count is not 4-5, captain index
                is out of range, or wire counts are invalid.
        """
        if seed is not None:
            random.seed(seed)

        player_count = len(player_names)
        if not (4 <= player_count <= 5):
            raise ValueError(f"Player count must be 4-5, got {player_count}")
        if not (0 <= captain < player_count):
            raise ValueError(
                f"Captain index must be 0-{player_count - 1}, "
                f"got {captain}"
            )

        # Build wire pool
        pool: list[Wire] = create_all_blue_wires()
        markers: list[Marker] = []

        for color, spec in [
            (WireColor.YELLOW, yellow_wires),
            (WireColor.RED, red_wires),
        ]:
            if spec is None:
                continue

            all_colored = (
                create_all_red_wires()
                if color == WireColor.RED
                else create_all_yellow_wires()
            )

            if isinstance(spec, tuple):
                count, pool_size = spec
                # Validate
                if pool_size < count:
                    raise ValueError(
                        f"Pool size ({pool_size}) must be >= count "
                        f"({count}) for {color.name.lower()} wires"
                    )
                if pool_size > 11:
                    raise ValueError(
                        f"Pool size ({pool_size}) exceeds available "
                        f"{color.name.lower()} wires (11)"
                    )
                # "X of Y" mode: draw pool_size, keep count
                drawn = random.sample(all_colored, pool_size)
                for w in drawn:
                    markers.append(
                        Marker(color, w.sort_value, MarkerState.UNCERTAIN),
                    )
                random.shuffle(drawn)
                pool.extend(drawn[:count])
            else:
                count = spec
                # Direct inclusion: draw exactly count wires
                drawn = random.sample(all_colored, count)
                for w in drawn:
                    markers.append(
                        Marker(color, w.sort_value, MarkerState.KNOWN),
                    )
                pool.extend(drawn)

        # Record all wires in play before shuffling
        wires_in_play = list(pool)

        # Shuffle and deal (starting with the captain, clockwise)
        random.shuffle(pool)
        total_wires = len(pool)
        base_count = total_wires // player_count
        extra = total_wires % player_count

        hands: list[list[Wire]] = [[] for _ in range(player_count)]
        idx = 0
        for deal_offset in range(player_count):
            p = (captain + deal_offset) % player_count
            count = base_count + (1 if deal_offset < extra else 0)
            hands[p] = pool[idx:idx + count]
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
            markers=markers,
            equipment=[],
            history=TurnHistory(),
            wires_in_play=wires_in_play,
            current_player_index=captain,
            captain_index=captain,
        )

    # -----------------------------------------------------------------
    # Factory: Calculator / Mid-Game Mode
    # -----------------------------------------------------------------

    @classmethod
    def from_partial_state(
        cls,
        player_names: list[str],
        stands: list[TileStand],
        mistakes_remaining: int | None = None,
        equipment: list[Equipment] | None = None,
        character_cards: list[CharacterCard | None] | None = None,
        history: TurnHistory | None = None,
        active_player_index: int = 0,
        captain: int = 0,
        blue_wires: list[Wire] | tuple[int, int] | None = None,
        yellow_wires: list[int] | tuple[list[int], int] | None = None,
        red_wires: list[int] | tuple[list[int], int] | None = None,
        constraints: list[SlotConstraint] | None = None,
        validate_stand_sizes: bool = True,
    ) -> GameState:
        """Create a game state from partial mid-game information.

        Use this to enter a game state during a real game for probability
        calculations, without needing to replay all turns.

        Wire configuration uses one parameter per color. The internal
        ``wires_in_play``, ``markers``, and ``uncertain_wire_groups``
        attributes are all auto-derived from these parameters.

        Args:
            player_names: Names of all players.
            stands: List of TileStand objects, one per player. Use
                ``TileStand.from_string()`` for quick entry or build
                manually with ``TileStand(slots=[...])``.
            mistakes_remaining: How many more mistakes the team can
                survive. Defaults to ``player_count - 1`` (a fresh
                mission).
            equipment: Equipment cards in play.
            character_cards: Character card for each player (or None).
            history: Optional turn history for deduction.
            active_player_index: Index of the player whose turn it is.
                Display output is rendered from this player's perspective
                (their wires visible, others masked as '?'). Defaults to 0.
            captain: Player index of the captain. Defaults to 0.
            blue_wires: Blue wire pool. ``None`` (default) = all blue
                1-12 (48 wires). ``(low, high)`` tuple = blue wires for
                values low through high (4 copies each).
                ``list[Wire]`` = custom wire list.
            yellow_wires: Yellow wire specification. ``None`` (default)
                = no yellow wires. ``[4, 7]`` = Y4 and Y7 definitely
                in play (KNOWN markers). ``([2, 3, 9], 2)`` = 2-of-3
                uncertain (UNCERTAIN markers, solver handles
                combinatorics).
            red_wires: Red wire specification. Same semantics as
                ``yellow_wires``. ``[4]`` = R4 definitely in play.
                ``([3, 7], 1)`` = 1-of-2 uncertain.
            constraints: List of ``SlotConstraint`` objects (e.g.,
                ``AdjacentNotEqual``, ``MustHaveValue``). These are
                passed through to ``GameState.slot_constraints``.
            validate_stand_sizes: If True (default), verify that each
                player's tile stand has the expected number of wires
                based on the wire pool and captain dealing order.
                Set to False for tests or scenarios where stand sizes
                don't follow standard dealing (e.g., Grappling Hook).

        Returns:
            A GameState initialized from the provided partial information.
        """
        player_count = len(player_names)
        if len(stands) != player_count:
            raise ValueError(
                f"Number of stands ({len(stands)}) must match "
                f"number of players ({player_count})"
            )
        if not (0 <= captain < player_count):
            raise ValueError(
                f"Captain index must be 0-{player_count - 1}, "
                f"got {captain}"
            )

        cards = character_cards or [None] * player_count
        players = [
            Player(
                name=player_names[i],
                tile_stand=stands[i],
                character_card=cards[i] if i < len(cards) else None,
            )
            for i in range(player_count)
        ]

        max_failures = player_count - 1
        if mistakes_remaining is None:
            mistakes_remaining = max_failures
        failures = max_failures - mistakes_remaining

        wires_in_play, markers, uncertain_groups = _build_wire_config(
            blue_wires, yellow_wires, red_wires,
        )

        # Validate tile stand sizes match the dealing distribution.
        # Wires are dealt starting with the captain, clockwise. If
        # the total doesn't divide evenly, earlier players get one
        # extra wire.
        if validate_stand_sizes:
            total_wires = len(wires_in_play) + sum(
                g.count_in_play for g in uncertain_groups
            )
            base_count = total_wires // player_count
            extra = total_wires % player_count
            for deal_offset in range(player_count):
                p_idx = (captain + deal_offset) % player_count
                expected = base_count + (1 if deal_offset < extra else 0)
                actual = len(stands[p_idx].slots)
                if actual != expected:
                    raise ValueError(
                        f"Player {p_idx} ({player_names[p_idx]}) has "
                        f"{actual} wires but expected {expected} "
                        f"({total_wires} total wires, captain={captain})"
                    )

        return cls(
            players=players,
            detonator=Detonator(
                failures=failures,
                max_failures=max_failures,
            ),
            markers=markers,
            equipment=equipment or [],
            history=history or TurnHistory(),
            wires_in_play=wires_in_play,
            uncertain_wire_groups=uncertain_groups,
            current_player_index=active_player_index,
            active_player_index=active_player_index,
            captain_index=captain,
            slot_constraints=constraints or [],
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

        # Validate actor has a matching uncut wire (hidden or info-revealed)
        actor_matching_slots = [
            (i, s) for i, s in enumerate(actor.tile_stand.slots)
            if not s.is_cut and s.wire is not None
            and s.wire.gameplay_value == guessed_value
        ]
        if not actor_matching_slots:
            raise ValueError(
                f"Active player has no uncut wire with value {guessed_value}"
            )

        # Validate target slot (allow HIDDEN and INFO_REVEALED, reject CUT)
        target_slot = target.tile_stand.slots[target_slot_index]
        if target_slot.is_cut:
            raise ValueError(f"Target slot {target_slot_index} is already cut")

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
            if second_slot.is_cut:
                raise ValueError(
                    f"Second target slot {second_target_slot_index} is already cut"
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
        """Check and unlock any equipment cards tied to this value.

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

    def can_solo_cut(
        self,
        player_index: int,
        value: int | str,
        fast_pass: bool = False,
    ) -> bool:
        """Check if a player can perform a solo cut for a given value.

        Solo cut requires that ALL remaining uncut wires of this value
        are in the player's hand. With ``fast_pass=True`` (Fast Pass
        Card, equipment #9.9), the player can solo cut 2 identical
        wires even if they are not the last remaining wires of that
        value.

        In calculator mode, other players' hidden wires are ``None``
        (unknown), so we cannot inspect them directly. Instead, we
        count the total wires of this value in the game and verify
        that all are accounted for — either on the active player's
        own stand or publicly visible on other stands (cut or
        info-revealed).

        Args:
            player_index: The player to check.
            value: The wire value (int 1-12 or 'YELLOW').
            fast_pass: If True, skip the "all remaining must be in
                hand" check. Only require 2 hidden wires of this value.

        Returns:
            True if the player can solo cut this value.
        """
        player = self.players[player_index]

        # Count hidden wires of this value in the player's hand
        # (only hidden wires can be included in a solo cut action).
        player_hidden_count = sum(
            1 for s in player.tile_stand.slots
            if s.is_hidden
            and s.wire is not None
            and s.wire.gameplay_value == value
        )

        if player_hidden_count < 2:
            return False

        # Fast Pass: only need 2 matching hidden wires, no ownership check
        if fast_pass:
            return True

        # Total wires of this value known to exist in the game.
        total_in_game = sum(
            1 for w in self.wires_in_play
            if w.gameplay_value == value
        )
        for group in self.uncertain_wire_groups:
            if any(w.gameplay_value == value for w in group.candidates):
                total_in_game += group.count_in_play

        # Accounted-for wires: those whose location is known.
        # 1. All of the active player's own wires of this value
        #    (the active player knows their entire hand).
        player_total = sum(
            1 for s in player.tile_stand.slots
            if s.wire is not None
            and s.wire.gameplay_value == value
        )

        # 2. Publicly visible wires on other players' stands
        #    (cut or info-revealed with known identity).
        other_visible = 0
        for i, p in enumerate(self.players):
            if i == player_index:
                continue
            for s in p.tile_stand.slots:
                if s.is_cut:
                    if (
                        s.wire is not None
                        and s.wire.gameplay_value == value
                    ):
                        other_visible += 1
                elif s.is_info_revealed:
                    if (
                        s.wire is not None
                        and s.wire.gameplay_value == value
                    ):
                        other_visible += 1
                    elif s.info_token == value:
                        other_visible += 1

        accounted = player_total + other_visible

        # If any wires of this value are unaccounted for, they could
        # be on other players' hidden slots — cannot guarantee solo cut.
        if total_in_game - accounted > 0:
            return False

        return player_hidden_count in (2, 4)

    def available_solo_cuts(
        self,
        player_index: int,
        fast_pass: bool = False,
    ) -> list[int | str]:
        """List all values the player can solo cut.

        Args:
            player_index: The player to check.
            fast_pass: If True, use Fast Pass rules (see ``can_solo_cut``).

        Returns:
            List of gameplay values that can be solo cut.
        """
        possible_values: set[int | str] = set()
        player = self.players[player_index]
        for slot in player.tile_stand.slots:
            if slot.is_hidden and slot.wire is not None:
                possible_values.add(slot.wire.gameplay_value)

        return [
            v for v in possible_values
            if self.can_solo_cut(player_index, v, fast_pass=fast_pass)
        ]

    # -----------------------------------------------------------------
    # Game State Modification Methods
    # -----------------------------------------------------------------

    def place_info_token(self, player_index: int, slot_index: int) -> None:
        """Mark a hidden blue wire as INFO_REVEALED with its value.

        Used by Post-It equipment (#4) and other effects that publicly
        reveal a wire's identity. The wire must be a hidden blue wire
        with a known identity (not None).

        Args:
            player_index: Index of the player whose stand to modify.
            slot_index: The slot index to reveal.

        Raises:
            ValueError: If the slot is not hidden or the wire is not
                a known blue wire.
        """
        player = self.players[player_index]
        slot = player.tile_stand.slots[slot_index]
        if not slot.is_hidden:
            raise ValueError(
                f"Slot {slot_index} on player {player_index} is not hidden"
            )
        if slot.wire is None:
            raise ValueError(
                f"Slot {slot_index} on player {player_index} has unknown "
                f"wire (calculator mode — use TileStand.place_info_token "
                f"directly for known values)"
            )
        if slot.wire.color != WireColor.BLUE:
            raise ValueError(
                f"Slot {slot_index} on player {player_index} is not a "
                f"blue wire (is {slot.wire.color.name})"
            )
        value = slot.wire.gameplay_value
        player.tile_stand.place_info_token(slot_index, value)

    def adjust_detonator(self, delta: int) -> None:
        """Change the detonator failure count by delta.

        Positive delta advances the detonator (more failures).
        Negative delta moves it back (used by Rewinder, equipment #6).
        The result is clamped to [0, max_failures].

        Args:
            delta: Amount to change failures by (positive or negative).
        """
        new_failures = self.detonator.failures + delta
        self.detonator.failures = max(
            0, min(new_failures, self.detonator.max_failures)
        )

    def set_detonator(self, mistakes_remaining: int) -> None:
        """Set the detonator to a specific mistakes-remaining value.

        Args:
            mistakes_remaining: Number of remaining mistakes allowed.

        Raises:
            ValueError: If the value is out of valid range.
        """
        if not (0 <= mistakes_remaining <= self.detonator.max_failures):
            raise ValueError(
                f"mistakes_remaining must be 0-{self.detonator.max_failures}, "
                f"got {mistakes_remaining}"
            )
        self.detonator.failures = (
            self.detonator.max_failures - mistakes_remaining
        )

    def reactivate_character_cards(
        self, player_indices: list[int]
    ) -> None:
        """Reset character cards from used to available.

        Used by Emergency Batteries equipment (#7) and Emergency Drop
        (#3.3). Reactivates the character card for each specified player.

        Args:
            player_indices: Indices of players whose character cards
                should be reactivated.

        Raises:
            ValueError: If a player has no character card.
        """
        for p_idx in player_indices:
            card = self.players[p_idx].character_card
            if card is None:
                raise ValueError(
                    f"Player {p_idx} has no character card"
                )
            card.used = False

    def set_current_player(self, player_index: int) -> None:
        """Change which player's turn it is.

        Used by Coffee Mug equipment (#11).

        Args:
            player_index: Index of the new active player.

        Raises:
            ValueError: If the index is out of range.
        """
        if not (0 <= player_index < len(self.players)):
            raise ValueError(
                f"Player index must be 0-{len(self.players) - 1}, "
                f"got {player_index}"
            )
        self.current_player_index = player_index

    def add_must_have(
        self,
        player_index: int,
        value: int | str,
        source: str = "",
    ) -> None:
        """Add a MustHaveValue constraint for a player.

        Used when General Radar (#8) gets a "yes" response: the player
        has at least one uncut wire of this value.

        Args:
            player_index: The player this constraint applies to.
            value: The gameplay value (int 1-12 or 'YELLOW').
            source: Optional description of how this was determined.
        """
        self.slot_constraints.append(
            MustHaveValue(
                player_index=player_index,
                value=value,
                source=source,
            )
        )

    def add_must_not_have(
        self,
        player_index: int,
        value: int | str,
        source: str = "",
    ) -> None:
        """Add a MustNotHaveValue constraint for a player.

        Used when General Radar (#8) gets a "no" response: the player
        does NOT have any uncut wire of this value.

        Args:
            player_index: The player this constraint applies to.
            value: The gameplay value (int 1-12 or 'YELLOW').
            source: Optional description of how this was determined.
        """
        self.slot_constraints.append(
            MustNotHaveValue(
                player_index=player_index,
                value=value,
                source=source,
            )
        )

    def add_adjacent_not_equal(
        self,
        player_index: int,
        slot_left: int,
        slot_right: int,
    ) -> None:
        """Add an AdjacentNotEqual constraint (Label != equipment #1).

        The two slots must be adjacent (slot_right = slot_left + 1).

        Args:
            player_index: The player whose stand is constrained.
            slot_left: Left slot index.
            slot_right: Right slot index.

        Raises:
            ValueError: If slots are not adjacent or out of range.
        """
        stand = self.players[player_index].tile_stand
        if slot_right != slot_left + 1:
            raise ValueError(
                f"Slots must be adjacent: got {slot_left} and {slot_right}"
            )
        if slot_left < 0 or slot_right >= len(stand.slots):
            raise ValueError(
                f"Slot indices out of range for stand with "
                f"{len(stand.slots)} slots"
            )
        self.slot_constraints.append(
            AdjacentNotEqual(
                player_index=player_index,
                slot_index_left=slot_left,
                slot_index_right=slot_right,
            )
        )

    def add_adjacent_equal(
        self,
        player_index: int,
        slot_left: int,
        slot_right: int,
    ) -> None:
        """Add an AdjacentEqual constraint (Label = equipment #12).

        The two slots must be adjacent (slot_right = slot_left + 1).

        Args:
            player_index: The player whose stand is constrained.
            slot_left: Left slot index.
            slot_right: Right slot index.

        Raises:
            ValueError: If slots are not adjacent or out of range.
        """
        stand = self.players[player_index].tile_stand
        if slot_right != slot_left + 1:
            raise ValueError(
                f"Slots must be adjacent: got {slot_left} and {slot_right}"
            )
        if slot_left < 0 or slot_right >= len(stand.slots):
            raise ValueError(
                f"Slot indices out of range for stand with "
                f"{len(stand.slots)} slots"
            )
        self.slot_constraints.append(
            AdjacentEqual(
                player_index=player_index,
                slot_index_left=slot_left,
                slot_index_right=slot_right,
            )
        )

    def cut_all_of_value(self, value: int | str) -> None:
        """Cut all remaining uncut wires of a given value across all stands.

        Used by Disintegrator equipment (#10.10). Marks all hidden wires
        with the matching gameplay value as CUT.

        Args:
            value: The gameplay value to cut (int 1-12 or 'YELLOW').
        """
        for player in self.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_hidden
                    and slot.wire is not None
                    and slot.wire.gameplay_value == value
                ):
                    slot.state = SlotState.CUT
        # Check validation tokens for blue wires
        if isinstance(value, int):
            self._check_validation(value)
        self._check_win()

    def apply_walkie_talkies(self, *args: object, **kwargs: object) -> None:
        """Swap wires between two players (Walkie-Talkies, equipment #2).

        Not yet implemented. Walkie-Talkies change tile stand composition
        and require re-sorting, which touches many game invariants.
        Workaround: enter the post-swap state via ``from_partial_state()``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Walkie-Talkies wire swapping is not yet implemented. "
            "Enter the post-swap game state via from_partial_state() instead."
        )

    def apply_grappling_hook(self, *args: object, **kwargs: object) -> None:
        """Take a teammate's wire (Grappling Hook, equipment #11.11).

        Not yet implemented. Grappling Hook changes tile stand composition
        and requires re-sorting, which touches many game invariants.
        Workaround: enter the post-transfer state via ``from_partial_state()``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Grappling Hook wire transfer is not yet implemented. "
            "Enter the post-transfer game state via from_partial_state() "
            "instead."
        )

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------

    def __str__(self) -> str:
        lines = [
            f"{_Colors.BOLD}=== Bomb Busters ==={_Colors.RESET}",
            str(self.detonator),
            "",
        ]

        # Blue wires in play (only shown when not the standard 1-12)
        blue_values_in_play = sorted({
            int(w.sort_value) for w in self.wires_in_play
            if w.color == WireColor.BLUE
        })
        standard_range = list(range(1, 13))
        if blue_values_in_play and blue_values_in_play != standard_range:
            lines.append(
                f"Blue wires in play: "
                f"{', '.join(str(v) for v in blue_values_in_play)}"
            )

        # Validated / cleared wires
        # Blue: validation tokens (all 4 copies cut)
        # Yellow/Red: unique wires that have been cut
        validated_parts: list[tuple[float, str]] = []
        for v in sorted(self.validation_tokens):
            label = f"{_Colors.BLUE}{v}{_Colors.RESET}"
            validated_parts.append((float(v), label))
        # Scan for cut yellow and red wires
        for player in self.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_cut
                    and slot.wire is not None
                    and slot.wire.color == WireColor.YELLOW
                ):
                    color = _Colors.YELLOW
                    label = f"{color}Y{slot.wire.base_number}{_Colors.RESET}"
                    validated_parts.append((slot.wire.sort_value, label))
                elif (
                    slot.is_cut
                    and slot.wire is not None
                    and slot.wire.color == WireColor.RED
                ):
                    color = _Colors.RED
                    label = f"{color}R{slot.wire.base_number}{_Colors.RESET}"
                    validated_parts.append((slot.wire.sort_value, label))
        # Deduplicate (same wire cut by multiple copies shouldn't repeat)
        seen_sort_values: set[float] = set()
        unique_parts: list[tuple[float, str]] = []
        for sv, label in validated_parts:
            if sv not in seen_sort_values:
                seen_sort_values.add(sv)
                unique_parts.append((sv, label))
        unique_parts.sort(key=lambda x: x[0])
        if unique_parts:
            lines.append(
                f"Validated: {', '.join(label for _, label in unique_parts)}"
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

        # Constraints
        if self.slot_constraints:
            lines.append("Constraints:")
            for c in self.slot_constraints:
                lines.append(f"  {c.describe()}")

        lines.append("")

        # Players
        indent = "    "
        highlight_index = (
            self.active_player_index if self.active_player_index is not None
            else self.current_player_index
        )
        for i, player in enumerate(self.players):
            crown = " 👑" if i == self.captain_index else ""
            if i == highlight_index:
                lines.append(f"{_Colors.BOLD}>>> Player {i}: {player}{crown}{_Colors.RESET}")
            else:
                lines.append(f"{indent}Player {i}: {player}{crown}")
            # Mask hidden wires for non-active players
            mask = (
                self.active_player_index is not None
                and i != self.active_player_index
            )
            status_line, values_line, letters_line = player.tile_stand.stand_lines(
                mask_hidden=mask,
            )
            prefix = indent + "  "
            lines.append(f"{prefix}{status_line}")
            lines.append(f"{prefix}{values_line}")
            lines.append(f"{prefix}{letters_line}")
            lines.append("")

        lines.append("")

        # Game status
        if self.game_over:
            if self.game_won:
                lines.append(f"{_Colors.GREEN}{_Colors.BOLD}MISSION SUCCESS!{_Colors.RESET}")
            else:
                lines.append(f"{_Colors.RED}{_Colors.BOLD}MISSION FAILED!{_Colors.RESET}")

        return "\n".join(lines)
