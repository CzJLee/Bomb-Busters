"""Bomb Busters mission definitions.

Defines the ``Mission`` dataclass for representing mission configurations,
an equipment catalog with all game equipment cards, and a registry of
missions 1-30 with their setup parameters. Missions are static game data
that can be referenced to help initialize a ``GameState`` faster.

Future integration: ``GameState.create_game()`` and
``GameState.from_partial_state()`` will accept an optional ``mission``
parameter to inherit wire configuration, equipment availability, and
other setup details from a mission definition.
"""

from __future__ import annotations

import dataclasses
import enum


# =============================================================================
# Enums
# =============================================================================

class IndicatorTokenType(enum.Enum):
    """Which type of indicator tokens are used in a mission.

    Missions use exactly one indicator token type. The type determines
    what information is revealed when a token is placed on a wire.

    INFO is the default for most missions. EVEN_ODD and MULTIPLICITY
    replace the standard info tokens entirely when in play.
    """
    INFO = enum.auto()           # Standard numbered 1-12 + yellow
    EVEN_ODD = enum.auto()       # Even/Odd tokens (replaces info tokens)
    MULTIPLICITY = enum.auto()   # x1/x2/x3 tokens (replaces info tokens)


class IndicationRule(enum.Enum):
    """How the indication phase works at the start of a mission.

    The indication phase occurs during setup after wires are dealt.
    Most missions use the standard rule where each player picks one
    blue wire to indicate. Some missions modify this phase.
    """
    STANDARD = enum.auto()       # Each player picks 1 blue wire to indicate
    RANDOM_TOKEN = enum.auto()   # Random token from pool, placed if matching
    CAPTAIN_FAKE = enum.auto()   # Captain places 2 fake (must-not-be) tokens
    SKIP = enum.auto()           # No indication phase
    NEGATIVE = enum.auto()       # Place 2 tokens of values NOT in hand


# =============================================================================
# Equipment Catalog
# =============================================================================

@dataclasses.dataclass(frozen=True)
class EquipmentCard:
    """Static definition of an equipment card.

    This is reference data describing what an equipment card does and
    when it unlocks. It is NOT a game-state object — the actual
    ``Equipment`` objects used during gameplay are created by
    ``GameState``.

    Attributes:
        card_id: Unique identifier string. Standard equipment uses
            ``"1"``-``"12"``, yellow uses ``"Y"``, double-numbered
            uses ``"2.2"``, ``"3.3"``, ``"9.9"``, ``"10.10"``,
            ``"11.11"``.
        name: Display name of the equipment card.
        description: Brief description of the equipment's effect.
        unlock_value: Blue wire value (1-12) that unlocks this card.
            For standard equipment, 2 wires of this value must be cut.
            For double-numbered equipment, all 4 wires must be cut.
            Yellow equipment uses 0 (unlocked when 2 yellow wires cut).
        is_double: True for double-numbered equipment cards that are
            unlocked after beating mission 54.
    """
    card_id: str
    name: str
    description: str
    unlock_value: int
    is_double: bool = False


EQUIPMENT_CATALOG: dict[str, EquipmentCard] = {
    # ── Standard equipment (1-12) ─────────────────────────────
    "1": EquipmentCard(
        "1", "Label !=",
        "Put the != token in front of 2 adjacent wires of different values.",
        unlock_value=1,
    ),
    "2": EquipmentCard(
        "2", "Walkie-Talkies",
        "Swap 1 uncut wire with a teammate.",
        unlock_value=2,
    ),
    "3": EquipmentCard(
        "3", "Triple Detector",
        "During a Dual Cut, state a value and point to 3 wires on a "
        "teammate's stand.",
        unlock_value=3,
    ),
    "4": EquipmentCard(
        "4", "Post-It",
        "Put 1 info token in front of 1 of your blue wires.",
        unlock_value=4,
    ),
    "5": EquipmentCard(
        "5", "Super Detector",
        "During a Dual Cut, state a value and point at a teammate's "
        "whole tile stand.",
        unlock_value=5,
    ),
    "6": EquipmentCard(
        "6", "Rewinder",
        "Move the detonator dial back 1 space.",
        unlock_value=6,
    ),
    "7": EquipmentCard(
        "7", "Emergency Batteries",
        "Choose 1 or 2 used character cards and flip them faceup.",
        unlock_value=7,
    ),
    "8": EquipmentCard(
        "8", "General Radar",
        "State a number 1-12. All players say Yes if they have that "
        "value.",
        unlock_value=8,
    ),
    "9": EquipmentCard(
        "9", "Stabilizer",
        "If the next Dual Cut fails, the detonator does not move and "
        "red wires do not explode.",
        unlock_value=9,
    ),
    "10": EquipmentCard(
        "10", "X or Y ray",
        "During a Dual Cut, state 2 values when pointing at 1 wire.",
        unlock_value=10,
    ),
    "11": EquipmentCard(
        "11", "Coffee Mug",
        "Skip your turn and choose who the next active player is.",
        unlock_value=11,
    ),
    "12": EquipmentCard(
        "12", "Label =",
        "Put the = token in front of 2 adjacent wires of the same value.",
        unlock_value=12,
    ),
    # ── Yellow equipment ──────────────────────────────────────
    "Y": EquipmentCard(
        "Y", "False Bottom",
        "Instant effect: take 2 equipment cards and add them to the game.",
        unlock_value=0,
    ),
    # ── Double-numbered equipment (post-mission 54) ───────────
    "2.2": EquipmentCard(
        "2.2", "Single Wire Label",
        "Put a x1 token in front of 1 of your blue wires.",
        unlock_value=2,
        is_double=True,
    ),
    "3.3": EquipmentCard(
        "3.3", "Emergency Drop",
        "Instant effect: flip all used equipment cards faceup.",
        unlock_value=3,
        is_double=True,
    ),
    "9.9": EquipmentCard(
        "9.9", "Fast Pass Card",
        "Solo cut 2 identical wires even if they are not the last "
        "remaining wires of that value.",
        unlock_value=9,
        is_double=True,
    ),
    "10.10": EquipmentCard(
        "10.10", "Disintegrator",
        "Instant effect: draw a random info token number, all players "
        "cut matching wires.",
        unlock_value=10,
        is_double=True,
    ),
    "11.11": EquipmentCard(
        "11.11", "Grappling Hook",
        "Take a teammate's wire without revealing it and add it to "
        "your hand.",
        unlock_value=11,
        is_double=True,
    ),
}


def get_equipment(card_id: str) -> EquipmentCard:
    """Look up an equipment card by its ID.

    Args:
        card_id: The equipment card identifier (e.g., ``"1"``,
            ``"Y"``, ``"9.9"``).

    Returns:
        The corresponding EquipmentCard.

    Raises:
        ValueError: If the card_id is not in the catalog.
    """
    card = EQUIPMENT_CATALOG.get(card_id)
    if card is None:
        raise ValueError(
            f"Unknown equipment card ID: {card_id!r}. "
            f"Valid IDs: {', '.join(sorted(EQUIPMENT_CATALOG.keys()))}"
        )
    return card


# =============================================================================
# Mission Dataclass
# =============================================================================

@dataclasses.dataclass(frozen=True)
class Mission:
    """A mission configuration defining setup parameters.

    Missions are static, immutable game data. Each mission specifies
    which wires, equipment, indicator tokens, and special rules are
    in play. Most fields have sensible defaults matching the standard
    post-training configuration (all blue 1-12, standard info tokens,
    standard indication phase).

    Wire configuration parameters (``yellow_wires``, ``red_wires``)
    use the same type signature as ``GameState.create_game()`` for
    seamless future integration.

    Equipment availability is inferred from the mission number by
    default (see ``available_equipment()``). Only exceptions need to
    be specified via ``equipment_forbidden`` or ``equipment_override``.

    Attributes:
        number: Mission number (1-66).
        name: Mission name/title.
        blue_wire_range: Tuple of (low, high) for blue wire values
            in play. 4 copies of each value. Default ``(1, 12)``
            (all 48 blue wires).
        yellow_wires: Yellow wire specification. ``None`` = no yellow
            wires. ``int`` = direct inclusion of that many random
            yellow wires. ``tuple[int, int]`` = ``(count, pool_size)``
            for X-of-Y uncertain selection.
        red_wires: Red wire specification. Same semantics as
            ``yellow_wires``.
        equipment_forbidden: Equipment card IDs explicitly forbidden
            for this mission. Removed from the default pool computed
            by ``available_equipment()``.
        equipment_override: If set, use EXACTLY these equipment card
            IDs instead of the default pool. Mutually exclusive with
            ``equipment_forbidden``.
        double_detector_disabled: If True, all players' Double
            Detectors are disabled for this mission.
        captain_double_detector_disabled: If True, only the captain's
            Double Detector is disabled.
        indicator_type: Which indicator token system is used.
        indication_rule: How the indication phase works during setup.
        x_tokens: Whether X indicator tokens are in play (1 per
            player, marking an unsorted wire).
        number_cards: Whether number cards are used in this mission.
        sequence_card: Whether a sequence card is used.
        timer_minutes: Optional time limit in minutes.
        red_wires_dealt_separately: If True, red wires are dealt
            one at a time starting with captain (not shuffled into
            the main pool).
        constraint_cards: Whether constraint cards are used.
            Reserved for future missions.
        challenge_cards: Whether challenge cards are used.
            Reserved for future missions (post-54).
        notes: Free-text field for special rules or setup details
            that don't fit into other fields.
    """
    # ── Identity ──────────────────────────────────────────────
    number: int
    name: str

    # ── Wire configuration ────────────────────────────────────
    blue_wire_range: tuple[int, int] = (1, 12)
    yellow_wires: int | tuple[int, int] | None = None
    red_wires: int | tuple[int, int] | None = None

    # ── Equipment ─────────────────────────────────────────────
    equipment_forbidden: tuple[str, ...] = ()
    equipment_override: tuple[str, ...] | None = None

    # ── Character cards ───────────────────────────────────────
    double_detector_disabled: bool = False
    captain_double_detector_disabled: bool = False

    # ── Indicator tokens ──────────────────────────────────────
    indicator_type: IndicatorTokenType = IndicatorTokenType.INFO
    indication_rule: IndicationRule = IndicationRule.STANDARD
    x_tokens: bool = False

    # ── Special components ────────────────────────────────────
    number_cards: bool = False
    sequence_card: bool = False
    timer_minutes: int | None = None
    red_wires_dealt_separately: bool = False

    # ── Future-proofing stubs ─────────────────────────────────
    constraint_cards: bool = False
    challenge_cards: bool = False

    # ── Free-text notes ───────────────────────────────────────
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate mission configuration."""
        # ── Number ────────────────────────────────────────────
        if not isinstance(self.number, int) or self.number < 1:
            raise ValueError(
                f"Mission number must be a positive integer, "
                f"got {self.number!r}"
            )

        # ── Blue wire range ───────────────────────────────────
        low, high = self.blue_wire_range
        if not (1 <= low <= 12 and 1 <= high <= 12 and low <= high):
            raise ValueError(
                f"blue_wire_range must satisfy 1 <= low <= high <= 12, "
                f"got ({low}, {high})"
            )

        # ── Yellow wires ──────────────────────────────────────
        if self.yellow_wires is not None:
            if isinstance(self.yellow_wires, tuple):
                count, pool_size = self.yellow_wires
                if not (1 <= count <= 4):
                    raise ValueError(
                        f"Yellow wire count must be 1-4, got {count}"
                    )
                if pool_size < count:
                    raise ValueError(
                        f"Yellow pool size ({pool_size}) must be >= "
                        f"count ({count})"
                    )
                if pool_size > 11:
                    raise ValueError(
                        f"Yellow pool size ({pool_size}) exceeds "
                        f"available yellow wires (11)"
                    )
            elif isinstance(self.yellow_wires, int):
                if not (1 <= self.yellow_wires <= 4):
                    raise ValueError(
                        f"Yellow wire count must be 1-4, "
                        f"got {self.yellow_wires}"
                    )
            else:
                raise ValueError(
                    f"yellow_wires must be int, tuple[int, int], or "
                    f"None, got {type(self.yellow_wires).__name__}"
                )

        # ── Red wires ─────────────────────────────────────────
        if self.red_wires is not None:
            if isinstance(self.red_wires, tuple):
                count, pool_size = self.red_wires
                if not (1 <= count <= 3):
                    raise ValueError(
                        f"Red wire count must be 1-3, got {count}"
                    )
                if pool_size < count:
                    raise ValueError(
                        f"Red pool size ({pool_size}) must be >= "
                        f"count ({count})"
                    )
                if pool_size > 11:
                    raise ValueError(
                        f"Red pool size ({pool_size}) exceeds "
                        f"available red wires (11)"
                    )
            elif isinstance(self.red_wires, int):
                if not (1 <= self.red_wires <= 3):
                    raise ValueError(
                        f"Red wire count must be 1-3, "
                        f"got {self.red_wires}"
                    )
            else:
                raise ValueError(
                    f"red_wires must be int, tuple[int, int], or "
                    f"None, got {type(self.red_wires).__name__}"
                )

        # ── Equipment mutual exclusion ────────────────────────
        if self.equipment_forbidden and self.equipment_override is not None:
            raise ValueError(
                "equipment_forbidden and equipment_override are "
                "mutually exclusive — set one or the other, not both"
            )

        # ── Equipment IDs exist in catalog ────────────────────
        for eid in self.equipment_forbidden:
            if eid not in EQUIPMENT_CATALOG:
                raise ValueError(
                    f"Unknown equipment ID in equipment_forbidden: "
                    f"{eid!r}"
                )
        if self.equipment_override is not None:
            for eid in self.equipment_override:
                if eid not in EQUIPMENT_CATALOG:
                    raise ValueError(
                        f"Unknown equipment ID in equipment_override: "
                        f"{eid!r}"
                    )

    def available_equipment(self) -> list[str]:
        """Compute the list of available equipment card IDs.

        The default equipment pool is inferred from the mission number:

        - Missions 1-2: no equipment.
        - Mission 3: equipment 1-10.
        - Missions 4-8: equipment 1-12.
        - Missions 9+: equipment 1-12 + yellow (if yellow wires in
          setup).
        - Missions 55+: equipment 1-12 + yellow + double-numbered.

        ``equipment_override`` replaces the default pool entirely.
        ``equipment_forbidden`` removes specific IDs from the default.

        Returns:
            List of equipment card ID strings available for this
            mission.
        """
        if self.equipment_override is not None:
            return list(self.equipment_override)

        if self.number <= 2:
            pool: list[str] = []
        elif self.number == 3:
            pool = [str(i) for i in range(1, 11)]
        else:
            pool = [str(i) for i in range(1, 13)]

        # Yellow equipment from mission 9+ (if yellow wires present)
        if self.number >= 9 and self.yellow_wires is not None:
            pool.append("Y")

        # Double-numbered from mission 55+
        if self.number >= 55:
            pool.extend(["2.2", "3.3", "9.9", "10.10", "11.11"])

        # Apply forbidden
        if self.equipment_forbidden:
            forbidden = set(self.equipment_forbidden)
            pool = [eid for eid in pool if eid not in forbidden]

        return pool

    def wire_config(self) -> dict[str, object]:
        """Return wire configuration as a dict.

        Keys match ``GameState.from_partial_state()`` parameter names.
        Only non-default values are included.

        Returns:
            Dict with keys ``blue_wires``, ``yellow_wires``, and/or
            ``red_wires`` as needed. Empty dict if all defaults.
        """
        result: dict[str, object] = {}
        low, high = self.blue_wire_range
        if (low, high) != (1, 12):
            result["blue_wires"] = (low, high)
        if self.yellow_wires is not None:
            result["yellow_wires"] = self.yellow_wires
        if self.red_wires is not None:
            result["red_wires"] = self.red_wires
        return result

    def __str__(self) -> str:
        return f"Mission {self.number}: {self.name}"


# =============================================================================
# Mission Registry
# =============================================================================

MISSIONS: dict[int, Mission] = {}


def get_mission(number: int) -> Mission:
    """Look up a mission by its number.

    Args:
        number: The mission number to look up.

    Returns:
        The Mission object for that number.

    Raises:
        ValueError: If no mission with that number is defined.
    """
    mission = MISSIONS.get(number)
    if mission is None:
        defined = sorted(MISSIONS.keys())
        raise ValueError(
            f"Mission {number} is not defined. "
            f"Defined missions: {defined}"
        )
    return mission


def all_missions() -> list[Mission]:
    """Return all defined missions sorted by mission number.

    Returns:
        List of Mission objects in ascending order by number.
    """
    return [MISSIONS[k] for k in sorted(MISSIONS.keys())]


# =============================================================================
# Mission Definitions
# =============================================================================

# ── Training Missions (1-8) ───────────────────────────────────

MISSIONS[1] = Mission(
    number=1,
    name="TRAINING, Day 1",
    blue_wire_range=(1, 6),
)

MISSIONS[2] = Mission(
    number=2,
    name="TRAINING, Day 2",
    blue_wire_range=(1, 8),
    yellow_wires=2,
)

MISSIONS[3] = Mission(
    number=3,
    name="TRAINING, Day 3",
    blue_wire_range=(1, 10),
    red_wires=1,
)

MISSIONS[4] = Mission(
    number=4,
    name="TRAINING: First Day in the Field",
    red_wires=1,
    yellow_wires=2,
)

MISSIONS[5] = Mission(
    number=5,
    name="TRAINING: Second Day in the Field",
    red_wires=1,
    yellow_wires=(2, 3),
)

MISSIONS[6] = Mission(
    number=6,
    name="TRAINING: THIRD Day in the Field",
    red_wires=1,
    yellow_wires=4,
)

MISSIONS[7] = Mission(
    number=7,
    name="TRAINING: Last Day of Class",
    red_wires=(1, 2),
)

MISSIONS[8] = Mission(
    number=8,
    name="FINAL EXAM",
    red_wires=(1, 2),
    yellow_wires=(2, 3),
)

# ── Post-Training Missions (9-30) ────────────────────────────

MISSIONS[9] = Mission(
    number=9,
    name="A Sense of Priorities",
    red_wires=1,
    yellow_wires=2,
    number_cards=True,
    sequence_card=True,
)

MISSIONS[10] = Mission(
    number=10,
    name="A Rough Patch",
    red_wires=1,
    yellow_wires=4,
    equipment_forbidden=("11",),
    timer_minutes=15,
)

MISSIONS[11] = Mission(
    number=11,
    name="Blue on Red, Looks Like We Are Dead",
    yellow_wires=2,
    number_cards=True,
)

MISSIONS[12] = Mission(
    number=12,
    name="Wrapped in Red Tape",
    red_wires=1,
    yellow_wires=4,
    number_cards=True,
    notes="One number card per equipment card in play.",
)

MISSIONS[13] = Mission(
    number=13,
    name="Red Alert!",
    red_wires=3,
    red_wires_dealt_separately=True,
    indication_rule=IndicationRule.RANDOM_TOKEN,
    notes=(
        "Red wires dealt 1-at-a-time starting with captain. "
        "During indication, each player draws a random info token "
        "and places it if they have that value, or to the side if not."
    ),
)

MISSIONS[14] = Mission(
    number=14,
    name="High-Risk Bomb Disposal Expert (aka. NOOB)",
    red_wires=2,
    yellow_wires=(2, 3),
)

MISSIONS[15] = Mission(
    number=15,
    name="Mission in Новосибирск",
    red_wires=(1, 3),
    number_cards=True,
)

MISSIONS[16] = Mission(
    number=16,
    name="Time to Reprioritize (Is this deja vu?)",
    red_wires=1,
    yellow_wires=(2, 3),
    number_cards=True,
    sequence_card=True,
)

MISSIONS[17] = Mission(
    number=17,
    name="Rhett Herrings",
    red_wires=(2, 3),
    indication_rule=IndicationRule.CAPTAIN_FAKE,
    notes=(
        "Captain places 2 fake info tokens (must-not-be-value "
        "constraints, cannot be placed in front of red wires). "
        "Other players indicate normally."
    ),
)

MISSIONS[18] = Mission(
    number=18,
    name="BAT-Helping-Hand",
    red_wires=2,
    equipment_override=("8",),
    indication_rule=IndicationRule.SKIP,
    number_cards=True,
)

MISSIONS[19] = Mission(
    number=19,
    name="In the Belly of the Beast",
    red_wires=1,
    yellow_wires=(2, 3),
)

MISSIONS[20] = Mission(
    number=20,
    name="The Big Bad Wolf",
    red_wires=2,
    yellow_wires=2,
    equipment_forbidden=("2",),
    x_tokens=True,
    notes=(
        "Last wire dealt to each stand is not sorted; placed at "
        "far right with X token. The X wire can be any color."
    ),
)

MISSIONS[21] = Mission(
    number=21,
    name="Death by Haggis",
    red_wires=(1, 2),
    indicator_type=IndicatorTokenType.EVEN_ODD,
)

MISSIONS[22] = Mission(
    number=22,
    name="Negative Impressions",
    red_wires=1,
    yellow_wires=4,
    indication_rule=IndicationRule.NEGATIVE,
    notes=(
        "Each player takes 2 tokens of values they do NOT have "
        "and puts them next to their tile stand."
    ),
)

MISSIONS[23] = Mission(
    number=23,
    name="Defusing in Fordwich (381 inhabitants, 64 miles from London)",
    red_wires=(1, 3),
    number_cards=True,
    notes=(
        "No equipment during setup. Instead, create a deck of 7 "
        "random equipment cards placed facedown. Select a random "
        "number card and place faceup next to mission card."
    ),
)

MISSIONS[24] = Mission(
    number=24,
    name="Tally Ho!",
    red_wires=2,
    indicator_type=IndicatorTokenType.MULTIPLICITY,
    notes="x1/x2/x3 tokens cannot be placed in front of red wires.",
)

MISSIONS[25] = Mission(
    number=25,
    name="The Better to Hear You with...",
    red_wires=2,
)

MISSIONS[26] = Mission(
    number=26,
    name="Speaking of the Wolf...",
    red_wires=2,
    equipment_forbidden=("10",),
    number_cards=True,
)

MISSIONS[27] = Mission(
    number=27,
    name="Playing with Wire",
    red_wires=1,
    yellow_wires=4,
    double_detector_disabled=True,
    equipment_forbidden=("7",),
)

MISSIONS[28] = Mission(
    number=28,
    name="Captain Careless",
    red_wires=2,
    yellow_wires=4,
    captain_double_detector_disabled=True,
)

MISSIONS[29] = Mission(
    number=29,
    name="Guessing Game",
    red_wires=3,
    number_cards=True,
    notes=(
        "Shuffle and deal 2 number cards facedown to each player "
        "(3 cards to the player on captain's right), rest facedown "
        "in a deck."
    ),
)

MISSIONS[30] = Mission(
    number=30,
    name="Speed Mission!",
    red_wires=(1, 2),
    yellow_wires=4,
    number_cards=True,
)
