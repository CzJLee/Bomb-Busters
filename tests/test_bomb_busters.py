"""Unit tests for bomb_busters game model."""

import unittest

import bomb_busters


class TestWire(unittest.TestCase):
    """Tests for the bomb_busters.Wire dataclass."""

    def test_blue_wire_creation(self) -> None:
        w = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        self.assertEqual(w.gameplay_value, 5)
        self.assertEqual(w.base_number, 5)
        self.assertEqual(w.color, bomb_busters.WireColor.BLUE)

    def test_yellow_wire_creation(self) -> None:
        w = bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 5.1)
        self.assertEqual(w.gameplay_value, "YELLOW")
        self.assertEqual(w.base_number, 5)
        self.assertEqual(w.color, bomb_busters.WireColor.YELLOW)

    def test_red_wire_creation(self) -> None:
        w = bomb_busters.Wire(bomb_busters.WireColor.RED, 5.5)
        self.assertEqual(w.gameplay_value, "RED")
        self.assertEqual(w.base_number, 5)
        self.assertEqual(w.color, bomb_busters.WireColor.RED)

    def test_wire_sorting(self) -> None:
        wires = [
            bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
            bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 2.1),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
        ]
        sorted_wires = sorted(wires)
        expected = [1.0, 2.0, 2.1, 2.5, 3.0]
        self.assertEqual([w.sort_value for w in sorted_wires], expected)

    def test_full_sort_order(self) -> None:
        """Verify blue < yellow < red for the same base number."""
        blue = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        yellow = bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 5.1)
        red = bomb_busters.Wire(bomb_busters.WireColor.RED, 5.5)
        self.assertLess(blue, yellow)
        self.assertLess(yellow, red)
        self.assertLess(blue, red)

    def test_cross_number_sort(self) -> None:
        """Red 2.5 < Blue 3.0."""
        red2 = bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5)
        blue3 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)
        self.assertLess(red2, blue3)

    def test_wire_equality(self) -> None:
        w1 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        w2 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        self.assertEqual(w1, w2)
        self.assertIsNot(w1, w2)

    def test_wire_inequality(self) -> None:
        w1 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        w2 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)
        self.assertNotEqual(w1, w2)

    def test_wire_hash(self) -> None:
        w1 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        w2 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        self.assertEqual(hash(w1), hash(w2))
        # Can be used in sets
        s = {w1, w2}
        self.assertEqual(len(s), 1)

    def test_wire_frozen(self) -> None:
        w = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        with self.assertRaises(AttributeError):
            w.color = bomb_busters.WireColor.RED  # type: ignore[misc]

    def test_str_output(self) -> None:
        blue = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        yellow = bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 3.1)
        red = bomb_busters.Wire(bomb_busters.WireColor.RED, 7.5)
        # Just verify they produce non-empty strings without errors
        self.assertIn("5", str(blue))
        self.assertIn("3", str(yellow))
        self.assertIn("7", str(red))


class TestWireFactories(unittest.TestCase):
    """Tests for wire factory functions."""

    def test_blue_wire_count(self) -> None:
        blues = bomb_busters.create_all_blue_wires()
        self.assertEqual(len(blues), 48)

    def test_blue_wire_distribution(self) -> None:
        blues = bomb_busters.create_all_blue_wires()
        for number in range(1, 13):
            count = sum(1 for w in blues if w.sort_value == float(number))
            self.assertEqual(count, 4, f"Expected 4 blue-{number} wires")

    def test_red_wire_count(self) -> None:
        reds = bomb_busters.create_all_red_wires()
        self.assertEqual(len(reds), 11)

    def test_red_wire_values(self) -> None:
        reds = bomb_busters.create_all_red_wires()
        expected = [n + 0.5 for n in range(1, 12)]
        self.assertEqual([w.sort_value for w in reds], expected)

    def test_yellow_wire_count(self) -> None:
        yellows = bomb_busters.create_all_yellow_wires()
        self.assertEqual(len(yellows), 11)

    def test_yellow_wire_values(self) -> None:
        yellows = bomb_busters.create_all_yellow_wires()
        expected = [n + 0.1 for n in range(1, 12)]
        self.assertEqual([w.sort_value for w in yellows], expected)


class TestSlot(unittest.TestCase):
    """Tests for the bomb_busters.Slot dataclass."""

    def test_default_state(self) -> None:
        s = bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0))
        self.assertTrue(s.is_hidden)
        self.assertFalse(s.is_cut)
        self.assertFalse(s.is_info_revealed)
        self.assertIsNone(s.info_token)

    def test_cut_state(self) -> None:
        s = bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), state=bomb_busters.SlotState.CUT)
        self.assertTrue(s.is_cut)
        self.assertFalse(s.is_hidden)

    def test_info_revealed_state(self) -> None:
        s = bomb_busters.Slot(
            wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
            state=bomb_busters.SlotState.INFO_REVEALED,
            info_token=5,
        )
        self.assertTrue(s.is_info_revealed)
        self.assertEqual(s.info_token, 5)

    def test_unknown_wire_slot(self) -> None:
        s = bomb_busters.Slot(wire=None)
        self.assertTrue(s.is_hidden)
        self.assertIsNone(s.wire)

    def test_str_hidden(self) -> None:
        s = bomb_busters.Slot(wire=None)
        self.assertIn("?", str(s))

    def test_str_cut(self) -> None:
        s = bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), state=bomb_busters.SlotState.CUT)
        output = str(s)
        self.assertIn("5", output)

    def test_yellow_info_token_display(self) -> None:
        """Yellow info token displays as 'Y', not 'YELLOW'."""
        # Simulation mode (wire is known)
        s = bomb_busters.Slot(
            wire=bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 4.1),
            state=bomb_busters.SlotState.INFO_REVEALED,
            info_token="YELLOW",
        )
        plain, colored = s.value_label()
        self.assertEqual(plain, "Y")
        self.assertIn("Y", colored)
        self.assertNotIn("YELLOW", colored)

        # Calculator mode (wire is None)
        s_calc = bomb_busters.Slot(
            wire=None,
            state=bomb_busters.SlotState.INFO_REVEALED,
            info_token="YELLOW",
        )
        plain_calc, colored_calc = s_calc.value_label()
        self.assertEqual(plain_calc, "Y")
        self.assertIn("Y", colored_calc)
        self.assertNotIn("YELLOW", colored_calc)

    def test_blue_info_token_display_color(self) -> None:
        """Blue info token displays in blue, not white."""
        # With wire set (normal case after parser fix)
        s = bomb_busters.Slot(
            wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
            state=bomb_busters.SlotState.INFO_REVEALED,
            info_token=6,
        )
        _, colored = s.value_label()
        self.assertIn("\033[94m", colored)  # BLUE ANSI code

        # Defensive: wire=None but info_token is int (manually constructed)
        s_calc = bomb_busters.Slot(
            wire=None,
            state=bomb_busters.SlotState.INFO_REVEALED,
            info_token=6,
        )
        _, colored_calc = s_calc.value_label()
        self.assertIn("\033[94m", colored_calc)  # BLUE ANSI code


class TestTileStand(unittest.TestCase):
    """Tests for the bomb_busters.TileStand class."""

    def test_from_wires_sorts(self) -> None:
        wires = [
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
        ]
        stand = bomb_busters.TileStand.from_wires(wires)
        values = [s.wire.sort_value for s in stand.slots if s.wire]
        self.assertEqual(values, [1.0, 3.0, 7.0, 10.0])

    def test_from_wires_mixed_colors(self) -> None:
        wires = [
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
            bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 2.1),
            bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5),
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
        ]
        stand = bomb_busters.TileStand.from_wires(wires)
        values = [s.wire.sort_value for s in stand.slots if s.wire]
        self.assertEqual(values, [1.0, 2.1, 2.5, 3.0])

    def test_hidden_slots(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i)) for i in range(1, 5)]
        stand = bomb_busters.TileStand.from_wires(wires)
        self.assertEqual(len(stand.hidden_slots), 4)

    def test_cut_wire_at(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i)) for i in range(1, 5)]
        stand = bomb_busters.TileStand.from_wires(wires)
        stand.cut_wire_at(0)
        self.assertEqual(len(stand.hidden_slots), 3)
        self.assertEqual(len(stand.cut_slots), 1)
        self.assertTrue(stand.slots[0].is_cut)

    def test_cut_wire_at_invalid_index(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        with self.assertRaises(IndexError):
            stand.cut_wire_at(5)

    def test_cut_wire_at_already_cut(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        stand.cut_wire_at(0)
        with self.assertRaises(ValueError):
            stand.cut_wire_at(0)

    def test_place_info_token(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        stand.place_info_token(0, 5)
        self.assertTrue(stand.slots[0].is_info_revealed)
        self.assertEqual(stand.slots[0].info_token, 5)
        self.assertEqual(len(stand.info_revealed_slots), 1)

    def test_is_empty(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        self.assertFalse(stand.is_empty)
        stand.cut_wire_at(0)
        self.assertFalse(stand.is_empty)
        stand.cut_wire_at(1)
        self.assertTrue(stand.is_empty)

    def test_remaining_count(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i)) for i in range(1, 4)]
        stand = bomb_busters.TileStand.from_wires(wires)
        self.assertEqual(stand.remaining_count, 3)
        stand.cut_wire_at(0)
        self.assertEqual(stand.remaining_count, 2)

    def test_wire_stays_in_position_after_cut(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        stand.cut_wire_at(0)
        # bomb_busters.Wire is still at index 0, just marked as cut
        self.assertEqual(len(stand.slots), 2)
        self.assertEqual(stand.slots[0].wire, bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0))
        self.assertTrue(stand.slots[0].is_cut)

    def test_str_output(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        output = str(stand)
        self.assertIn("A", output)
        self.assertIn("B", output)

    def test_stand_lines(self) -> None:
        wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)]
        stand = bomb_busters.TileStand.from_wires(wires)
        _, _, letters = stand.stand_lines()
        self.assertIn("A", letters)
        self.assertIn("B", letters)


class TestTileStandFromString(unittest.TestCase):
    """Tests for TileStand.from_string() shorthand notation."""

    # -- CUT wires (no prefix) --

    def test_cut_blue_wires(self) -> None:
        """Plain numbers produce CUT blue wires."""
        stand = bomb_busters.TileStand.from_string("1 5 12")
        self.assertEqual(len(stand.slots), 3)
        for slot in stand.slots:
            self.assertEqual(slot.state, bomb_busters.SlotState.CUT)
            self.assertEqual(slot.wire.color, bomb_busters.WireColor.BLUE)
        self.assertEqual(stand.slots[0].wire.sort_value, 1.0)
        self.assertEqual(stand.slots[1].wire.sort_value, 5.0)
        self.assertEqual(stand.slots[2].wire.sort_value, 12.0)

    def test_cut_yellow_wire(self) -> None:
        """'Y' prefix produces a CUT yellow wire."""
        stand = bomb_busters.TileStand.from_string("Y4")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.CUT)
        self.assertEqual(slot.wire.color, bomb_busters.WireColor.YELLOW)
        self.assertEqual(slot.wire.sort_value, 4.1)

    def test_cut_red_wire(self) -> None:
        """'R' prefix produces a CUT red wire."""
        stand = bomb_busters.TileStand.from_string("R5")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.CUT)
        self.assertEqual(slot.wire.color, bomb_busters.WireColor.RED)
        self.assertEqual(slot.wire.sort_value, 5.5)

    # -- HIDDEN wires ('?' prefix) --

    def test_hidden_unknown(self) -> None:
        """'?' alone produces a HIDDEN slot with wire=None."""
        stand = bomb_busters.TileStand.from_string("? ? ?")
        self.assertEqual(len(stand.slots), 3)
        for slot in stand.slots:
            self.assertEqual(slot.state, bomb_busters.SlotState.HIDDEN)
            self.assertIsNone(slot.wire)

    def test_hidden_known_blue(self) -> None:
        """'?N' produces a HIDDEN blue wire known to observer."""
        stand = bomb_busters.TileStand.from_string("?4 ?8")
        self.assertEqual(len(stand.slots), 2)
        for slot in stand.slots:
            self.assertEqual(slot.state, bomb_busters.SlotState.HIDDEN)
            self.assertEqual(slot.wire.color, bomb_busters.WireColor.BLUE)
        self.assertEqual(stand.slots[0].wire.sort_value, 4.0)
        self.assertEqual(stand.slots[1].wire.sort_value, 8.0)

    def test_hidden_yellow(self) -> None:
        """'?YN' produces a HIDDEN yellow wire."""
        stand = bomb_busters.TileStand.from_string("?Y4")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.HIDDEN)
        self.assertEqual(slot.wire.color, bomb_busters.WireColor.YELLOW)
        self.assertEqual(slot.wire.sort_value, 4.1)

    def test_hidden_red(self) -> None:
        """'?RN' produces a HIDDEN red wire."""
        stand = bomb_busters.TileStand.from_string("?R5")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.HIDDEN)
        self.assertEqual(slot.wire.color, bomb_busters.WireColor.RED)
        self.assertEqual(slot.wire.sort_value, 5.5)

    # -- INFO_REVEALED wires ('i' prefix) --

    def test_info_blue(self) -> None:
        """'iN' produces INFO_REVEALED with blue wire and info token."""
        stand = bomb_busters.TileStand.from_string("i5")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.INFO_REVEALED)
        self.assertIsNotNone(slot.wire)
        self.assertEqual(
            slot.wire,
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
        )
        self.assertEqual(slot.info_token, 5)

    def test_info_yellow(self) -> None:
        """'iY' produces INFO_REVEALED with yellow info token, wire unknown."""
        stand = bomb_busters.TileStand.from_string("iY")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.INFO_REVEALED)
        self.assertIsNone(slot.wire)
        self.assertEqual(slot.info_token, "YELLOW")

    def test_info_yellow_with_number(self) -> None:
        """'iYN' produces INFO_REVEALED yellow with known wire identity."""
        stand = bomb_busters.TileStand.from_string("iY4")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.INFO_REVEALED)
        self.assertIsNotNone(slot.wire)
        self.assertEqual(
            slot.wire,
            bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 4.1),
        )
        self.assertEqual(slot.info_token, "YELLOW")

    def test_info_yellow_with_number_case_insensitive(self) -> None:
        """'iy4' is case-insensitive."""
        stand = bomb_busters.TileStand.from_string("iy4")
        slot = stand.slots[0]
        self.assertEqual(slot.state, bomb_busters.SlotState.INFO_REVEALED)
        self.assertEqual(
            slot.wire,
            bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 4.1),
        )
        self.assertEqual(slot.info_token, "YELLOW")

    def test_info_red_raises(self) -> None:
        """'iR' and 'iRN' are invalid — no red info tokens exist."""
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("iR")
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("iR5")
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("ir3")

    # -- Case insensitivity --

    def test_case_insensitive(self) -> None:
        """Prefixes Y, R, i are case-insensitive."""
        stand = bomb_busters.TileStand.from_string("y4 r5 iy i5")
        self.assertEqual(stand.slots[0].wire.color, bomb_busters.WireColor.YELLOW)
        self.assertEqual(stand.slots[0].state, bomb_busters.SlotState.CUT)
        self.assertEqual(stand.slots[1].wire.color, bomb_busters.WireColor.RED)
        self.assertEqual(stand.slots[1].state, bomb_busters.SlotState.CUT)
        self.assertEqual(stand.slots[2].info_token, "YELLOW")
        self.assertEqual(stand.slots[3].info_token, 5)
        self.assertEqual(stand.slots[3].wire.color, bomb_busters.WireColor.BLUE)

    # -- Mixed stands (simulate.py examples) --

    def test_alice_stand(self) -> None:
        """Observer's own stand: mix of cut and hidden-known wires."""
        stand = bomb_busters.TileStand.from_string(
            "1 2 ?4 ?Y4 ?6 ?7 ?8 ?8 9 11 12",
        )
        self.assertEqual(len(stand.slots), 11)
        # Cut slots: indices 0,1,8,9,10
        for i in (0, 1, 8, 9, 10):
            self.assertEqual(stand.slots[i].state, bomb_busters.SlotState.CUT)
        # Hidden slots: indices 2,3,4,5,6,7
        for i in (2, 3, 4, 5, 6, 7):
            self.assertEqual(stand.slots[i].state, bomb_busters.SlotState.HIDDEN)
            self.assertIsNotNone(stand.slots[i].wire)
        # Yellow wire at index 3
        self.assertEqual(
            stand.slots[3].wire.color, bomb_busters.WireColor.YELLOW,
        )
        self.assertEqual(stand.slots[3].wire.sort_value, 4.1)

    def test_bob_stand(self) -> None:
        """Other player's stand: mix of cut and unknown hidden wires."""
        stand = bomb_busters.TileStand.from_string("1 3 ? ? ? 8 9 ? ? 12")
        self.assertEqual(len(stand.slots), 10)
        # Cut slots
        for i in (0, 1, 5, 6, 9):
            self.assertEqual(stand.slots[i].state, bomb_busters.SlotState.CUT)
        # Hidden unknown slots
        for i in (2, 3, 4, 7, 8):
            self.assertEqual(stand.slots[i].state, bomb_busters.SlotState.HIDDEN)
            self.assertIsNone(stand.slots[i].wire)

    def test_diana_stand_with_info_token(self) -> None:
        """Stand with an info token among cut and hidden wires."""
        stand = bomb_busters.TileStand.from_string("2 3 ? ? i6 ? ? 9 ? 11")
        self.assertEqual(len(stand.slots), 10)
        # Info-revealed at index 4
        self.assertEqual(
            stand.slots[4].state, bomb_busters.SlotState.INFO_REVEALED,
        )
        self.assertEqual(stand.slots[4].info_token, 6)
        self.assertEqual(
            stand.slots[4].wire,
            bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
        )

    # -- Custom separator --

    def test_custom_separator(self) -> None:
        """sep parameter changes the token delimiter."""
        stand = bomb_busters.TileStand.from_string("1,2,3", sep=",")
        self.assertEqual(len(stand.slots), 3)
        for slot in stand.slots:
            self.assertEqual(slot.state, bomb_busters.SlotState.CUT)
            self.assertEqual(slot.wire.color, bomb_busters.WireColor.BLUE)

    # -- num_tiles validation --

    def test_num_tiles_correct(self) -> None:
        """num_tiles passes when count matches."""
        stand = bomb_busters.TileStand.from_string("1 2 3", num_tiles=3)
        self.assertEqual(len(stand.slots), 3)

    def test_num_tiles_mismatch(self) -> None:
        """num_tiles raises ValueError when count doesn't match."""
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("1 2 3", num_tiles=5)

    # -- Error cases --

    def test_empty_string(self) -> None:
        """Empty notation raises ValueError."""
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("")

    def test_whitespace_only(self) -> None:
        """Whitespace-only notation raises ValueError."""
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("   ")

    def test_invalid_token(self) -> None:
        """Unrecognizable token raises ValueError."""
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("1 abc 3")

    def test_info_token_missing_value(self) -> None:
        """Bare 'i' with no value raises ValueError."""
        with self.assertRaises(ValueError):
            bomb_busters.TileStand.from_string("i")


class TestSortValueBounds(unittest.TestCase):
    """Tests for the bomb_busters.get_sort_value_bounds helper."""

    def test_unconstrained(self) -> None:
        slots = [bomb_busters.Slot(wire=None), bomb_busters.Slot(wire=None), bomb_busters.Slot(wire=None)]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 13.0)

    def test_bounded_by_neighbors(self) -> None:
        slots = [
            bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), state=bomb_busters.SlotState.CUT),
            bomb_busters.Slot(wire=None),
            bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), state=bomb_busters.SlotState.CUT),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 7.0)

    def test_left_bound_only(self) -> None:
        slots = [
            bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), state=bomb_busters.SlotState.CUT),
            bomb_busters.Slot(wire=None),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 13.0)

    def test_right_bound_only(self) -> None:
        slots = [
            bomb_busters.Slot(wire=None),
            bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), state=bomb_busters.SlotState.CUT),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 7.0)

    def test_skips_none_wires(self) -> None:
        slots = [
            bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), state=bomb_busters.SlotState.CUT),
            bomb_busters.Slot(wire=None),
            bomb_busters.Slot(wire=None),  # target
            bomb_busters.Slot(wire=None),
            bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0), state=bomb_busters.SlotState.CUT),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 2)
        self.assertEqual(lower, 2.0)
        self.assertEqual(upper, 9.0)

    def test_skips_hidden_wires_in_simulation_mode(self) -> None:
        """Hidden slots with wires (simulation mode) must not be used as bounds.

        In simulation mode, hidden slots have actual Wire objects, but
        the observer cannot see them. Only CUT and INFO_REVEALED slots
        should provide bounds.
        """
        slots = [
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
                state=bomb_busters.SlotState.CUT,
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                state=bomb_busters.SlotState.HIDDEN,
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
                state=bomb_busters.SlotState.HIDDEN,  # target
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
                state=bomb_busters.SlotState.HIDDEN,
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0),
                state=bomb_busters.SlotState.CUT,
            ),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 2)
        # Should use CUT neighbors (2.0 and 9.0), NOT hidden neighbors (5.0 and 7.0)
        self.assertEqual(lower, 2.0)
        self.assertEqual(upper, 9.0)

    def test_uses_info_revealed_as_bounds(self) -> None:
        """INFO_REVEALED slots should be used as bounds (they are public)."""
        slots = [
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
                state=bomb_busters.SlotState.INFO_REVEALED,
                info_token=3,
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                state=bomb_busters.SlotState.HIDDEN,  # target
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0),
                state=bomb_busters.SlotState.INFO_REVEALED,
                info_token=8,
            ),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 8.0)

    def test_info_revealed_blue_wire_none_uses_info_token(self) -> None:
        """INFO_REVEALED blue slot with wire=None uses info_token as bound.

        In calculator mode, manually constructed INFO_REVEALED slots may
        have wire=None but info_token set to an int. The sort_value should
        be inferred from the info_token.
        """
        slots = [
            bomb_busters.Slot(
                wire=None,
                state=bomb_busters.SlotState.INFO_REVEALED,
                info_token=3,
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                state=bomb_busters.SlotState.HIDDEN,  # target
            ),
            bomb_busters.Slot(
                wire=None,
                state=bomb_busters.SlotState.INFO_REVEALED,
                info_token=8,
            ),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 8.0)

    def test_yellow_info_revealed_not_used_as_bound(self) -> None:
        """INFO_REVEALED yellow slot with wire=None doesn't provide bounds.

        Yellow info tokens show 'YELLOW' but not a specific sort_value,
        so they cannot narrow bounds for adjacent hidden slots.
        """
        slots = [
            bomb_busters.Slot(
                wire=None,
                state=bomb_busters.SlotState.INFO_REVEALED,
                info_token="YELLOW",
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                state=bomb_busters.SlotState.HIDDEN,  # target
            ),
            bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0),
                state=bomb_busters.SlotState.CUT,
            ),
        ]
        lower, upper = bomb_busters.get_sort_value_bounds(slots, 1)
        # Yellow info (wire=None, no numeric token) cannot provide lower bound
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 9.0)


class TestCharacterCard(unittest.TestCase):
    """Tests for the bomb_busters.CharacterCard class."""

    def test_creation(self) -> None:
        card = bomb_busters.create_double_detector()
        self.assertEqual(card.name, "Double Detector")
        self.assertFalse(card.used)

    def test_use(self) -> None:
        card = bomb_busters.create_double_detector()
        card.use()
        self.assertTrue(card.used)

    def test_use_twice_raises(self) -> None:
        card = bomb_busters.create_double_detector()
        card.use()
        with self.assertRaises(ValueError):
            card.use()

    def test_str_available(self) -> None:
        card = bomb_busters.create_double_detector()
        self.assertIn("available", str(card))

    def test_str_used(self) -> None:
        card = bomb_busters.create_double_detector()
        card.use()
        self.assertIn("used", str(card))


class TestDetonator(unittest.TestCase):
    """Tests for the bomb_busters.Detonator class."""

    def test_initial_state(self) -> None:
        d = bomb_busters.Detonator(max_failures=4)
        self.assertEqual(d.failures, 0)
        self.assertEqual(d.remaining_failures, 4)
        self.assertFalse(d.is_exploded)

    def test_advance(self) -> None:
        d = bomb_busters.Detonator(max_failures=4)
        self.assertFalse(d.advance())  # 1
        self.assertEqual(d.remaining_failures, 3)
        self.assertFalse(d.advance())  # 2
        self.assertFalse(d.advance())  # 3
        self.assertTrue(d.advance())   # 4 -> exploded
        self.assertTrue(d.is_exploded)
        self.assertEqual(d.remaining_failures, 0)

    def test_five_players(self) -> None:
        """5 players = 4 failures allowed."""
        d = bomb_busters.Detonator(max_failures=4)
        for _ in range(3):
            self.assertFalse(d.advance())
        self.assertTrue(d.advance())

    def test_four_players(self) -> None:
        """4 players = 3 failures allowed."""
        d = bomb_busters.Detonator(max_failures=3)
        self.assertFalse(d.advance())
        self.assertFalse(d.advance())
        self.assertTrue(d.advance())

    def test_str_output(self) -> None:
        d = bomb_busters.Detonator(max_failures=4)
        d.advance()
        output = str(d)
        self.assertIn("Mistakes remaining:", output)
        self.assertIn("3", output)


class TestEquipment(unittest.TestCase):
    """Tests for the bomb_busters.Equipment class."""

    def test_creation(self) -> None:
        e = bomb_busters.Equipment(
            name="Walkie-Talkies",
            description="Exchange wires",
            unlock_value=2,
        )
        self.assertFalse(e.used)
        self.assertFalse(e.unlocked)

    def test_unlock_and_use(self) -> None:
        e = bomb_busters.Equipment(name="Test", description="", unlock_value=3)
        e.unlock()
        self.assertTrue(e.unlocked)
        e.use()
        self.assertTrue(e.used)

    def test_use_locked_raises(self) -> None:
        e = bomb_busters.Equipment(name="Test", description="", unlock_value=3)
        with self.assertRaises(ValueError):
            e.use()

    def test_use_twice_raises(self) -> None:
        e = bomb_busters.Equipment(name="Test", description="", unlock_value=3)
        e.unlock()
        e.use()
        with self.assertRaises(ValueError):
            e.use()


class TestMarker(unittest.TestCase):
    """Tests for the bomb_busters.Marker class."""

    def test_known_marker(self) -> None:
        m = bomb_busters.Marker(bomb_busters.WireColor.RED, 3.5, bomb_busters.MarkerState.KNOWN)
        self.assertEqual(m.base_number, 3)
        self.assertEqual(m.state, bomb_busters.MarkerState.KNOWN)

    def test_uncertain_marker(self) -> None:
        m = bomb_busters.Marker(bomb_busters.WireColor.YELLOW, 5.1, bomb_busters.MarkerState.UNCERTAIN)
        self.assertEqual(m.state, bomb_busters.MarkerState.UNCERTAIN)
        self.assertIn("?", str(m))


class TestWireConfig(unittest.TestCase):
    """Tests for the bomb_busters.WireConfig class."""

    def test_valid_yellow(self) -> None:
        config = bomb_busters.WireConfig(bomb_busters.WireColor.YELLOW, count=2)
        self.assertEqual(config.count, 2)
        self.assertIsNone(config.pool_size)

    def test_valid_red_x_of_y(self) -> None:
        config = bomb_busters.WireConfig(bomb_busters.WireColor.RED, count=2, pool_size=3)
        self.assertEqual(config.count, 2)
        self.assertEqual(config.pool_size, 3)

    def test_blue_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.WireConfig(bomb_busters.WireColor.BLUE, count=4)

    def test_yellow_too_many_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.WireConfig(bomb_busters.WireColor.YELLOW, count=5)

    def test_red_too_many_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.WireConfig(bomb_busters.WireColor.RED, count=4)

    def test_pool_less_than_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.WireConfig(bomb_busters.WireColor.YELLOW, count=3, pool_size=2)


class TestTurnHistory(unittest.TestCase):
    """Tests for the bomb_busters.TurnHistory class."""

    def test_empty_history(self) -> None:
        h = bomb_busters.TurnHistory()
        self.assertEqual(len(h.actions), 0)

    def test_record_and_query(self) -> None:
        h = bomb_busters.TurnHistory()
        action = bomb_busters.DualCutAction(
            actor_index=0,
            target_player_index=1,
            target_slot_index=3,
            guessed_value=5,
            result=bomb_busters.ActionResult.FAIL_BLUE_YELLOW,
        )
        h.record(action)
        self.assertEqual(len(h.actions), 1)

        failed = h.failed_dual_cuts_by_player(0)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].guessed_value, 5)

    def test_failed_cuts_filters_correctly(self) -> None:
        h = bomb_busters.TurnHistory()
        # Failed dual cut by player 0
        h.record(bomb_busters.DualCutAction(
            actor_index=0, target_player_index=1,
            target_slot_index=0, guessed_value=3,
            result=bomb_busters.ActionResult.FAIL_BLUE_YELLOW,
        ))
        # Successful dual cut by player 0
        h.record(bomb_busters.DualCutAction(
            actor_index=0, target_player_index=1,
            target_slot_index=1, guessed_value=5,
            result=bomb_busters.ActionResult.SUCCESS,
        ))
        # Failed dual cut by player 1
        h.record(bomb_busters.DualCutAction(
            actor_index=1, target_player_index=0,
            target_slot_index=0, guessed_value=7,
            result=bomb_busters.ActionResult.FAIL_BLUE_YELLOW,
        ))
        # Solo cut by player 0
        h.record(bomb_busters.SoloCutAction(
            actor_index=0, value=9,
            slot_indices=(0, 1), wire_count=2,
        ))

        # bomb_busters.Player 0 failed cuts: only the one for value 3
        failed_0 = h.failed_dual_cuts_by_player(0)
        self.assertEqual(len(failed_0), 1)
        self.assertEqual(failed_0[0].guessed_value, 3)

        # bomb_busters.Player 1 failed cuts: only the one for value 7
        failed_1 = h.failed_dual_cuts_by_player(1)
        self.assertEqual(len(failed_1), 1)
        self.assertEqual(failed_1[0].guessed_value, 7)


class TestGameStateCreation(unittest.TestCase):
    """Tests for bomb_busters.GameState.create_game."""

    def test_create_4_players(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["Alice", "Bob", "Carol", "Dave"],
            seed=42,
        )
        self.assertEqual(len(game.players), 4)
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 48)
        self.assertEqual(game.detonator.max_failures, 3)
        self.assertFalse(game.game_over)
        self.assertFalse(game.game_won)

    def test_create_5_players(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D", "E"],
            seed=42,
        )
        self.assertEqual(len(game.players), 5)
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 48)
        self.assertEqual(game.detonator.max_failures, 4)

    def test_create_with_red_wires(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            red_wires=3,
            seed=42,
        )
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 51)
        # 3 KNOWN red markers
        red_markers = [m for m in game.markers if m.color == bomb_busters.WireColor.RED]
        self.assertEqual(len(red_markers), 3)
        self.assertTrue(all(m.state == bomb_busters.MarkerState.KNOWN for m in red_markers))

    def test_create_with_yellow_x_of_y(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            yellow_wires=(2, 3),
            seed=42,
        )
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 50)  # 48 blue + 2 yellow
        # 3 UNCERTAIN yellow markers
        yellow_markers = [m for m in game.markers if m.color == bomb_busters.WireColor.YELLOW]
        self.assertEqual(len(yellow_markers), 3)
        self.assertTrue(
            all(m.state == bomb_busters.MarkerState.UNCERTAIN for m in yellow_markers)
        )

    def test_invalid_player_count(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.GameState.create_game(player_names=["A", "B", "C"])

    def test_each_player_has_character_card(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        for player in game.players:
            self.assertIsNotNone(player.character_card)
            self.assertEqual(player.character_card.name, "Double Detector")
            self.assertFalse(player.character_card.used)

    def test_stands_are_sorted(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        for player in game.players:
            values = [
                s.wire.sort_value
                for s in player.tile_stand.slots
                if s.wire is not None
            ]
            self.assertEqual(values, sorted(values))

    def test_deal_distribution(self) -> None:
        """With 48 wires and 4 players: all get 12."""
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        for player in game.players:
            self.assertEqual(player.tile_stand.remaining_count, 12)

    def test_deal_distribution_uneven(self) -> None:
        """With 50 wires and 4 players: 2 get 13, 2 get 12."""
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            yellow_wires=2,
            seed=42,
        )
        counts = [p.tile_stand.remaining_count for p in game.players]
        self.assertEqual(sum(counts), 50)
        self.assertEqual(sorted(counts), [12, 12, 13, 13])

    def test_wires_in_play_tracked(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        self.assertEqual(len(game.wires_in_play), 48)

    def test_seed_reproducibility(self) -> None:
        game1 = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"], seed=123
        )
        game2 = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"], seed=123
        )
        for p1, p2 in zip(game1.players, game2.players):
            wires1 = [s.wire for s in p1.tile_stand.slots]
            wires2 = [s.wire for s in p2.tile_stand.slots]
            self.assertEqual(wires1, wires2)

    def test_str_output(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["Alice", "Bob", "Carol", "Dave"],
            seed=42,
        )
        output = str(game)
        self.assertIn("Bomb Busters", output)
        self.assertIn("Alice", output)
        self.assertIn("Mistakes remaining", output)


class TestGameStatePartialState(unittest.TestCase):
    """Tests for bomb_busters.GameState.from_partial_state."""

    def test_basic_partial_state(self) -> None:
        stands = [
            # bomb_busters.Player 0 (observer): knows own wires
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)),
            ]),
            # bomb_busters.Player 1: one hidden, one cut
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(wire=None),
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), state=bomb_busters.SlotState.CUT),
            ]),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["Me", "Them"],
            stands=stands,
        )
        self.assertEqual(len(game.players), 2)
        self.assertIsNotNone(game.players[0].tile_stand.slots[0].wire)
        self.assertIsNone(game.players[1].tile_stand.slots[0].wire)

    def test_partial_with_mistakes_remaining(self) -> None:
        stands = [
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0))]),
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=None)]),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B"],
            stands=stands,
            mistakes_remaining=0,
        )
        # 2 players → max_failures=1, mistakes_remaining=0 → failures=1
        self.assertEqual(game.detonator.failures, 1)
        self.assertEqual(game.detonator.remaining_failures, 0)

    def test_partial_defaults_to_max_mistakes(self) -> None:
        stands = [
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0))]),
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=None)]),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B"],
            stands=stands,
        )
        # 2 players → max_failures=1, default mistakes_remaining=1 → failures=0
        self.assertEqual(game.detonator.failures, 0)
        self.assertEqual(game.detonator.remaining_failures, 1)

    def test_validation_tokens_computed_from_stands(self) -> None:
        """validation_tokens is computed from cut blue wires on stands."""
        # 4 cut blue-5 wires across two players → value 5 is validated.
        # 2 cut blue-3 wires → value 3 is NOT validated.
        stands = [
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(
                    wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                    state=bomb_busters.SlotState.CUT,
                ),
                bomb_busters.Slot(
                    wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                    state=bomb_busters.SlotState.CUT,
                ),
                bomb_busters.Slot(
                    wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
                    state=bomb_busters.SlotState.CUT,
                ),
            ]),
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(
                    wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                    state=bomb_busters.SlotState.CUT,
                ),
                bomb_busters.Slot(
                    wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                    state=bomb_busters.SlotState.CUT,
                ),
                bomb_busters.Slot(
                    wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
                    state=bomb_busters.SlotState.CUT,
                ),
            ]),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B"],
            stands=stands,
        )
        self.assertIn(5, game.validation_tokens)
        self.assertNotIn(3, game.validation_tokens)

    def test_mismatched_stands_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
                player_names=["A", "B"],
                stands=[bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=None)])],  # Only 1 stand for 2 players
            )


class TestDualCutExecution(unittest.TestCase):
    """Tests for bomb_busters.GameState.execute_dual_cut."""

    def _make_game(self) -> bomb_busters.GameState:
        """Create a small known game for testing."""
        # bomb_busters.Player 0: [blue-1, blue-2, blue-3]
        # bomb_busters.Player 1: [blue-1, blue-2, blue-3]
        # bomb_busters.Player 2: [blue-1, blue-2, blue-3]
        # bomb_busters.Player 3: [blue-1, blue-2, blue-3]
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        return bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )

    def test_successful_dual_cut(self) -> None:
        game = self._make_game()
        # bomb_busters.Player 0 guesses player 1's slot 0 is a 1 (correct!)
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.SUCCESS)
        # Target's slot 0 is now cut
        self.assertTrue(game.players[1].tile_stand.slots[0].is_cut)
        # Actor also cut one of their 1s
        self.assertIsNotNone(action.actor_cut_slot_index)

    def test_failed_dual_cut_blue(self) -> None:
        game = self._make_game()
        # bomb_busters.Player 0 guesses player 1's slot 2 (blue-3) is a 2 (wrong!)
        # bomb_busters.Player 0 has a 2, so the guess is valid but incorrect.
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=2,
            guessed_value=2,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.FAIL_BLUE_YELLOW)
        # bomb_busters.Detonator advanced
        self.assertEqual(game.detonator.failures, 1)
        # Info token placed showing the real value (3)
        self.assertTrue(game.players[1].tile_stand.slots[2].is_info_revealed)
        self.assertEqual(game.players[1].tile_stand.slots[2].info_token, 3)

    def test_failed_dual_cut_yellow(self) -> None:
        """Failed dual cut on yellow wire stores 'YELLOW' info token."""
        # P0 has yellow-3.1, P1 has yellow-5.1 at slot 1
        # P0 guesses P1 slot 1 is YELLOW (correct — it IS yellow),
        # but P0 also needs a yellow to cut. For a failure scenario,
        # P0 guesses blue value 5 on P1 slot 1 (which is actually yellow).
        hands = [
            [
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
            ],
            [
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
                bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 5.1),
            ],
            [
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0),
            ],
            [
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
                bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0),
            ],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        # P0 guesses P1 slot 1 (yellow-5.1) is blue-5 — wrong!
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=1,
            guessed_value=5,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.FAIL_BLUE_YELLOW)
        self.assertEqual(game.detonator.failures, 1)
        # Info token shows "YELLOW" (not the numeric value)
        slot = game.players[1].tile_stand.slots[1]
        self.assertTrue(slot.is_info_revealed)
        self.assertEqual(slot.info_token, "YELLOW")
        # Display should show "Y", not "YELLOW"
        plain, _ = slot.value_label()
        self.assertEqual(plain, "Y")

    def test_failed_dual_cut_red_explodes(self) -> None:
        # Create game with a red wire
        # bomb_busters.Player 0 has [blue-1, blue-2], bomb_busters.Player 1 has [red-1.5, blue-3]
        # bomb_busters.Player 0 guesses P1's slot 0 (red-1.5) is a 1. P0 has a 1.
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 1.5), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[bomb_busters.Marker(bomb_busters.WireColor.RED, 1.5, bomb_busters.MarkerState.KNOWN)],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        # bomb_busters.Player 0 guesses player 1's slot 0 (red-1.5) is a 1
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.FAIL_RED)
        self.assertTrue(game.game_over)
        self.assertFalse(game.game_won)

    def test_cannot_cut_own_wires(self) -> None:
        game = self._make_game()
        with self.assertRaises(ValueError):
            game.execute_dual_cut(
                target_player_index=0,
                target_slot_index=0,
                guessed_value=1,
            )

    def test_actor_needs_matching_wire(self) -> None:
        game = self._make_game()
        with self.assertRaises(ValueError):
            game.execute_dual_cut(
                target_player_index=1,
                target_slot_index=0,
                guessed_value=10,  # bomb_busters.Player 0 doesn't have a 10
            )

    def test_dual_cut_advances_turn(self) -> None:
        game = self._make_game()
        self.assertEqual(game.current_player_index, 0)
        game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
        )
        self.assertEqual(game.current_player_index, 1)

    def test_validation_token_placed(self) -> None:
        # Each player has [1, 1, 5, 5] so we can do two dual cuts for 1
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        # P0 dual cuts P1's slot 0 (blue-1) for value 1 → success
        game.execute_dual_cut(1, 0, 1)
        self.assertEqual(game.get_cut_count_for_value(1), 2)
        self.assertNotIn(1, game.validation_tokens)
        # P1 dual cuts P0's slot 1 (blue-1) for value 1 → success
        # P1 still has a 1 at slot 1
        game.execute_dual_cut(0, 1, 1)
        self.assertEqual(game.get_cut_count_for_value(1), 4)
        self.assertIn(1, game.validation_tokens)

    def test_history_recorded(self) -> None:
        game = self._make_game()
        game.execute_dual_cut(1, 0, 1)
        self.assertEqual(len(game.history.actions), 1)

    def test_dual_cut_info_revealed_target(self) -> None:
        """Dual cut on an INFO_REVEALED target slot should succeed.

        Info-revealed slots are uncut wires with known values. A player
        can dual cut them just like hidden wires.
        """
        game = self._make_game()
        # First, fail a dual cut to create an info-revealed slot.
        # P0 guesses P1's slot 0 (blue-1) is a 2 → wrong, creates info token.
        action = game.execute_dual_cut(1, 0, 2)
        self.assertEqual(action.result, bomb_busters.ActionResult.FAIL_BLUE_YELLOW)
        self.assertTrue(game.players[1].tile_stand.slots[0].is_info_revealed)
        self.assertEqual(game.players[1].tile_stand.slots[0].info_token, 1)

        # Now P1 can target P1's info-revealed slot 0 (which we know is 1).
        # P1 (current player) dual cuts P0's slot 0 (blue-1) for value 1.
        # This succeeds, cutting P1's own blue-1 at slot 0 (which is
        # info-revealed, not hidden) and P0's slot 0.
        action2 = game.execute_dual_cut(0, 0, 1)
        self.assertEqual(action2.result, bomb_busters.ActionResult.SUCCESS)
        # P1's info-revealed slot 0 should now be cut (used as actor's match)
        self.assertTrue(game.players[1].tile_stand.slots[0].is_cut)

        # Alternatively: another player can target an info-revealed slot.
        # P2 (current) fails guessing P0's slot 1 (blue-2) is 3 → info token.
        self.assertEqual(game.current_player_index, 2)
        action3 = game.execute_dual_cut(0, 1, 3)
        self.assertEqual(action3.result, bomb_busters.ActionResult.FAIL_BLUE_YELLOW)
        self.assertTrue(game.players[0].tile_stand.slots[1].is_info_revealed)

        # P3 (current) dual cuts P0's info-revealed slot 1 for value 2.
        self.assertEqual(game.current_player_index, 3)
        action4 = game.execute_dual_cut(0, 1, 2)
        self.assertEqual(action4.result, bomb_busters.ActionResult.SUCCESS)
        self.assertTrue(game.players[0].tile_stand.slots[1].is_cut)

    def test_dual_cut_already_cut_raises(self) -> None:
        """Dual cut on an already-cut slot should raise ValueError."""
        game = self._make_game()
        # Cut P1's slot 0.
        game.execute_dual_cut(1, 0, 1)
        self.assertTrue(game.players[1].tile_stand.slots[0].is_cut)
        # Trying to cut it again should fail.
        with self.assertRaises(ValueError):
            game.execute_dual_cut(1, 0, 1)


class TestDoubleDectectorExecution(unittest.TestCase):
    """Tests for Double Detector in dual cut."""

    def _make_game(self) -> bomb_busters.GameState:
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        return bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )

    def test_dd_success_primary(self) -> None:
        game = self._make_game()
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
            is_double_detector=True,
            second_target_slot_index=1,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.SUCCESS)
        self.assertTrue(action.is_double_detector)
        # Character card used
        self.assertTrue(game.players[0].character_card.used)

    def test_dd_success_secondary(self) -> None:
        game = self._make_game()
        # Guess 5, point at slots 0 (blue-1) and 1 (blue-5) — secondary matches
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=5,
            is_double_detector=True,
            second_target_slot_index=1,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.SUCCESS)

    def test_dd_failure(self) -> None:
        game = self._make_game()
        # Guess 3, point at P2's slots (blue-2 and blue-6) — neither matches
        # First skip to P0's turn (already there)
        action = game.execute_dual_cut(
            target_player_index=2,
            target_slot_index=0,
            guessed_value=1,
            is_double_detector=True,
            second_target_slot_index=1,
        )
        self.assertEqual(action.result, bomb_busters.ActionResult.FAIL_BLUE_YELLOW)
        self.assertEqual(game.detonator.failures, 1)

    def test_dd_cannot_use_yellow(self) -> None:
        game = self._make_game()
        with self.assertRaises(ValueError):
            game.execute_dual_cut(
                target_player_index=1,
                target_slot_index=0,
                guessed_value="YELLOW",
                is_double_detector=True,
                second_target_slot_index=1,
            )

    def test_dd_already_used_raises(self) -> None:
        game = self._make_game()
        game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
            is_double_detector=True,
            second_target_slot_index=1,
        )
        # Try to use DD again (on player 1's turn now)
        with self.assertRaises(ValueError):
            game.execute_dual_cut(
                target_player_index=0,
                target_slot_index=0,
                guessed_value=5,
                is_double_detector=True,
                second_target_slot_index=1,
            )


class TestSoloCutExecution(unittest.TestCase):
    """Tests for bomb_busters.GameState.execute_solo_cut."""

    def _make_game_for_solo(self) -> bomb_busters.GameState:
        """bomb_busters.Player 0 has all four 1s."""
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        return bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )

    def test_solo_cut_4(self) -> None:
        game = self._make_game_for_solo()
        action = game.execute_solo_cut(1, [0, 1, 2, 3])
        self.assertEqual(action.wire_count, 4)
        self.assertTrue(all(
            game.players[0].tile_stand.slots[i].is_cut for i in range(4)
        ))
        self.assertIn(1, game.validation_tokens)

    def test_solo_cut_2(self) -> None:
        """After 2 are cut elsewhere, can solo cut remaining 2."""
        game = self._make_game_for_solo()
        # Manually cut 2 of the 1s first
        game.players[0].tile_stand.cut_wire_at(0)
        game.players[0].tile_stand.cut_wire_at(1)
        action = game.execute_solo_cut(1, [2, 3])
        self.assertEqual(action.wire_count, 2)

    def test_solo_cut_invalid_count(self) -> None:
        game = self._make_game_for_solo()
        with self.assertRaises(ValueError):
            game.execute_solo_cut(1, [0, 1, 2])  # Must be 2 or 4

    def test_cannot_solo_cut_if_others_have_it(self) -> None:
        # Create game where player 0 has two 2s, but so does player 1
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        with self.assertRaises(ValueError):
            game.execute_solo_cut(2, [0, 1])


class TestRevealRedExecution(unittest.TestCase):
    """Tests for bomb_busters.GameState.execute_reveal_red."""

    def test_reveal_all_red(self) -> None:
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 1.5), bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        action = game.execute_reveal_red()
        self.assertEqual(len(action.slot_indices), 2)
        self.assertTrue(game.players[0].is_finished)

    def test_cannot_reveal_if_not_all_red(self) -> None:
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 1.5), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        with self.assertRaises(ValueError):
            game.execute_reveal_red()


class TestGameStateHelpers(unittest.TestCase):
    """Tests for bomb_busters.GameState helper methods."""

    def _make_game(self) -> bomb_busters.GameState:
        hands = [
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hands[i]),
                character_card=bomb_busters.create_double_detector(),
            )
            for i in range(4)
        ]
        return bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(max_failures=3),

            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )

    def test_get_all_cut_wires(self) -> None:
        game = self._make_game()
        self.assertEqual(len(game.get_all_cut_wires()), 0)
        game.execute_dual_cut(1, 0, 1)  # Cuts two 1s
        self.assertEqual(len(game.get_all_cut_wires()), 2)

    def test_get_cut_count_for_value(self) -> None:
        game = self._make_game()
        self.assertEqual(game.get_cut_count_for_value(1), 0)
        game.execute_dual_cut(1, 0, 1)
        self.assertEqual(game.get_cut_count_for_value(1), 2)
        self.assertEqual(game.get_cut_count_for_value(2), 0)

    def test_can_solo_cut(self) -> None:
        game = self._make_game()
        # Initially nobody can solo cut (others have matching wires)
        self.assertFalse(game.can_solo_cut(0, 1))

    def test_available_solo_cuts_empty(self) -> None:
        game = self._make_game()
        self.assertEqual(game.available_solo_cuts(0), [])

    def test_win_condition(self) -> None:
        """Game is won when all stands are empty."""
        game = self._make_game()
        # Manually cut all wires
        for player in game.players:
            for i in range(len(player.tile_stand.slots)):
                player.tile_stand.slots[i].state = bomb_busters.SlotState.CUT
        game._check_win()
        self.assertTrue(game.game_over)
        self.assertTrue(game.game_won)

    def test_turn_skips_finished_players(self) -> None:
        game = self._make_game()
        # Finish player 1
        for slot in game.players[1].tile_stand.slots:
            slot.state = bomb_busters.SlotState.CUT
        # bomb_busters.Player 0's turn, do a cut
        game.execute_dual_cut(2, 0, 1)
        # Should skip player 1, go to player 2
        self.assertEqual(game.current_player_index, 2)

    def test_can_solo_cut_calculator_mode_rejects_unknown_wires(self) -> None:
        """In calculator mode, hidden wires on other stands are None.

        can_solo_cut must not falsely allow a solo cut when unaccounted
        wires could be on other players' hidden slots.
        """
        # Player 3 has two 1s, but other players have unknown hidden wires.
        # Total blue-1 in game = 4. Player has 2. Unaccounted = 2.
        me = bomb_busters.TileStand.from_string("?1 ?1 ?3 ?4")
        other = bomb_busters.TileStand.from_string("? ? ? ?")
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=[other, other, other, me],
            active_player_index=3,
            blue_wires=(1, 4),
        )
        self.assertFalse(game.can_solo_cut(3, 1))
        self.assertEqual(game.available_solo_cuts(3), [])

    def test_can_solo_cut_calculator_mode_allows_when_all_accounted(self) -> None:
        """Solo cut is allowed when all wires of the value are accounted for."""
        # Player 3 has all 4 blue-1 wires. No unaccounted copies.
        me = bomb_busters.TileStand.from_string("?1 ?1 ?1 ?1 ?3")
        other = bomb_busters.TileStand.from_string("? ? ? ?")
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=[other, other, other, me],
            active_player_index=3,
            blue_wires=(1, 4),
        )
        self.assertTrue(game.can_solo_cut(3, 1))
        self.assertIn(1, game.available_solo_cuts(3))

    def test_can_solo_cut_calculator_mode_with_cut_wires_elsewhere(self) -> None:
        """Solo cut allowed when remaining copies are all in player's hand."""
        # 2 blue-3s cut on other stands, player has the remaining 2.
        me = bomb_busters.TileStand.from_string("?2 ?3 ?3 ?4")
        p1 = bomb_busters.TileStand.from_string("3 ? ? ?")
        p2 = bomb_busters.TileStand.from_string("? 3 ? ?")
        p3 = bomb_busters.TileStand.from_string("? ? ? ?")
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "Me"],
            stands=[p1, p2, p3, me],
            active_player_index=3,
            blue_wires=(1, 4),
        )
        self.assertTrue(game.can_solo_cut(3, 3))

    def test_can_solo_cut_calculator_mode_with_info_token_elsewhere(self) -> None:
        """Info tokens on other stands account for wires of that value."""
        # Player has 2 blue-3s. One blue-3 is info-revealed on another stand.
        # 4 total - 2 player - 1 info-revealed = 1 unaccounted → no solo cut.
        me = bomb_busters.TileStand.from_string("?2 ?3 ?3 ?4")
        p1 = bomb_busters.TileStand.from_string("? i3 ? ?")
        p2 = bomb_busters.TileStand.from_string("? ? ? ?")
        p3 = bomb_busters.TileStand.from_string("? ? ? ?")
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "Me"],
            stands=[p1, p2, p3, me],
            active_player_index=3,
            blue_wires=(1, 4),
        )
        self.assertFalse(game.can_solo_cut(3, 3))

    def test_can_solo_cut_calculator_mode_info_tokens_complete_set(self) -> None:
        """Solo cut allowed when info tokens + player's hand cover all copies."""
        # Player has 2 blue-3s. The other 2 are info-revealed on other stands.
        # 4 total - 2 player - 2 info-revealed = 0 unaccounted → solo cut OK.
        me = bomb_busters.TileStand.from_string("?2 ?3 ?3 ?4")
        p1 = bomb_busters.TileStand.from_string("? i3 ? ?")
        p2 = bomb_busters.TileStand.from_string("i3 ? ? ?")
        p3 = bomb_busters.TileStand.from_string("? ? ? ?")
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "Me"],
            stands=[p1, p2, p3, me],
            active_player_index=3,
            blue_wires=(1, 4),
        )
        self.assertTrue(game.can_solo_cut(3, 3))


class TestUncertainWireGroup(unittest.TestCase):
    """Tests for the bomb_busters.UncertainWireGroup class."""

    def test_yellow_factory(self) -> None:
        group = bomb_busters.UncertainWireGroup.yellow([2, 3, 9], count=2)
        self.assertEqual(len(group.candidates), 3)
        self.assertEqual(group.count_in_play, 2)
        self.assertEqual(group.color, bomb_busters.WireColor.YELLOW)
        self.assertEqual(group.discard_count, 1)
        self.assertEqual(
            group.candidates[0],
            bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 2.1),
        )

    def test_red_factory(self) -> None:
        group = bomb_busters.UncertainWireGroup.red([3, 7], count=1)
        self.assertEqual(len(group.candidates), 2)
        self.assertEqual(group.count_in_play, 1)
        self.assertEqual(group.color, bomb_busters.WireColor.RED)
        self.assertEqual(group.discard_count, 1)
        self.assertEqual(
            group.candidates[1],
            bomb_busters.Wire(bomb_busters.WireColor.RED, 7.5),
        )

    def test_blue_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.UncertainWireGroup(
                candidates=[bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
                count_in_play=1,
            )

    def test_empty_candidates_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.UncertainWireGroup(candidates=[], count_in_play=0)

    def test_mixed_colors_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.UncertainWireGroup(
                candidates=[
                    bomb_busters.Wire(bomb_busters.WireColor.YELLOW, 2.1),
                    bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
                ],
                count_in_play=1,
            )

    def test_count_exceeds_candidates_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.UncertainWireGroup.yellow([2, 3], count=3)

    def test_yellow_count_exceeds_game_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.UncertainWireGroup.yellow([1, 2, 3, 4, 5], count=5)

    def test_red_count_exceeds_game_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.UncertainWireGroup.red([1, 2, 3, 4], count=4)

    def test_zero_count_in_play(self) -> None:
        group = bomb_busters.UncertainWireGroup.yellow([2, 3, 9], count=0)
        self.assertEqual(group.count_in_play, 0)
        self.assertEqual(group.discard_count, 3)

    def test_from_partial_state_auto_generates_markers(self) -> None:
        """yellow_wires=([2, 3, 9], 2) creates 3 UNCERTAIN markers."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            yellow_wires=([2, 3, 9], 2),
        )
        yellow_markers = [
            m for m in game.markers
            if m.color == bomb_busters.WireColor.YELLOW
        ]
        self.assertEqual(len(yellow_markers), 3)
        self.assertTrue(
            all(
                m.state == bomb_busters.MarkerState.UNCERTAIN
                for m in yellow_markers
            )
        )
        marker_bases = sorted(m.base_number for m in yellow_markers)
        self.assertEqual(marker_bases, [2, 3, 9])

    def test_convenience_known_yellow_markers(self) -> None:
        """yellow_wires=[4, 7] creates 2 KNOWN markers."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            yellow_wires=[4, 7],
        )
        yellow_markers = [
            m for m in game.markers
            if m.color == bomb_busters.WireColor.YELLOW
        ]
        self.assertEqual(len(yellow_markers), 2)
        self.assertTrue(
            all(
                m.state == bomb_busters.MarkerState.KNOWN
                for m in yellow_markers
            )
        )
        marker_bases = sorted(m.base_number for m in yellow_markers)
        self.assertEqual(marker_bases, [4, 7])
        # Known yellow wires should be in wires_in_play
        yellow_in_play = [
            w for w in game.wires_in_play
            if w.color == bomb_busters.WireColor.YELLOW
        ]
        self.assertEqual(len(yellow_in_play), 2)

    def test_convenience_known_red(self) -> None:
        """red_wires=[4] creates KNOWN marker and adds R4 to wires_in_play."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            red_wires=[4],
        )
        self.assertEqual(len(game.markers), 1)
        self.assertEqual(game.markers[0].color, bomb_busters.WireColor.RED)
        self.assertEqual(game.markers[0].sort_value, 4.5)
        self.assertEqual(game.markers[0].state, bomb_busters.MarkerState.KNOWN)
        red_in_play = [w for w in game.wires_in_play if w.color == bomb_busters.WireColor.RED]
        self.assertEqual(len(red_in_play), 1)
        self.assertEqual(red_in_play[0].sort_value, 4.5)

    def test_convenience_uncertain_red(self) -> None:
        """red_wires=([3, 7], 1) creates UNCERTAIN markers and group."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            red_wires=([3, 7], 1),
        )
        self.assertEqual(len(game.markers), 2)
        self.assertTrue(all(m.state == bomb_busters.MarkerState.UNCERTAIN for m in game.markers))
        self.assertTrue(all(m.color == bomb_busters.WireColor.RED for m in game.markers))
        self.assertEqual(len(game.uncertain_wire_groups), 1)
        self.assertEqual(game.uncertain_wire_groups[0].count_in_play, 1)
        # Red wires should NOT be in wires_in_play (they're uncertain)
        red_in_play = [w for w in game.wires_in_play if w.color == bomb_busters.WireColor.RED]
        self.assertEqual(len(red_in_play), 0)

    def test_convenience_mixed(self) -> None:
        """Known red + uncertain yellow together."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            yellow_wires=([2, 3, 9], 2),
            red_wires=[4],
        )
        # 3 yellow UNCERTAIN + 1 red KNOWN = 4 markers
        self.assertEqual(len(game.markers), 4)
        yellow_markers = [m for m in game.markers if m.color == bomb_busters.WireColor.YELLOW]
        red_markers = [m for m in game.markers if m.color == bomb_busters.WireColor.RED]
        self.assertEqual(len(yellow_markers), 3)
        self.assertTrue(all(m.state == bomb_busters.MarkerState.UNCERTAIN for m in yellow_markers))
        self.assertEqual(len(red_markers), 1)
        self.assertEqual(red_markers[0].state, bomb_busters.MarkerState.KNOWN)
        # Red wire in wires_in_play, yellow not
        red_in_play = [w for w in game.wires_in_play if w.color == bomb_busters.WireColor.RED]
        self.assertEqual(len(red_in_play), 1)
        yellow_in_play = [w for w in game.wires_in_play if w.color == bomb_busters.WireColor.YELLOW]
        self.assertEqual(len(yellow_in_play), 0)

    def test_convenience_blue_range(self) -> None:
        """blue_wires=(1, 8) creates 32 blue wires."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 8),
        )
        blue_in_play = [w for w in game.wires_in_play if w.color == bomb_busters.WireColor.BLUE]
        self.assertEqual(len(blue_in_play), 32)
        values = sorted({int(w.sort_value) for w in blue_in_play})
        self.assertEqual(values, list(range(1, 9)))

    def test_convenience_custom_blue_list(self) -> None:
        """blue_wires=[Wire(...)] passes through a custom pool."""
        custom = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i)) for i in [1, 1, 2, 2]]
        stands = [
            bomb_busters.TileStand.from_string("?1 ?2"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=custom,
        )
        self.assertEqual(len(game.wires_in_play), 4)

    def test_convenience_default_blue(self) -> None:
        """No blue_wires param -> all 48 blue wires (1-12)."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?5"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
            bomb_busters.TileStand.from_string("? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
        )
        blue_in_play = [w for w in game.wires_in_play if w.color == bomb_busters.WireColor.BLUE]
        self.assertEqual(len(blue_in_play), 48)


class TestCaptainIndex(unittest.TestCase):
    """Tests for captain_index on GameState."""

    def test_create_game_captain_defaults_to_zero(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        self.assertEqual(game.captain_index, 0)
        self.assertEqual(game.current_player_index, 0)

    def test_create_game_captain_explicit(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
            captain=2,
        )
        self.assertEqual(game.captain_index, 2)
        self.assertEqual(game.current_player_index, 2)

    def test_create_game_captain_invalid_index(self) -> None:
        with self.assertRaises(ValueError):
            bomb_busters.GameState.create_game(
                player_names=["A", "B", "C", "D"],
                captain=5,
            )
        with self.assertRaises(ValueError):
            bomb_busters.GameState.create_game(
                player_names=["A", "B", "C", "D"],
                captain=-1,
            )

    def test_create_game_captain_deals_first(self) -> None:
        """Captain and next players get extra wires when uneven."""
        # 50 wires (48 blue + 2 yellow) among 4 players = 12+12+13+13
        # With captain=2: players 2,3 get 13; players 0,1 get 12
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            yellow_wires=2,
            seed=42,
            captain=2,
        )
        counts = [p.tile_stand.remaining_count for p in game.players]
        self.assertEqual(sum(counts), 50)
        # Captain (index 2) and next clockwise (index 3) get 13
        self.assertEqual(counts[2], 13)
        self.assertEqual(counts[3], 13)
        # Others get 12
        self.assertEqual(counts[0], 12)
        self.assertEqual(counts[1], 12)

    def test_create_game_captain_deals_first_wraps(self) -> None:
        """Extra wires wrap around when captain is near end."""
        # 50 wires, 4 players, captain=3: players 3,0 get 13; 1,2 get 12
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            yellow_wires=2,
            seed=42,
            captain=3,
        )
        counts = [p.tile_stand.remaining_count for p in game.players]
        self.assertEqual(sum(counts), 50)
        self.assertEqual(counts[3], 13)
        self.assertEqual(counts[0], 13)
        self.assertEqual(counts[1], 12)
        self.assertEqual(counts[2], 12)

    def test_from_partial_state_captain_defaults_to_zero(self) -> None:
        stands = [
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
            ]),
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=None)]),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B"],
            stands=stands,
        )
        self.assertEqual(game.captain_index, 0)

    def test_from_partial_state_captain_explicit(self) -> None:
        stands = [
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
            ]),
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=None)]),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B"],
            stands=stands,
            captain=1,
        )
        self.assertEqual(game.captain_index, 1)

    def test_from_partial_state_captain_invalid(self) -> None:
        stands = [
            bomb_busters.TileStand(slots=[
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
            ]),
            bomb_busters.TileStand(slots=[bomb_busters.Slot(wire=None)]),
        ]
        with self.assertRaises(ValueError):
            bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
                player_names=["A", "B"],
                stands=stands,
                captain=2,
            )

    def test_crown_in_display(self) -> None:
        """Captain's crown icon appears in __str__ output."""
        game = bomb_busters.GameState.create_game(
            player_names=["Alice", "Bob", "Carol", "Dave"],
            seed=42,
            captain=1,
        )
        output = str(game)
        # Crown should appear on Bob's line (captain)
        for line in output.split("\n"):
            if "Bob" in line:
                self.assertIn("\U0001f451", line)  # 👑
            elif any(name in line for name in ["Alice", "Carol", "Dave"]):
                self.assertNotIn("\U0001f451", line)


class TestSlotConstraints(unittest.TestCase):
    """Tests for constraint class hierarchy."""

    def test_adjacent_not_equal_construction(self) -> None:
        c = bomb_busters.AdjacentNotEqual(
            player_index=1, slot_index_left=2, slot_index_right=3,
        )
        self.assertEqual(c.player_index, 1)
        self.assertEqual(c.slot_index_left, 2)
        self.assertEqual(c.slot_index_right, 3)

    def test_adjacent_equal_construction(self) -> None:
        c = bomb_busters.AdjacentEqual(
            player_index=0, slot_index_left=4, slot_index_right=5,
        )
        self.assertEqual(c.player_index, 0)
        self.assertEqual(c.slot_index_left, 4)

    def test_must_have_value_construction(self) -> None:
        c = bomb_busters.MustHaveValue(
            player_index=2, value=7, source="General Radar",
        )
        self.assertEqual(c.value, 7)
        self.assertEqual(c.source, "General Radar")

    def test_must_have_value_default_source(self) -> None:
        c = bomb_busters.MustHaveValue(player_index=0, value=3)
        self.assertEqual(c.source, "")

    def test_must_not_have_value_construction(self) -> None:
        c = bomb_busters.MustNotHaveValue(
            player_index=1, value="YELLOW", source="General Radar",
        )
        self.assertEqual(c.value, "YELLOW")

    def test_slot_parity_construction(self) -> None:
        c = bomb_busters.SlotParity(
            player_index=0, slot_index=3, parity=bomb_busters.Parity.EVEN,
        )
        self.assertEqual(c.parity, bomb_busters.Parity.EVEN)

    def test_value_multiplicity_construction(self) -> None:
        c = bomb_busters.ValueMultiplicity(
            player_index=0, slot_index=2,
            multiplicity=bomb_busters.Multiplicity.DOUBLE,
        )
        self.assertEqual(c.multiplicity, bomb_busters.Multiplicity.DOUBLE)
        self.assertEqual(c.multiplicity.value, 2)

    def test_unsorted_slot_construction(self) -> None:
        c = bomb_busters.UnsortedSlot(player_index=1, slot_index=5)
        self.assertEqual(c.slot_index, 5)

    def test_constraints_are_frozen(self) -> None:
        c = bomb_busters.AdjacentNotEqual(
            player_index=0, slot_index_left=0, slot_index_right=1,
        )
        with self.assertRaises(AttributeError):
            c.player_index = 1  # type: ignore[misc]

    def test_constraints_are_hashable(self) -> None:
        c1 = bomb_busters.MustHaveValue(player_index=0, value=5)
        c2 = bomb_busters.MustHaveValue(player_index=0, value=5)
        self.assertEqual(hash(c1), hash(c2))
        self.assertEqual(c1, c2)
        s = {c1, c2}
        self.assertEqual(len(s), 1)

    def test_base_class_describe_raises(self) -> None:
        c = bomb_busters.SlotConstraint(player_index=0)
        with self.assertRaises(NotImplementedError):
            c.describe()

    def test_adjacent_not_equal_describe(self) -> None:
        c = bomb_busters.AdjacentNotEqual(
            player_index=1, slot_index_left=0, slot_index_right=1,
        )
        desc = c.describe()
        self.assertIn("!=", desc)
        self.assertIn("A", desc)
        self.assertIn("B", desc)

    def test_adjacent_equal_describe(self) -> None:
        c = bomb_busters.AdjacentEqual(
            player_index=0, slot_index_left=2, slot_index_right=3,
        )
        desc = c.describe()
        self.assertIn("=", desc)
        self.assertIn("C", desc)
        self.assertIn("D", desc)

    def test_must_have_describe(self) -> None:
        c = bomb_busters.MustHaveValue(
            player_index=2, value=7, source="General Radar",
        )
        desc = c.describe()
        self.assertIn("has 7", desc)
        self.assertIn("General Radar", desc)

    def test_must_not_have_describe(self) -> None:
        c = bomb_busters.MustNotHaveValue(player_index=1, value=3)
        desc = c.describe()
        self.assertIn("does not have 3", desc)

    def test_slot_parity_describe(self) -> None:
        c = bomb_busters.SlotParity(
            player_index=0, slot_index=0, parity=bomb_busters.Parity.ODD,
        )
        self.assertIn("odd", c.describe())

    def test_value_multiplicity_describe(self) -> None:
        c = bomb_busters.ValueMultiplicity(
            player_index=0, slot_index=1,
            multiplicity=bomb_busters.Multiplicity.TRIPLE,
        )
        self.assertIn("x3", c.describe())

    def test_unsorted_slot_describe(self) -> None:
        c = bomb_busters.UnsortedSlot(player_index=0, slot_index=4)
        self.assertIn("unsorted", c.describe())

    def test_parity_enum_values(self) -> None:
        self.assertEqual(len(bomb_busters.Parity), 2)
        self.assertIn(bomb_busters.Parity.EVEN, bomb_busters.Parity)
        self.assertIn(bomb_busters.Parity.ODD, bomb_busters.Parity)

    def test_multiplicity_enum_values(self) -> None:
        self.assertEqual(bomb_busters.Multiplicity.SINGLE.value, 1)
        self.assertEqual(bomb_busters.Multiplicity.DOUBLE.value, 2)
        self.assertEqual(bomb_busters.Multiplicity.TRIPLE.value, 3)


class TestGameStateConstraints(unittest.TestCase):
    """Tests for GameState constraint management."""

    def _make_game(
        self,
        constraints: list[bomb_busters.SlotConstraint] | None = None,
    ) -> bomb_busters.GameState:
        """Create a minimal 4-player game for constraint testing."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?2 ?3 ?4 ?5"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
        ]
        return bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 5),
            constraints=constraints,
        )

    def test_add_constraint(self) -> None:
        game = self._make_game()
        self.assertEqual(len(game.slot_constraints), 0)
        c = bomb_busters.MustHaveValue(player_index=1, value=3)
        game.add_constraint(c)
        self.assertEqual(len(game.slot_constraints), 1)
        self.assertIs(game.slot_constraints[0], c)

    def test_get_constraints_for_player(self) -> None:
        c1 = bomb_busters.MustHaveValue(player_index=1, value=3)
        c2 = bomb_busters.MustNotHaveValue(player_index=2, value=5)
        c3 = bomb_busters.AdjacentNotEqual(
            player_index=1, slot_index_left=0, slot_index_right=1,
        )
        game = self._make_game(constraints=[c1, c2, c3])
        p1 = game.get_constraints_for_player(1)
        self.assertEqual(len(p1), 2)
        self.assertIn(c1, p1)
        self.assertIn(c3, p1)
        p2 = game.get_constraints_for_player(2)
        self.assertEqual(len(p2), 1)
        self.assertIn(c2, p2)
        p0 = game.get_constraints_for_player(0)
        self.assertEqual(len(p0), 0)

    def test_from_partial_state_with_constraints(self) -> None:
        constraints = [
            bomb_busters.MustHaveValue(player_index=1, value=3),
            bomb_busters.AdjacentEqual(
                player_index=2, slot_index_left=0, slot_index_right=1,
            ),
        ]
        game = self._make_game(constraints=constraints)
        self.assertEqual(len(game.slot_constraints), 2)

    def test_from_partial_state_no_constraints(self) -> None:
        game = self._make_game()
        self.assertEqual(game.slot_constraints, [])

    def test_constraints_in_str_output(self) -> None:
        c = bomb_busters.MustHaveValue(
            player_index=1, value=3, source="Radar",
        )
        game = self._make_game(constraints=[c])
        output = str(game)
        self.assertIn("Constraints:", output)
        self.assertIn("has 3", output)

    def test_no_constraints_section_when_empty(self) -> None:
        game = self._make_game()
        output = str(game)
        self.assertNotIn("Constraints:", output)


class TestPlaceInfoToken(unittest.TestCase):
    """Tests for GameState.place_info_token()."""

    def test_place_info_token_on_hidden_blue(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        # Find a hidden blue wire on player 0
        slot = game.players[0].tile_stand.slots[0]
        self.assertTrue(slot.is_hidden)
        self.assertEqual(slot.wire.color, bomb_busters.WireColor.BLUE)
        game.place_info_token(0, 0)
        self.assertTrue(slot.is_info_revealed)
        self.assertEqual(slot.info_token, slot.wire.gameplay_value)

    def test_place_info_token_on_cut_slot_raises(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.players[0].tile_stand.cut_wire_at(0)
        with self.assertRaises(ValueError):
            game.place_info_token(0, 0)

    def test_place_info_token_on_unknown_wire_raises(self) -> None:
        stands = [
            bomb_busters.TileStand.from_string("?1 ?2 ?3"),
            bomb_busters.TileStand.from_string("? ? ?"),
            bomb_busters.TileStand.from_string("? ? ?"),
            bomb_busters.TileStand.from_string("? ? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 3),
        )
        # Player 1 slot 0 is unknown (wire=None)
        with self.assertRaises(ValueError):
            game.place_info_token(1, 0)

    def test_place_info_token_on_non_blue_raises(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
            yellow_wires=2,
        )
        # Find a yellow wire
        for p_idx, player in enumerate(game.players):
            for s_idx, slot in enumerate(player.tile_stand.slots):
                if (
                    slot.is_hidden
                    and slot.wire is not None
                    and slot.wire.color == bomb_busters.WireColor.YELLOW
                ):
                    with self.assertRaises(ValueError):
                        game.place_info_token(p_idx, s_idx)
                    return
        self.skipTest("No yellow wire found on any stand")


class TestAdjustDetonator(unittest.TestCase):
    """Tests for GameState.adjust_detonator()."""

    def test_advance_detonator(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        initial = game.detonator.remaining_failures
        game.adjust_detonator(1)
        self.assertEqual(game.detonator.remaining_failures, initial - 1)

    def test_rewind_detonator(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.adjust_detonator(2)  # advance by 2
        remaining_after = game.detonator.remaining_failures
        game.adjust_detonator(-1)  # rewind by 1
        self.assertEqual(
            game.detonator.remaining_failures, remaining_after + 1,
        )

    def test_clamps_to_zero(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.adjust_detonator(-100)
        self.assertEqual(game.detonator.failures, 0)
        self.assertEqual(
            game.detonator.remaining_failures,
            game.detonator.max_failures,
        )

    def test_clamps_to_max(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.adjust_detonator(100)
        self.assertEqual(
            game.detonator.failures, game.detonator.max_failures,
        )


class TestSetDetonator(unittest.TestCase):
    """Tests for GameState.set_detonator()."""

    def test_set_detonator(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.set_detonator(1)
        self.assertEqual(game.detonator.remaining_failures, 1)

    def test_set_detonator_zero(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.set_detonator(0)
        self.assertEqual(game.detonator.remaining_failures, 0)

    def test_set_detonator_out_of_range(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        with self.assertRaises(ValueError):
            game.set_detonator(100)
        with self.assertRaises(ValueError):
            game.set_detonator(-1)


class TestReactivateCharacterCards(unittest.TestCase):
    """Tests for GameState.reactivate_character_cards()."""

    def test_reactivate_used_card(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.players[0].character_card.use()
        self.assertTrue(game.players[0].character_card.used)
        game.reactivate_character_cards([0])
        self.assertFalse(game.players[0].character_card.used)

    def test_reactivate_multiple(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.players[0].character_card.use()
        game.players[2].character_card.use()
        game.reactivate_character_cards([0, 2])
        self.assertFalse(game.players[0].character_card.used)
        self.assertFalse(game.players[2].character_card.used)

    def test_reactivate_no_card_raises(self) -> None:
        stands = [
            bomb_busters.TileStand.from_string("?1 ?2 ?3"),
            bomb_busters.TileStand.from_string("? ? ?"),
            bomb_busters.TileStand.from_string("? ? ?"),
            bomb_busters.TileStand.from_string("? ? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 3),
        )
        with self.assertRaises(ValueError):
            game.reactivate_character_cards([0])


class TestSetCurrentPlayer(unittest.TestCase):
    """Tests for GameState.set_current_player()."""

    def test_set_current_player(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        self.assertEqual(game.current_player_index, 0)
        game.set_current_player(2)
        self.assertEqual(game.current_player_index, 2)

    def test_set_current_player_out_of_range(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        with self.assertRaises(ValueError):
            game.set_current_player(4)
        with self.assertRaises(ValueError):
            game.set_current_player(-1)


class TestAddMustHave(unittest.TestCase):
    """Tests for GameState.add_must_have()."""

    def test_add_must_have(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.add_must_have(1, 5, source="General Radar")
        self.assertEqual(len(game.slot_constraints), 1)
        c = game.slot_constraints[0]
        self.assertIsInstance(c, bomb_busters.MustHaveValue)
        self.assertEqual(c.player_index, 1)
        self.assertEqual(c.value, 5)
        self.assertEqual(c.source, "General Radar")


class TestAddMustNotHave(unittest.TestCase):
    """Tests for GameState.add_must_not_have()."""

    def test_add_must_not_have(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.add_must_not_have(2, 8, source="General Radar")
        self.assertEqual(len(game.slot_constraints), 1)
        c = game.slot_constraints[0]
        self.assertIsInstance(c, bomb_busters.MustNotHaveValue)
        self.assertEqual(c.player_index, 2)
        self.assertEqual(c.value, 8)


class TestAddAdjacentConstraints(unittest.TestCase):
    """Tests for add_adjacent_not_equal and add_adjacent_equal."""

    def _make_game(self) -> bomb_busters.GameState:
        stands = [
            bomb_busters.TileStand.from_string("?1 ?2 ?3 ?4 ?5"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
        ]
        return bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 5),
        )

    def test_add_adjacent_not_equal(self) -> None:
        game = self._make_game()
        game.add_adjacent_not_equal(1, 2, 3)
        self.assertEqual(len(game.slot_constraints), 1)
        c = game.slot_constraints[0]
        self.assertIsInstance(c, bomb_busters.AdjacentNotEqual)
        self.assertEqual(c.slot_index_left, 2)
        self.assertEqual(c.slot_index_right, 3)

    def test_add_adjacent_equal(self) -> None:
        game = self._make_game()
        game.add_adjacent_equal(1, 0, 1)
        c = game.slot_constraints[0]
        self.assertIsInstance(c, bomb_busters.AdjacentEqual)

    def test_non_adjacent_raises(self) -> None:
        game = self._make_game()
        with self.assertRaises(ValueError):
            game.add_adjacent_not_equal(1, 0, 2)
        with self.assertRaises(ValueError):
            game.add_adjacent_equal(1, 0, 3)

    def test_out_of_range_raises(self) -> None:
        game = self._make_game()
        with self.assertRaises(ValueError):
            game.add_adjacent_not_equal(1, -1, 0)
        with self.assertRaises(ValueError):
            game.add_adjacent_equal(1, 4, 5)


class TestCutAllOfValue(unittest.TestCase):
    """Tests for GameState.cut_all_of_value()."""

    def test_cut_all_of_value(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        # Count how many hidden wires of value 3 exist
        hidden_3s = 0
        for player in game.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_hidden
                    and slot.wire is not None
                    and slot.wire.gameplay_value == 3
                ):
                    hidden_3s += 1
        self.assertEqual(hidden_3s, 4)

        game.cut_all_of_value(3)

        # All should be cut now
        remaining_3s = 0
        for player in game.players:
            for slot in player.tile_stand.slots:
                if (
                    slot.is_hidden
                    and slot.wire is not None
                    and slot.wire.gameplay_value == 3
                ):
                    remaining_3s += 1
        self.assertEqual(remaining_3s, 0)

    def test_cut_all_triggers_validation(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        game.cut_all_of_value(7)
        self.assertIn(7, game.validation_tokens)


class TestFastPassSoloCut(unittest.TestCase):
    """Tests for fast_pass parameter on can_solo_cut/available_solo_cuts."""

    def test_fast_pass_allows_cut_without_all_wires(self) -> None:
        """Fast Pass allows solo cut with 2 wires even if others exist."""
        # Player 0 has two 3s but other 3s exist elsewhere
        stands = [
            bomb_busters.TileStand.from_string("?1 ?3 ?3 ?5 ?7"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 8),
        )
        # Normal solo cut should fail (other 3s elsewhere)
        self.assertFalse(game.can_solo_cut(0, 3))
        # Fast pass should succeed (has 2 matching hidden wires)
        self.assertTrue(game.can_solo_cut(0, 3, fast_pass=True))

    def test_fast_pass_needs_two_wires(self) -> None:
        """Fast Pass still requires at least 2 hidden wires."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?3 ?5 ?7 ?9"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 10),
        )
        # Only one 3, can't solo cut even with fast pass
        self.assertFalse(game.can_solo_cut(0, 3, fast_pass=True))

    def test_available_solo_cuts_fast_pass(self) -> None:
        """available_solo_cuts with fast_pass returns more values."""
        stands = [
            bomb_busters.TileStand.from_string("?1 ?3 ?3 ?5 ?5"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
            bomb_busters.TileStand.from_string("? ? ? ? ?"),
        ]
        game = bomb_busters.GameState.from_partial_state(validate_stand_sizes=False,
            player_names=["A", "B", "C", "D"],
            stands=stands,
            blue_wires=(1, 8),
        )
        normal = game.available_solo_cuts(0)
        fast = game.available_solo_cuts(0, fast_pass=True)
        # Fast pass should include 3 and 5 (both have 2 hidden)
        self.assertIn(3, fast)
        self.assertIn(5, fast)
        # Normal should not (other copies exist elsewhere)
        self.assertNotIn(3, normal)
        self.assertNotIn(5, normal)


class TestWalkieTalkiesStub(unittest.TestCase):
    """Tests for apply_walkie_talkies stub."""

    def test_raises_not_implemented(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        with self.assertRaises(NotImplementedError):
            game.apply_walkie_talkies()


class TestGrapplingHookStub(unittest.TestCase):
    """Tests for apply_grappling_hook stub."""

    def test_raises_not_implemented(self) -> None:
        game = bomb_busters.GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        with self.assertRaises(NotImplementedError):
            game.apply_grappling_hook()


if __name__ == "__main__":
    unittest.main()
