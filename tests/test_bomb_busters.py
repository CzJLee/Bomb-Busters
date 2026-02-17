"""Unit tests for bomb_busters game model."""

import unittest

from bomb_busters import (
    ActionResult,
    CharacterCard,
    Detonator,
    Equipment,
    GameState,
    InfoTokenPool,
    Marker,
    MarkerState,
    Player,
    Slot,
    SlotState,
    TileStand,
    TurnHistory,
    Wire,
    WireColor,
    WireConfig,
    create_all_blue_wires,
    create_all_red_wires,
    create_all_yellow_wires,
    create_double_detector,
    get_sort_value_bounds,
)


class TestWire(unittest.TestCase):
    """Tests for the Wire dataclass."""

    def test_blue_wire_creation(self) -> None:
        w = Wire(WireColor.BLUE, 5.0)
        self.assertEqual(w.gameplay_value, 5)
        self.assertEqual(w.base_number, 5)
        self.assertEqual(w.color, WireColor.BLUE)

    def test_yellow_wire_creation(self) -> None:
        w = Wire(WireColor.YELLOW, 5.1)
        self.assertEqual(w.gameplay_value, "YELLOW")
        self.assertEqual(w.base_number, 5)
        self.assertEqual(w.color, WireColor.YELLOW)

    def test_red_wire_creation(self) -> None:
        w = Wire(WireColor.RED, 5.5)
        self.assertEqual(w.gameplay_value, "RED")
        self.assertEqual(w.base_number, 5)
        self.assertEqual(w.color, WireColor.RED)

    def test_wire_sorting(self) -> None:
        wires = [
            Wire(WireColor.RED, 2.5),
            Wire(WireColor.BLUE, 2.0),
            Wire(WireColor.YELLOW, 2.1),
            Wire(WireColor.BLUE, 1.0),
            Wire(WireColor.BLUE, 3.0),
        ]
        sorted_wires = sorted(wires)
        expected = [1.0, 2.0, 2.1, 2.5, 3.0]
        self.assertEqual([w.sort_value for w in sorted_wires], expected)

    def test_full_sort_order(self) -> None:
        """Verify blue < yellow < red for the same base number."""
        blue = Wire(WireColor.BLUE, 5.0)
        yellow = Wire(WireColor.YELLOW, 5.1)
        red = Wire(WireColor.RED, 5.5)
        self.assertLess(blue, yellow)
        self.assertLess(yellow, red)
        self.assertLess(blue, red)

    def test_cross_number_sort(self) -> None:
        """Red 2.5 < Blue 3.0."""
        red2 = Wire(WireColor.RED, 2.5)
        blue3 = Wire(WireColor.BLUE, 3.0)
        self.assertLess(red2, blue3)

    def test_wire_equality(self) -> None:
        w1 = Wire(WireColor.BLUE, 5.0)
        w2 = Wire(WireColor.BLUE, 5.0)
        self.assertEqual(w1, w2)
        self.assertIsNot(w1, w2)

    def test_wire_inequality(self) -> None:
        w1 = Wire(WireColor.BLUE, 5.0)
        w2 = Wire(WireColor.BLUE, 6.0)
        self.assertNotEqual(w1, w2)

    def test_wire_hash(self) -> None:
        w1 = Wire(WireColor.BLUE, 5.0)
        w2 = Wire(WireColor.BLUE, 5.0)
        self.assertEqual(hash(w1), hash(w2))
        # Can be used in sets
        s = {w1, w2}
        self.assertEqual(len(s), 1)

    def test_wire_frozen(self) -> None:
        w = Wire(WireColor.BLUE, 5.0)
        with self.assertRaises(AttributeError):
            w.color = WireColor.RED  # type: ignore[misc]

    def test_str_output(self) -> None:
        blue = Wire(WireColor.BLUE, 5.0)
        yellow = Wire(WireColor.YELLOW, 3.1)
        red = Wire(WireColor.RED, 7.5)
        # Just verify they produce non-empty strings without errors
        self.assertIn("5", str(blue))
        self.assertIn("3", str(yellow))
        self.assertIn("7", str(red))


class TestWireFactories(unittest.TestCase):
    """Tests for wire factory functions."""

    def test_blue_wire_count(self) -> None:
        blues = create_all_blue_wires()
        self.assertEqual(len(blues), 48)

    def test_blue_wire_distribution(self) -> None:
        blues = create_all_blue_wires()
        for number in range(1, 13):
            count = sum(1 for w in blues if w.sort_value == float(number))
            self.assertEqual(count, 4, f"Expected 4 blue-{number} wires")

    def test_red_wire_count(self) -> None:
        reds = create_all_red_wires()
        self.assertEqual(len(reds), 11)

    def test_red_wire_values(self) -> None:
        reds = create_all_red_wires()
        expected = [n + 0.5 for n in range(1, 12)]
        self.assertEqual([w.sort_value for w in reds], expected)

    def test_yellow_wire_count(self) -> None:
        yellows = create_all_yellow_wires()
        self.assertEqual(len(yellows), 11)

    def test_yellow_wire_values(self) -> None:
        yellows = create_all_yellow_wires()
        expected = [n + 0.1 for n in range(1, 12)]
        self.assertEqual([w.sort_value for w in yellows], expected)


class TestSlot(unittest.TestCase):
    """Tests for the Slot dataclass."""

    def test_default_state(self) -> None:
        s = Slot(wire=Wire(WireColor.BLUE, 5.0))
        self.assertTrue(s.is_hidden)
        self.assertFalse(s.is_cut)
        self.assertFalse(s.is_info_revealed)
        self.assertIsNone(s.info_token)

    def test_cut_state(self) -> None:
        s = Slot(wire=Wire(WireColor.BLUE, 5.0), state=SlotState.CUT)
        self.assertTrue(s.is_cut)
        self.assertFalse(s.is_hidden)

    def test_info_revealed_state(self) -> None:
        s = Slot(
            wire=Wire(WireColor.BLUE, 5.0),
            state=SlotState.INFO_REVEALED,
            info_token=5,
        )
        self.assertTrue(s.is_info_revealed)
        self.assertEqual(s.info_token, 5)

    def test_unknown_wire_slot(self) -> None:
        s = Slot(wire=None)
        self.assertTrue(s.is_hidden)
        self.assertIsNone(s.wire)

    def test_str_hidden(self) -> None:
        s = Slot(wire=None)
        self.assertIn("?", str(s))

    def test_str_cut(self) -> None:
        s = Slot(wire=Wire(WireColor.BLUE, 5.0), state=SlotState.CUT)
        output = str(s)
        self.assertIn("5", output)


class TestTileStand(unittest.TestCase):
    """Tests for the TileStand class."""

    def test_from_wires_sorts(self) -> None:
        wires = [
            Wire(WireColor.BLUE, 7.0),
            Wire(WireColor.BLUE, 3.0),
            Wire(WireColor.BLUE, 10.0),
            Wire(WireColor.BLUE, 1.0),
        ]
        stand = TileStand.from_wires(wires)
        values = [s.wire.sort_value for s in stand.slots if s.wire]
        self.assertEqual(values, [1.0, 3.0, 7.0, 10.0])

    def test_from_wires_mixed_colors(self) -> None:
        wires = [
            Wire(WireColor.BLUE, 3.0),
            Wire(WireColor.YELLOW, 2.1),
            Wire(WireColor.RED, 2.5),
            Wire(WireColor.BLUE, 1.0),
        ]
        stand = TileStand.from_wires(wires)
        values = [s.wire.sort_value for s in stand.slots if s.wire]
        self.assertEqual(values, [1.0, 2.1, 2.5, 3.0])

    def test_hidden_slots(self) -> None:
        wires = [Wire(WireColor.BLUE, float(i)) for i in range(1, 5)]
        stand = TileStand.from_wires(wires)
        self.assertEqual(len(stand.hidden_slots), 4)

    def test_cut_wire_at(self) -> None:
        wires = [Wire(WireColor.BLUE, float(i)) for i in range(1, 5)]
        stand = TileStand.from_wires(wires)
        stand.cut_wire_at(0)
        self.assertEqual(len(stand.hidden_slots), 3)
        self.assertEqual(len(stand.cut_slots), 1)
        self.assertTrue(stand.slots[0].is_cut)

    def test_cut_wire_at_invalid_index(self) -> None:
        wires = [Wire(WireColor.BLUE, 1.0)]
        stand = TileStand.from_wires(wires)
        with self.assertRaises(IndexError):
            stand.cut_wire_at(5)

    def test_cut_wire_at_already_cut(self) -> None:
        wires = [Wire(WireColor.BLUE, 1.0)]
        stand = TileStand.from_wires(wires)
        stand.cut_wire_at(0)
        with self.assertRaises(ValueError):
            stand.cut_wire_at(0)

    def test_place_info_token(self) -> None:
        wires = [Wire(WireColor.BLUE, 5.0)]
        stand = TileStand.from_wires(wires)
        stand.place_info_token(0, 5)
        self.assertTrue(stand.slots[0].is_info_revealed)
        self.assertEqual(stand.slots[0].info_token, 5)
        self.assertEqual(len(stand.info_revealed_slots), 1)

    def test_is_empty(self) -> None:
        wires = [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)]
        stand = TileStand.from_wires(wires)
        self.assertFalse(stand.is_empty)
        stand.cut_wire_at(0)
        self.assertFalse(stand.is_empty)
        stand.cut_wire_at(1)
        self.assertTrue(stand.is_empty)

    def test_remaining_count(self) -> None:
        wires = [Wire(WireColor.BLUE, float(i)) for i in range(1, 4)]
        stand = TileStand.from_wires(wires)
        self.assertEqual(stand.remaining_count, 3)
        stand.cut_wire_at(0)
        self.assertEqual(stand.remaining_count, 2)

    def test_wire_stays_in_position_after_cut(self) -> None:
        wires = [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 5.0)]
        stand = TileStand.from_wires(wires)
        stand.cut_wire_at(0)
        # Wire is still at index 0, just marked as cut
        self.assertEqual(len(stand.slots), 2)
        self.assertEqual(stand.slots[0].wire, Wire(WireColor.BLUE, 1.0))
        self.assertTrue(stand.slots[0].is_cut)

    def test_str_output(self) -> None:
        wires = [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 5.0)]
        stand = TileStand.from_wires(wires)
        output = str(stand)
        self.assertIn("A", output)
        self.assertIn("B", output)

    def test_stand_lines(self) -> None:
        wires = [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 5.0)]
        stand = TileStand.from_wires(wires)
        _, _, letters = stand.stand_lines()
        self.assertIn("A", letters)
        self.assertIn("B", letters)


class TestSortValueBounds(unittest.TestCase):
    """Tests for the get_sort_value_bounds helper."""

    def test_unconstrained(self) -> None:
        slots = [Slot(wire=None), Slot(wire=None), Slot(wire=None)]
        lower, upper = get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 13.0)

    def test_bounded_by_neighbors(self) -> None:
        slots = [
            Slot(wire=Wire(WireColor.BLUE, 3.0), state=SlotState.CUT),
            Slot(wire=None),
            Slot(wire=Wire(WireColor.BLUE, 7.0), state=SlotState.CUT),
        ]
        lower, upper = get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 7.0)

    def test_left_bound_only(self) -> None:
        slots = [
            Slot(wire=Wire(WireColor.BLUE, 3.0), state=SlotState.CUT),
            Slot(wire=None),
        ]
        lower, upper = get_sort_value_bounds(slots, 1)
        self.assertEqual(lower, 3.0)
        self.assertEqual(upper, 13.0)

    def test_right_bound_only(self) -> None:
        slots = [
            Slot(wire=None),
            Slot(wire=Wire(WireColor.BLUE, 7.0), state=SlotState.CUT),
        ]
        lower, upper = get_sort_value_bounds(slots, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 7.0)

    def test_skips_none_wires(self) -> None:
        slots = [
            Slot(wire=Wire(WireColor.BLUE, 2.0), state=SlotState.CUT),
            Slot(wire=None),
            Slot(wire=None),  # target
            Slot(wire=None),
            Slot(wire=Wire(WireColor.BLUE, 9.0), state=SlotState.CUT),
        ]
        lower, upper = get_sort_value_bounds(slots, 2)
        self.assertEqual(lower, 2.0)
        self.assertEqual(upper, 9.0)


class TestCharacterCard(unittest.TestCase):
    """Tests for the CharacterCard class."""

    def test_creation(self) -> None:
        card = create_double_detector()
        self.assertEqual(card.name, "Double Detector")
        self.assertFalse(card.used)

    def test_use(self) -> None:
        card = create_double_detector()
        card.use()
        self.assertTrue(card.used)

    def test_use_twice_raises(self) -> None:
        card = create_double_detector()
        card.use()
        with self.assertRaises(ValueError):
            card.use()

    def test_str_available(self) -> None:
        card = create_double_detector()
        self.assertIn("available", str(card))

    def test_str_used(self) -> None:
        card = create_double_detector()
        card.use()
        self.assertIn("used", str(card))


class TestDetonator(unittest.TestCase):
    """Tests for the Detonator class."""

    def test_initial_state(self) -> None:
        d = Detonator(max_failures=4)
        self.assertEqual(d.failures, 0)
        self.assertEqual(d.remaining_failures, 4)
        self.assertFalse(d.is_exploded)

    def test_advance(self) -> None:
        d = Detonator(max_failures=4)
        self.assertFalse(d.advance())  # 1
        self.assertEqual(d.remaining_failures, 3)
        self.assertFalse(d.advance())  # 2
        self.assertFalse(d.advance())  # 3
        self.assertTrue(d.advance())   # 4 -> exploded
        self.assertTrue(d.is_exploded)
        self.assertEqual(d.remaining_failures, 0)

    def test_five_players(self) -> None:
        """5 players = 4 failures allowed."""
        d = Detonator(max_failures=4)
        for _ in range(3):
            self.assertFalse(d.advance())
        self.assertTrue(d.advance())

    def test_four_players(self) -> None:
        """4 players = 3 failures allowed."""
        d = Detonator(max_failures=3)
        self.assertFalse(d.advance())
        self.assertFalse(d.advance())
        self.assertTrue(d.advance())

    def test_str_output(self) -> None:
        d = Detonator(max_failures=4)
        d.advance()
        output = str(d)
        self.assertIn("Mistakes remaining: 3", output)


class TestInfoTokenPool(unittest.TestCase):
    """Tests for the InfoTokenPool class."""

    def test_create_full(self) -> None:
        pool = InfoTokenPool.create_full()
        self.assertEqual(pool.yellow_tokens, 2)
        for n in range(1, 13):
            self.assertEqual(pool.blue_tokens[n], 2)

    def test_use_blue_token(self) -> None:
        pool = InfoTokenPool.create_full()
        self.assertTrue(pool.use_blue_token(5))
        self.assertEqual(pool.blue_tokens[5], 1)
        self.assertTrue(pool.use_blue_token(5))
        self.assertEqual(pool.blue_tokens[5], 0)
        self.assertFalse(pool.use_blue_token(5))

    def test_use_yellow_token(self) -> None:
        pool = InfoTokenPool.create_full()
        self.assertTrue(pool.use_yellow_token())
        self.assertTrue(pool.use_yellow_token())
        self.assertFalse(pool.use_yellow_token())


class TestEquipment(unittest.TestCase):
    """Tests for the Equipment class."""

    def test_creation(self) -> None:
        e = Equipment(
            name="Walkie-Talkies",
            description="Exchange wires",
            unlock_value=2,
        )
        self.assertFalse(e.used)
        self.assertFalse(e.unlocked)

    def test_unlock_and_use(self) -> None:
        e = Equipment(name="Test", description="", unlock_value=3)
        e.unlock()
        self.assertTrue(e.unlocked)
        e.use()
        self.assertTrue(e.used)

    def test_use_locked_raises(self) -> None:
        e = Equipment(name="Test", description="", unlock_value=3)
        with self.assertRaises(ValueError):
            e.use()

    def test_use_twice_raises(self) -> None:
        e = Equipment(name="Test", description="", unlock_value=3)
        e.unlock()
        e.use()
        with self.assertRaises(ValueError):
            e.use()


class TestMarker(unittest.TestCase):
    """Tests for the Marker class."""

    def test_known_marker(self) -> None:
        m = Marker(WireColor.RED, 3.5, MarkerState.KNOWN)
        self.assertEqual(m.base_number, 3)
        self.assertEqual(m.state, MarkerState.KNOWN)

    def test_uncertain_marker(self) -> None:
        m = Marker(WireColor.YELLOW, 5.1, MarkerState.UNCERTAIN)
        self.assertEqual(m.state, MarkerState.UNCERTAIN)
        self.assertIn("?", str(m))


class TestWireConfig(unittest.TestCase):
    """Tests for the WireConfig class."""

    def test_valid_yellow(self) -> None:
        config = WireConfig(WireColor.YELLOW, count=2)
        self.assertEqual(config.count, 2)
        self.assertIsNone(config.pool_size)

    def test_valid_red_x_of_y(self) -> None:
        config = WireConfig(WireColor.RED, count=2, pool_size=3)
        self.assertEqual(config.count, 2)
        self.assertEqual(config.pool_size, 3)

    def test_blue_raises(self) -> None:
        with self.assertRaises(ValueError):
            WireConfig(WireColor.BLUE, count=4)

    def test_yellow_too_many_raises(self) -> None:
        with self.assertRaises(ValueError):
            WireConfig(WireColor.YELLOW, count=5)

    def test_red_too_many_raises(self) -> None:
        with self.assertRaises(ValueError):
            WireConfig(WireColor.RED, count=4)

    def test_pool_less_than_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            WireConfig(WireColor.YELLOW, count=3, pool_size=2)


class TestTurnHistory(unittest.TestCase):
    """Tests for the TurnHistory class."""

    def test_empty_history(self) -> None:
        h = TurnHistory()
        self.assertEqual(len(h.actions), 0)

    def test_record_and_query(self) -> None:
        from bomb_busters import DualCutAction

        h = TurnHistory()
        action = DualCutAction(
            actor_index=0,
            target_player_index=1,
            target_slot_index=3,
            guessed_value=5,
            result=ActionResult.FAIL_BLUE_YELLOW,
        )
        h.record(action)
        self.assertEqual(len(h.actions), 1)

        failed = h.failed_dual_cuts_by_player(0)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].guessed_value, 5)

    def test_failed_cuts_filters_correctly(self) -> None:
        from bomb_busters import DualCutAction, SoloCutAction

        h = TurnHistory()
        # Failed dual cut by player 0
        h.record(DualCutAction(
            actor_index=0, target_player_index=1,
            target_slot_index=0, guessed_value=3,
            result=ActionResult.FAIL_BLUE_YELLOW,
        ))
        # Successful dual cut by player 0
        h.record(DualCutAction(
            actor_index=0, target_player_index=1,
            target_slot_index=1, guessed_value=5,
            result=ActionResult.SUCCESS,
        ))
        # Failed dual cut by player 1
        h.record(DualCutAction(
            actor_index=1, target_player_index=0,
            target_slot_index=0, guessed_value=7,
            result=ActionResult.FAIL_BLUE_YELLOW,
        ))
        # Solo cut by player 0
        h.record(SoloCutAction(
            actor_index=0, value=9,
            slot_indices=(0, 1), wire_count=2,
        ))

        # Player 0 failed cuts: only the one for value 3
        failed_0 = h.failed_dual_cuts_by_player(0)
        self.assertEqual(len(failed_0), 1)
        self.assertEqual(failed_0[0].guessed_value, 3)

        # Player 1 failed cuts: only the one for value 7
        failed_1 = h.failed_dual_cuts_by_player(1)
        self.assertEqual(len(failed_1), 1)
        self.assertEqual(failed_1[0].guessed_value, 7)


class TestGameStateCreation(unittest.TestCase):
    """Tests for GameState.create_game."""

    def test_create_4_players(self) -> None:
        game = GameState.create_game(
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
        game = GameState.create_game(
            player_names=["A", "B", "C", "D", "E"],
            seed=42,
        )
        self.assertEqual(len(game.players), 5)
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 48)
        self.assertEqual(game.detonator.max_failures, 4)

    def test_create_with_red_wires(self) -> None:
        game = GameState.create_game(
            player_names=["A", "B", "C", "D"],
            wire_configs=[WireConfig(WireColor.RED, count=3)],
            seed=42,
        )
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 51)
        # 3 KNOWN red markers
        red_markers = [m for m in game.markers if m.color == WireColor.RED]
        self.assertEqual(len(red_markers), 3)
        self.assertTrue(all(m.state == MarkerState.KNOWN for m in red_markers))

    def test_create_with_yellow_x_of_y(self) -> None:
        game = GameState.create_game(
            player_names=["A", "B", "C", "D"],
            wire_configs=[WireConfig(WireColor.YELLOW, count=2, pool_size=3)],
            seed=42,
        )
        total = sum(p.tile_stand.remaining_count for p in game.players)
        self.assertEqual(total, 50)  # 48 blue + 2 yellow
        # 3 UNCERTAIN yellow markers
        yellow_markers = [m for m in game.markers if m.color == WireColor.YELLOW]
        self.assertEqual(len(yellow_markers), 3)
        self.assertTrue(
            all(m.state == MarkerState.UNCERTAIN for m in yellow_markers)
        )

    def test_invalid_player_count(self) -> None:
        with self.assertRaises(ValueError):
            GameState.create_game(player_names=["A", "B", "C"])

    def test_each_player_has_character_card(self) -> None:
        game = GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        for player in game.players:
            self.assertIsNotNone(player.character_card)
            self.assertEqual(player.character_card.name, "Double Detector")
            self.assertFalse(player.character_card.used)

    def test_stands_are_sorted(self) -> None:
        game = GameState.create_game(
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
        game = GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        for player in game.players:
            self.assertEqual(player.tile_stand.remaining_count, 12)

    def test_deal_distribution_uneven(self) -> None:
        """With 50 wires and 4 players: 2 get 13, 2 get 12."""
        game = GameState.create_game(
            player_names=["A", "B", "C", "D"],
            wire_configs=[WireConfig(WireColor.YELLOW, count=2)],
            seed=42,
        )
        counts = [p.tile_stand.remaining_count for p in game.players]
        self.assertEqual(sum(counts), 50)
        self.assertEqual(sorted(counts), [12, 12, 13, 13])

    def test_wires_in_play_tracked(self) -> None:
        game = GameState.create_game(
            player_names=["A", "B", "C", "D"],
            seed=42,
        )
        self.assertEqual(len(game.wires_in_play), 48)

    def test_seed_reproducibility(self) -> None:
        game1 = GameState.create_game(
            player_names=["A", "B", "C", "D"], seed=123
        )
        game2 = GameState.create_game(
            player_names=["A", "B", "C", "D"], seed=123
        )
        for p1, p2 in zip(game1.players, game2.players):
            wires1 = [s.wire for s in p1.tile_stand.slots]
            wires2 = [s.wire for s in p2.tile_stand.slots]
            self.assertEqual(wires1, wires2)

    def test_str_output(self) -> None:
        game = GameState.create_game(
            player_names=["Alice", "Bob", "Carol", "Dave"],
            seed=42,
        )
        output = str(game)
        self.assertIn("Bomb Busters", output)
        self.assertIn("Alice", output)
        self.assertIn("Mistakes remaining", output)


class TestGameStatePartialState(unittest.TestCase):
    """Tests for GameState.from_partial_state."""

    def test_basic_partial_state(self) -> None:
        stands = [
            # Player 0 (observer): knows own wires
            [
                Slot(wire=Wire(WireColor.BLUE, 1.0)),
                Slot(wire=Wire(WireColor.BLUE, 5.0)),
            ],
            # Player 1: one hidden, one cut
            [
                Slot(wire=None),
                Slot(wire=Wire(WireColor.BLUE, 3.0), state=SlotState.CUT),
            ],
        ]
        game = GameState.from_partial_state(
            player_names=["Me", "Them"],
            stands=stands,
        )
        self.assertEqual(len(game.players), 2)
        self.assertIsNotNone(game.players[0].tile_stand.slots[0].wire)
        self.assertIsNone(game.players[1].tile_stand.slots[0].wire)

    def test_partial_with_detonator(self) -> None:
        stands = [
            [Slot(wire=Wire(WireColor.BLUE, 1.0))],
            [Slot(wire=None)],
        ]
        game = GameState.from_partial_state(
            player_names=["A", "B"],
            stands=stands,
            detonator_failures=2,
        )
        self.assertEqual(game.detonator.failures, 2)

    def test_partial_with_validation(self) -> None:
        stands = [
            [Slot(wire=Wire(WireColor.BLUE, 1.0))],
            [Slot(wire=None)],
        ]
        game = GameState.from_partial_state(
            player_names=["A", "B"],
            stands=stands,
            validation_tokens={1, 5, 9},
        )
        self.assertEqual(game.validation_tokens, {1, 5, 9})

    def test_mismatched_stands_raises(self) -> None:
        with self.assertRaises(ValueError):
            GameState.from_partial_state(
                player_names=["A", "B"],
                stands=[[Slot(wire=None)]],  # Only 1 stand for 2 players
            )


class TestDualCutExecution(unittest.TestCase):
    """Tests for GameState.execute_dual_cut."""

    def _make_game(self) -> GameState:
        """Create a small known game for testing."""
        # Player 0: [blue-1, blue-2, blue-3]
        # Player 1: [blue-1, blue-2, blue-3]
        # Player 2: [blue-1, blue-2, blue-3]
        # Player 3: [blue-1, blue-2, blue-3]
        hands = [
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0),
             Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0),
             Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0),
             Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0),
             Wire(WireColor.BLUE, 3.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        return GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )

    def test_successful_dual_cut(self) -> None:
        game = self._make_game()
        # Player 0 guesses player 1's slot 0 is a 1 (correct!)
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
        )
        self.assertEqual(action.result, ActionResult.SUCCESS)
        # Target's slot 0 is now cut
        self.assertTrue(game.players[1].tile_stand.slots[0].is_cut)
        # Actor also cut one of their 1s
        self.assertIsNotNone(action.actor_cut_slot_index)

    def test_failed_dual_cut_blue(self) -> None:
        game = self._make_game()
        # Player 0 guesses player 1's slot 2 (blue-3) is a 2 (wrong!)
        # Player 0 has a 2, so the guess is valid but incorrect.
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=2,
            guessed_value=2,
        )
        self.assertEqual(action.result, ActionResult.FAIL_BLUE_YELLOW)
        # Detonator advanced
        self.assertEqual(game.detonator.failures, 1)
        # Info token placed showing the real value (3)
        self.assertTrue(game.players[1].tile_stand.slots[2].is_info_revealed)
        self.assertEqual(game.players[1].tile_stand.slots[2].info_token, 3)

    def test_failed_dual_cut_red_explodes(self) -> None:
        # Create game with a red wire
        # Player 0 has [blue-1, blue-2], Player 1 has [red-1.5, blue-3]
        # Player 0 guesses P1's slot 0 (red-1.5) is a 1. P0 has a 1.
        hands = [
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.RED, 1.5), Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        game = GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[Marker(WireColor.RED, 1.5, MarkerState.KNOWN)],
            equipment=[],
            history=TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        # Player 0 guesses player 1's slot 0 (red-1.5) is a 1
        action = game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=1,
        )
        self.assertEqual(action.result, ActionResult.FAIL_RED)
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
                guessed_value=10,  # Player 0 doesn't have a 10
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
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 1.0),
             Wire(WireColor.BLUE, 5.0), Wire(WireColor.BLUE, 5.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 1.0),
             Wire(WireColor.BLUE, 5.0), Wire(WireColor.BLUE, 5.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 3.0),
             Wire(WireColor.BLUE, 6.0), Wire(WireColor.BLUE, 7.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 3.0),
             Wire(WireColor.BLUE, 6.0), Wire(WireColor.BLUE, 7.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        game = GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
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


class TestDoubleDectectorExecution(unittest.TestCase):
    """Tests for Double Detector in dual cut."""

    def _make_game(self) -> GameState:
        hands = [
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 5.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 5.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 6.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 6.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        return GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
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
        self.assertEqual(action.result, ActionResult.SUCCESS)
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
        self.assertEqual(action.result, ActionResult.SUCCESS)

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
        self.assertEqual(action.result, ActionResult.FAIL_BLUE_YELLOW)
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
    """Tests for GameState.execute_solo_cut."""

    def _make_game_for_solo(self) -> GameState:
        """Player 0 has all four 1s."""
        hands = [
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 1.0),
             Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 1.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 3.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        return GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
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
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 2.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 3.0), Wire(WireColor.BLUE, 4.0)],
            [Wire(WireColor.BLUE, 3.0), Wire(WireColor.BLUE, 4.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        game = GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        with self.assertRaises(ValueError):
            game.execute_solo_cut(2, [0, 1])


class TestRevealRedExecution(unittest.TestCase):
    """Tests for GameState.execute_reveal_red."""

    def test_reveal_all_red(self) -> None:
        hands = [
            [Wire(WireColor.RED, 1.5), Wire(WireColor.RED, 2.5)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 3.0), Wire(WireColor.BLUE, 4.0)],
            [Wire(WireColor.BLUE, 5.0), Wire(WireColor.BLUE, 6.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        game = GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        action = game.execute_reveal_red()
        self.assertEqual(len(action.slot_indices), 2)
        self.assertTrue(game.players[0].is_finished)

    def test_cannot_reveal_if_not_all_red(self) -> None:
        hands = [
            [Wire(WireColor.RED, 1.5), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 3.0)],
            [Wire(WireColor.BLUE, 4.0), Wire(WireColor.BLUE, 5.0)],
            [Wire(WireColor.BLUE, 6.0), Wire(WireColor.BLUE, 7.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        game = GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
            wires_in_play=[w for hand in hands for w in hand],
        )
        with self.assertRaises(ValueError):
            game.execute_reveal_red()


class TestGameStateHelpers(unittest.TestCase):
    """Tests for GameState helper methods."""

    def _make_game(self) -> GameState:
        hands = [
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
            [Wire(WireColor.BLUE, 1.0), Wire(WireColor.BLUE, 2.0)],
        ]
        players = [
            Player(
                name=f"P{i}",
                tile_stand=TileStand.from_wires(hands[i]),
                character_card=create_double_detector(),
            )
            for i in range(4)
        ]
        return GameState(
            players=players,
            detonator=Detonator(max_failures=3),
            info_token_pool=InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=TurnHistory(),
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
                player.tile_stand.slots[i].state = SlotState.CUT
        game._check_win()
        self.assertTrue(game.game_over)
        self.assertTrue(game.game_won)

    def test_turn_skips_finished_players(self) -> None:
        game = self._make_game()
        # Finish player 1
        for slot in game.players[1].tile_stand.slots:
            slot.state = SlotState.CUT
        # Player 0's turn, do a cut
        game.execute_dual_cut(2, 0, 1)
        # Should skip player 1, go to player 2
        self.assertEqual(game.current_player_index, 2)


if __name__ == "__main__":
    unittest.main()
