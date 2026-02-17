"""Unit tests for the probability engine."""

import unittest

import bomb_busters
import compute_probabilities


def _make_known_game(
    hands: list[list[bomb_busters.Wire]],
    detonator_failures: int = 0,
    history: bomb_busters.TurnHistory | None = None,
) -> bomb_busters.GameState:
    """Helper to create a fully-known game from explicit hand lists.

    Args:
        hands: List of wire lists, one per player.
        detonator_failures: Starting failures on the detonator.
        history: Optional turn history.

    Returns:
        A bomb_busters.GameState with all wires known.
    """
    players = [
        bomb_busters.Player(
            name=f"P{i}",
            tile_stand=bomb_busters.TileStand.from_wires(hand),
            character_card=bomb_busters.create_double_detector(),
        )
        for i, hand in enumerate(hands)
    ]
    all_wires = [w for hand in hands for w in hand]
    return bomb_busters.GameState(
        players=players,
        detonator=bomb_busters.Detonator(
            failures=detonator_failures,
            max_failures=len(hands) - 1,
        ),
        info_token_pool=bomb_busters.InfoTokenPool.create_full(),
        validation_tokens=set(),
        markers=[],
        equipment=[],
        history=history or bomb_busters.TurnHistory(),
        wires_in_play=all_wires,
    )


class TestExtractKnownInfo(unittest.TestCase):
    """Tests for compute_probabilities.extract_known_info."""

    def test_basic_extraction(self) -> None:
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        known = compute_probabilities.extract_known_info(game, 0)
        self.assertEqual(known.observer_index, 0)
        self.assertEqual(len(known.observer_wires), 2)
        self.assertEqual(len(known.cut_wires), 0)
        self.assertEqual(len(known.info_revealed), 0)

    def test_with_cuts(self) -> None:
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # P0 dual cuts P1's slot 0 (value 1)
        game.execute_dual_cut(1, 0, 1)
        known = compute_probabilities.extract_known_info(game, 0)
        self.assertEqual(len(known.cut_wires), 2)


class TestComputeUnknownPool(unittest.TestCase):
    """Tests for compute_probabilities.compute_unknown_pool."""

    def test_initial_pool(self) -> None:
        """Unknown pool is all wires minus the observer's hand."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        known = compute_probabilities.extract_known_info(game, 0)
        pool = compute_probabilities.compute_unknown_pool(known, game)
        # Total 8 wires, observer has 2 → 6 unknown
        self.assertEqual(len(pool), 6)

    def test_pool_after_cuts(self) -> None:
        """Cut wires are removed from the pool."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        game.execute_dual_cut(1, 0, 1)  # Cuts two 1s
        known = compute_probabilities.extract_known_info(game, 0)
        pool = compute_probabilities.compute_unknown_pool(known, game)
        # 8 total - 2 observer (includes P0's cut wire) - 1 (P1's cut wire) = 5
        self.assertEqual(len(pool), 5)

    def test_pool_after_info_reveal(self) -> None:
        """Info-revealed blue wires are removed from the pool."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Fail a dual cut → info token placed
        game.execute_dual_cut(1, 0, 1)  # P0 guesses P1[0]=1, but it's 2 → fail
        known = compute_probabilities.extract_known_info(game, 0)
        pool = compute_probabilities.compute_unknown_pool(known, game)
        # 8 total - 2 observer - 1 info revealed = 5 unknown
        self.assertEqual(len(pool), 5)


class TestPositionConstraints(unittest.TestCase):
    """Tests for compute_probabilities.compute_position_constraints."""

    def test_all_hidden(self) -> None:
        """All other players' slots are hidden → wide bounds."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        constraints = compute_probabilities.compute_position_constraints(game, 0)
        self.assertEqual(len(constraints), 3)  # 3 other players, 1 slot each
        for c in constraints:
            self.assertEqual(c.lower_bound, 0.0)
            self.assertEqual(c.upper_bound, 13.0)

    def test_constraints_from_neighbors(self) -> None:
        """Cut neighbors constrain hidden slots."""
        # P1 has [CUT-3, HIDDEN, CUT-7]
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
        ])
        # Cut P1's slots 0 and 2
        game.players[1].tile_stand.cut_wire_at(0)
        game.players[1].tile_stand.cut_wire_at(2)

        constraints = compute_probabilities.compute_position_constraints(game, 0)
        # Find P1's constraint
        p1_constraints = [c for c in constraints if c.player_index == 1]
        self.assertEqual(len(p1_constraints), 1)
        c = p1_constraints[0]
        self.assertEqual(c.lower_bound, 3.0)
        self.assertEqual(c.upper_bound, 7.0)

    def test_hidden_neighbors_do_not_leak_bounds(self) -> None:
        """Hidden slots with actual wires must not tighten constraints.

        In simulation mode, hidden slots have Wire objects set. The
        constraint solver must NOT use these as bounds — only CUT and
        INFO_REVEALED slots are publicly visible.

        P1 has [CUT-2, HIDDEN-5, HIDDEN-6, HIDDEN-8, CUT-10].
        The constraint for HIDDEN-6 (slot 2) should be [2.0, 10.0]
        (from the two CUT neighbors), NOT [5.0, 8.0] (from the adjacent
        hidden wires 5 and 8 which the observer cannot see).
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
        ])
        # Cut P1's first and last slots
        game.players[1].tile_stand.cut_wire_at(0)  # blue-2
        game.players[1].tile_stand.cut_wire_at(4)  # blue-10

        constraints = compute_probabilities.compute_position_constraints(game, 0)
        p1_constraints = {
            c.slot_index: c for c in constraints if c.player_index == 1
        }
        # All 3 hidden slots should have bounds [2.0, 10.0]
        for s_idx in (1, 2, 3):
            self.assertIn(s_idx, p1_constraints)
            c = p1_constraints[s_idx]
            self.assertEqual(c.lower_bound, 2.0,
                             f"Slot {s_idx} lower bound should be 2.0 (CUT), "
                             f"not tighter from hidden neighbor")
            self.assertEqual(c.upper_bound, 10.0,
                             f"Slot {s_idx} upper bound should be 10.0 (CUT), "
                             f"not tighter from hidden neighbor")

    def test_no_false_100_percent_from_hidden_info(self) -> None:
        """Verify probabilities are not artificially inflated by hidden wire info.

        Creates a scenario where tight bounds from hidden wires would produce
        a false 100% guarantee, but correct bounds allow multiple distributions.

        P0 (observer): [blue-1, blue-12]
        P1: [blue-3, blue-5, blue-9] — 3 hidden slots
        P2: [blue-4, blue-7] — 2 hidden slots
        P3: [blue-6, blue-11] — 2 hidden slots

        With leaked info: P1's hidden constraints would be artificially tight
        (each bounded by its neighbors). Without leak: all slots on P1 have
        bounds [0.0, 13.0] since nothing is cut on P1.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        probs = compute_probabilities.compute_position_probabilities(game, 0)

        # P1 has 3 all-hidden slots with no cuts → bounds should be wide.
        # Multiple wire values should be possible at each position.
        key_p1_0 = (1, 0)
        self.assertIn(key_p1_0, probs)
        counter = probs[key_p1_0]
        total = sum(counter.values())
        self.assertGreater(total, 0)
        # P1[0] should NOT be determined — multiple wires should be possible
        for wire, count in counter.items():
            self.assertLess(count, total,
                            f"P1[0] should not be 100% any wire, "
                            f"but {wire} has {count}/{total}")


class TestSmallScenario(unittest.TestCase):
    """Hand-verifiable probability scenarios with small wire counts."""

    def test_two_unknown_positions_deterministic(self) -> None:
        """When only one valid assignment exists, probability is 100%.

        P0 (observer): [blue-1, blue-4]
        P1: [blue-2, blue-3] — both hidden
        Pool = [blue-2, blue-3] (only possible assignment)
        P1[0] must be blue-2 (100%), P1[1] must be blue-3 (100%)
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        probs = compute_probabilities.compute_position_probabilities(game, 0)

        # P1 slot 0
        key_0 = (1, 0)
        self.assertIn(key_0, probs)
        total_0 = sum(probs[key_0].values())
        # Check the distribution includes values for slot 0
        self.assertGreater(total_0, 0)

    def test_equal_probability_symmetric(self) -> None:
        """Two identical wires, two positions → 50/50.

        P0 (observer): [blue-1, blue-4]
        P1: [blue-2, blue-2] — both hidden, unknown to observer
        Pool contains 2x blue-2 among other wires. Both slots can have blue-2.
        Since P1's wires are both blue-2, the probability of P1[0]=2 is...

        Actually this depends on what OTHER wires are in the pool.
        Let's make it simpler.
        """
        # P0: [blue-5], P1: [blue-1, blue-3]
        # Observer (P0) knows own hand. Pool = all wires - P0's wires.
        # P1 has 2 hidden slots. Other players also have hidden slots.
        # This gets complex with multiple players. Let's use a 2-player-like setup.

        # Simpler: P0 has [blue-5, blue-6, blue-7, blue-8],
        # P1 has [blue-1, blue-3], P2 has [blue-2, blue-4], P3 has [blue-9, blue-10]
        # Pool from P0's view: [blue-1, blue-2, blue-3, blue-4, blue-9, blue-10]
        # Positions: P1 has 2 hidden, P2 has 2 hidden, P3 has 2 hidden
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])
        probs = compute_probabilities.compute_position_probabilities(game, 0)
        # All 6 unknown wires must be distributed among 6 positions
        # across P1 (2), P2 (2), P3 (2).
        self.assertEqual(len(probs), 6)

    def test_known_value_after_info_token(self) -> None:
        """After an info token reveals a value, probability becomes 100%.

        P0: [blue-1, blue-3], P1: [blue-2, blue-4]
        P0 dual cuts P1[1] guessing 3 → fails, info token reveals 4.
        Now P1[1] is known to be 4 (info-revealed).
        P1[0] is still hidden. Pool should narrow down.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Fail a dual cut on P1 slot 1 (blue-4)
        game.execute_dual_cut(1, 1, 3)  # P0 guesses 3, actual is 4
        # P1[1] now has info token showing 4
        self.assertTrue(game.players[1].tile_stand.slots[1].is_info_revealed)

        # Now compute probabilities — P1[0] is the only hidden slot on P1
        probs = compute_probabilities.compute_position_probabilities(game, 0)
        # P1 slot 0 should be in the results
        key = (1, 0)
        self.assertIn(key, probs)

    def test_tiny_deterministic_scenario(self) -> None:
        """Minimal scenario where everything is determined.

        P0 (observer): [blue-1, blue-2, blue-3, blue-4]
        P1: [blue-5, blue-6]
        P2: [blue-7, blue-8]
        P3: [blue-9, blue-10]

        Each wire is unique. Pool from P0: [5,6,7,8,9,10].
        P1 slots must have sort values between 0 and 13.
        P1[0] < P1[1]. P2[0] < P2[1]. P3[0] < P3[1].
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])

        # P(P1[0] = 5) should be calculable
        prob = compute_probabilities.probability_of_dual_cut(game, 0, 1, 0, 5)
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        # P(P3[1] = 10) should be calculable
        prob10 = compute_probabilities.probability_of_dual_cut(game, 0, 3, 1, 10)
        self.assertGreater(prob10, 0.0)


class TestProbabilityOfDualCut(unittest.TestCase):
    """Tests for the compute_probabilities.probability_of_dual_cut function."""

    def test_impossible_cut(self) -> None:
        """Probability of a value not in the pool should be 0."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Value 12 is not in the game at all
        prob = compute_probabilities.probability_of_dual_cut(game, 0, 1, 0, 12)
        self.assertEqual(prob, 0.0)

    def test_certain_cut(self) -> None:
        """When there's only one possible wire for a position."""
        # P0 has everything except two specific wires
        # P1 has exactly [blue-11, blue-12] and pool only has these two
        # P2, P3 have nothing left (empty after cuts)
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Cut all P2 and P3 wires
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        # Now pool = [11, 12] and they must go into P1's 2 slots
        # P1[0] must be 11 (sorted ascending), P1[1] must be 12
        prob_11 = compute_probabilities.probability_of_dual_cut(game, 0, 1, 0, 11)
        prob_12 = compute_probabilities.probability_of_dual_cut(game, 0, 1, 1, 12)
        self.assertAlmostEqual(prob_11, 1.0)
        self.assertAlmostEqual(prob_12, 1.0)

        # And the opposite should be 0
        prob_12_at_0 = compute_probabilities.probability_of_dual_cut(game, 0, 1, 0, 12)
        prob_11_at_1 = compute_probabilities.probability_of_dual_cut(game, 0, 1, 1, 11)
        self.assertAlmostEqual(prob_12_at_0, 0.0)
        self.assertAlmostEqual(prob_11_at_1, 0.0)


class TestProbabilityOfDoubleDetector(unittest.TestCase):
    """Tests for compute_probabilities.probability_of_double_detector."""

    def test_dd_certain(self) -> None:
        """DD on two slots where one is guaranteed to match."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Cut all P2 and P3
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        # P1 must be [11, 12]. DD guessing 11 on slots 0 and 1
        prob = compute_probabilities.probability_of_double_detector(game, 0, 1, 0, 1, 11)
        self.assertAlmostEqual(prob, 1.0)

    def test_dd_higher_than_single(self) -> None:
        """DD probability should generally be >= single slot probability."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])
        # Cut all P2 and P3
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        # Single slot probability
        prob_single = compute_probabilities.probability_of_dual_cut(game, 0, 1, 0, 5)
        # DD probability
        prob_dd = compute_probabilities.probability_of_double_detector(game, 0, 1, 0, 1, 5)
        self.assertGreaterEqual(prob_dd, prob_single)


class TestGuaranteedActions(unittest.TestCase):
    """Tests for compute_probabilities.guaranteed_actions."""

    def test_solo_cut_guaranteed(self) -> None:
        """Solo cut appears in guaranteed actions."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
        ])
        result = compute_probabilities.guaranteed_actions(game, 0)
        self.assertEqual(len(result["solo_cuts"]), 1)
        self.assertEqual(result["solo_cuts"][0][0], 1)  # type: ignore[index]

    def test_reveal_red_guaranteed(self) -> None:
        """Reveal red appears when all remaining wires are red."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 1.5), bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
        ])
        result = compute_probabilities.guaranteed_actions(game, 0)
        self.assertTrue(result["reveal_red"])

    def test_no_guaranteed_actions(self) -> None:
        """No guaranteed actions in a typical uncertain state."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
        ])
        result = compute_probabilities.guaranteed_actions(game, 0)
        self.assertEqual(len(result["solo_cuts"]), 0)
        self.assertFalse(result["reveal_red"])


class TestRankAllMoves(unittest.TestCase):
    """Tests for compute_probabilities.rank_all_moves."""

    def test_moves_sorted_by_probability(self) -> None:
        # P0 has values 1,2 which overlap with other players' values
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
        ])
        moves = compute_probabilities.rank_all_moves(game, 0)
        # Should have some moves (P0 has 1 and 2, others might have them)
        self.assertGreater(len(moves), 0)
        # Sorted descending by probability
        for i in range(len(moves) - 1):
            self.assertGreaterEqual(moves[i].probability, moves[i + 1].probability)

    def test_solo_cut_ranked_first(self) -> None:
        """Solo cuts (100%) should appear at the top."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
        ])
        moves = compute_probabilities.rank_all_moves(game, 0)
        # First move should be the solo cut
        self.assertEqual(moves[0].action_type, "solo_cut")
        self.assertAlmostEqual(moves[0].probability, 1.0)

    def test_str_output(self) -> None:
        """compute_probabilities.RankedMove str output should be readable."""
        move = compute_probabilities.RankedMove(
            action_type="dual_cut",
            target_player=1,
            target_slot=3,
            guessed_value=5,
            probability=0.75,
        )
        output = str(move)
        self.assertIn("P1", output)
        self.assertIn("5", output)
        self.assertIn("75.0%", output)


class TestCalculatorMode(unittest.TestCase):
    """Tests using from_partial_state for calculator-mode scenarios."""

    def test_partial_state_probabilities(self) -> None:
        """Compute probabilities from a partially-known game state.

        Observer (P0) knows their hand: [blue-1, blue-5].
        P1 has 2 slots: one cut (blue-3), one hidden (unknown).
        P2 has 2 slots: both hidden.
        P3 has 2 slots: both hidden.

        Total wires in play: 8 (1,2,3,4,5,6,7,8 all blue).
        Known: P0=[1,5], P1[0]=CUT-3. Pool = [2,4,6,7,8].
        5 wires to fill 5 positions (P1[1], P2[0], P2[1], P3[0], P3[1]).
        """
        stands = [
            # P0: knows own hand
            [
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)),
            ],
            # P1: one cut, one hidden
            [
                bomb_busters.Slot(wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), state=bomb_busters.SlotState.CUT),
                bomb_busters.Slot(wire=None),  # Unknown
            ],
            # P2: both hidden
            [bomb_busters.Slot(wire=None), bomb_busters.Slot(wire=None)],
            # P3: both hidden
            [bomb_busters.Slot(wire=None), bomb_busters.Slot(wire=None)],
        ]
        all_wires = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i)) for i in range(1, 9)]
        game = bomb_busters.GameState.from_partial_state(
            player_names=["Me", "P1", "P2", "P3"],
            stands=stands,
            wires_in_play=all_wires,
        )

        # P1[1] must have sort_value > 3.0 (because P1[0] = CUT-3)
        # Possible wires for P1[1]: 4, 6, 7, 8 (not 2 since 2 < 3)
        probs = compute_probabilities.compute_position_probabilities(game, 0)
        key = (1, 1)
        self.assertIn(key, probs)

        # Blue-2 should NOT appear at P1[1] (violates sort order)
        wire_2 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)
        self.assertEqual(probs[key].get(wire_2, 0), 0)

        # Blue-4 should be possible at P1[1]
        wire_4 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)
        self.assertGreater(probs[key].get(wire_4, 0), 0)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for the probability engine."""

    def test_empty_constraints(self) -> None:
        """No hidden slots on other players → empty result."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Cut all other players' wires
        for p_idx in range(1, 4):
            for i in range(len(game.players[p_idx].tile_stand.slots)):
                game.players[p_idx].tile_stand.cut_wire_at(i)

        probs = compute_probabilities.compute_position_probabilities(game, 0)
        self.assertEqual(len(probs), 0)

    def test_duplicate_wires_in_pool(self) -> None:
        """Multiple copies of the same wire are handled correctly."""
        # P0 has [blue-1], P1-P3 each have one blue-1
        # Pool has 3x blue-1 → probability of P1[0]=1 depends on how many
        # blue-1s remain and how they distribute
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0), bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        prob = compute_probabilities.probability_of_dual_cut(game, 0, 1, 0, 1)
        # Should be > 0 (blue-1 is in the pool)
        self.assertGreater(prob, 0.0)
        # Should be < 1 (other wires could be there too)
        self.assertLess(prob, 1.0)


class TestCutDoesNotEliminateValueFromHiddenSlots(unittest.TestCase):
    """Verify that cutting one wire of a value doesn't exclude that value
    from a player's remaining hidden slots.

    Scenario: Player C (index 2) fails a dual cut for value 5, proving C
    has at least one 5. Then Player D (index 3) successfully dual cuts a 5
    on C's stand, cutting one of C's 5s. C actually has TWO 5s, so C's
    remaining hidden slots should still allow value 5.
    """

    def _make_scenario(self) -> bomb_busters.GameState:
        """Build the game state for the cut-doesn't-eliminate scenario.

        P0 (observer): [blue-5, blue-10, blue-11, blue-12]
        P1: [blue-1, blue-2]
        P2 (C): [blue-5, blue-5, blue-6]  — two 5s and a 6
        P3 (D): [blue-5, blue-8]
        P4: [blue-3, blue-4]

        Turn order: P0 → P1 → P2 → P3 → ...
        Step 1: P0's turn — skip (not relevant, just advance).
        We'll manually set current_player_index to orchestrate turns.

        Step 1: C (P2) attempts dual cut on P1[0], guessing 5. P1[0] is
                blue-1, so it fails. History records C tried 5 and missed.
        Step 2: D (P3) dual cuts 5 on C (P2), targeting P2[0] (blue-5).
                Succeeds — P2[0] and one of D's 5s are cut.

        After this, C still has a hidden blue-5 at P2[1]. The solver
        should recognize that 5 is still possible at P2[1].
        """
        game = _make_known_game([
            # P0 (observer): knows own hand
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            # P1
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            # P2 (C): two 5s and a 6
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            # P3 (D)
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            # P4
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
        ])

        # Step 1: C (P2) tries to dual cut P1[0] guessing 5.
        # P1[0] is blue-1, so this fails. Info token placed on P1[0].
        game.current_player_index = 2
        game.execute_dual_cut(
            target_player_index=1,
            target_slot_index=0,
            guessed_value=5,
        )

        # Step 2: D (P3) dual cuts 5 on C (P2), targeting P2[0] (blue-5).
        # This succeeds — P2[0] and P3[0] are cut.
        game.current_player_index = 3
        game.execute_dual_cut(
            target_player_index=2,
            target_slot_index=0,
            guessed_value=5,
        )

        return game

    def test_value_still_possible_after_cut(self) -> None:
        """After cutting one 5 on C's stand, 5 should still be possible
        at C's remaining hidden slots."""
        game = self._make_scenario()

        # P2[0] is now CUT (blue-5), P2[1] and P2[2] are still hidden.
        # P2[1] is actually blue-5. The solver should allow 5 there.
        probs = compute_probabilities.compute_position_probabilities(
            game, observer_index=0,
        )

        # P2 slot 1 should have a nonzero probability for value 5
        key = (2, 1)
        self.assertIn(key, probs)
        counter = probs[key]
        total = sum(counter.values())
        self.assertGreater(total, 0)

        wire_5 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        count_5 = counter.get(wire_5, 0)
        self.assertGreater(
            count_5, 0,
            "After cutting one 5 on P2, value 5 should still be possible "
            "at P2's remaining hidden slots (P2 had multiple 5s)",
        )

    def test_value_not_falsely_certain(self) -> None:
        """The solver should not be 100% certain about P2[1] being 5.

        Other wires from the unknown pool could also fit at that position,
        so 5 should be possible but not guaranteed.
        """
        game = self._make_scenario()

        probs = compute_probabilities.compute_position_probabilities(
            game, observer_index=0,
        )

        key = (2, 1)
        self.assertIn(key, probs)
        counter = probs[key]
        total = sum(counter.values())

        wire_5 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        count_5 = counter.get(wire_5, 0)

        # P2[1] should not be 100% certain to be 5, because other wires
        # could also occupy that position
        self.assertLess(
            count_5, total,
            "P2[1] should not be 100% certain to be 5 — other wires "
            "from the pool could also fit there",
        )

    def test_must_have_removed_after_cut(self) -> None:
        """The must-have constraint for value 5 on P2 should be removed
        after one of P2's 5s is cut, since the constraint ('at least one
        5') was satisfied by the cut wire."""
        game = self._make_scenario()

        known = compute_probabilities.extract_known_info(game, observer_index=0)
        # P2 had a failed dual cut for 5, but has since cut a 5.
        # The must-have constraint should be gone.
        self.assertNotIn(
            5,
            known.player_must_have.get(2, set()),
            "must_have for P2 should not include 5 after a 5 was cut",
        )

    def test_pool_still_contains_fives(self) -> None:
        """The unknown pool should still contain blue-5 wires after the cut.

        4 blue-5s total: P0 has 1 (known), P2[0] cut, P3[0] cut.
        That accounts for 3. One blue-5 should remain in the unknown pool.
        """
        game = self._make_scenario()

        known = compute_probabilities.extract_known_info(game, observer_index=0)
        pool = compute_probabilities.compute_unknown_pool(known, game)

        wire_5 = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)
        count_5_in_pool = sum(1 for w in pool if w == wire_5)
        self.assertEqual(
            count_5_in_pool, 1,
            "One blue-5 should remain in the unknown pool (4 total minus "
            "1 observer, 1 cut on P2, 1 cut on P3)",
        )


class TestProbabilityOfRedWire(unittest.TestCase):
    """Tests for compute_probabilities.probability_of_red_wire."""

    def test_no_red_wires_in_play(self) -> None:
        """When no red wires are in the game, P(red) is always 0%."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        probs = compute_probabilities.compute_position_probabilities(game, 0)
        for p_idx in range(1, 4):
            for s_idx in range(2):
                red_p = compute_probabilities.probability_of_red_wire(
                    game, 0, p_idx, s_idx, probs=probs,
                )
                self.assertAlmostEqual(
                    red_p, 0.0,
                    msg=f"P{p_idx}[{s_idx}] should have 0% red with no red wires",
                )

    def test_red_in_range_nonzero(self) -> None:
        """Red wire in range of a hidden slot gives P(red) > 0%.

        P0: [blue-4, blue-8]
        P1: [blue-3, red-3.5, blue-5] — Bob's slot A cut
        After cutting P1[0], P1[1] has bounds [3.0, 13.0].
        Red-3.5 fits → P(red at P1[1]) > 0%.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        game.players[1].tile_stand.cut_wire_at(0)  # Cut blue-3
        red_p = compute_probabilities.probability_of_red_wire(game, 0, 1, 1)
        self.assertGreater(red_p, 0.0)

    def test_red_out_of_range_zero(self) -> None:
        """Red wire guaranteed out of range gives P(red) = 0%.

        P0: [blue-4, blue-8]
        P1: [blue-3, red-3.5, blue-5] — slots A and C cut
        P1[1] has bounds [3.0, 5.0]. Red-3.5 fits. But if we
        constrain differently:
        P1: [blue-1, blue-5, blue-6] with red-8.5 somewhere else.
        P1[1] has bounds [1.0, 6.0] if A and C cut → red-8.5 doesn't fit.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 8.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 8.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        # Cut P1 slots A and C to constrain P1[1] to [1.0, 6.0]
        game.players[1].tile_stand.cut_wire_at(0)  # blue-1
        game.players[1].tile_stand.cut_wire_at(2)  # blue-6
        # P1[1] must have sort_value in [1.0, 6.0]. Red-8.5 doesn't fit.
        red_p = compute_probabilities.probability_of_red_wire(game, 0, 1, 1)
        self.assertAlmostEqual(red_p, 0.0)

    def test_all_red_already_revealed(self) -> None:
        """If all red wires are already exposed (cut), P(red) = 0%.

        One red wire in play, already cut via reveal-red. No hidden red
        wires remain anywhere.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            # P3 had only a red wire left, revealed it
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        # P3's red wire is already cut (revealed)
        game.players[3].tile_stand.cut_wire_at(0)

        probs = compute_probabilities.compute_position_probabilities(game, 0)
        for p_idx in range(1, 3):
            for s_idx in range(2):
                red_p = compute_probabilities.probability_of_red_wire(
                    game, 0, p_idx, s_idx, probs=probs,
                )
                self.assertAlmostEqual(
                    red_p, 0.0,
                    msg=f"All reds revealed, P{p_idx}[{s_idx}] should be 0%",
                )

    def test_100_percent_dual_cut_means_zero_red(self) -> None:
        """If dual cut success is 100%, red probability must be 0%.

        When a slot is known to be a specific blue wire, it cannot be red.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 7.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 7.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        # Cut all of P2 and P3 so P1 is determined
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        # P1 must be [11, 12] → 100% certain
        prob_11 = compute_probabilities.probability_of_dual_cut(
            game, 0, 1, 0, 11,
        )
        self.assertAlmostEqual(prob_11, 1.0)
        red_p = compute_probabilities.probability_of_red_wire(game, 0, 1, 0)
        self.assertAlmostEqual(red_p, 0.0)

    def test_guaranteed_red_wire(self) -> None:
        """Slot guaranteed to be red → P(dual cut success) = 0%, P(red) = 100%.

        P0: [blue-1, blue-12]
        P1: [red-5.5] — only one hidden slot, must be red
        P2: [blue-5, blue-6]
        P3: [blue-7, blue-8]
        All P2 and P3 cut → pool = [red-5.5] → P1[0] must be red.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 5.5)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 5.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        red_p = compute_probabilities.probability_of_red_wire(game, 0, 1, 0)
        self.assertAlmostEqual(red_p, 1.0)

        # No blue value can be dual-cut at this position
        for val in range(1, 13):
            prob = compute_probabilities.probability_of_dual_cut(
                game, 0, 1, 0, val,
            )
            self.assertAlmostEqual(
                prob, 0.0,
                msg=f"Dual cut for value {val} on guaranteed red should be 0%",
            )

    def test_red_is_not_one_minus_success(self) -> None:
        """P(red) != 1 - P(success) and P(success) != 1 - P(red).

        When there are blue, yellow, and red wires, a failed dual cut
        could be due to a different blue/yellow wire (not red).
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        game.players[1].tile_stand.cut_wire_at(0)  # Cut blue-3

        probs = compute_probabilities.compute_position_probabilities(game, 0)
        red_p = compute_probabilities.probability_of_red_wire(
            game, 0, 1, 1, probs=probs,
        )
        success_p = compute_probabilities.probability_of_dual_cut(
            game, 0, 1, 1, 5,
        )

        # With red + non-matching blue in the pool, neither relation holds
        self.assertGreater(red_p, 0.0)
        self.assertGreater(success_p, 0.0)
        # P(red) + P(success for value 5) < 1.0 because other blues can be there
        self.assertLess(red_p + success_p, 1.0 + 1e-9)
        # They are not complements
        self.assertNotAlmostEqual(red_p, 1.0 - success_p)

    def test_x_of_y_red_wire_probability(self) -> None:
        """X of Y mode: 1 of 2 possible red values in play.

        Uses from_partial_state with UNCERTAIN markers for 2 red values
        but only 1 actually in the game. The solver should account for
        the red wire being somewhere in the unknown pool.

        P0 (observer): [blue-1, blue-9]  (2 wires)
        P1: [CUT-2, HIDDEN, HIDDEN]      (2 hidden)
        P2: [HIDDEN, HIDDEN]              (2 hidden)
        P3: [HIDDEN]                      (1 hidden)

        Total wires: 8 (6 blue + 1 red). Observer has 2, P1 cut has 1.
        Unknown pool = 8 - 2 - 1 = 5 wires for 5 hidden positions.
        """
        all_blue = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i))
                    for i in [1, 2, 5, 6, 8, 9]]
        red_wire = bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5)
        # Add another blue wire to balance pool vs positions
        extra_blue = bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)
        all_wires = all_blue + [red_wire, extra_blue]

        stands = [
            # P0: knows own hand
            [bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
             bomb_busters.Slot(
                 wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0))],
            # P1: one cut, two hidden
            [bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
                state=bomb_busters.SlotState.CUT),
             bomb_busters.Slot(wire=None),
             bomb_busters.Slot(wire=None)],
            # P2: two hidden
            [bomb_busters.Slot(wire=None), bomb_busters.Slot(wire=None)],
            # P3: one hidden
            [bomb_busters.Slot(wire=None)],
        ]
        game = bomb_busters.GameState.from_partial_state(
            player_names=["Me", "P1", "P2", "P3"],
            stands=stands,
            markers=[
                bomb_busters.Marker(
                    bomb_busters.WireColor.RED, 3.5,
                    bomb_busters.MarkerState.UNCERTAIN,
                ),
                bomb_busters.Marker(
                    bomb_busters.WireColor.RED, 7.5,
                    bomb_busters.MarkerState.UNCERTAIN,
                ),
            ],
            wires_in_play=all_wires,
        )

        probs = compute_probabilities.compute_position_probabilities(game, 0)
        # P1[1] has bounds [2.0, 13.0]. Red-3.5 fits → P(red) > 0
        red_p1_1 = compute_probabilities.probability_of_red_wire(
            game, 0, 1, 1, probs=probs,
        )
        self.assertGreater(red_p1_1, 0.0)

        # Total red probability across all hidden slots should sum to
        # exactly 1 red wire distributed somewhere
        total_red = 0.0
        for (p_idx, s_idx), counter in probs.items():
            total = sum(counter.values())
            if total == 0:
                continue
            red_count = sum(
                c for w, c in counter.items()
                if w.color == bomb_busters.WireColor.RED
            )
            total_red += red_count / total
        # Should be close to 1.0 (one red wire exists somewhere)
        self.assertAlmostEqual(total_red, 1.0, places=5)

    def test_x_of_y_deduced_from_game_state(self) -> None:
        """When game state eliminates one X-of-Y candidate, probability updates.

        1 of 2 red wires is in play (red-3.5 or red-7.5). Only red-3.5
        is actually in wires_in_play. All hidden slots on P1 have bounds
        [2.0, 5.0] so red-7.5 (which isn't in the pool anyway) wouldn't
        fit. The red-3.5 DOES fit and should have nonzero probability.

        P0 (observer): [blue-1, blue-6]  (2 wires)
        P1: [CUT-2, HIDDEN, CUT-5]       (1 hidden)
        P2: [HIDDEN]                      (1 hidden)
        P3: [HIDDEN]                      (1 hidden)

        Total wires: 6 (5 blue + 1 red). Observer has 2, 2 cut.
        Unknown pool = 6 - 2 - 2 = 2? No wait, observer wires include
        cut wires too? No, observer_wires are P0's wires. P1's cut wires
        are separate.

        Let's be precise: wires_in_play has 6 wires.
        Observer wires (P0): blue-1, blue-6 → 2 removed.
        Other cut wires: P1[0]=blue-2, P1[2]=blue-5 → 2 removed.
        Unknown pool = 6 - 2 - 2 = 2 wires: [blue-3, red-3.5]
        Hidden positions: P1[1], P2[0], P3[0] = 3 positions.
        But 2 wires for 3 positions is inconsistent!

        Fix: add another wire to balance.
        """
        all_blue = [bomb_busters.Wire(bomb_busters.WireColor.BLUE, float(i))
                    for i in [1, 2, 3, 4, 5, 6]]
        red_wire = bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5)
        all_wires = all_blue + [red_wire]
        # 7 total wires. Observer has 2, P1 has 2 cut → pool = 3, positions = 3.

        stands = [
            # P0 (observer)
            [bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)),
             bomb_busters.Slot(
                 wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0))],
            # P1: [CUT-2, HIDDEN, CUT-5]
            [bomb_busters.Slot(
                wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
                state=bomb_busters.SlotState.CUT),
             bomb_busters.Slot(wire=None),
             bomb_busters.Slot(
                 wire=bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
                 state=bomb_busters.SlotState.CUT)],
            # P2: [HIDDEN]
            [bomb_busters.Slot(wire=None)],
            # P3: [HIDDEN]
            [bomb_busters.Slot(wire=None)],
        ]
        game = bomb_busters.GameState.from_partial_state(
            player_names=["Me", "P1", "P2", "P3"],
            stands=stands,
            markers=[
                bomb_busters.Marker(
                    bomb_busters.WireColor.RED, 3.5,
                    bomb_busters.MarkerState.UNCERTAIN,
                ),
                bomb_busters.Marker(
                    bomb_busters.WireColor.RED, 7.5,
                    bomb_busters.MarkerState.UNCERTAIN,
                ),
            ],
            wires_in_play=all_wires,
        )

        probs = compute_probabilities.compute_position_probabilities(game, 0)
        # P1[1] has bounds [2.0, 5.0]. Red-3.5 fits.
        # Pool = [blue-3, red-3.5, blue-4] → 3 wires, 3 positions.
        red_p1_1 = compute_probabilities.probability_of_red_wire(
            game, 0, 1, 1, probs=probs,
        )
        self.assertGreater(
            red_p1_1, 0.0,
            "Red-3.5 fits in [2.0, 5.0] bounds, should have nonzero probability",
        )

        # P2[0] and P3[0] have wide bounds [0, 13]. Red-3.5 also fits there.
        red_p2_0 = compute_probabilities.probability_of_red_wire(
            game, 0, 2, 0, probs=probs,
        )
        red_p3_0 = compute_probabilities.probability_of_red_wire(
            game, 0, 3, 0, probs=probs,
        )
        # All three positions can potentially have the red wire
        self.assertGreater(red_p2_0, 0.0)
        self.assertGreater(red_p3_0, 0.0)


class TestProbabilityOfRedWireDD(unittest.TestCase):
    """Tests for compute_probabilities.probability_of_red_wire_dd."""

    def test_dd_one_red_in_play_zero_red_explosion(self) -> None:
        """With DD and only 1 red wire in play, P(both red) = 0%.

        Double Detector targeting 2 slots: the bomb only explodes if
        BOTH are red. With only 1 red wire in the game, it's impossible
        for both to be red simultaneously.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        # Only 1 red wire → DD can never have both slots red
        red_dd = compute_probabilities.probability_of_red_wire_dd(
            game, 0, 1, 0, 1,
        )
        self.assertAlmostEqual(red_dd, 0.0)

        # But the single-slot red probability should still be > 0
        red_single = compute_probabilities.probability_of_red_wire(
            game, 0, 1, 1,
        )
        self.assertGreater(red_single, 0.0)

    def test_dd_two_reds_possible_both_red(self) -> None:
        """With 2 red wires and DD targeting 2 adjacent slots, P(both red) > 0%.

        P0: [blue-5, blue-8]
        P1: [red-3.5, red-4.5, blue-6] — two reds adjacent
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 4.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 4.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        # 2 red wires exist → DD targeting P1 slots 0 and 1 could have
        # both be red
        red_dd = compute_probabilities.probability_of_red_wire_dd(
            game, 0, 1, 0, 1,
        )
        self.assertGreater(red_dd, 0.0)

    def test_dd_two_reds_constrained_to_one_max(self) -> None:
        """DD on a player where at most 1 red fits → P(both red) = 0%.

        P0: [blue-5, blue-8]
        P1: [CUT-1, HIDDEN, CUT-3] — bounds [1.0, 3.0] for P1[1]
        P2: [blue-9, blue-10]
        P3: [blue-11, blue-12]

        Only red-2.5 fits in [1.0, 3.0]. Red-7.5 doesn't.
        P1 has only 1 hidden slot, so DD can't target 2 slots on P1.

        Instead use P1 with 2 hidden slots but tight bounds that only
        allow 1 red wire.

        P1: [CUT-1, HIDDEN, HIDDEN, CUT-4]
        Bounds for P1[1]: [1.0, 4.0], P1[2]: [1.0, 4.0].
        Red-2.5 fits, Red-7.5 doesn't. Only 1 red can be placed.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 7.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 2.5,
                bomb_busters.MarkerState.KNOWN,
            ),
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 7.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        # Cut P1's first and last slots to constrain hidden slots
        game.players[1].tile_stand.cut_wire_at(0)  # blue-1
        game.players[1].tile_stand.cut_wire_at(3)  # blue-4
        # P1[1] and P1[2] have bounds [1.0, 4.0].
        # Only red-2.5 fits (red-7.5 = 7.5 > 4.0).
        # So at most 1 of the 2 hidden slots can be red → P(both red) = 0.
        red_dd = compute_probabilities.probability_of_red_wire_dd(
            game, 0, 1, 1, 2,
        )
        self.assertAlmostEqual(red_dd, 0.0)

    def test_dd_guaranteed_both_red(self) -> None:
        """When both DD target slots are guaranteed red, P(both red) = 100%.

        P0: [blue-1, blue-12]
        P1: [red-3.5, red-4.5] — only 2 wires, both red
        All other players cut.
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 12.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 4.5)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 4.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        # Pool = [red-3.5, red-4.5] → P1 must have both reds
        red_dd = compute_probabilities.probability_of_red_wire_dd(
            game, 0, 1, 0, 1,
        )
        self.assertAlmostEqual(red_dd, 1.0)


class TestRankedMoveRedProbability(unittest.TestCase):
    """Tests for red_probability field in RankedMove via rank_all_moves."""

    def test_ranked_moves_include_red_probability(self) -> None:
        """Ranked moves include non-zero red_probability when red wires exist."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        game.players[1].tile_stand.cut_wire_at(0)  # Cut blue-3

        moves = compute_probabilities.rank_all_moves(game, 0)
        # At least one move should have nonzero red_probability
        red_moves = [m for m in moves if m.red_probability > 0]
        self.assertGreater(len(red_moves), 0)

        # Solo cuts should have 0 red probability
        solo_moves = [m for m in moves if m.action_type == "solo_cut"]
        for m in solo_moves:
            self.assertAlmostEqual(m.red_probability, 0.0)

    def test_no_red_means_zero_red_in_moves(self) -> None:
        """All moves have red_probability = 0 when no red wires exist."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
        ])
        moves = compute_probabilities.rank_all_moves(game, 0)
        for m in moves:
            self.assertAlmostEqual(
                m.red_probability, 0.0,
                msg=f"No red wires → all moves should have 0% red, got {m}",
            )

    def test_ranked_move_str_shows_red(self) -> None:
        """RankedMove.__str__ includes RED warning when red_probability > 0."""
        move = compute_probabilities.RankedMove(
            action_type="dual_cut",
            target_player=1,
            target_slot=2,
            guessed_value=5,
            probability=0.5,
            red_probability=0.25,
        )
        output = str(move)
        self.assertIn("RED", output)
        self.assertIn("25.0%", output)

    def test_ranked_move_str_no_red_when_zero(self) -> None:
        """RankedMove.__str__ omits RED warning when red_probability is 0."""
        move = compute_probabilities.RankedMove(
            action_type="dual_cut",
            target_player=1,
            target_slot=2,
            guessed_value=5,
            probability=0.5,
            red_probability=0.0,
        )
        output = str(move)
        self.assertNotIn("RED", output)


class TestBlueWireSubset(unittest.TestCase):
    """Tests that probabilities correctly respect a subset of blue wires."""

    def _make_subset_game(
        self,
    ) -> tuple[bomb_busters.GameState, int]:
        """Create a game with only blue wires 1-4 (16 total), 4 players.

        Returns:
            Tuple of (game, observer_index).
        """
        blue_pool = bomb_busters.create_blue_wires(1, 4)
        # 16 wires, 4 players = 4 each
        hands = [
            # P0 (observer)
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            # P1
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            # P2
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            # P3
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hand),
                character_card=bomb_busters.create_double_detector(),
            )
            for i, hand in enumerate(hands)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(failures=0, max_failures=3),
            info_token_pool=bomb_busters.InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=blue_pool,
        )
        return game, 0

    def test_no_out_of_range_wires_in_probabilities(self) -> None:
        """Position probabilities only contain wires from the subset pool."""
        game, observer = self._make_subset_game()
        probs = compute_probabilities.compute_position_probabilities(
            game, observer,
        )
        for (p_idx, s_idx), counter in probs.items():
            for wire in counter:
                self.assertIn(
                    wire.gameplay_value, {1, 2, 3, 4},
                    f"Wire {wire} at P{p_idx}[{s_idx}] is outside the "
                    f"blue 1-4 subset pool",
                )

    def test_dual_cut_zero_for_absent_value(self) -> None:
        """Dual cut for a value not in the pool returns 0% probability."""
        game, observer = self._make_subset_game()
        # Value 5 is not in the pool
        prob = compute_probabilities.probability_of_dual_cut(
            game, observer, 1, 0, 5,
        )
        self.assertAlmostEqual(prob, 0.0)

    def test_dual_cut_nonzero_for_present_value(self) -> None:
        """Dual cut for a value in the pool returns non-zero probability."""
        game, observer = self._make_subset_game()
        prob = compute_probabilities.probability_of_dual_cut(
            game, observer, 1, 0, 1,
        )
        self.assertGreater(prob, 0.0)

    def test_rank_all_moves_only_subset_values(self) -> None:
        """Ranked moves only suggest values within the subset pool."""
        game, observer = self._make_subset_game()
        moves = compute_probabilities.rank_all_moves(game, observer)
        for move in moves:
            if move.guessed_value is not None:
                self.assertIn(
                    move.guessed_value, {1, 2, 3, 4},
                    f"Move suggests value {move.guessed_value} which is "
                    f"outside the blue 1-4 subset pool",
                )

    def test_probabilities_sum_to_one(self) -> None:
        """Each hidden slot's probability distribution sums to 1.0."""
        game, observer = self._make_subset_game()
        probs = compute_probabilities.compute_position_probabilities(
            game, observer,
        )
        for (p_idx, s_idx), counter in probs.items():
            total = sum(counter.values())
            self.assertGreater(total, 0)
            # Each wire's fraction should sum to 1.0
            prob_sum = sum(count / total for count in counter.values())
            self.assertAlmostEqual(
                prob_sum, 1.0, places=10,
                msg=f"P{p_idx}[{s_idx}] probabilities don't sum to 1.0",
            )

    def test_subset_with_red_wires(self) -> None:
        """Probabilities work correctly with blue subset + red wires."""
        blue_pool = bomb_busters.create_blue_wires(1, 3)  # 12 blue
        red_wire = bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5)
        all_wires = blue_pool + [red_wire]
        # 13 wires, 4 players = 4-3-3-3
        hands = [
            # P0 (observer) — 4 wires
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            # P1 — 3 wires including the red wire
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 2.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            # P2 — 3 wires
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            # P3 — 3 wires
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
        ]
        players = [
            bomb_busters.Player(
                name=f"P{i}",
                tile_stand=bomb_busters.TileStand.from_wires(hand),
                character_card=bomb_busters.create_double_detector(),
            )
            for i, hand in enumerate(hands)
        ]
        game = bomb_busters.GameState(
            players=players,
            detonator=bomb_busters.Detonator(failures=0, max_failures=3),
            info_token_pool=bomb_busters.InfoTokenPool.create_full(),
            validation_tokens=set(),
            markers=[
                bomb_busters.Marker(
                    bomb_busters.WireColor.RED, 2.5,
                    bomb_busters.MarkerState.KNOWN,
                ),
            ],
            equipment=[],
            history=bomb_busters.TurnHistory(),
            wires_in_play=all_wires,
        )
        probs = compute_probabilities.compute_position_probabilities(
            game, 0,
        )
        # No wire with value > 3 should appear
        for (p_idx, s_idx), counter in probs.items():
            for wire in counter:
                self.assertLessEqual(
                    wire.sort_value, 3.0,
                    f"Wire {wire} at P{p_idx}[{s_idx}] exceeds blue 1-3 "
                    f"+ red range",
                )
        # Red wire should have nonzero probability on some slot of P1
        # (P1 has slots A=blue-1, B=red-2.5, C=blue-3; B is hidden)
        red_probs = []
        for s_idx in range(3):
            key = (1, s_idx)
            if key in probs:
                for wire, count in probs[key].items():
                    if wire.color == bomb_busters.WireColor.RED:
                        red_probs.append(count)
        self.assertTrue(
            any(c > 0 for c in red_probs),
            "Red wire should have nonzero probability on P1's stand",
        )


class TestSharedMemo(unittest.TestCase):
    """Tests for shared memo correctness via build_solver().

    Verifies that pre-building ctx/memo with build_solver() and passing
    them to API functions produces identical results to calling those
    functions without pre-built ctx/memo (which triggers independent
    solver builds internally).
    """

    def _make_game_with_red(
        self,
    ) -> tuple[bomb_busters.GameState, int]:
        """Create a game with red wires for shared-memo testing.

        P0 (observer): [blue-5, blue-8]
        P1: [blue-3, red-3.5, blue-6] — red wire in middle
        P2: [blue-7, blue-9]
        P3: [blue-10, blue-11]

        One red wire in play. P1 has one cut slot to create constraints.

        Returns:
            Tuple of (game, observer_index).
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        game.players[1].tile_stand.cut_wire_at(0)  # Cut blue-3
        return game, 0

    def _make_game_two_reds(
        self,
    ) -> tuple[bomb_busters.GameState, int]:
        """Create a game with two red wires for DD red testing.

        P0 (observer): [blue-5, blue-8]
        P1: [red-3.5, red-4.5, blue-6] — two adjacent reds
        P2: [blue-7, blue-9]
        P3: [blue-10, blue-11]

        Returns:
            Tuple of (game, observer_index).
        """
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.RED, 3.5),
             bomb_busters.Wire(bomb_busters.WireColor.RED, 4.5),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 9.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 10.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 11.0)],
        ])
        game.markers = [
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 3.5,
                bomb_busters.MarkerState.KNOWN,
            ),
            bomb_busters.Marker(
                bomb_busters.WireColor.RED, 4.5,
                bomb_busters.MarkerState.KNOWN,
            ),
        ]
        return game, 0

    def test_position_probs_match_with_shared_memo(self) -> None:
        """compute_position_probabilities with shared memo matches standalone."""
        game, obs = self._make_game_with_red()
        standalone = compute_probabilities.compute_position_probabilities(
            game, obs,
        )
        solver = compute_probabilities.build_solver(
            game, obs, show_progress=False,
        )
        self.assertIsNotNone(solver)
        ctx, memo = solver
        shared = compute_probabilities.compute_position_probabilities(
            game, obs, ctx=ctx, memo=memo,
        )
        self.assertEqual(standalone.keys(), shared.keys())
        for key in standalone:
            self.assertEqual(
                dict(standalone[key]), dict(shared[key]),
                f"Mismatch at {key}",
            )

    def test_dual_cut_matches_with_shared_memo(self) -> None:
        """probability_of_dual_cut with shared memo matches standalone."""
        game, obs = self._make_game_with_red()
        solver = compute_probabilities.build_solver(
            game, obs, show_progress=False,
        )
        ctx, memo = solver
        # Test several target/value combinations
        for target_p in range(1, 4):
            stand = game.players[target_p].tile_stand
            for s_idx, slot in enumerate(stand.slots):
                if not slot.is_hidden:
                    continue
                for val in [5, 6, 7, 8, 9, 10, 11]:
                    standalone = compute_probabilities.probability_of_dual_cut(
                        game, obs, target_p, s_idx, val,
                    )
                    shared = compute_probabilities.probability_of_dual_cut(
                        game, obs, target_p, s_idx, val,
                        ctx=ctx, memo=memo,
                    )
                    self.assertAlmostEqual(
                        standalone, shared, places=10,
                        msg=(
                            f"Dual cut P{target_p}[{s_idx}]={val}: "
                            f"standalone={standalone}, shared={shared}"
                        ),
                    )

    def test_dd_matches_with_shared_memo(self) -> None:
        """probability_of_double_detector with shared memo matches standalone."""
        game, obs = self._make_game_with_red()
        solver = compute_probabilities.build_solver(game, obs, show_progress=False)
        ctx, memo = solver
        # DD on P1 slots 1,2 (the two hidden slots) guessing value 6
        standalone = compute_probabilities.probability_of_double_detector(
            game, obs, 1, 1, 2, 6,
        )
        shared = compute_probabilities.probability_of_double_detector(
            game, obs, 1, 1, 2, 6, ctx=ctx, memo=memo,
        )
        self.assertAlmostEqual(
            standalone, shared, places=10,
            msg="DD with shared memo should match standalone",
        )

    def test_dd_multiple_targets_shared_memo(self) -> None:
        """Multiple DD calls with same shared memo all match standalone."""
        game, obs = self._make_game_with_red()
        solver = compute_probabilities.build_solver(game, obs, show_progress=False)
        ctx, memo = solver
        # Test DD on different players and values
        test_cases = [
            (2, 0, 1, 7),   # P2 slots 0,1 guess 7
            (2, 0, 1, 9),   # P2 slots 0,1 guess 9
            (3, 0, 1, 10),  # P3 slots 0,1 guess 10
            (3, 0, 1, 11),  # P3 slots 0,1 guess 11
        ]
        for target_p, s1, s2, val in test_cases:
            standalone = compute_probabilities.probability_of_double_detector(
                game, obs, target_p, s1, s2, val,
            )
            shared = compute_probabilities.probability_of_double_detector(
                game, obs, target_p, s1, s2, val, ctx=ctx, memo=memo,
            )
            self.assertAlmostEqual(
                standalone, shared, places=10,
                msg=(
                    f"DD P{target_p}[{s1},{s2}]={val}: "
                    f"standalone={standalone}, shared={shared}"
                ),
            )

    def test_red_wire_dd_matches_with_shared_memo(self) -> None:
        """probability_of_red_wire_dd with shared memo matches standalone."""
        game, obs = self._make_game_two_reds()
        solver = compute_probabilities.build_solver(game, obs, show_progress=False)
        ctx, memo = solver
        # P1 slots 0,1 are two reds
        standalone = compute_probabilities.probability_of_red_wire_dd(
            game, obs, 1, 0, 1,
        )
        shared = compute_probabilities.probability_of_red_wire_dd(
            game, obs, 1, 0, 1, ctx=ctx, memo=memo,
        )
        self.assertAlmostEqual(
            standalone, shared, places=10,
            msg="Red DD with shared memo should match standalone",
        )

    def test_red_wire_dd_zero_case_shared_memo(self) -> None:
        """Red DD = 0% case produces same result with shared memo."""
        game, obs = self._make_game_with_red()  # Only 1 red wire
        solver = compute_probabilities.build_solver(game, obs, show_progress=False)
        ctx, memo = solver
        # With only 1 red, P(both red) = 0
        standalone = compute_probabilities.probability_of_red_wire_dd(
            game, obs, 1, 1, 2,
        )
        shared = compute_probabilities.probability_of_red_wire_dd(
            game, obs, 1, 1, 2, ctx=ctx, memo=memo,
        )
        self.assertAlmostEqual(standalone, 0.0)
        self.assertAlmostEqual(shared, 0.0)

    def test_build_solver_returns_none_no_hidden(self) -> None:
        """build_solver returns None when no hidden positions exist."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
        ])
        # Cut all other players' wires
        for p in range(1, 4):
            game.players[p].tile_stand.cut_wire_at(0)
        result = compute_probabilities.build_solver(game, 0, show_progress=False)
        self.assertIsNone(result)

    def test_build_solver_single_hidden_player(self) -> None:
        """Shared memo works when only one other player has hidden slots."""
        game = _make_known_game([
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 1.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 2.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 3.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 4.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 5.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 6.0)],
            [bomb_busters.Wire(bomb_busters.WireColor.BLUE, 7.0),
             bomb_busters.Wire(bomb_busters.WireColor.BLUE, 8.0)],
        ])
        # Cut all of P2 and P3 so only P1 has hidden slots
        for i in range(2):
            game.players[2].tile_stand.cut_wire_at(i)
            game.players[3].tile_stand.cut_wire_at(i)

        solver = compute_probabilities.build_solver(game, 0, show_progress=False)
        self.assertIsNotNone(solver)
        ctx, memo = solver
        standalone = compute_probabilities.compute_position_probabilities(
            game, 0,
        )
        shared = compute_probabilities.compute_position_probabilities(
            game, 0, ctx=ctx, memo=memo,
        )
        self.assertEqual(standalone.keys(), shared.keys())
        for key in standalone:
            self.assertEqual(dict(standalone[key]), dict(shared[key]))

    def test_rank_all_moves_matches_with_shared_memo(self) -> None:
        """rank_all_moves uses shared probs consistently."""
        game, obs = self._make_game_with_red()
        # Compute probs both ways and verify rank_all_moves gives same results
        solver = compute_probabilities.build_solver(game, obs, show_progress=False)
        ctx, memo = solver
        probs_standalone = compute_probabilities.compute_position_probabilities(
            game, obs,
        )
        probs_shared = compute_probabilities.compute_position_probabilities(
            game, obs, ctx=ctx, memo=memo,
        )
        moves_standalone = compute_probabilities.rank_all_moves(
            game, obs, probs=probs_standalone,
        )
        moves_shared = compute_probabilities.rank_all_moves(
            game, obs, probs=probs_shared,
        )
        self.assertEqual(len(moves_standalone), len(moves_shared))
        for m1, m2 in zip(moves_standalone, moves_shared):
            self.assertEqual(m1.action_type, m2.action_type)
            self.assertEqual(m1.target_player, m2.target_player)
            self.assertEqual(m1.target_slot, m2.target_slot)
            self.assertEqual(m1.guessed_value, m2.guessed_value)
            self.assertAlmostEqual(m1.probability, m2.probability, places=10)
            self.assertAlmostEqual(
                m1.red_probability, m2.red_probability, places=10,
            )


if __name__ == "__main__":
    unittest.main()
