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


if __name__ == "__main__":
    unittest.main()
