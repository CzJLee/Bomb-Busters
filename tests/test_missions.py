"""Unit tests for the missions module."""

import unittest

import missions


class TestIndicatorTokenType(unittest.TestCase):
    """Test IndicatorTokenType enum values."""

    def test_info_exists(self) -> None:
        self.assertIsNotNone(missions.IndicatorTokenType.INFO)

    def test_even_odd_exists(self) -> None:
        self.assertIsNotNone(missions.IndicatorTokenType.EVEN_ODD)

    def test_multiplicity_exists(self) -> None:
        self.assertIsNotNone(missions.IndicatorTokenType.MULTIPLICITY)

    def test_member_count(self) -> None:
        self.assertEqual(len(missions.IndicatorTokenType), 3)


class TestIndicationRule(unittest.TestCase):
    """Test IndicationRule enum values."""

    def test_standard_exists(self) -> None:
        self.assertIsNotNone(missions.IndicationRule.STANDARD)

    def test_random_token_exists(self) -> None:
        self.assertIsNotNone(missions.IndicationRule.RANDOM_TOKEN)

    def test_captain_fake_exists(self) -> None:
        self.assertIsNotNone(missions.IndicationRule.CAPTAIN_FAKE)

    def test_skip_exists(self) -> None:
        self.assertIsNotNone(missions.IndicationRule.SKIP)

    def test_negative_exists(self) -> None:
        self.assertIsNotNone(missions.IndicationRule.NEGATIVE)

    def test_member_count(self) -> None:
        self.assertEqual(len(missions.IndicationRule), 5)


class TestEquipmentCard(unittest.TestCase):
    """Test EquipmentCard creation and properties."""

    def test_standard_card_creation(self) -> None:
        card = missions.EquipmentCard(
            card_id="1", name="Label !=",
            description="Test description", unlock_value=1,
        )
        self.assertEqual(card.card_id, "1")
        self.assertEqual(card.name, "Label !=")
        self.assertEqual(card.unlock_value, 1)
        self.assertFalse(card.is_double)

    def test_double_card_creation(self) -> None:
        card = missions.EquipmentCard(
            card_id="9.9", name="Fast Pass Card",
            description="Test", unlock_value=9, is_double=True,
        )
        self.assertTrue(card.is_double)
        self.assertEqual(card.unlock_value, 9)

    def test_frozen_immutability(self) -> None:
        card = missions.EquipmentCard(
            card_id="1", name="Test", description="d", unlock_value=1,
        )
        with self.assertRaises(AttributeError):
            card.name = "Changed"  # type: ignore[misc]


class TestEquipmentCatalog(unittest.TestCase):
    """Test the EQUIPMENT_CATALOG and get_equipment()."""

    def test_catalog_has_18_entries(self) -> None:
        # 12 standard + 1 yellow + 5 double-numbered = 18
        self.assertEqual(len(missions.EQUIPMENT_CATALOG), 18)

    def test_standard_equipment_ids(self) -> None:
        for i in range(1, 13):
            self.assertIn(str(i), missions.EQUIPMENT_CATALOG)

    def test_yellow_equipment_exists(self) -> None:
        self.assertIn("Y", missions.EQUIPMENT_CATALOG)

    def test_double_equipment_ids(self) -> None:
        for eid in ("2.2", "3.3", "9.9", "10.10", "11.11"):
            self.assertIn(eid, missions.EQUIPMENT_CATALOG)

    def test_double_equipment_is_double_flag(self) -> None:
        for eid in ("2.2", "3.3", "9.9", "10.10", "11.11"):
            card = missions.EQUIPMENT_CATALOG[eid]
            self.assertTrue(
                card.is_double, f"Expected is_double=True for {eid}",
            )

    def test_standard_equipment_not_double(self) -> None:
        for i in range(1, 13):
            card = missions.EQUIPMENT_CATALOG[str(i)]
            self.assertFalse(
                card.is_double, f"Expected is_double=False for {i}",
            )

    def test_yellow_equipment_not_double(self) -> None:
        self.assertFalse(missions.EQUIPMENT_CATALOG["Y"].is_double)

    def test_unlock_values(self) -> None:
        for i in range(1, 13):
            card = missions.EQUIPMENT_CATALOG[str(i)]
            self.assertEqual(card.unlock_value, i)

    def test_yellow_unlock_value(self) -> None:
        self.assertEqual(missions.EQUIPMENT_CATALOG["Y"].unlock_value, 0)

    def test_card_id_matches_key(self) -> None:
        for key, card in missions.EQUIPMENT_CATALOG.items():
            self.assertEqual(card.card_id, key)

    def test_get_equipment_valid(self) -> None:
        card = missions.get_equipment("5")
        self.assertEqual(card.name, "Super Detector")

    def test_get_equipment_yellow(self) -> None:
        card = missions.get_equipment("Y")
        self.assertEqual(card.name, "False Bottom")

    def test_get_equipment_double(self) -> None:
        card = missions.get_equipment("9.9")
        self.assertEqual(card.name, "Fast Pass Card")

    def test_get_equipment_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            missions.get_equipment("99")

    def test_get_equipment_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            missions.get_equipment("")

    def test_all_cards_have_names(self) -> None:
        for key, card in missions.EQUIPMENT_CATALOG.items():
            self.assertTrue(
                card.name, f"Equipment {key} has no name",
            )

    def test_all_cards_have_descriptions(self) -> None:
        for key, card in missions.EQUIPMENT_CATALOG.items():
            self.assertTrue(
                card.description, f"Equipment {key} has no description",
            )


class TestMissionDefaults(unittest.TestCase):
    """Test Mission default field values."""

    def setUp(self) -> None:
        self.mission = missions.Mission(number=99, name="Test Mission")

    def test_default_blue_wire_range(self) -> None:
        self.assertEqual(self.mission.blue_wire_range, (1, 12))

    def test_default_yellow_wires(self) -> None:
        self.assertIsNone(self.mission.yellow_wires)

    def test_default_red_wires(self) -> None:
        self.assertIsNone(self.mission.red_wires)

    def test_default_equipment_forbidden(self) -> None:
        self.assertEqual(self.mission.equipment_forbidden, ())

    def test_default_equipment_override(self) -> None:
        self.assertIsNone(self.mission.equipment_override)

    def test_default_double_detector_disabled(self) -> None:
        self.assertFalse(self.mission.double_detector_disabled)

    def test_default_captain_double_detector_disabled(self) -> None:
        self.assertFalse(self.mission.captain_double_detector_disabled)

    def test_default_indicator_type(self) -> None:
        self.assertEqual(
            self.mission.indicator_type,
            missions.IndicatorTokenType.INFO,
        )

    def test_default_indication_rule(self) -> None:
        self.assertEqual(
            self.mission.indication_rule,
            missions.IndicationRule.STANDARD,
        )

    def test_default_x_tokens(self) -> None:
        self.assertFalse(self.mission.x_tokens)

    def test_default_number_cards(self) -> None:
        self.assertFalse(self.mission.number_cards)

    def test_default_sequence_card(self) -> None:
        self.assertFalse(self.mission.sequence_card)

    def test_default_timer_minutes(self) -> None:
        self.assertIsNone(self.mission.timer_minutes)

    def test_default_red_wires_dealt_separately(self) -> None:
        self.assertFalse(self.mission.red_wires_dealt_separately)

    def test_default_constraint_cards(self) -> None:
        self.assertFalse(self.mission.constraint_cards)

    def test_default_challenge_cards(self) -> None:
        self.assertFalse(self.mission.challenge_cards)

    def test_default_notes(self) -> None:
        self.assertEqual(self.mission.notes, "")

    def test_frozen_immutability(self) -> None:
        with self.assertRaises(AttributeError):
            self.mission.name = "Changed"  # type: ignore[misc]

    def test_str_representation(self) -> None:
        self.assertEqual(str(self.mission), "Mission 99: Test Mission")


class TestMissionValidation(unittest.TestCase):
    """Test Mission __post_init__ validation."""

    def test_invalid_mission_number_zero(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=0, name="Bad")

    def test_invalid_mission_number_negative(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=-1, name="Bad")

    def test_invalid_blue_range_low_too_low(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", blue_wire_range=(0, 12))

    def test_invalid_blue_range_high_too_high(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", blue_wire_range=(1, 13))

    def test_invalid_blue_range_inverted(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", blue_wire_range=(8, 3))

    def test_valid_blue_range_single_value(self) -> None:
        m = missions.Mission(number=1, name="OK", blue_wire_range=(5, 5))
        self.assertEqual(m.blue_wire_range, (5, 5))

    def test_invalid_yellow_count_too_high(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", yellow_wires=5)

    def test_invalid_yellow_count_zero(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", yellow_wires=0)

    def test_invalid_yellow_tuple_count_too_high(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", yellow_wires=(5, 6))

    def test_invalid_yellow_tuple_pool_less_than_count(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", yellow_wires=(3, 2))

    def test_invalid_yellow_tuple_pool_too_large(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", yellow_wires=(2, 12))

    def test_invalid_red_count_too_high(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", red_wires=4)

    def test_invalid_red_count_zero(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", red_wires=0)

    def test_invalid_red_tuple_count_too_high(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", red_wires=(4, 5))

    def test_invalid_red_tuple_pool_less_than_count(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", red_wires=(2, 1))

    def test_invalid_red_tuple_pool_too_large(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(number=1, name="Bad", red_wires=(1, 12))

    def test_equipment_forbidden_and_override_mutual_exclusion(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(
                number=10, name="Bad",
                equipment_forbidden=("1",),
                equipment_override=("2",),
            )

    def test_invalid_equipment_forbidden_id(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(
                number=10, name="Bad",
                equipment_forbidden=("99",),
            )

    def test_invalid_equipment_override_id(self) -> None:
        with self.assertRaises(ValueError):
            missions.Mission(
                number=10, name="Bad",
                equipment_override=("nonexistent",),
            )

    def test_valid_equipment_forbidden(self) -> None:
        m = missions.Mission(
            number=10, name="OK",
            equipment_forbidden=("1", "11"),
        )
        self.assertEqual(m.equipment_forbidden, ("1", "11"))

    def test_valid_equipment_override(self) -> None:
        m = missions.Mission(
            number=10, name="OK",
            equipment_override=("8",),
        )
        self.assertEqual(m.equipment_override, ("8",))


class TestMissionRegistry(unittest.TestCase):
    """Test get_mission() and all_missions() registry functions."""

    def test_get_mission_valid(self) -> None:
        m = missions.get_mission(1)
        self.assertEqual(m.number, 1)
        self.assertEqual(m.name, "TRAINING, Day 1")

    def test_get_mission_last(self) -> None:
        m = missions.get_mission(30)
        self.assertEqual(m.number, 30)

    def test_get_mission_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            missions.get_mission(999)

    def test_get_mission_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            missions.get_mission(0)

    def test_all_missions_returns_list(self) -> None:
        result = missions.all_missions()
        self.assertIsInstance(result, list)

    def test_all_missions_count(self) -> None:
        result = missions.all_missions()
        self.assertEqual(len(result), 30)

    def test_all_missions_sorted_by_number(self) -> None:
        result = missions.all_missions()
        numbers = [m.number for m in result]
        self.assertEqual(numbers, sorted(numbers))

    def test_all_missions_contains_all_defined(self) -> None:
        result = missions.all_missions()
        numbers = {m.number for m in result}
        self.assertEqual(numbers, set(range(1, 31)))

    def test_registry_key_matches_mission_number(self) -> None:
        for key, mission in missions.MISSIONS.items():
            self.assertEqual(
                key, mission.number,
                f"Registry key {key} != mission.number {mission.number}",
            )


class TestMissionDefinitions(unittest.TestCase):
    """Test specific mission definition properties."""

    def test_all_missions_1_to_30_defined(self) -> None:
        for i in range(1, 31):
            self.assertIn(i, missions.MISSIONS, f"Mission {i} not defined")

    def test_mission_1_blue_range(self) -> None:
        m = missions.get_mission(1)
        self.assertEqual(m.blue_wire_range, (1, 6))
        self.assertIsNone(m.yellow_wires)
        self.assertIsNone(m.red_wires)

    def test_mission_2_yellow_wires(self) -> None:
        m = missions.get_mission(2)
        self.assertEqual(m.blue_wire_range, (1, 8))
        self.assertEqual(m.yellow_wires, 2)

    def test_mission_3_red_wires(self) -> None:
        m = missions.get_mission(3)
        self.assertEqual(m.blue_wire_range, (1, 10))
        self.assertEqual(m.red_wires, 1)

    def test_mission_4_full_blue_range(self) -> None:
        m = missions.get_mission(4)
        self.assertEqual(m.blue_wire_range, (1, 12))
        self.assertEqual(m.red_wires, 1)
        self.assertEqual(m.yellow_wires, 2)

    def test_mission_5_uncertain_yellow(self) -> None:
        m = missions.get_mission(5)
        self.assertEqual(m.yellow_wires, (2, 3))

    def test_mission_7_uncertain_red(self) -> None:
        m = missions.get_mission(7)
        self.assertEqual(m.red_wires, (1, 2))

    def test_mission_8_both_uncertain(self) -> None:
        m = missions.get_mission(8)
        self.assertEqual(m.red_wires, (1, 2))
        self.assertEqual(m.yellow_wires, (2, 3))

    def test_mission_9_number_and_sequence_cards(self) -> None:
        m = missions.get_mission(9)
        self.assertTrue(m.number_cards)
        self.assertTrue(m.sequence_card)

    def test_mission_10_timer(self) -> None:
        m = missions.get_mission(10)
        self.assertEqual(m.timer_minutes, 15)
        self.assertEqual(m.equipment_forbidden, ("11",))

    def test_mission_13_indication_rule(self) -> None:
        m = missions.get_mission(13)
        self.assertEqual(
            m.indication_rule, missions.IndicationRule.RANDOM_TOKEN,
        )
        self.assertTrue(m.red_wires_dealt_separately)
        self.assertEqual(m.red_wires, 3)

    def test_mission_17_captain_fake(self) -> None:
        m = missions.get_mission(17)
        self.assertEqual(
            m.indication_rule, missions.IndicationRule.CAPTAIN_FAKE,
        )

    def test_mission_18_skip_indication(self) -> None:
        m = missions.get_mission(18)
        self.assertEqual(
            m.indication_rule, missions.IndicationRule.SKIP,
        )
        self.assertEqual(m.equipment_override, ("8",))

    def test_mission_20_x_tokens(self) -> None:
        m = missions.get_mission(20)
        self.assertTrue(m.x_tokens)
        self.assertEqual(m.equipment_forbidden, ("2",))

    def test_mission_21_even_odd_tokens(self) -> None:
        m = missions.get_mission(21)
        self.assertEqual(
            m.indicator_type, missions.IndicatorTokenType.EVEN_ODD,
        )

    def test_mission_22_negative_indication(self) -> None:
        m = missions.get_mission(22)
        self.assertEqual(
            m.indication_rule, missions.IndicationRule.NEGATIVE,
        )

    def test_mission_24_multiplicity_tokens(self) -> None:
        m = missions.get_mission(24)
        self.assertEqual(
            m.indicator_type, missions.IndicatorTokenType.MULTIPLICITY,
        )

    def test_mission_27_double_detector_disabled(self) -> None:
        m = missions.get_mission(27)
        self.assertTrue(m.double_detector_disabled)
        self.assertEqual(m.equipment_forbidden, ("7",))

    def test_mission_28_captain_dd_disabled(self) -> None:
        m = missions.get_mission(28)
        self.assertTrue(m.captain_double_detector_disabled)
        self.assertFalse(m.double_detector_disabled)

    def test_all_missions_have_names(self) -> None:
        for m in missions.all_missions():
            self.assertTrue(
                m.name, f"Mission {m.number} has no name",
            )

    def test_all_missions_have_unique_numbers(self) -> None:
        numbers = [m.number for m in missions.all_missions()]
        self.assertEqual(len(numbers), len(set(numbers)))

    def test_all_missions_have_unique_names(self) -> None:
        names = [m.name for m in missions.all_missions()]
        self.assertEqual(len(names), len(set(names)))

    def test_training_missions_have_restricted_blue(self) -> None:
        """Missions 1-3 have restricted blue ranges; 4+ use full."""
        self.assertEqual(missions.get_mission(1).blue_wire_range, (1, 6))
        self.assertEqual(missions.get_mission(2).blue_wire_range, (1, 8))
        self.assertEqual(missions.get_mission(3).blue_wire_range, (1, 10))
        for i in range(4, 31):
            m = missions.get_mission(i)
            self.assertEqual(
                m.blue_wire_range, (1, 12),
                f"Mission {i} blue_wire_range is {m.blue_wire_range}",
            )


class TestAvailableEquipment(unittest.TestCase):
    """Test Mission.available_equipment() default logic."""

    def test_mission_1_no_equipment(self) -> None:
        m = missions.get_mission(1)
        self.assertEqual(m.available_equipment(), [])

    def test_mission_2_no_equipment(self) -> None:
        m = missions.get_mission(2)
        self.assertEqual(m.available_equipment(), [])

    def test_mission_3_equipment_1_to_10(self) -> None:
        m = missions.get_mission(3)
        expected = [str(i) for i in range(1, 11)]
        self.assertEqual(m.available_equipment(), expected)

    def test_mission_4_equipment_1_to_12(self) -> None:
        m = missions.get_mission(4)
        expected = [str(i) for i in range(1, 13)]
        # Mission 4 has yellow wires but number < 9, so no yellow equipment
        self.assertEqual(m.available_equipment(), expected)

    def test_mission_9_includes_yellow_equipment(self) -> None:
        m = missions.get_mission(9)
        equip = m.available_equipment()
        self.assertIn("Y", equip)
        # Standard 1-12 + yellow
        expected = [str(i) for i in range(1, 13)] + ["Y"]
        self.assertEqual(equip, expected)

    def test_mission_without_yellow_wires_no_yellow_equipment(self) -> None:
        """Mission 25 has red but no yellow wires; no yellow equipment."""
        m = missions.get_mission(25)
        self.assertIsNone(m.yellow_wires)
        equip = m.available_equipment()
        self.assertNotIn("Y", equip)

    def test_forbidden_removes_equipment(self) -> None:
        m = missions.get_mission(10)
        equip = m.available_equipment()
        self.assertNotIn("11", equip)
        # Other standard equipment should be present
        self.assertIn("1", equip)
        self.assertIn("12", equip)

    def test_override_replaces_default(self) -> None:
        m = missions.get_mission(18)
        self.assertEqual(m.available_equipment(), ["8"])

    def test_custom_mission_no_equipment_before_3(self) -> None:
        m = missions.Mission(number=2, name="Test")
        self.assertEqual(m.available_equipment(), [])

    def test_custom_mission_3_has_1_to_10(self) -> None:
        m = missions.Mission(number=3, name="Test")
        self.assertEqual(
            m.available_equipment(),
            [str(i) for i in range(1, 11)],
        )

    def test_custom_mission_4_has_1_to_12(self) -> None:
        m = missions.Mission(number=4, name="Test")
        self.assertEqual(
            m.available_equipment(),
            [str(i) for i in range(1, 13)],
        )

    def test_custom_mission_9_with_yellow_includes_y(self) -> None:
        m = missions.Mission(number=9, name="Test", yellow_wires=2)
        equip = m.available_equipment()
        self.assertIn("Y", equip)

    def test_custom_mission_9_without_yellow_no_y(self) -> None:
        m = missions.Mission(number=9, name="Test")
        equip = m.available_equipment()
        self.assertNotIn("Y", equip)

    def test_custom_mission_55_includes_double(self) -> None:
        m = missions.Mission(number=55, name="Test", yellow_wires=2)
        equip = m.available_equipment()
        for eid in ("2.2", "3.3", "9.9", "10.10", "11.11"):
            self.assertIn(eid, equip, f"Expected {eid} in equipment")
        self.assertIn("Y", equip)

    def test_custom_mission_54_no_double(self) -> None:
        m = missions.Mission(number=54, name="Test")
        equip = m.available_equipment()
        for eid in ("2.2", "3.3", "9.9", "10.10", "11.11"):
            self.assertNotIn(eid, equip, f"Unexpected {eid} in equipment")

    def test_override_ignores_number_rules(self) -> None:
        """Override with equipment_override should ignore defaults."""
        m = missions.Mission(
            number=1, name="Test",
            equipment_override=("8", "9"),
        )
        self.assertEqual(m.available_equipment(), ["8", "9"])

    def test_forbidden_with_yellow_equipment(self) -> None:
        m = missions.Mission(
            number=10, name="Test", yellow_wires=2,
            equipment_forbidden=("Y",),
        )
        equip = m.available_equipment()
        self.assertNotIn("Y", equip)
        self.assertIn("1", equip)


class TestWireConfig(unittest.TestCase):
    """Test Mission.wire_config() extraction."""

    def test_default_returns_empty(self) -> None:
        m = missions.Mission(number=99, name="Test")
        self.assertEqual(m.wire_config(), {})

    def test_restricted_blue_range(self) -> None:
        m = missions.get_mission(1)
        config = m.wire_config()
        self.assertEqual(config["blue_wires"], (1, 6))

    def test_yellow_wires_direct(self) -> None:
        m = missions.get_mission(4)
        config = m.wire_config()
        self.assertEqual(config["yellow_wires"], 2)
        self.assertEqual(config["red_wires"], 1)

    def test_yellow_wires_uncertain(self) -> None:
        m = missions.get_mission(5)
        config = m.wire_config()
        self.assertEqual(config["yellow_wires"], (2, 3))

    def test_red_wires_uncertain(self) -> None:
        m = missions.get_mission(7)
        config = m.wire_config()
        self.assertEqual(config["red_wires"], (1, 2))

    def test_full_blue_range_omitted(self) -> None:
        """Full blue 1-12 range should not appear in wire_config."""
        m = missions.get_mission(4)
        config = m.wire_config()
        self.assertNotIn("blue_wires", config)

    def test_no_colored_wires_omitted(self) -> None:
        m = missions.Mission(number=99, name="Test")
        config = m.wire_config()
        self.assertNotIn("yellow_wires", config)
        self.assertNotIn("red_wires", config)

    def test_mission_8_all_wires(self) -> None:
        """Mission 8 has both uncertain red and yellow."""
        m = missions.get_mission(8)
        config = m.wire_config()
        self.assertEqual(config["red_wires"], (1, 2))
        self.assertEqual(config["yellow_wires"], (2, 3))
        self.assertNotIn("blue_wires", config)

    def test_mission_1_has_blue_and_nothing_else(self) -> None:
        m = missions.get_mission(1)
        config = m.wire_config()
        self.assertEqual(config, {"blue_wires": (1, 6)})


if __name__ == "__main__":
    unittest.main()
