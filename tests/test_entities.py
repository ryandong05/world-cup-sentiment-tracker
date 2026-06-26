import unittest

from src.data.entities import infer_entities, load_entity_reference


class EntityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_entity_reference()

    def test_ronaldo_maps_to_portugal_and_cristiano_ronaldo(self):
        entities = infer_entities("Ronaldo scored again", self.reference)

        self.assertIn("Portugal", entities["inferred_teams"])
        self.assertIn("Cristiano Ronaldo", entities["inferred_players"])
        self.assertIn("ronaldo", entities["matched_entities"])

    def test_cr7_maps_to_portugal_and_cristiano_ronaldo(self):
        entities = infer_entities("CR7 is inevitable", self.reference)

        self.assertIn("Portugal", entities["inferred_teams"])
        self.assertIn("Cristiano Ronaldo", entities["inferred_players"])
        self.assertIn("cr7", entities["matched_entities"])

    def test_elongated_siu_maps_to_portugal_and_cristiano_ronaldo(self):
        entities = infer_entities("SIUUUUUUUUUU", self.reference)

        self.assertIn("Portugal", entities["inferred_teams"])
        self.assertIn("Cristiano Ronaldo", entities["inferred_players"])
        self.assertIn("siuuu", entities["matched_entities"])

    def test_uzbekistan_maps_to_uzbekistan(self):
        entities = infer_entities("Relax, it is just against Uzbekistan", self.reference)

        self.assertEqual(entities["inferred_teams"], ["Uzbekistan"])
        self.assertEqual(entities["inferred_players"], [])

    def test_messi_and_mbappe_can_infer_multiple_players(self):
        entities = infer_entities("Messi and Mbappe right now", self.reference)

        self.assertIn("Argentina", entities["inferred_teams"])
        self.assertIn("France", entities["inferred_teams"])
        self.assertIn("Lionel Messi", entities["inferred_players"])
        self.assertIn("Kylian Mbappé", entities["inferred_players"])

    def test_manager_can_be_inferred_with_team_context(self):
        entities = infer_entities(
            "Martinez got the formation wrong",
            self.reference,
            context_teams=["Portugal"],
        )

        self.assertIn("Portugal", entities["inferred_teams"])
        self.assertIn("Roberto Martínez", entities["inferred_managers"])

    def test_unrelated_ad_text_has_no_entity_matches(self):
        entities = infer_entities("Get iPhone 17 by switching to T-Mobile", self.reference)

        self.assertEqual(entities["matched_entities"], [])
        self.assertEqual(entities["inferred_teams"], [])
        self.assertEqual(entities["inferred_players"], [])
        self.assertEqual(entities["inferred_managers"], [])
        self.assertEqual(entities["entity_confidence"], 0)


if __name__ == "__main__":
    unittest.main()
