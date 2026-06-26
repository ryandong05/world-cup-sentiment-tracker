import unittest

import pandas as pd

from src.data.match_linking import (
    find_candidate_matches,
    link_replies_to_matches,
    link_reply_to_match,
    load_match_schedule,
)


class MatchLinkingTests(unittest.TestCase):
    def test_load_match_schedule_parses_reference_file(self):
        schedule_df = load_match_schedule()

        self.assertEqual(len(schedule_df), 104)
        self.assertEqual(schedule_df["match_id"].nunique(), 104)
        self.assertIn("match_id", schedule_df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(schedule_df["kickoff_utc"]))

    def test_ronaldo_portugal_reply_links_to_portugal_uzbekistan(self):
        schedule_df = _schedule(
            [
                ("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z"),
                ("2026_POR_ESP", "Portugal", "Spain", "2026-06-26T19:00:00Z"),
            ]
        )
        row = pd.Series(
            {
                "timestamp": "2026-06-23T20:00:00Z",
                "inferred_teams": ["Portugal"],
                "inferred_players": ["Cristiano Ronaldo"],
                "match": "multiple",
            }
        )

        result = link_reply_to_match(row, schedule_df)

        self.assertEqual(result["linked_match_id"], "2026_POR_UZB")
        self.assertEqual(result["linked_match"], "Portugal vs Uzbekistan")
        self.assertIn(result["linked_match_confidence"], {"high", "medium"})

    def test_uzbekistan_only_reply_links_to_portugal_uzbekistan(self):
        schedule_df = _schedule(
            [
                ("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z"),
                ("2026_ARG_FRA", "Argentina", "France", "2026-06-23T22:00:00Z"),
            ]
        )
        row = pd.Series(
            {
                "timestamp": "2026-06-23T22:00:00Z",
                "inferred_teams": ["Uzbekistan"],
                "match": "multiple",
            }
        )

        result = link_reply_to_match(row, schedule_df)

        self.assertEqual(result["linked_match_id"], "2026_POR_UZB")

    def test_messi_and_mbappe_reply_can_be_ambiguous(self):
        schedule_df = _schedule(
            [
                ("2026_ARG_GER", "Argentina", "Germany", "2026-06-23T22:00:00Z"),
                ("2026_FRA_CAN", "France", "Canada", "2026-06-23T22:00:00Z"),
            ]
        )
        row = pd.Series(
            {
                "timestamp": "2026-06-23T22:30:00Z",
                "inferred_teams": ["Argentina", "France"],
                "inferred_players": ["Lionel Messi", "Kylian Mbappe"],
                "match": "multiple",
            }
        )

        result = link_reply_to_match(row, schedule_df)

        self.assertIsNone(result["linked_match_id"])
        self.assertEqual(result["linked_match_confidence"], "ambiguous")
        self.assertEqual(len(result["match_candidates"]), 2)

    def test_unrelated_text_returns_no_linked_match(self):
        schedule_df = _schedule(
            [("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z")]
        )
        row = pd.Series({"timestamp": "2026-06-23T20:00:00Z", "inferred_teams": []})

        result = link_reply_to_match(row, schedule_df)

        self.assertIsNone(result["linked_match_id"])
        self.assertEqual(result["linked_match_confidence"], "none")
        self.assertEqual(result["linked_match_method"], "no_team_context")

    def test_two_team_matches_choose_nearest_only_when_clearly_closer(self):
        schedule_df = _schedule(
            [
                ("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z"),
                ("2026_POR_ESP", "Portugal", "Spain", "2026-06-24T10:00:00Z"),
            ]
        )
        row = pd.Series(
            {
                "timestamp": "2026-06-23T20:00:00Z",
                "inferred_teams": ["Portugal"],
                "match": "multiple",
            }
        )

        result = link_reply_to_match(row, schedule_df)

        self.assertEqual(result["linked_match_id"], "2026_POR_UZB")
        self.assertEqual(result["linked_match_confidence"], "medium")

    def test_close_team_matches_are_ambiguous(self):
        schedule_df = _schedule(
            [
                ("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z"),
                ("2026_POR_ESP", "Portugal", "Spain", "2026-06-23T21:00:00Z"),
            ]
        )
        row = pd.Series(
            {
                "timestamp": "2026-06-23T20:00:00Z",
                "inferred_teams": ["Portugal"],
                "match": "multiple",
            }
        )

        result = link_reply_to_match(row, schedule_df)

        self.assertIsNone(result["linked_match_id"])
        self.assertEqual(result["linked_match_confidence"], "ambiguous")

    def test_link_replies_to_matches_adds_link_columns(self):
        schedule_df = _schedule(
            [("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z")]
        )
        df = pd.DataFrame(
            [
                {
                    "timestamp": "2026-06-23T20:00:00Z",
                    "inferred_teams": ["Portugal"],
                    "match": "multiple",
                }
            ]
        )

        linked = link_replies_to_matches(df, schedule_df)

        self.assertEqual(linked.loc[0, "linked_match_id"], "2026_POR_UZB")
        self.assertIn("match_candidates", linked.columns)

    def test_find_candidate_matches_filters_by_time_window(self):
        schedule_df = _schedule(
            [
                ("2026_POR_UZB", "Portugal", "Uzbekistan", "2026-06-23T19:00:00Z"),
                ("2026_POR_ESP", "Portugal", "Spain", "2026-06-30T19:00:00Z"),
            ]
        )

        candidates = find_candidate_matches(
            teams=["Portugal"],
            timestamp="2026-06-23T20:00:00Z",
            schedule_df=schedule_df,
            window_hours=48,
        )

        self.assertEqual(candidates["match_id"].tolist(), ["2026_POR_UZB"])


def _schedule(rows):
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "tournament": "World Cup",
                "stage": "Group Stage",
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_utc": pd.Timestamp(kickoff_utc),
            }
            for match_id, home_team, away_team, kickoff_utc in rows
        ]
    )


if __name__ == "__main__":
    unittest.main()
