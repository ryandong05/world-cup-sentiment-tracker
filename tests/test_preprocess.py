import unittest

import pandas as pd

from src.data.preprocess import (
    annotate_replies,
    clean_text,
    is_match_relevant,
    is_spam_text,
    is_usable_text,
    preprocess_replies,
    preprocessing_summary,
    relevance_score,
)


class PreprocessTests(unittest.TestCase):
    def test_clean_text_removes_urls_mentions_html_and_extra_space(self):
        self.assertEqual(
            clean_text("  <b>@fan</b> What a goal!!! https://t.co/test  "),
            "What a goal!!!",
        )

    def test_is_usable_text_filters_media_only_and_short_replies(self):
        self.assertFalse(is_usable_text("🔥🔥🔥"))
        self.assertFalse(is_usable_text("https://t.co/test"))
        self.assertFalse(is_usable_text("@fan"))
        self.assertFalse(is_usable_text("GIF"))
        self.assertTrue(is_usable_text("What a finish"))

    def test_relevance_score_uses_match_context_without_notebook_keywords(self):
        score = relevance_score(
            "Ronaldo with a ridiculous finish",
            match="Portugal vs Spain",
            team="Portugal",
            player="Ronaldo",
            event="goal",
        )

        self.assertGreaterEqual(score, 3)
        self.assertTrue(
            is_match_relevant(
                "Ronaldo with a ridiculous finish",
                match="Portugal vs Spain",
                team="Portugal",
                player="Ronaldo",
                event="goal",
            )
        )

    def test_relevance_score_penalizes_spam(self):
        self.assertLess(
            relevance_score("Follow me for crypto giveaway", match="Portugal vs Spain"),
            1,
        )

    def test_is_spam_text_detects_platform_and_ad_noise(self):
        self.assertTrue(is_spam_text("Ask Grok is currently available on X"))
        self.assertTrue(is_spam_text("Get iPhone 17 by switching to T-Mobile"))
        self.assertFalse(is_spam_text("No one can achieve what this man achieved"))

    def test_preprocess_replies_filters_and_keeps_audit_columns(self):
        df = pd.DataFrame(
            [
                {
                    "text": "Ronaldo with a ridiculous finish",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Ronaldo",
                    "event": "goal",
                },
                {
                    "text": "https://t.co/test",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Ronaldo",
                    "event": "goal",
                },
                {
                    "text": "Follow me for crypto giveaway",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Ronaldo",
                    "event": "goal",
                },
            ]
        )

        analysis_df = preprocess_replies(df)

        self.assertEqual(len(analysis_df), 1)
        self.assertEqual(analysis_df.loc[0, "clean_text"], "Ronaldo with a ridiculous finish")
        self.assertEqual(analysis_df.loc[0, "filter_reason"], "keep")

    def test_annotate_replies_keeps_filtered_rows_for_audit(self):
        df = pd.DataFrame(
            [
                {"text": "What a goal", "match": "Portugal vs Spain", "event": "goal"},
                {"text": "GIF", "match": "Portugal vs Spain", "event": "goal"},
            ]
        )

        annotated = annotate_replies(df)

        self.assertEqual(len(annotated), 2)
        self.assertEqual(annotated["filter_reason"].tolist(), ["keep", "unusable_text"])

    def test_preprocessing_summary_computes_filter_counts(self):
        df = pd.DataFrame(
            [
                {"text": "What a goal", "match": "Portugal vs Spain", "event": "goal"},
                {"text": "GIF", "match": "Portugal vs Spain", "event": "goal"},
            ]
        )
        analysis_df = preprocess_replies(df)

        summary = preprocessing_summary(df, analysis_df)

        self.assertEqual(summary["raw_rows"], 2)
        self.assertEqual(summary["analysis_rows"], 1)
        self.assertEqual(summary["removed_rows"], 1)
        self.assertEqual(summary["filter_reasons"], {"keep": 1, "unusable_text": 1})

    def test_entity_only_reply_is_kept_and_enriched(self):
        df = pd.DataFrame(
            [
                {
                    "text": "Ronaldo",
                    "match": "multiple",
                    "team": None,
                    "player": None,
                    "event": None,
                }
            ]
        )

        analysis_df = preprocess_replies(df)

        self.assertEqual(len(analysis_df), 1)
        self.assertEqual(analysis_df.loc[0, "inferred_teams"], ["Portugal"])
        self.assertEqual(analysis_df.loc[0, "inferred_players"], ["Cristiano Ronaldo"])
        self.assertGreaterEqual(analysis_df.loc[0, "entity_confidence"], 1)

    def test_manager_context_and_tactical_terms_are_relevant(self):
        df = pd.DataFrame(
            [
                {
                    "text": "Martinez got the formation wrong",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": None,
                    "event": None,
                }
            ]
        )

        analysis_df = preprocess_replies(df)

        self.assertEqual(len(analysis_df), 1)
        self.assertEqual(analysis_df.loc[0, "inferred_managers"], ["Roberto Martínez"])
        self.assertIn("formation", analysis_df.loc[0, "matched_entities"])


    def test_vague_pronoun_reply_is_kept_with_parent_player_context(self):
        df = pd.DataFrame(
            [
                {
                    "text": "No one can ever achieve what this man has achieved",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Cristiano Ronaldo",
                    "event": "record",
                }
            ]
        )

        analysis_df = preprocess_replies(df)

        self.assertEqual(len(analysis_df), 1)
        self.assertGreaterEqual(analysis_df.loc[0, "parent_context_confidence"], 1)
        self.assertGreaterEqual(analysis_df.loc[0, "context_relevance_boost"], 1)
        self.assertTrue(analysis_df.loc[0, "needs_context_review"])

    def test_record_reply_is_kept_with_parent_context(self):
        df = pd.DataFrame(
            [
                {
                    "text": "Another record set!",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Cristiano Ronaldo",
                    "event": "record",
                }
            ]
        )

        analysis_df = preprocess_replies(df)

        self.assertEqual(len(analysis_df), 1)
        self.assertIn("record", analysis_df.loc[0, "contextual_terms"])

    def test_media_only_reply_stays_unusable_even_with_parent_context(self):
        df = pd.DataFrame(
            [
                {
                    "text": "@FIFAWorldCup https://x.com/i/status/123",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Cristiano Ronaldo",
                    "event": "record",
                }
            ]
        )

        annotated = annotate_replies(df)

        self.assertEqual(annotated.loc[0, "filter_reason"], "unusable_text")

    def test_spam_reply_stays_removed_even_with_parent_context(self):
        df = pd.DataFrame(
            [
                {
                    "text": "Ask Grok is currently available on X",
                    "match": "Portugal vs Spain",
                    "team": "Portugal",
                    "player": "Cristiano Ronaldo",
                    "event": "record",
                }
            ]
        )

        annotated = annotate_replies(df)

        self.assertEqual(annotated.loc[0, "filter_reason"], "spam")

if __name__ == "__main__":
    unittest.main()
