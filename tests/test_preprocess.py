import unittest

import pandas as pd

from src.data.preprocess import (
    annotate_replies,
    clean_text,
    is_match_relevant,
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


if __name__ == "__main__":
    unittest.main()
