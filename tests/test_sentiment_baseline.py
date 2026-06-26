import unittest

import pandas as pd

from src.models.sentiment_baseline import add_sentiment


class SentimentBaselineTests(unittest.TestCase):
    def test_add_sentiment_uses_clean_text_and_preserves_input(self):
        df = pd.DataFrame({"clean_text": ["Great goal", "Terrible miss"]})

        def classifier(text):
            label = "POSITIVE" if "Great" in text else "NEGATIVE"
            return [{"label": label, "score": 0.9}]

        scored = add_sentiment(df, classifier=classifier)

        self.assertNotIn("sentiment_label", df.columns)
        self.assertEqual(scored["sentiment_label"].tolist(), ["POSITIVE", "NEGATIVE"])
        self.assertEqual(scored["sentiment_score"].tolist(), [0.9, 0.9])


if __name__ == "__main__":
    unittest.main()
