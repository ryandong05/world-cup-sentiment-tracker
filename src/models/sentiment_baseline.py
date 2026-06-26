from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_sentiment_model():
    """Load the baseline HuggingFace sentiment model."""
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def predict_sentiment(text: str, classifier=None) -> dict:
    """Predict sentiment label and confidence score for one text input."""
    if classifier is None:
        classifier = load_sentiment_model()

    result = classifier(text)[0]
    
    return {
        "sentiment_label": result["label"],
        "sentiment_score": result["score"]
    }


def add_sentiment(df, text_column: str = "clean_text", classifier=None):
    """Add baseline sentiment columns to a dataframe without mutating input."""
    if text_column not in df.columns:
        raise ValueError(f"add_sentiment requires a '{text_column}' column.")

    if classifier is None:
        classifier = load_sentiment_model()

    scored = df.copy()
    results = scored[text_column].apply(
        lambda text: predict_sentiment(text, classifier=classifier)
    )
    sentiment_df = scored_from_results(results)

    return scored.join(sentiment_df)


def scored_from_results(results):
    """Convert predict_sentiment dictionaries into dataframe columns."""
    import pandas as pd

    return pd.DataFrame(results.tolist(), index=results.index)
