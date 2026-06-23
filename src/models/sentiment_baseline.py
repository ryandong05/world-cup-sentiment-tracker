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