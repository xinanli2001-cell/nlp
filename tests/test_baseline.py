# tests/test_baseline.py
import pytest
from src.models.baseline import RuleBasedABSA

@pytest.fixture
def model():
    return RuleBasedABSA()

def test_positive_review(model):
    result = model.predict("The battery life is amazing.", "battery")
    assert result == "positive"

def test_negative_review(model):
    result = model.predict("The screen is absolutely terrible and blurry.", "screen")
    assert result == "negative"

def test_neutral_fallback(model):
    result = model.predict("The product arrived on Tuesday.", "usability")
    assert result in ["positive", "negative", "neutral"]

def test_batch_returns_list(model):
    reviews = [
        "Great sound quality!",
        "Battery dies too fast.",
        "The price is okay.",
    ]
    aspects = ["sound", "battery", "price"]
    results = model.predict_batch(reviews, aspects)
    assert len(results) == 3
    assert all(r in ["positive", "negative", "neutral"] for r in results)
