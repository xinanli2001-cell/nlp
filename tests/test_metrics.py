import pytest
from src.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    golds = ["positive", "negative", "neutral", "positive"]
    preds = ["positive", "negative", "neutral", "positive"]
    result = compute_metrics(golds, preds)
    assert result["macro_f1"] == pytest.approx(1.0, abs=1e-4)
    assert result["accuracy"] == pytest.approx(1.0, abs=1e-4)


def test_known_f1():
    golds = ["positive", "negative"]
    preds = ["positive", "positive"]
    result = compute_metrics(golds, preds)
    assert result["accuracy"] == pytest.approx(0.5, abs=1e-4)
    assert result["macro_f1"] == pytest.approx(0.333, abs=0.01)


def test_per_class_keys():
    golds = ["positive", "neutral"]
    preds = ["positive", "positive"]
    result = compute_metrics(golds, preds)
    assert "precision" in result
    assert "recall" in result
    assert "per_class" in result
