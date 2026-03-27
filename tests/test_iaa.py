# tests/test_iaa.py
import pytest
from src.data.iaa import compute_cohens_kappa, load_annotations

def test_perfect_agreement():
    labels_a = ["positive", "negative", "neutral", "positive"]
    labels_b = ["positive", "negative", "neutral", "positive"]
    kappa = compute_cohens_kappa(labels_a, labels_b)
    assert abs(kappa - 1.0) < 1e-6

def test_random_agreement_near_zero():
    import random
    random.seed(42)
    choices = ["positive", "negative", "neutral"]
    labels_a = [random.choice(choices) for _ in range(300)]
    labels_b = [random.choice(choices) for _ in range(300)]
    kappa = compute_cohens_kappa(labels_a, labels_b)
    assert -0.15 < kappa < 0.15

def test_load_annotations_returns_aligned_lists(tmp_path):
    csv1 = tmp_path / "a1.csv"
    csv2 = tmp_path / "a2.csv"
    csv1.write_text("id,sentiment\n1,positive\n2,negative\n3,neutral\n")
    csv2.write_text("id,sentiment\n1,positive\n2,positive\n3,neutral\n")
    a, b = load_annotations(str(csv1), str(csv2))
    assert a == ["positive", "negative", "neutral"]
    assert b == ["positive", "positive", "neutral"]
