from pathlib import Path

from src.evaluation.error_analysis import analyse_errors, classify_error


def test_classify_implicit_aspect():
    review = "It disconnects randomly all day long."
    assert classify_error(review, "connectivity", "negative", "positive") == "implicit_aspect"


def test_classify_aspect_mismatch():
    review = "The battery is amazing but the screen is terrible."
    assert classify_error(review, "screen", "negative", "positive") == "aspect_mismatch"


def test_analyse_errors_writes_json(tmp_path: Path):
    output_path = tmp_path / "error_report.json"
    summary = analyse_errors(
        reviews=["Yeah right, the battery is just perfect and died in an hour."],
        aspects=["battery"],
        golds=["negative"],
        preds=["positive"],
        model_name="unit-test-model",
        output_path=output_path,
    )
    assert summary["total_errors"] == 1
    assert output_path.exists()
