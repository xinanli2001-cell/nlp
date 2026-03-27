from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)

LABELS = ["positive", "negative", "neutral"]


def compute_metrics(golds: list[str], preds: list[str]) -> dict:
    """Return accuracy, macro P/R/F1, and per-class breakdown."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        golds, preds, average="macro", zero_division=0
    )
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        golds, preds, labels=LABELS, average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(golds, preds),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "per_class": {
            label: round(float(f), 4)
            for label, f in zip(LABELS, per_class_f1)
        },
    }


def print_report(golds: list[str], preds: list[str], model_name: str = "") -> None:
    result = compute_metrics(golds, preds)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Macro F1:  {result['macro_f1']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print("\nPer-class F1:")
    for label, score in result["per_class"].items():
        print(f"  {label}: {score:.4f}")
