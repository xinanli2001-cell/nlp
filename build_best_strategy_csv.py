"""Build a single long-format CSV that uses the best training strategy per model.

Selection protocol:
- Standard BERT: fixed epoch 5 (plateaus by epoch 3; the 5-epoch CSV is used)
- Extended BERT: fixed epoch 10 (cross-domain curves still rising at epoch 5)
- RoBERTa:       fixed epoch 10 (same as above)

The output schema matches outputs/evaluation/multi_seed_full_cuda/full_benchmark.csv
so that make_figures.py can consume either file by switching FULL_CSV.
"""

from __future__ import annotations
import csv
from pathlib import Path

CURVES_5 = Path("outputs/evaluation/all_models_curves/curves.csv")
CURVES_10 = Path("outputs/evaluation/all_models_curves/curves_10ep.csv")
OUT = Path("outputs/evaluation/best_strategy/best_strategy.csv")

# (model, final_epoch_to_select)
FINAL_EPOCH = {
    "Standard BERT ABSA": 5,
    "Extended BERT ABSA": 10,
    "RoBERTa ABSA": 10,
}

FIELDS = [
    "model", "domain", "status", "n_examples", "seed",
    "accuracy", "precision", "recall", "macro_f1",
    "positive_f1", "negative_f1", "neutral_f1",
]


def _load(path: Path) -> list[dict]:
    """Read a per-epoch curves CSV into a list of dict rows."""
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Merge the 5-epoch and 10-epoch curves into a single best-strategy CSV."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    curves5 = _load(CURVES_5)
    curves10 = _load(CURVES_10) if CURVES_10.exists() else []

    # Build a single stream that selects one epoch per (model, seed, domain).
    selected: list[dict] = []
    for r in curves5:
        if r["model"] == "Standard BERT ABSA" and int(r["epoch"]) == FINAL_EPOCH[r["model"]]:
            selected.append(r)
    for r in curves10:
        if r["model"] in ("Extended BERT ABSA", "RoBERTa ABSA") and int(r["epoch"]) == FINAL_EPOCH[r["model"]]:
            selected.append(r)

    # Re-project into the target schema expected by make_figures.py
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in selected:
            writer.writerow({
                "model": r["model"],
                "domain": r["domain"],
                "status": "ok",
                "n_examples": r["n_examples"],
                "seed": r["seed"],
                "accuracy": r["accuracy"],
                "precision": "",   # not recorded in curves CSV
                "recall": "",
                "macro_f1": r["macro_f1"],
                "positive_f1": r["positive_f1"],
                "negative_f1": r["negative_f1"],
                "neutral_f1": r["neutral_f1"],
            })
    print(f"Wrote {len(selected)} rows to {OUT}")


if __name__ == "__main__":
    main()
