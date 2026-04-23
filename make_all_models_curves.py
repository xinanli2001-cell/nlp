"""Per-epoch learning curves for Standard BERT, Extended BERT, and RoBERTa.

Reads the 1200-row CSV produced by train_all_models_curves.py and produces
a 3x4 grid (models x domains) of macro F1 curves over epochs. Each cell
shows individual seeds as thin lines and the 20-seed mean with a
1-sigma shaded band.

Output: outputs/figures/fig6_all_models_learning_curves.{pdf,png}
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("outputs/evaluation/all_models_curves/curves.csv")
CSV_10EP_PATH = Path("outputs/evaluation/all_models_curves/curves_10ep.csv")
OUT_DIR = Path("outputs/figures")

MODEL_ORDER = ["Standard BERT ABSA", "Extended BERT ABSA", "RoBERTa ABSA"]
MODEL_SHORT = {
    "Standard BERT ABSA": "Standard BERT",
    "Extended BERT ABSA": "Extended BERT",
    "RoBERTa ABSA": "RoBERTa",
}
DOMAIN_ORDER = ["val", "in_domain", "laptop", "restaurant"]
DOMAIN_TITLE = {
    "val":        "Validation (122)",
    "in_domain":  "In-domain test (123)",
    "laptop":     "SemEval Laptop (2313)",
    "restaurant": "SemEval Restaurant (3602)",
}
COLOR = {
    "Standard BERT": "#4C72B0",
    "Extended BERT": "#55A868",
    "RoBERTa":       "#C44E52",
}
# Per-model max epoch: Standard BERT plateaus by e3 so 5 is enough;
# Extended BERT and RoBERTa show upward trend on cross-domain at e5
# so we extended them to 10 epochs.
MAX_EPOCHS = {
    "Standard BERT ABSA": 5,
    "Extended BERT ABSA": 10,
    "RoBERTa ABSA": 10,
}
X_AXIS_MAX = max(MAX_EPOCHS.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_curves() -> dict[tuple[str, str, int], dict[int, float]]:
    curves: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    # Standard BERT: 5-epoch data from the original CSV
    with CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != "Standard BERT ABSA":
                continue
            key = (row["model"], row["domain"], int(row["seed"]))
            curves[key][int(row["epoch"])] = float(row["macro_f1"])
    # Extended BERT + RoBERTa: prefer the 10-epoch extension when available
    if CSV_10EP_PATH.exists():
        with CSV_10EP_PATH.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["model"], row["domain"], int(row["seed"]))
                curves[key][int(row["epoch"])] = float(row["macro_f1"])
    else:
        # Fall back to original 5-epoch CSV
        with CSV_PATH.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == "Standard BERT ABSA":
                    continue
                key = (row["model"], row["domain"], int(row["seed"]))
                curves[key][int(row["epoch"])] = float(row["macro_f1"])
    return curves


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    fig, axes = plt.subplots(len(MODEL_ORDER), len(DOMAIN_ORDER),
                             figsize=(13, 8.4), sharex="row", sharey=True)

    for i, model in enumerate(MODEL_ORDER):
        short = MODEL_SHORT[model]
        color = COLOR[short]
        n_epochs = MAX_EPOCHS[model]
        epochs = np.arange(1, n_epochs + 1)
        for j, domain in enumerate(DOMAIN_ORDER):
            ax = axes[i, j]
            seed_curves = {seed: curves[(model, domain, seed)]
                           for (m, d, seed) in curves
                           if m == model and d == domain}
            matrix = np.array([
                [seed_curves[s][e] for e in range(1, n_epochs + 1)]
                for s in sorted(seed_curves)
            ])
            for row in matrix:
                ax.plot(epochs, row, color=color, alpha=0.18, linewidth=0.8)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1)
            ax.plot(epochs, mean, color=color, linewidth=2.3,
                    label=f"Mean ({matrix.shape[0]} seeds)")
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=0.18, label=r"$\pm 1\sigma$")

            if i == 0:
                ax.set_title(DOMAIN_TITLE[domain])
            if j == 0:
                suffix = f"({n_epochs} ep)"
                ax.set_ylabel(f"{short} {suffix}\nMacro F1")
            # Each row gets its own x range: Standard BERT 1-5, the others 1-10.
            ax.set_xlim(0.7, n_epochs + 0.3)
            ax.set_xticks(range(1, n_epochs + 1))
            ax.set_xlabel("Training epoch")
            ax.set_ylim(0.05, 0.85)
            ax.grid(axis="y", alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", frameon=False)

    fig.suptitle("Per-epoch Macro F1 across 20 seeds  —  three architectures × four evaluation domains",
                 fontsize=12, y=0.998)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig6_all_models_learning_curves.{ext}")
    plt.close(fig)
    for p in sorted(OUT_DIR.glob("fig6*")):
        print(p)


if __name__ == "__main__":
    main()
