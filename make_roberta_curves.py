"""Plot training curves and comparison for the improved RoBERTa experiment.

Produces under outputs/figures/:
- fig6_roberta_curves.{pdf,png}
    Four subplots: val Macro F1 + three test-domain Macro F1 across 10 epochs.
    Thin coloured lines are individual seeds; thick line is the 20-seed mean;
    shaded band is ± 1 standard deviation.
- fig7_roberta_improved_vs_original.{pdf,png}
    Violin + mean marker comparison of improved vs original RoBERTa for all
    three domains (val-peak aggregation for both).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IMPROVED_CSV = Path("outputs/evaluation/roberta_improved/roberta_curves.csv")
ORIGINAL_CSV = Path("outputs/evaluation/multi_seed_full_cuda/full_benchmark.csv")
OUT_DIR = Path("outputs/figures")

DOMAINS = ["val", "in_domain", "laptop", "restaurant"]
DOMAIN_TITLE = {
    "val":        "Validation (122)",
    "in_domain":  "Test — In-domain (123)",
    "laptop":     "Test — SemEval Laptop (2313)",
    "restaurant": "Test — SemEval Restaurant (3602)",
}
DOMAIN_COLOR = {
    "val":        "#4C72B0",
    "in_domain":  "#55A868",
    "laptop":     "#C44E52",
    "restaurant": "#8172B2",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_curves() -> dict[str, dict[int, dict[int, float]]]:
    """Return {domain: {seed: {epoch: macro_f1}}}."""
    data: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with IMPROVED_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            epoch = int(row["epoch"])
            domain = row["domain"]
            data[domain][seed][epoch] = float(row["macro_f1"])
    return data


def fig6_training_curves(curves):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    epochs = np.arange(1, 11)
    val_peak_epochs = []
    # Determine val peak epoch per seed for annotating.
    for seed, seed_curve in curves["val"].items():
        best_epoch = max(seed_curve.items(), key=lambda x: x[1])[0]
        val_peak_epochs.append(best_epoch)

    for ax, domain in zip(axes.flat, DOMAINS):
        seeds = sorted(curves[domain].keys())
        matrix = np.array([[curves[domain][s][e] for e in range(1, 11)] for s in seeds])
        # Individual seed lines, faint.
        for i, row in enumerate(matrix):
            ax.plot(epochs, row, color=DOMAIN_COLOR[domain], alpha=0.18, linewidth=0.8)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        ax.plot(epochs, mean, color=DOMAIN_COLOR[domain], linewidth=2.5, label=f"Mean (n={len(seeds)})")
        ax.fill_between(epochs, mean - std, mean + std, color=DOMAIN_COLOR[domain], alpha=0.18, label="±1σ")

        # Mark median val-peak epoch as a vertical dashed line on val subplot only.
        if domain == "val":
            median_peak = int(np.median(val_peak_epochs))
            ax.axvline(median_peak, linestyle="--", color="#555", linewidth=1,
                       label=f"Median val peak: epoch {median_peak}")

        ax.set_title(DOMAIN_TITLE[domain])
        ax.set_ylim(0.0, 0.85)
        ax.set_xticks(epochs)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", frameon=False)

    for ax in axes[-1]:
        ax.set_xlabel("Training epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro F1")

    fig.suptitle("Improved RoBERTa training curves  —  lr=1e-5, weight_decay=0.01, 10 epochs × 20 seeds",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig6_roberta_curves.{ext}")
    plt.close(fig)


def val_peak_results(csv_path: Path, model_name_filter: str | None = None) -> dict[str, list[float]]:
    """Read a benchmark CSV and return per-domain list of val-peak-selected test Macro F1.

    For the improved RoBERTa CSV (no model column), pass model_name_filter=None.
    For the original benchmark CSV, pass model_name_filter="RoBERTa ABSA".
    """
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if model_name_filter and r.get("model") != model_name_filter:
                continue
            rows.append(r)

    is_improved = "epoch" in rows[0]
    if is_improved:
        # (seed, epoch, domain) -> macro_f1 ; val used to select epoch
        by_seed: dict[int, dict[tuple[int, str], float]] = defaultdict(dict)
        for r in rows:
            by_seed[int(r["seed"])][(int(r["epoch"]), r["domain"])] = float(r["macro_f1"])
        out: dict[str, list[float]] = {"in_domain": [], "laptop": [], "restaurant": []}
        for seed, metrics in by_seed.items():
            val_items = [(e, metrics[(e, "val")]) for (e, d) in metrics if d == "val"]
            peak_epoch = max(val_items, key=lambda x: x[1])[0]
            for dom in out:
                out[dom].append(metrics[(peak_epoch, dom)])
        return out
    else:
        # Original benchmark: best-of-5-epoch checkpoint per seed already, no val selection needed here.
        # Just pool per-domain test macro_f1 across seeds.
        out = {"in_domain": [], "laptop": [], "restaurant": []}
        for r in rows:
            if r["domain"] in out:
                out[r["domain"]].append(float(r["macro_f1"]))
        return out


def fig7_comparison(curves_csv: Path, original_csv: Path):
    improved = val_peak_results(curves_csv)
    original = val_peak_results(original_csv, model_name_filter="RoBERTa ABSA")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    domains = ["in_domain", "laptop", "restaurant"]
    titles = ["In-domain (123)", "SemEval Laptop (2313)", "SemEval Restaurant (3602)"]

    for ax, dom, title in zip(axes, domains, titles):
        data = [original[dom], improved[dom]]
        labels = ["Original\n(lr=2e-5, 5 ep)", "Improved\n(lr=1e-5, wd=0.01, 10 ep)"]
        parts = ax.violinplot(data, positions=[1, 2], widths=0.7, showmeans=False, showmedians=False)
        for body, color in zip(parts["bodies"], ["#C44E52", "#4C72B0"]):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.45)
        for pc in ("cbars", "cmins", "cmaxes"):
            if pc in parts:
                parts[pc].set_edgecolor("#555")
                parts[pc].set_linewidth(0.8)

        rng = np.random.default_rng(0)
        for i, (pos, vals, color) in enumerate(zip([1, 2], data, ["#C44E52", "#4C72B0"])):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       color=color, s=20, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            ax.hlines(mean, pos - 0.3, pos + 0.3, colors="black", linewidth=1.8, zorder=4)
            ax.text(pos, mean + 0.02, f"{mean:.3f}", ha="center", fontsize=9, fontweight="bold")
            ax.text(pos, 0.31, f"σ={std:.3f}", ha="center", fontsize=8, color="#555")

        ax.set_title(title)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0.30, 0.80)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Test Macro F1")
    fig.suptitle("RoBERTa: Original 20-seed vs Improved (lr↓ + weight decay + 10 epochs) — val-peak selection",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig7_roberta_improved_vs_original.{ext}")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    fig6_training_curves(curves)
    fig7_comparison(IMPROVED_CSV, ORIGINAL_CSV)
    for p in sorted(OUT_DIR.glob("fig[67]*")):
        print(p)


if __name__ == "__main__":
    main()
