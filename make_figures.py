"""Publication-style figures for the ABSA benchmark report.

Reads the 20-seed full benchmark CSV plus the one-shot baseline evaluation
results and produces five consolidated figures under outputs/figures/.

Figures:
    fig1_macro_f1_distribution.{pdf,png}
        Violin + strip plot of Macro F1 across 20 seeds per (model, domain).
        Rule-based baseline shown as a dashed horizontal reference.

    fig2_cross_domain_scatter.{pdf,png}
        Per-seed scatter of in-domain vs cross-domain Macro F1. Makes
        RoBERTa's flatter transfer curve visually obvious.

    fig3_per_class_f1.{pdf,png}
        Grouped bar chart of per-class F1 (positive / negative / neutral)
        with 1-sigma error bars, faceted by domain.

    fig4_variance.{pdf,png}
        Macro F1 and Neutral F1 standard deviation per (model, domain).
        Frames training stability as a first-class finding.

    fig5_data_distribution.{pdf,png}
        Sentiment-class distribution of each dataset (project 814, SemEval
        laptop 2313, SemEval restaurant 3602).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FULL_CSV = Path("outputs/evaluation/best_strategy/best_strategy.csv")
BASELINE_IN_CSV = Path("outputs/evaluation/metrics_summary.csv")
BASELINE_CROSS_CSV = Path("outputs/evaluation/cross_domain/metrics_summary.csv")
SEMEVAL_STATS = Path("data/semeval_stats.json")
OUT_DIR = Path("outputs/figures")

MODEL_ORDER = ["Standard BERT ABSA", "Extended BERT ABSA", "RoBERTa ABSA"]
MODEL_SHORT = {
    "Standard BERT ABSA": "Standard BERT",
    "Extended BERT ABSA": "Extended BERT",
    "RoBERTa ABSA": "RoBERTa",
    "Rule-Based Baseline": "Rule-Based",
}
DOMAIN_ORDER = ["in_domain", "laptop", "restaurant"]
DOMAIN_LABEL = {
    "in_domain": "In-domain\n(project, 123)",
    "laptop": "Cross-domain\nLaptop (2313)",
    "restaurant": "Cross-domain\nRestaurant (3602)",
}

# Accessible palette with distinguishable luminance.
COLOR = {
    "Standard BERT": "#4C72B0",
    "Extended BERT": "#55A868",
    "RoBERTa": "#C44E52",
    "Rule-Based": "#8C8C8C",
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


def _load_full() -> list[dict]:
    """Load the benchmark CSV as a list of rows with numeric fields cast to float."""
    with FULL_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Drop val rows so downstream filters by in_domain / laptop / restaurant only.
    rows = [r for r in rows if r.get("domain") != "val"]
    for r in rows:
        r["seed"] = int(r["seed"])
        r["n_examples"] = int(r["n_examples"])
        for k in ("accuracy", "macro_f1",
                  "positive_f1", "negative_f1", "neutral_f1"):
            r[k] = float(r[k])
    return rows


def _load_baseline() -> dict:
    """Return {domain: {macro_f1, positive_f1, ...}} for the rule-based row."""
    out: dict[str, dict] = {}
    with BASELINE_IN_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "Rule-Based Baseline":
                out["in_domain"] = {
                    "accuracy": float(r["accuracy"]),
                    "macro_f1": float(r["macro_f1"]),
                    "positive_f1": float(r["positive_f1"]),
                    "negative_f1": float(r["negative_f1"]),
                    "neutral_f1": float(r["neutral_f1"]),
                }
    with BASELINE_CROSS_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "Rule-Based Baseline":
                out[r["domain"]] = {
                    "accuracy": float(r["accuracy"]),
                    "macro_f1": float(r["macro_f1"]),
                    "positive_f1": float(r["positive_f1"]),
                    "negative_f1": float(r["negative_f1"]),
                    "neutral_f1": float(r["neutral_f1"]),
                }
    return out


def _grouped(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rows by (model, domain) for quick lookup when plotting."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["domain"])].append(r)
    return grouped


# ──────────────────────────────────────────────────────────────────────────
def fig1_macro_f1_distribution(rows, baseline):
    """Render fig1: violin + strip plot of Macro F1 per (model, domain)."""
    grouped = _grouped(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    positions = [1, 2, 3]
    for ax, domain in zip(axes, DOMAIN_ORDER):
        data = [[r["macro_f1"] for r in grouped[(m, domain)]] for m in MODEL_ORDER]
        parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, widths=0.7)
        for body, m in zip(parts["bodies"], MODEL_ORDER):
            body.set_facecolor(COLOR[MODEL_SHORT[m]])
            body.set_edgecolor("black")
            body.set_alpha(0.4)
        for pc in ("cbars", "cmins", "cmaxes"):
            if pc in parts:
                parts[pc].set_edgecolor("#555")
                parts[pc].set_linewidth(0.7)

        # Strip dots for individual seeds.
        rng = np.random.default_rng(0)
        for i, (pos, vals, m) in enumerate(zip(positions, data, MODEL_ORDER)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       color=COLOR[MODEL_SHORT[m]], s=18, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
            mean = np.mean(vals)
            ax.hlines(mean, pos - 0.3, pos + 0.3, colors="black", linewidth=1.8, zorder=4)

        # Baseline reference.
        ax.axhline(baseline[domain]["macro_f1"], linestyle="--", linewidth=1.1,
                   color=COLOR["Rule-Based"],
                   label=f"Rule-Based ({baseline[domain]['macro_f1']:.3f})")

        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=0)
        ax.set_title(DOMAIN_LABEL[domain])
        ax.set_ylim(0.2, 0.85)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", frameon=False)

    axes[0].set_ylabel("Macro F1")
    fig.suptitle("Macro F1 across 20 training seeds  —  violin + strip + baseline reference",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig1_macro_f1_distribution.{ext}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
def fig2_cross_domain_scatter(rows):
    """Render fig2: in-domain vs cross-domain Macro F1 scatter with y=x reference."""
    grouped = _grouped(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, sharex=True)
    for ax, cross in zip(axes, ["laptop", "restaurant"]):
        for m in MODEL_ORDER:
            in_vals = {r["seed"]: r["macro_f1"] for r in grouped[(m, "in_domain")]}
            cross_vals = {r["seed"]: r["macro_f1"] for r in grouped[(m, cross)]}
            seeds = sorted(in_vals)
            xs = [in_vals[s] for s in seeds]
            ys = [cross_vals[s] for s in seeds]
            ax.scatter(xs, ys, s=46, color=COLOR[MODEL_SHORT[m]],
                       edgecolor="black", linewidth=0.5,
                       label=f"{MODEL_SHORT[m]}  ({np.mean(ys):.3f} mean)",
                       alpha=0.9)
        lo, hi = 0.35, 0.85
        ax.plot([lo, hi], [lo, hi], color="#999", linestyle="--", linewidth=1, zorder=0, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("In-domain Macro F1")
        ax.set_title(f"Cross-domain: {cross.capitalize()}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].set_ylabel("Cross-domain Macro F1")
    fig.suptitle("In-domain vs cross-domain Macro F1 per seed  —  points below y=x mean the model lost transferability",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_cross_domain_scatter.{ext}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
def fig3_per_class_f1(rows, baseline):
    """Render fig3: grouped per-class F1 bars (positive/negative/neutral) with 1-sigma error bars."""
    grouped = _grouped(rows)
    classes = [("positive_f1", "Positive F1"), ("negative_f1", "Negative F1"), ("neutral_f1", "Neutral F1")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    models_plus_baseline = MODEL_ORDER + ["Rule-Based Baseline"]
    x_positions = np.arange(len(models_plus_baseline))
    width = 0.25
    for ax, domain in zip(axes, DOMAIN_ORDER):
        for i, (key, label) in enumerate(classes):
            means, stds, colors = [], [], []
            for m in models_plus_baseline:
                if m == "Rule-Based Baseline":
                    means.append(baseline[domain][key])
                    stds.append(0.0)
                else:
                    vals = [r[key] for r in grouped[(m, domain)]]
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals, ddof=1)))
                colors.append(COLOR[MODEL_SHORT[m]])
            offset = (i - 1) * width
            ax.bar(x_positions + offset, means, width=width, yerr=stds,
                   color=colors, edgecolor="black", linewidth=0.5,
                   error_kw={"elinewidth": 1, "capsize": 3}, label=label,
                   alpha=[0.9, 0.65, 0.45][i])  # varying alpha distinguishes classes
        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_SHORT[m] for m in models_plus_baseline], rotation=20, ha="right")
        ax.set_title(DOMAIN_LABEL[domain])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("F1")
    # Legend: show class distinction using grey alphas, since per-bar color is model.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#777", alpha=0.9, edgecolor="black", label="Positive"),
        Patch(facecolor="#777", alpha=0.65, edgecolor="black", label="Negative"),
        Patch(facecolor="#777", alpha=0.45, edgecolor="black", label="Neutral"),
    ]
    axes[-1].legend(handles=legend_handles, loc="upper right", frameon=False, title="Class (alpha)")
    fig.suptitle("Per-class F1  —  mean ± 1σ across 20 seeds (baseline is single-run)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig3_per_class_f1.{ext}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
def fig4_variance(rows):
    """Render fig4: per-(model, domain) standard deviation of Macro F1 and Neutral F1."""
    grouped = _grouped(rows)
    metrics = [("macro_f1", "Macro F1"), ("neutral_f1", "Neutral F1")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x_positions = np.arange(len(MODEL_ORDER))
    width = 0.25
    for ax, (key, label) in zip(axes, metrics):
        for i, domain in enumerate(DOMAIN_ORDER):
            stds = []
            for m in MODEL_ORDER:
                vals = [r[key] for r in grouped[(m, domain)]]
                stds.append(float(np.std(vals, ddof=1)))
            offset = (i - 1) * width
            ax.bar(x_positions + offset, stds, width=width,
                   label=DOMAIN_LABEL[domain].replace("\n", " "),
                   edgecolor="black", linewidth=0.5,
                   color=["#7A9AC2", "#C79C9C", "#C2B97A"][i])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
        ax.set_title(f"Standard deviation of {label}")
        ax.set_ylabel("σ across 20 seeds")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("Training stability  —  a lower σ means the architecture is more reproducible across seeds",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_variance.{ext}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
def fig5_data_distribution():
    """Render fig5: sentiment-class pie charts for project data vs SemEval splits."""
    with SEMEVAL_STATS.open(encoding="utf-8") as f:
        stats = json.load(f)
    dataset_order = [
        ("Project (Amazon)", stats["project"]["sentiments"]),
        ("SemEval Laptop", stats["laptop"]["sentiments"]),
        ("SemEval Restaurant", stats["restaurant"]["sentiments"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    sent_colors = {"positive": "#3A8F49", "negative": "#C44E52", "neutral": "#B8B8B8"}
    sent_order = ["positive", "negative", "neutral"]

    for ax, (name, dist) in zip(axes, dataset_order):
        total = sum(dist.values())
        sizes = [dist[s] for s in sent_order]
        percents = [v / total * 100 for v in sizes]
        colors = [sent_colors[s] for s in sent_order]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 9, "color": "white", "fontweight": "bold"})
        ax.set_title(f"{name}\n(n={total})", fontsize=10)
        ax.set_aspect("equal")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=sent_colors[s], edgecolor="white", label=s.capitalize())
                      for s in sent_order]
    fig.legend(handles=legend_handles, loc="center right", frameon=False, fontsize=10,
               bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Sentiment class distribution by dataset  —  project is heavily imbalanced",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig5_data_distribution.{ext}")
    plt.close(fig)


def main() -> None:
    """Regenerate all five report figures from the best-strategy CSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_full()
    baseline = _load_baseline()
    fig1_macro_f1_distribution(rows, baseline)
    fig2_cross_domain_scatter(rows)
    fig3_per_class_f1(rows, baseline)
    fig4_variance(rows)
    fig5_data_distribution()
    generated = sorted(OUT_DIR.glob("*"))
    print(f"Generated {len(generated)} files:")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
