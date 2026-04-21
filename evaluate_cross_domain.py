"""Cross-domain evaluation on SemEval 2014 Task 4 datasets.

Loads checkpoints trained on the project's Amazon electronics reviews and
runs inference on SemEval laptop and restaurant ABSA data. This measures
how well models generalise to aspect vocabulary and review styles outside
their training distribution.

Artifacts written to ``outputs/evaluation/cross_domain/``:
- metrics_summary.csv / metrics_summary.json
- confusion_matrices/*.png
- error_reports/*.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import ABSADataset, ID2LABEL, load_csv
from src.data.extended_dataset import ExtendedABSADataset
from src.data.roberta_dataset import RobertaABSADataset
from src.evaluation.error_analysis import analyse_errors
from src.evaluation.metrics import compute_metrics, print_report
from src.models.baseline import RuleBasedABSA
from src.models.bert_absa import BertABSA
from src.models.extended_bert import ExtendedBertABSA
from src.models.robertaabsa import RobertaABSA

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs/evaluation/cross_domain")
ERROR_DIR = OUTPUT_DIR / "error_reports"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
SUMMARY_CSV = OUTPUT_DIR / "metrics_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "metrics_summary.json"
LABELS = ["positive", "negative", "neutral"]

DOMAINS = {
    "laptop": Path("data/semeval_laptop.csv"),
    "restaurant": Path("data/semeval_restaurant.csv"),
}


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def _ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


def _save_confusion_matrix(golds, preds, label: str) -> Path:
    cm = confusion_matrix(golds, preds, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {label}")
    fig.tight_layout()
    path = CONFUSION_DIR / f"{_slugify(label)}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _load_checkpoint(model: torch.nn.Module, path: Path) -> bool:
    if not path.exists():
        print(f"[WARN] Missing checkpoint: {path}. Skipping.")
        return False
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return True


def _summary_row(model_name: str, domain: str, metrics: dict, n: int, status: str = "ok") -> dict:
    return {
        "model": model_name,
        "domain": domain,
        "status": status,
        "n_examples": n,
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": metrics["macro_f1"],
        "positive_f1": metrics["per_class"]["positive"],
        "negative_f1": metrics["per_class"]["negative"],
        "neutral_f1": metrics["per_class"]["neutral"],
    }


def _persist(golds, preds, label: str, reviews, aspects) -> None:
    analyse_errors(
        reviews, aspects, golds, preds, model_name=label,
        output_path=ERROR_DIR / f"{_slugify(label)}.json",
    )
    _save_confusion_matrix(golds, preds, label)


def _infer(model: torch.nn.Module, loader: DataLoader, *, use_token_types: bool):
    golds, preds = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if use_token_types:
                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["token_type_ids"].to(DEVICE),
                )
            else:
                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                )
            preds.extend(ID2LABEL[i] for i in logits.argmax(-1).cpu().tolist())
            golds.extend(ID2LABEL[i] for i in batch["label"].cpu().tolist())
    return golds, preds


def _run_baseline(rows, domain):
    model_name = "Rule-Based Baseline"
    label = f"{model_name} ({domain})"
    reviews = [r["review"] for r in rows]
    aspects = [r["aspect"] for r in rows]
    golds = [r["sentiment"] for r in rows]
    preds = RuleBasedABSA().predict_batch(reviews, aspects)
    print_report(golds, preds, label)
    _persist(golds, preds, label, reviews, aspects)
    return _summary_row(model_name, domain, compute_metrics(golds, preds), len(rows))


def _run_bert_like(model_cls, model_name, ckpt_name, dataset, domain, rows, *, use_token_types):
    label = f"{model_name} ({domain})"
    model = model_cls().to(DEVICE)
    if not _load_checkpoint(model, CHECKPOINT_DIR / ckpt_name):
        return None
    loader = DataLoader(dataset, batch_size=32)
    golds, preds = _infer(model, loader, use_token_types=use_token_types)
    print_report(golds, preds, label)
    _persist(golds, preds, label,
             [r["review"] for r in rows],
             [r["aspect"] for r in rows])
    return _summary_row(model_name, domain, compute_metrics(golds, preds), len(golds))


def _save_summary(rows: list[dict]) -> None:
    fieldnames = ["model", "domain", "status", "n_examples", "accuracy",
                  "macro_f1", "positive_f1", "negative_f1", "neutral_f1"]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    _ensure_output_dirs()
    all_rows: list[dict] = []

    for domain, csv_path in DOMAINS.items():
        if not csv_path.exists():
            print(f"[WARN] Missing dataset: {csv_path}. Skipping domain {domain}.")
            continue

        rows = load_csv(str(csv_path))
        print(f"\n{'#' * 60}\n# Domain: {domain}  ({len(rows)} examples)\n{'#' * 60}")

        all_rows.append(_run_baseline(rows, domain))

        for result in (
            _run_bert_like(BertABSA, "Standard BERT ABSA", "bert_absa_best.pt",
                           ABSADataset(rows), domain, rows, use_token_types=True),
            _run_bert_like(ExtendedBertABSA, "Extended BERT ABSA", "extended_bert_best.pt",
                           ExtendedABSADataset(rows), domain, rows, use_token_types=True),
            _run_bert_like(RobertaABSA, "RoBERTa ABSA", "roberta_absa_best.pt",
                           RobertaABSADataset(rows), domain, rows, use_token_types=False),
        ):
            if result is not None:
                all_rows.append(result)

    _save_summary(all_rows)
    print(f"\nSaved cross-domain metrics to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
