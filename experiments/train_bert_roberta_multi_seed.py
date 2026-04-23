"""Multi-seed training for Standard BERT and RoBERTa ABSA.

Trains each model independently across the same 5 seeds used for Extended
BERT, so that variance is directly comparable across the three BERT-family
architectures. For every (model, seed) pair:
- Saves the best-val-macro-F1 checkpoint to checkpoints/{prefix}_seed_{seed}.pt
- Evaluates that checkpoint on the held-out test set
- Records both val and test metrics

The final combined summary joins these with the existing Extended BERT
multi-seed results to produce a single variance-analysis table.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import ABSADataset, ID2LABEL, load_csv
from src.data.roberta_dataset import RobertaABSADataset
from src.evaluation.metrics import compute_metrics
from src.models.bert_absa import BertABSA
from src.models.robertaabsa import RobertaABSA

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS = [42, 1, 7, 2024, 123]
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 5
LR = 2e-5
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs/evaluation/multi_seed")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class ModelConfig:
    """Per-model hyperparameter + forward-signature bundle."""
    name: str
    model_cls: type
    dataset_cls: type
    ckpt_prefix: str
    uses_token_types: bool


CONFIGS = [
    ModelConfig("Standard BERT ABSA", BertABSA, ABSADataset, "bert_absa", True),
    ModelConfig("RoBERTa ABSA", RobertaABSA, RobertaABSADataset, "roberta_absa", False),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_forward(model, batch, uses_token_types: bool):
    if uses_token_types:
        return model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["token_type_ids"].to(DEVICE),
        )
    return model(
        batch["input_ids"].to(DEVICE),
        batch["attention_mask"].to(DEVICE),
    )


def evaluate(model, loader, uses_token_types):
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = run_forward(model, batch, uses_token_types)
            preds += [ID2LABEL[p] for p in logits.argmax(dim=-1).cpu().tolist()]
            golds += [ID2LABEL[g] for g in batch["label"].cpu().tolist()]
    return golds, preds


def train_one_seed(cfg: ModelConfig, seed: int, train_rows, val_rows, test_rows) -> dict:
    set_seed(seed)

    train_dl = DataLoader(cfg.dataset_cls(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(cfg.dataset_cls(val_rows), batch_size=BATCH_SIZE)
    test_dl = DataLoader(cfg.dataset_cls(test_rows), batch_size=BATCH_SIZE)

    model = cfg.model_cls().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )
    weights = torch.tensor([1.0, 13.7, 17.0], device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            logits = run_forward(model, batch, cfg.uses_token_types)
            loss = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        golds, preds = evaluate(model, val_dl, cfg.uses_token_types)
        val_metrics = compute_metrics(golds, preds)
        print(f"  [{cfg.name}] seed {seed} | epoch {epoch}/{EPOCHS} | "
              f"train_loss={total_loss/len(train_dl):.4f} | val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    save_path = CHECKPOINT_DIR / f"{cfg.ckpt_prefix}_seed_{seed}.pt"
    torch.save(best_state, save_path)
    model.load_state_dict(best_state)
    test_golds, test_preds = evaluate(model, test_dl, cfg.uses_token_types)
    test_metrics = compute_metrics(test_golds, test_preds)
    print(f"  [{cfg.name}] seed {seed} | SAVED {save_path.name} | test_f1={test_metrics['macro_f1']:.4f}")

    return {
        "model": cfg.name,
        "seed": seed,
        "val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_positive_f1": round(test_metrics["per_class"]["positive"], 4),
        "test_negative_f1": round(test_metrics["per_class"]["negative"], 4),
        "test_neutral_f1": round(test_metrics["per_class"]["neutral"], 4),
    }


def summarise(rows: list[dict]) -> dict:
    """Grouped summary: per model, compute mean/std/min/max for each metric."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = {}
    numeric_keys = [k for k in rows[0] if k not in ("model", "seed")]
    for model_name, model_rows in by_model.items():
        summary[model_name] = {}
        for k in numeric_keys:
            vals = [r[k] for r in model_rows]
            summary[model_name][k] = {
                "mean": round(statistics.mean(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_csv("data/final/train.csv")
    val_rows = load_csv("data/final/val.csv")
    test_rows = load_csv("data/final/test.csv")
    print(f"Data: {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test")
    print(f"Device: {DEVICE}")
    print(f"Models: {[c.name for c in CONFIGS]}")
    print(f"Seeds: {SEEDS}\n")

    results = []
    for cfg in CONFIGS:
        print(f"\n{'#' * 60}\n# {cfg.name}\n{'#' * 60}")
        for seed in SEEDS:
            print(f"\n--- seed {seed} ---")
            results.append(train_one_seed(cfg, seed, train_rows, val_rows, test_rows))

    # Persist raw per-(model,seed) CSV.
    results_csv = OUTPUT_DIR / "bert_roberta_multi_seed.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Merge with existing Extended BERT multi-seed results (if present).
    existing_ext = OUTPUT_DIR / "extended_bert_multi_seed.csv"
    combined = list(results)
    if existing_ext.exists():
        with existing_ext.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ext_row = {"model": "Extended BERT ABSA", "seed": int(row["seed"])}
                for k in ("val_macro_f1", "test_accuracy", "test_macro_f1",
                          "test_positive_f1", "test_negative_f1", "test_neutral_f1"):
                    ext_row[k] = float(row[k])
                combined.append(ext_row)

    combined_csv = OUTPUT_DIR / "all_models_multi_seed.csv"
    with combined_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(combined)

    summary = summarise(combined)
    summary_json = OUTPUT_DIR / "all_models_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"results": combined, "summary": summary}, f, ensure_ascii=False, indent=2)

    print(f"\n{'#' * 60}\n# Cross-model variance summary (5 seeds each)\n{'#' * 60}")
    for model_name, stats in summary.items():
        m = stats["test_macro_f1"]
        n = stats["test_neutral_f1"]
        print(f"  {model_name:25s}  Macro F1: {m['mean']:.4f} ± {m['stdev']:.4f}  "
              f"(min {m['min']:.4f}, max {m['max']:.4f})   "
              f"Neutral F1: {n['mean']:.4f} ± {n['stdev']:.4f}")

    print(f"\nSaved per-seed CSV:    {results_csv}")
    print(f"Saved combined CSV:    {combined_csv}")
    print(f"Saved JSON summary:    {summary_json}")


if __name__ == "__main__":
    main()
