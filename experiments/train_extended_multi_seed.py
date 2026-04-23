"""Multi-seed training of Extended BERT ABSA for variance analysis.

Trains a fresh Extended BERT model for each seed in SEEDS. For every seed:
- Saves the best-val-macro-F1 checkpoint to checkpoints/extended_bert_seed_{seed}.pt
- Evaluates that checkpoint on the held-out test set
- Records both val and test metrics

The final summary (mean ± std across seeds) quantifies training variance
with a 569-example training set, directly supporting the report's
"training stochasticity dominates architectural choice" finding.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import ID2LABEL
from src.data.extended_dataset import ExtendedABSADataset, load_csv
from src.evaluation.metrics import compute_metrics
from src.models.extended_bert import ExtendedBertABSA

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS = [42, 1, 7, 2024, 123]
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 5
LR = 2e-5
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs/evaluation/multi_seed")
SUMMARY_CSV = OUTPUT_DIR / "extended_bert_multi_seed.csv"
SUMMARY_JSON = OUTPUT_DIR / "extended_bert_multi_seed.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Fix all sources of randomness to make a run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: torch.nn.Module, loader: DataLoader) -> tuple[list[str], list[str]]:
    """Run inference and return gold/pred label sequences."""
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE),
            )
            preds += [ID2LABEL[p] for p in logits.argmax(dim=-1).cpu().tolist()]
            golds += [ID2LABEL[g] for g in batch["label"].cpu().tolist()]
    return golds, preds


def train_one_seed(seed: int, train_rows, val_rows, test_rows) -> dict:
    """Train Extended BERT with a fixed seed and return val+test metrics."""
    set_seed(seed)

    train_dl = DataLoader(ExtendedABSADataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(ExtendedABSADataset(val_rows), batch_size=BATCH_SIZE)
    test_dl = DataLoader(ExtendedABSADataset(test_rows), batch_size=BATCH_SIZE)

    model = ExtendedBertABSA().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )
    weights = torch.tensor([1.0, 13.7, 17.0], device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    save_path = CHECKPOINT_DIR / f"extended_bert_seed_{seed}.pt"
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE),
            )
            loss = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        golds, preds = evaluate(model, val_dl)
        val_metrics = compute_metrics(golds, preds)
        print(f"  seed {seed} | epoch {epoch}/{EPOCHS} | "
              f"train_loss={total_loss/len(train_dl):.4f} | val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Persist best checkpoint and reload for clean test evaluation.
    assert best_state is not None, "No best epoch — should never happen"
    torch.save(best_state, save_path)
    model.load_state_dict(best_state)
    test_golds, test_preds = evaluate(model, test_dl)
    test_metrics = compute_metrics(test_golds, test_preds)
    print(f"  seed {seed} | SAVED {save_path.name} | test_f1={test_metrics['macro_f1']:.4f}")

    return {
        "seed": seed,
        "val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_positive_f1": round(test_metrics["per_class"]["positive"], 4),
        "test_negative_f1": round(test_metrics["per_class"]["negative"], 4),
        "test_neutral_f1": round(test_metrics["per_class"]["neutral"], 4),
    }


def summarise(results: list[dict]) -> dict:
    """Compute mean and stdev over seeds for each numeric metric."""
    keys = [k for k in results[0] if k != "seed"]
    summary = {}
    for k in keys:
        values = [r[k] for r in results]
        summary[k] = {
            "mean": round(statistics.mean(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
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
    print(f"Seeds: {SEEDS}\n")

    results = []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        results.append(train_one_seed(seed, train_rows, val_rows, test_rows))

    summary = summarise(results)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, ensure_ascii=False, indent=2)

    print(f"\n{'#' * 60}\n# Multi-seed summary (across {len(SEEDS)} seeds)\n{'#' * 60}")
    for metric, stats in summary.items():
        print(f"  {metric:22s}  mean={stats['mean']:.4f}  std={stats['stdev']:.4f}  "
              f"min={stats['min']:.4f}  max={stats['max']:.4f}")
    print(f"\nSaved summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
