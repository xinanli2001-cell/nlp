"""Extended training curves (10 epochs) for Extended BERT and RoBERTa.

Follow-up to train_all_models_curves.py. Standard BERT's validation and
test curves plateau by epoch 3, so it is not re-run here; Extended BERT
and RoBERTa are extended to 10 epochs to inspect whether the visible
upward trend in cross-domain macro F1 at epoch 5 continues.

All other hyperparameters match the original benchmark (lr=2e-5, batch
16, warmup 10%, inverse-frequency class weights, no weight decay).

Output: outputs/evaluation/all_models_curves/curves_10ep.csv
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import ID2LABEL, load_csv
from src.data.extended_dataset import ExtendedABSADataset
from src.data.roberta_dataset import RobertaABSADataset
from src.evaluation.metrics import compute_metrics
from src.models.extended_bert import ExtendedBertABSA
from src.models.robertaabsa import RobertaABSA

SEEDS = [
    42, 1, 7, 2024, 123,
    0, 2, 3, 5, 10,
    50, 99, 100, 314, 555,
    777, 999, 1234, 2023, 4096,
]
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 10
LR = 2e-5
WARMUP_FRAC = 0.10

OUTPUT_DIR = Path("outputs/evaluation/all_models_curves")
CURVES_CSV = OUTPUT_DIR / "curves_10ep.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class ModelConfig:
    name: str
    model_cls: type
    dataset_cls: type
    uses_token_types: bool


CONFIGS = [
    ModelConfig("Extended BERT ABSA", ExtendedBertABSA, ExtendedABSADataset, True),
    ModelConfig("RoBERTa ABSA", RobertaABSA, RobertaABSADataset, False),
]

DOMAINS = {
    "val":        Path("data/final/val.csv"),
    "in_domain":  Path("data/final/test.csv"),
    "laptop":     Path("data/semeval_laptop.csv"),
    "restaurant": Path("data/semeval_restaurant.csv"),
}


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch RNGs so a given seed reproduces the same run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward(model, batch, use_tti):
    """Call ``model`` with BERT- or RoBERTa-style inputs depending on ``use_tti``."""
    if use_tti:
        return model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["token_type_ids"].to(DEVICE),
        )
    return model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))


def evaluate(model, loader, use_tti):
    """Run inference over ``loader`` and return gold / prediction label lists."""
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = forward(model, batch, use_tti)
            preds += [ID2LABEL[i] for i in logits.argmax(dim=-1).cpu().tolist()]
            golds += [ID2LABEL[i] for i in batch["label"].cpu().tolist()]
    return golds, preds


def _load_existing() -> tuple[list[dict], set[tuple[str, int]]]:
    """Resume support: return existing rows plus (model, seed) pairs already complete."""
    if not CURVES_CSV.exists():
        return [], set()
    existing: list[dict] = []
    with CURVES_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("seed", "epoch", "n_examples"):
                r[k] = int(r[k])
            for k in ("train_loss", "accuracy", "macro_f1",
                      "positive_f1", "negative_f1", "neutral_f1"):
                r[k] = float(r[k])
            existing.append(r)
    counts: dict[tuple[str, int], int] = {}
    for r in existing:
        key = (r["model"], r["seed"])
        counts[key] = counts.get(key, 0) + 1
    needed = EPOCHS * len(DOMAINS)
    done = {k for k, n in counts.items() if n >= needed}
    return existing, done


def _write(rows: list[dict]) -> None:
    """Persist ``rows`` to the long-format per-epoch CSV."""
    if not rows:
        return
    with CURVES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(cfg: ModelConfig, seed: int, train_rows, eval_loaders) -> list[dict]:
    """Train one (model, seed) for 10 epochs and return the per-epoch evaluation rows."""
    set_seed(seed)
    train_dl = DataLoader(cfg.dataset_cls(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    model = cfg.model_cls().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRAC), total_steps,
    )
    weights = torch.tensor([1.0, 13.7, 17.0], device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    seed_rows: list[dict] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            logits = forward(model, batch, cfg.uses_token_types)
            loss = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        for domain, loader in eval_loaders.items():
            golds, preds = evaluate(model, loader, cfg.uses_token_types)
            m = compute_metrics(golds, preds)
            seed_rows.append({
                "model": cfg.name,
                "seed": seed,
                "epoch": epoch,
                "domain": domain,
                "n_examples": len(golds),
                "train_loss": round(avg_loss, 4),
                "accuracy": round(float(m["accuracy"]), 4),
                "macro_f1": round(m["macro_f1"], 4),
                "positive_f1": round(m["per_class"]["positive"], 4),
                "negative_f1": round(m["per_class"]["negative"], 4),
                "neutral_f1": round(m["per_class"]["neutral"], 4),
            })
    in_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "in_domain")
    lap_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "laptop")
    res_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "restaurant")
    val_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "val")
    print(f"  [{cfg.name:20s}] seed {seed:>4} | e{EPOCHS}: "
          f"val {val_f1:.4f} | in {in_f1:.4f} | lap {lap_f1:.4f} | res {res_f1:.4f}",
          flush=True)
    return seed_rows


def main() -> None:
    """Run the 20-seed x 2-model x 10-epoch curve collection end to end."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Models: {[c.name for c in CONFIGS]}")

    train_rows = load_csv("data/final/train.csv")
    eval_rows_by_domain = {name: load_csv(str(p)) for name, p in DOMAINS.items()}

    all_rows, done = _load_existing()
    if done:
        print(f"Resuming: skipping {len(done)} already-complete (model, seed) pairs")

    t0 = time.time()
    for seed_idx, seed in enumerate(SEEDS, start=1):
        print(f"\n=== Seed {seed}  ({seed_idx}/{len(SEEDS)}) ===")
        for cfg in CONFIGS:
            if (cfg.name, seed) in done:
                print(f"  [{cfg.name:20s}] seed {seed} — already done, skipping.")
                continue
            eval_loaders = {
                name: DataLoader(cfg.dataset_cls(rows), batch_size=BATCH_SIZE)
                for name, rows in eval_rows_by_domain.items()
            }
            rows = run_one(cfg, seed, train_rows, eval_loaders)
            all_rows.extend(rows)
            _write(all_rows)

    _write(all_rows)
    print(f"\nTotal wall time: {(time.time() - t0)/60:.1f} min")
    print(f"Saved: {CURVES_CSV}")


if __name__ == "__main__":
    main()
