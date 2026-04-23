"""Record per-epoch val + test curves for the three original BERT-family models.

Uses the same hyperparameters as train_full_benchmark.py (lr=2e-5,
5 epochs, no weight decay, batch 16) so that the resulting curves match
the canonical 20-seed benchmark in full_benchmark.csv. The only change
is that every epoch we evaluate on val AND the three test domains and
write one CSV row per (model, seed, epoch, domain).

No per-seed checkpoints are re-saved — the weights produced by this run
are functionally identical to those already in checkpoints/ from the
original benchmark.

Output: outputs/evaluation/all_models_curves/curves.csv
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

from src.data.dataset import ABSADataset, ID2LABEL, load_csv
from src.data.extended_dataset import ExtendedABSADataset
from src.data.roberta_dataset import RobertaABSADataset
from src.evaluation.metrics import compute_metrics
from src.models.bert_absa import BertABSA
from src.models.extended_bert import ExtendedBertABSA
from src.models.robertaabsa import RobertaABSA

# ── Config (matches train_full_benchmark.py) ────────────────────────────────
SEEDS = [
    42, 1, 7, 2024, 123,
    0, 2, 3, 5, 10,
    50, 99, 100, 314, 555,
    777, 999, 1234, 2023, 4096,
]
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 5
LR = 2e-5
WARMUP_FRAC = 0.10

OUTPUT_DIR = Path("outputs/evaluation/all_models_curves")
CURVES_CSV = OUTPUT_DIR / "curves.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class ModelConfig:
    name: str
    model_cls: type
    dataset_cls: type
    uses_token_types: bool


CONFIGS = [
    ModelConfig("Standard BERT ABSA", BertABSA, ABSADataset, True),
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward(model, batch, use_tti):
    if use_tti:
        return model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["token_type_ids"].to(DEVICE),
        )
    return model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))


def evaluate(model, loader, use_tti):
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = forward(model, batch, use_tti)
            preds += [ID2LABEL[i] for i in logits.argmax(dim=-1).cpu().tolist()]
            golds += [ID2LABEL[i] for i in batch["label"].cpu().tolist()]
    return golds, preds


def _load_existing() -> tuple[list[dict], set[tuple[str, int]]]:
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
    if not rows:
        return
    with CURVES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(cfg: ModelConfig, seed: int, train_rows, eval_loaders) -> list[dict]:
    set_seed(seed)
    train_dl = DataLoader(cfg.dataset_cls(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    model = cfg.model_cls().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)  # no weight_decay — match original
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
    val_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "val")
    in_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "in_domain")
    lap_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "laptop")
    res_f1 = next(r["macro_f1"] for r in seed_rows if r["epoch"] == EPOCHS and r["domain"] == "restaurant")
    print(f"  [{cfg.name:20s}] seed {seed:>4} | e{EPOCHS}: "
          f"val {val_f1:.4f} | in {in_f1:.4f} | lap {lap_f1:.4f} | res {res_f1:.4f}",
          flush=True)
    return seed_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")

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
