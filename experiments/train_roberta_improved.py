"""Improved RoBERTa ABSA training with per-epoch val + test curves.

Changes vs the baseline RoBERTa training (train_roberta.py / train_full_benchmark.py):
- learning rate 2e-5 → 1e-5 (halved for stability on small dataset)
- weight_decay 0 → 0.01 (AdamW regularisation)
- epochs 5 → 10
- Every epoch evaluates val + three test domains, writing one CSV row per
  (seed, epoch, domain). This produces full learning curves (not just peak)
  for post-hoc convergence / overfitting analysis.

Outputs:
- checkpoints/roberta_improved_seed_{seed}.pt  (per-seed best-val-F1 weights)
- outputs/evaluation/roberta_improved/roberta_curves.csv
  (one row per (seed, epoch, domain) — 20 seeds * 10 epochs * 4 domains = 800 rows)
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import ID2LABEL, load_csv
from src.data.roberta_dataset import RobertaABSADataset
from src.evaluation.metrics import compute_metrics
from src.models.robertaabsa import RobertaABSA

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS = [
    42, 1, 7, 2024, 123,
    0, 2, 3, 5, 10,
    50, 99, 100, 314, 555,
    777, 999, 1234, 2023, 4096,
]
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 10
LR = 1e-5                # was 2e-5
WEIGHT_DECAY = 0.01      # was 0
WARMUP_FRAC = 0.10

CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs/evaluation/roberta_improved")
CURVES_CSV = OUTPUT_DIR / "roberta_curves.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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


def forward(model, batch):
    return model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))


def evaluate(model, loader) -> tuple[list[str], list[str]]:
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = forward(model, batch)
            preds += [ID2LABEL[i] for i in logits.argmax(dim=-1).cpu().tolist()]
            golds += [ID2LABEL[i] for i in batch["label"].cpu().tolist()]
    return golds, preds


def _load_existing_rows() -> tuple[list[dict], set[int]]:
    """Resume: load existing CSV and collect seeds whose 10 epochs × 4 domains are all present."""
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
    counts: dict[int, int] = {}
    for r in existing:
        counts[r["seed"]] = counts.get(r["seed"], 0) + 1
    needed = EPOCHS * len(DOMAINS)
    done_seeds = {s for s, n in counts.items() if n >= needed}
    return existing, done_seeds


def _write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with CURVES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed(seed: int, all_rows: list[dict], dataloaders: dict) -> None:
    set_seed(seed)

    train_dl = DataLoader(
        RobertaABSADataset(dataloaders["_train_rows"]),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    model = RobertaABSA().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_FRAC),
        num_training_steps=total_steps,
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
            logits = forward(model, batch)
            loss = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dl)

        epoch_rows = []
        for domain in DOMAINS:
            loader = dataloaders[domain]
            golds, preds = evaluate(model, loader)
            m = compute_metrics(golds, preds)
            epoch_rows.append({
                "seed": seed,
                "epoch": epoch,
                "domain": domain,
                "n_examples": len(golds),
                "train_loss": round(avg_train_loss, 4),
                "accuracy": round(float(m["accuracy"]), 4),
                "macro_f1": round(m["macro_f1"], 4),
                "positive_f1": round(m["per_class"]["positive"], 4),
                "negative_f1": round(m["per_class"]["negative"], 4),
                "neutral_f1": round(m["per_class"]["neutral"], 4),
            })

        val_f1 = next(r["macro_f1"] for r in epoch_rows if r["domain"] == "val")
        in_f1 = next(r["macro_f1"] for r in epoch_rows if r["domain"] == "in_domain")
        lap_f1 = next(r["macro_f1"] for r in epoch_rows if r["domain"] == "laptop")
        res_f1 = next(r["macro_f1"] for r in epoch_rows if r["domain"] == "restaurant")
        print(f"  seed {seed:>4} | epoch {epoch:>2}/{EPOCHS} | "
              f"loss {avg_train_loss:.4f} | val {val_f1:.4f} | "
              f"in {in_f1:.4f} | lap {lap_f1:.4f} | res {res_f1:.4f}", flush=True)

        all_rows.extend(epoch_rows)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        _write_csv(all_rows)

    if best_state is not None:
        save_path = CHECKPOINT_DIR / f"roberta_improved_seed_{seed}.pt"
        torch.save(best_state, save_path)
        print(f"  seed {seed:>4} | saved {save_path.name} (best val F1 {best_val_f1:.4f})", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Hyperparameters: lr={LR}, weight_decay={WEIGHT_DECAY}, "
          f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, warmup={WARMUP_FRAC}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")

    train_rows = load_csv(str(DOMAINS["val"]).replace("val", "train"))
    dataloaders: dict = {"_train_rows": train_rows}
    for name, path in DOMAINS.items():
        rows = load_csv(str(path))
        dataloaders[name] = DataLoader(RobertaABSADataset(rows), batch_size=BATCH_SIZE)
        print(f"  {name}: {len(rows)}")

    all_rows, done_seeds = _load_existing_rows()
    if done_seeds:
        print(f"Resuming: skipping {len(done_seeds)} already-complete seeds: {sorted(done_seeds)}")

    t0 = time.time()
    for i, seed in enumerate(SEEDS, start=1):
        if seed in done_seeds:
            print(f"\n=== Seed {seed}  ({i}/{len(SEEDS)})  — already done, skipping ===")
            continue
        print(f"\n=== Seed {seed}  ({i}/{len(SEEDS)}) ===")
        run_seed(seed, all_rows, dataloaders)

    _write_csv(all_rows)
    print(f"\nTotal wall time: {(time.time() - t0)/60:.1f} min")
    print(f"Saved: {CURVES_CSV}")


if __name__ == "__main__":
    main()
