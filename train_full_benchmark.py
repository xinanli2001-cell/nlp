"""Full multi-seed benchmark: 20 seeds × 3 BERT variants × 3 evaluation domains.

For every (model, seed) combination:
- Train a fresh model
- Save the best-val-macro-F1 checkpoint as checkpoints/{prefix}_seed_{seed}.pt
- Evaluate that checkpoint on three domains:
    * in_domain  — data/final/test.csv              (123 examples)
    * laptop     — data/semeval_laptop.csv          (2313 examples)
    * restaurant — data/semeval_restaurant.csv      (3602 examples)

Outputs:
- outputs/evaluation/multi_seed_full/full_benchmark.csv (long format — one row per
  (model, seed, domain); usable for per-cell mean/std analysis)
- outputs/evaluation/multi_seed_full/full_benchmark_summary.json
  (per-(model, domain) mean/std/min/max for every metric)
"""

from __future__ import annotations

import csv
import json
import random
import statistics
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
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs/evaluation/multi_seed_full")
LONG_CSV = OUTPUT_DIR / "full_benchmark.csv"
SUMMARY_JSON = OUTPUT_DIR / "full_benchmark_summary.json"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class ModelConfig:
    name: str
    model_cls: type
    dataset_cls: type
    ckpt_prefix: str
    uses_token_types: bool


CONFIGS = [
    ModelConfig("Standard BERT ABSA", BertABSA, ABSADataset, "bert_absa", True),
    ModelConfig("Extended BERT ABSA", ExtendedBertABSA, ExtendedABSADataset, "extended_bert", True),
    ModelConfig("RoBERTa ABSA", RobertaABSA, RobertaABSADataset, "roberta_absa", False),
]

DOMAINS = {
    "in_domain": Path("data/final/test.csv"),
    "laptop": Path("data/semeval_laptop.csv"),
    "restaurant": Path("data/semeval_restaurant.csv"),
}


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch RNGs so one seed reproduces the same run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward(model, batch, use_tti):
    """Call ``model`` with the right input arity for BERT vs RoBERTa.

    BERT-family models take ``token_type_ids`` as the third argument;
    RoBERTa does not. ``use_tti`` picks between the two call signatures.
    """
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


def train_one(cfg: ModelConfig, seed: int, train_rows, val_rows):
    """Train one (model, seed) run and save the best-val-macro-F1 checkpoint.

    Returns ``(model, best_val_macro_f1, checkpoint_path)``. The model object is
    loaded with the best state so the caller can evaluate without reloading.
    """
    set_seed(seed)
    train_dl = DataLoader(cfg.dataset_cls(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(cfg.dataset_cls(val_rows), batch_size=BATCH_SIZE)
    model = cfg.model_cls().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )
    weights = torch.tensor([1.0, 13.7, 17.0], device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_f1 = 0.0
    best_state = None
    for _ in range(EPOCHS):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            logits = forward(model, batch, cfg.uses_token_types)
            loss = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        golds, preds = evaluate(model, val_dl, cfg.uses_token_types)
        f1 = compute_metrics(golds, preds)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    ckpt_path = CHECKPOINT_DIR / f"{cfg.ckpt_prefix}_seed_{seed}.pt"
    torch.save(best_state, ckpt_path)
    model.load_state_dict(best_state)
    return model, best_f1, ckpt_path


def eval_all_domains(cfg, model, val_macro_f1, seed, domain_data):
    """Evaluate one checkpoint on every domain in ``domain_data`` and build CSV rows."""
    rows = []
    for domain_name, rows_in_domain in domain_data.items():
        loader = DataLoader(cfg.dataset_cls(rows_in_domain), batch_size=32)
        golds, preds = evaluate(model, loader, cfg.uses_token_types)
        m = compute_metrics(golds, preds)
        rows.append({
            "model": cfg.name,
            "seed": seed,
            "domain": domain_name,
            "n_examples": len(golds),
            "val_macro_f1": round(val_macro_f1, 4),
            "accuracy": round(float(m["accuracy"]), 4),
            "macro_f1": round(m["macro_f1"], 4),
            "positive_f1": round(m["per_class"]["positive"], 4),
            "negative_f1": round(m["per_class"]["negative"], 4),
            "neutral_f1": round(m["per_class"]["neutral"], 4),
        })
    return rows


def _load_existing_rows() -> tuple[list[dict], set[tuple[str, int]]]:
    """Load previously saved rows so we can resume after a crash.

    Returns (existing_rows, done_pairs) where done_pairs is the set of
    (model_name, seed) tuples whose three-domain evaluation is already present.
    """
    if not LONG_CSV.exists():
        return [], set()
    existing: list[dict] = []
    with LONG_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in ("seed", "n_examples"):
                row[k] = int(row[k])
            for k in ("val_macro_f1", "accuracy", "macro_f1",
                      "positive_f1", "negative_f1", "neutral_f1"):
                row[k] = float(row[k])
            existing.append(row)

    domain_count: dict[tuple[str, int], int] = {}
    for r in existing:
        key = (r["model"], r["seed"])
        domain_count[key] = domain_count.get(key, 0) + 1
    # A (model, seed) is only considered "done" when all three domains logged.
    done = {key for key, n in domain_count.items() if n >= len(DOMAINS)}
    return existing, done


def main() -> None:
    """Run the 20-seed x 3-model x 3-domain benchmark end to end (resumable)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_csv("data/final/train.csv")
    val_rows = load_csv("data/final/val.csv")
    domain_data = {name: load_csv(str(p)) for name, p in DOMAINS.items()}
    print(f"Device: {DEVICE}")
    print(f"Data: {len(train_rows)} train / {len(val_rows)} val")
    for n, d in domain_data.items():
        print(f"  {n}: {len(d)}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")

    all_rows, done_pairs = _load_existing_rows()
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} (model, seed) pairs already complete, skipping them.")

    t0 = time.time()

    for seed_idx, seed in enumerate(SEEDS, start=1):
        print(f"\n=== Seed {seed}  ({seed_idx}/{len(SEEDS)}) ===")
        for cfg in CONFIGS:
            if (cfg.name, seed) in done_pairs:
                print(f"  [{cfg.name:20s}] seed {seed} — already done, skipping.")
                continue
            t_train_start = time.time()
            model, val_f1, ckpt = train_one(cfg, seed, train_rows, val_rows)
            t_train = time.time() - t_train_start

            t_eval_start = time.time()
            rows = eval_all_domains(cfg, model, val_f1, seed, domain_data)
            t_eval = time.time() - t_eval_start

            for r in rows:
                all_rows.append(r)
            print(f"  [{cfg.name:20s}] val_f1={val_f1:.4f}  "
                  f"in={rows[0]['macro_f1']:.4f}  "
                  f"lap={rows[1]['macro_f1']:.4f}  "
                  f"res={rows[2]['macro_f1']:.4f}  "
                  f"train={t_train:.1f}s eval={t_eval:.1f}s  "
                  f"saved {ckpt.name}")

            # Flush partial CSV after every (seed, model) combination so a crash
            # mid-run still leaves usable data.
            _write_long_csv(all_rows)

    _write_long_csv(all_rows)
    _write_summary_json(all_rows)
    print(f"\nTotal wall time: {(time.time() - t0)/60:.1f} min")
    print(f"Saved long CSV:   {LONG_CSV}")
    print(f"Saved summary:    {SUMMARY_JSON}")


def _write_long_csv(rows: list[dict]) -> None:
    """Persist ``rows`` as a long-format CSV (one line per model/seed/domain)."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with LONG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(rows: list[dict]) -> None:
    """Write per-(model, domain) mean, stdev, min and max across seeds to JSON."""
    if not rows:
        return
    numeric_keys = ["val_macro_f1", "accuracy", "macro_f1",
                    "positive_f1", "negative_f1", "neutral_f1"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["model"], r["domain"]), []).append(r)

    summary: dict[str, dict[str, dict]] = {}
    for (model_name, domain_name), subset in grouped.items():
        stats: dict[str, dict[str, float]] = {}
        for k in numeric_keys:
            vals = [r[k] for r in subset]
            stats[k] = {
                "mean": round(statistics.mean(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "n": len(vals),
            }
        summary.setdefault(model_name, {})[domain_name] = stats

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump({"n_seeds": len(SEEDS), "seeds": SEEDS, "summary": summary},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
