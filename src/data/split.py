# src/data/split.py
import csv
import random
from collections import defaultdict


def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = [k for k in rows[0].keys() if k is not None]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(
    rows: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratify by sentiment to preserve class distribution."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["sentiment"]].append(row)

    train, val, test = [], [], []
    for sentiment, bucket in buckets.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train += bucket[:n_train]
        val   += bucket[n_train:n_train + n_val]
        test  += bucket[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


if __name__ == "__main__":
    import os
    rows = load_csv("data/reviews_all.csv")
    train, val, test = stratified_split(rows)
    os.makedirs("data/final", exist_ok=True)
    save_csv(train, "data/final/train.csv")
    save_csv(val,   "data/final/val.csv")
    save_csv(test,  "data/final/test.csv")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
