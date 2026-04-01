# src/data/split.py
import csv
import random
from collections import Counter, defaultdict


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
    """Split rows by id groups while approximating sentiment distribution."""
    rng = random.Random(seed)
    split_names = ("train", "val", "test")

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["id"]].append(row)

    total_rows = len(rows)
    target_sizes = {
        "train": int(total_rows * train_ratio),
        "val": int(total_rows * val_ratio),
    }
    target_sizes["test"] = total_rows - target_sizes["train"] - target_sizes["val"]

    total_sentiments = Counter(row["sentiment"] for row in rows)
    target_sentiments = {
        "train": Counter({
            sentiment: int(count * train_ratio)
            for sentiment, count in total_sentiments.items()
        }),
        "val": Counter({
            sentiment: int(count * val_ratio)
            for sentiment, count in total_sentiments.items()
        }),
    }
    target_sentiments["test"] = Counter({
        sentiment: total_sentiments[sentiment]
        - target_sentiments["train"][sentiment]
        - target_sentiments["val"][sentiment]
        for sentiment in total_sentiments
    })

    group_items = list(grouped_rows.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_rows = {name: [] for name in split_names}
    split_sizes = Counter()
    split_sentiments = {name: Counter() for name in split_names}

    for _, group in group_items:
        group_size = len(group)
        group_sentiments = Counter(row["sentiment"] for row in group)
        best_split = None
        best_score = None

        for split_name in split_names:
            next_size = split_sizes[split_name] + group_size
            size_overflow = max(0, next_size - target_sizes[split_name])
            size_gap = abs(next_size - target_sizes[split_name])

            sentiment_overflow = 0
            sentiment_gap = 0
            for sentiment in total_sentiments:
                next_count = split_sentiments[split_name][sentiment] + group_sentiments.get(sentiment, 0)
                target_count = target_sentiments[split_name][sentiment]
                sentiment_overflow += max(0, next_count - target_count)
                sentiment_gap += abs(next_count - target_count)

            score = (
                size_overflow,
                sentiment_overflow,
                size_gap,
                sentiment_gap,
                split_sizes[split_name],
            )

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        split_rows[best_split].extend(group)
        split_sizes[best_split] += group_size
        split_sentiments[best_split].update(group_sentiments)

    for split_name in split_names:
        rng.shuffle(split_rows[split_name])

    return split_rows["train"], split_rows["val"], split_rows["test"]


if __name__ == "__main__":
    import os
    rows = load_csv("data/reviews_all.csv")
    train, val, test = stratified_split(rows)
    os.makedirs("data/final", exist_ok=True)
    save_csv(train, "data/final/train.csv")
    save_csv(val,   "data/final/val.csv")
    save_csv(test,  "data/final/test.csv")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
