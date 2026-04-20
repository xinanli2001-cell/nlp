# cli.py
"""
CLI for ABSA prediction.

Usage:
  python cli.py --review "The battery life is great." --aspect battery --model extended
  python cli.py --input_file data/final/test.csv --model bert --output results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ID2LABEL, load_csv
from src.models.baseline import RuleBasedABSA

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Module-level cache: model loaded once per process
_model_cache: dict = {}
_baseline_cache = None


def _get_baseline() -> RuleBasedABSA:
    global _baseline_cache
    if _baseline_cache is None:
        _baseline_cache = RuleBasedABSA()
    return _baseline_cache


def _get_bert(model_name: str):
    if model_name not in _model_cache:
        if model_name == "bert":
            from src.models.bert_absa import BertABSA
            from src.data.dataset import ABSADataset
            checkpoint_path = Path("checkpoints/bert_absa_best.pt")
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    "Missing checkpoint checkpoints/bert_absa_best.pt. "
                    "Train the model first or place the checkpoint in the checkpoints/ directory."
                )
            model = BertABSA().to(DEVICE)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            _model_cache[model_name] = (model, ABSADataset)
        elif model_name == "extended":
            from src.models.extended_bert import ExtendedBertABSA
            from src.data.extended_dataset import ExtendedABSADataset
            checkpoint_path = Path("checkpoints/extended_bert_best.pt")
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    "Missing checkpoint checkpoints/extended_bert_best.pt. "
                    "Train the model first or place the checkpoint in the checkpoints/ directory."
                )
            model = ExtendedBertABSA().to(DEVICE)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            _model_cache[model_name] = (model, ExtendedABSADataset)
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose: baseline, bert, extended")
    return _model_cache[model_name]


def predict_single(review: str, aspect: str, model_name: str) -> str:
    if model_name == "baseline":
        return _get_baseline().predict(review, aspect)

    model, DatasetClass = _get_bert(model_name)
    model.eval()
    ds = DatasetClass([{"review": review, "aspect": aspect, "sentiment": "positive"}])
    batch = next(iter(DataLoader(ds, batch_size=1)))
    with torch.no_grad():
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["token_type_ids"].to(DEVICE))
    return ID2LABEL[logits.argmax(-1).item()]


def predict_file(input_path: str, output_path: str, model_name: str) -> None:
    rows = load_csv(input_path)
    # Load model once, then run batch inference
    if model_name == "baseline":
        bl = _get_baseline()
        for row in rows:
            row["predicted"] = bl.predict(row["review"], row["aspect"])
    else:
        model, DatasetClass = _get_bert(model_name)
        model.eval()
        # Reuse the training dataset for tokenization, but inference inputs
        # may not carry gold sentiments.
        inference_rows = [
            {**row, "sentiment": row.get("sentiment", "positive")}
            for row in rows
        ]
        ds = DatasetClass(inference_rows)
        dl = DataLoader(ds, batch_size=32)
        all_preds = []
        with torch.no_grad():
            for batch in dl:
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["token_type_ids"].to(DEVICE))
                all_preds += [ID2LABEL[i] for i in logits.argmax(-1).cpu().tolist()]
        for row, pred in zip(rows, all_preds):
            row["predicted"] = pred
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ABSA Prediction CLI")
    parser.add_argument("--review",      type=str, help="Single review text")
    parser.add_argument("--aspect",      type=str, help="Aspect category (e.g. battery, screen)")
    parser.add_argument("--input_file",  type=str, help="CSV file with review,aspect columns")
    parser.add_argument("--output",      type=str, default="predictions.csv")
    parser.add_argument("--model",       type=str, default="extended",
                        choices=["baseline", "bert", "extended"],
                        help="Model to use (default: extended)")
    args = parser.parse_args()

    if args.review and args.aspect:
        result = predict_single(args.review, args.aspect, args.model)
        print(f"Sentiment: {result}")
    elif args.input_file:
        predict_file(args.input_file, args.output, args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
