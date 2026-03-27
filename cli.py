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
import torch

from src.data.dataset import ID2LABEL, load_csv
from src.models.baseline import RuleBasedABSA


def load_bert(model_name: str, device: str):
    if model_name == "bert":
        from src.models.bert_absa import BertABSA
        from src.data.dataset import ABSADataset
        model = BertABSA().to(device)
        model.load_state_dict(torch.load("checkpoints/bert_absa_best.pt", map_location=device))
        return model, ABSADataset
    elif model_name == "extended":
        from src.models.extended_bert import ExtendedBertABSA
        from src.data.extended_dataset import ExtendedABSADataset
        model = ExtendedBertABSA().to(device)
        model.load_state_dict(torch.load("checkpoints/extended_bert_best.pt", map_location=device))
        return model, ExtendedABSADataset
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: baseline, bert, extended")


def predict_single(review: str, aspect: str, model_name: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if model_name == "baseline":
        return RuleBasedABSA().predict(review, aspect)

    model, DatasetClass = load_bert(model_name, device)
    model.eval()
    from torch.utils.data import DataLoader
    ds = DatasetClass([{"review": review, "aspect": aspect, "sentiment": "positive"}])
    batch = next(iter(DataLoader(ds, batch_size=1)))
    with torch.no_grad():
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["token_type_ids"].to(device))
    return ID2LABEL[logits.argmax(-1).item()]


def predict_file(input_path: str, output_path: str, model_name: str) -> None:
    rows = load_csv(input_path)
    for row in rows:
        row["predicted"] = predict_single(row["review"], row["aspect"], model_name)
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
