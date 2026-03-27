# evaluate_baseline.py
import csv
from src.models.baseline import RuleBasedABSA
from src.evaluation.metrics import print_report

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

model = RuleBasedABSA()
test_rows = load_csv("data/final/test.csv")
reviews = [r["review"] for r in test_rows]
aspects = [r["aspect"]  for r in test_rows]
golds   = [r["sentiment"] for r in test_rows]

preds = model.predict_batch(reviews, aspects)
print_report(golds, preds, model_name="Rule-Based Baseline")
