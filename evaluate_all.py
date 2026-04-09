# evaluate_all.py
import csv, torch
from torch.utils.data import DataLoader

from src.models.baseline import RuleBasedABSA
from src.models.bert_absa import BertABSA
from src.models.extended_bert import ExtendedBertABSA
from src.models.robertaabsa import RobertaABSA
from src.data.dataset import ABSADataset, ID2LABEL, load_csv
from src.data.extended_dataset import ExtendedABSADataset
from src.data.roberta_dataset import RobertaABSADataset
from src.evaluation.metrics import print_report
from src.evaluation.error_analysis import analyse_errors

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
test_rows = load_csv("data/final/test.csv")
test_reviews = [r["review"] for r in test_rows]
test_aspects = [r["aspect"] for r in test_rows]
test_golds   = [r["sentiment"] for r in test_rows]

# --- Baseline ---
bl = RuleBasedABSA()
bl_preds = bl.predict_batch(test_reviews, test_aspects)
print_report(test_golds, bl_preds, "Rule-Based Baseline")

# --- Standard BERT ---
bert = BertABSA().to(DEVICE)
bert.load_state_dict(torch.load("checkpoints/bert_absa_best.pt", map_location=DEVICE))
bert.eval()
test_dl = DataLoader(ABSADataset(test_rows), batch_size=32)
golds, preds = [], []
with torch.no_grad():
    for b in test_dl:
        out = bert(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["token_type_ids"].to(DEVICE))
        preds += [ID2LABEL[i] for i in out.argmax(-1).cpu().tolist()]
        golds += [ID2LABEL[i] for i in b["label"].cpu().tolist()]
print_report(golds, preds, "Standard BERT ABSA")

# --- Extended BERT ---
ext = ExtendedBertABSA().to(DEVICE)
ext.load_state_dict(torch.load("checkpoints/extended_bert_best.pt", map_location=DEVICE))
ext.eval()
test_dl2 = DataLoader(ExtendedABSADataset(test_rows), batch_size=32)
golds2, preds2 = [], []
with torch.no_grad():
    for b in test_dl2:
        out = ext(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["token_type_ids"].to(DEVICE))
        preds2 += [ID2LABEL[i] for i in out.argmax(-1).cpu().tolist()]
        golds2 += [ID2LABEL[i] for i in b["label"].cpu().tolist()]
print_report(golds2, preds2, "Extended BERT ABSA")

# --- RoBERTa ---
rob = RobertaABSA().to(DEVICE)
rob.load_state_dict(torch.load("checkpoints/roberta_absa_best.pt", map_location=DEVICE))
rob.eval()
test_dl3 = DataLoader(RobertaABSADataset(test_rows), batch_size=32)
golds3, preds3 = [], []
with torch.no_grad():
    for b in test_dl3:
        out = rob(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        preds3 += [ID2LABEL[i] for i in out.argmax(-1).cpu().tolist()]
        golds3 += [ID2LABEL[i] for i in b["label"].cpu().tolist()]
print_report(golds3, preds3, "RoBERTa ABSA")

# Error analysis (requires src/evaluation/error_analysis.py — created in Task 9)
try:
    analyse_errors(test_reviews, test_aspects, test_golds, bl_preds,  "Rule-Based Baseline")
    analyse_errors(test_reviews, test_aspects, golds,      preds,     "Standard BERT")
    analyse_errors(test_reviews, test_aspects, golds2,     preds2,    "Extended BERT")
    analyse_errors(test_reviews, test_aspects, golds3,     preds3,    "RoBERTa")
except ImportError:
    print("Error analysis module not yet available.")
