# train_extended.py
"""Train extended BERT ABSA. Run: python train_extended.py"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data.extended_dataset import ExtendedABSADataset, load_csv
from src.data.dataset import ID2LABEL
from src.models.extended_bert import ExtendedBertABSA
from src.evaluation.metrics import compute_metrics

BATCH_SIZE  = 16
MAX_LEN     = 128
EPOCHS      = 5
LR          = 2e-5
SAVE_PATH   = "checkpoints/extended_bert_best.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

train_ds = ExtendedABSADataset(load_csv("data/final/train.csv"), max_len=MAX_LEN)
val_ds   = ExtendedABSADataset(load_csv("data/final/val.csv"),   max_len=MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

model = ExtendedBertABSA().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

weights = torch.tensor([1.0, 13.7, 17.0], device=DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

os.makedirs("checkpoints", exist_ok=True)
best_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_dl:
        optimizer.zero_grad()
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["token_type_ids"].to(DEVICE))
        loss = criterion(logits, batch["label"].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for batch in val_dl:
            logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["token_type_ids"].to(DEVICE))
            preds += [ID2LABEL[i] for i in logits.argmax(-1).cpu().tolist()]
            golds += [ID2LABEL[i] for i in batch["label"].cpu().tolist()]

    result = compute_metrics(golds, preds)
    print(f"Epoch {epoch}/{EPOCHS} | loss={total_loss/len(train_dl):.4f} | val macro_f1={result['macro_f1']:.4f}")

    if result["macro_f1"] > best_f1:
        best_f1 = result["macro_f1"]
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → Saved best model (f1={best_f1:.4f})")

print(f"\nBest val macro F1: {best_f1:.4f}")
