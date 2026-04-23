"""Continue training Extended BERT ABSA from the current best checkpoint.

Loads checkpoints/extended_bert_best.pt and trains for EXTRA_EPOCHS more
epochs with a fresh optimizer + scheduler. Saves when val macro F1 beats
the starting checkpoint's val F1.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import ID2LABEL
from src.data.extended_dataset import ExtendedABSADataset, load_csv
from src.models.extended_bert import ExtendedBertABSA
from src.evaluation.metrics import compute_metrics

BATCH_SIZE   = 16
MAX_LEN      = 128
EXTRA_EPOCHS = 5
LR           = 2e-5
LOAD_PATH    = "checkpoints/extended_bert_best.pt"
SAVE_PATH    = "checkpoints/extended_bert_best.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

train_ds = ExtendedABSADataset(load_csv("data/final/train.csv"), max_len=MAX_LEN)
val_ds   = ExtendedABSADataset(load_csv("data/final/val.csv"),   max_len=MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

model = ExtendedBertABSA().to(DEVICE)
model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
print(f"Loaded starting checkpoint from {LOAD_PATH}")

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_dl) * EXTRA_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

weights = torch.tensor([1.0, 13.7, 17.0], device=DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

# Measure val macro F1 of the starting checkpoint first.
model.eval()
golds, preds = [], []
with torch.no_grad():
    for batch in val_dl:
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["token_type_ids"].to(DEVICE),
        )
        preds += [ID2LABEL[p] for p in logits.argmax(dim=-1).cpu().tolist()]
        golds += [ID2LABEL[g] for g in batch["label"].cpu().tolist()]
best_f1 = compute_metrics(golds, preds)["macro_f1"]
print(f"Starting val macro F1: {best_f1:.4f}")

for epoch in range(1, EXTRA_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_dl:
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["token_type_ids"].to(DEVICE),
        )
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
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE),
            )
            preds += [ID2LABEL[p] for p in logits.argmax(dim=-1).cpu().tolist()]
            golds += [ID2LABEL[g] for g in batch["label"].cpu().tolist()]

    result = compute_metrics(golds, preds)
    print(f"Epoch {epoch}/{EXTRA_EPOCHS} | loss={total_loss/len(train_dl):.4f} | val macro_f1={result['macro_f1']:.4f}")

    if result["macro_f1"] > best_f1:
        best_f1 = result["macro_f1"]
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → Saved improved model (f1={best_f1:.4f})")

print(f"\nFinal best val macro F1: {best_f1:.4f}")
