# src/data/roberta_dataset.py
import csv
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

_TOKENIZER = None


def get_tokenizer() -> RobertaTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
    return _TOKENIZER


class RobertaABSADataset(Dataset):
    """Standard RoBERTa ABSA: [CLS] review [SEP] aspect [SEP]"""

    def __init__(self, rows: list[dict], max_len: int = 128):
        self._rows = rows
        self._max_len = max_len
        self._tok = get_tokenizer()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        enc = self._tok(
            row["review"],
            row["aspect"],
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(LABEL2ID[row["sentiment"]], dtype=torch.long),
        }


def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
