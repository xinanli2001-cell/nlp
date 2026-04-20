# src/data/extended_dataset.py
"""
Extended ABSA dataset: marks aspect mention in review with [ASPECT] ... [/ASPECT].
"""

import csv
import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.data.dataset import LABEL2ID
from src.models.aspects import ASPECT_KEYWORDS, get_aspect_keywords

_TOKENIZER = None

def get_tokenizer() -> BertTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        tok.add_special_tokens({"additional_special_tokens": ["[ASPECT]", "[/ASPECT]"]})
        _TOKENIZER = tok
    return _TOKENIZER


def mark_aspect_in_text(review: str, aspect: str) -> str:
    """
    Find the first occurrence of an aspect keyword in the review and wrap it
    with [ASPECT] ... [/ASPECT]. Returns original text if not found.
    """
    keywords = get_aspect_keywords(aspect)
    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        match = pattern.search(review)
        if match:
            s, e = match.start(), match.end()
            return review[:s] + "[ASPECT] " + review[s:e] + " [/ASPECT]" + review[e:]
    return review  # fallback: no marking


class ExtendedABSADataset(Dataset):
    """[CLS] marked_review [SEP] aspect [SEP] → sentiment"""

    def __init__(self, rows: list[dict], max_len: int = 128):
        self._rows = rows
        self._max_len = max_len
        self._tok = get_tokenizer()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        marked = mark_aspect_in_text(row["review"], row["aspect"])
        enc = self._tok(
            marked,
            row["aspect"],
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self._max_len, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(LABEL2ID[row["sentiment"]], dtype=torch.long),
        }


def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
