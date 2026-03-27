# tests/test_bert_dataset.py
import pytest
from src.data.dataset import ABSADataset, LABEL2ID

def test_label_mapping():
    assert LABEL2ID["positive"] == 0
    assert LABEL2ID["negative"] == 1
    assert LABEL2ID["neutral"]  == 2

def test_dataset_len():
    rows = [
        {"review": "Great battery life!", "aspect": "battery", "sentiment": "positive"},
        {"review": "Screen is awful.",    "aspect": "screen",  "sentiment": "negative"},
    ]
    ds = ABSADataset(rows, max_len=64)
    assert len(ds) == 2

def test_dataset_item_keys():
    rows = [{"review": "Good price.", "aspect": "price", "sentiment": "positive"}]
    ds = ABSADataset(rows, max_len=64)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item
    assert item["label"] == 0  # positive → 0

def test_input_ids_length():
    rows = [{"review": "Amazing product.", "aspect": "overall", "sentiment": "neutral"}]
    ds = ABSADataset(rows, max_len=32)
    item = ds[0]
    assert item["input_ids"].shape[0] == 32
