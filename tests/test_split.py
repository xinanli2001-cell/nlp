# tests/test_split.py
import os
import csv
import tempfile
import pytest
from src.data.split import stratified_split

def test_split_sizes():
    rows = [{"id": str(i), "review": f"r{i}", "aspect": "battery",
             "sentiment": ["positive","negative","neutral"][i % 3]} for i in range(100)]
    train, val, test = stratified_split(rows, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert abs(len(train) - 70) <= 5
    assert abs(len(val) - 15) <= 5
    assert abs(len(test) - 15) <= 5
    assert len(train) + len(val) + len(test) == 100

def test_no_id_overlap():
    rows = [{"id": str(i), "review": f"r{i}", "aspect": "usability",
             "sentiment": "positive"} for i in range(90)]
    train, val, test = stratified_split(rows, train_ratio=0.7, val_ratio=0.15, seed=0)
    train_ids = {r["id"] for r in train}
    val_ids   = {r["id"] for r in val}
    test_ids  = {r["id"] for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
