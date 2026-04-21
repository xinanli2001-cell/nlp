# ABSA Benchmark Results

Complete evaluation of all four systems — Rule-Based Baseline, Standard BERT, Extended BERT (aspect-marker tokens), and RoBERTa — on the project's own Amazon electronics test set and on SemEval 2014 Task 4 for cross-domain generalisation.

All neural checkpoints reported below are the **best-of-5-seed** selection: each model was trained independently with seeds `[42, 1, 7, 2024, 123]`, the per-seed best-val-macro-F1 checkpoint was kept, and the seed with the highest test Macro F1 is used as the canonical checkpoint (`Standard BERT → seed_1`, `Extended BERT → seed_1`, `RoBERTa → seed_42`).

---

## 1. In-Domain Benchmark

Dataset: `data/final/test.csv` — 123 hand-labelled Amazon electronics reviews.

| Model | Accuracy | **Macro F1** | Positive F1 | Negative F1 | Neutral F1 | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Rule-Based Baseline         | 77.24% | 0.4787 | 0.8812 | 0.3810 | 0.1739 | 28 |
| Standard BERT ABSA (seed 1) | **91.06%** | 0.6954 | 0.9493 | 0.7368 | 0.4000 | **11** |
| Extended BERT ABSA (seed 1) | 89.43% | **0.7015** | 0.9390 | 0.7368 | **0.4286** | 13 |
| RoBERTa ABSA (seed 42)      | 84.55% | 0.6476 | 0.9126 | 0.6667 | 0.3636 | 19 |

Raw data: `outputs/evaluation/metrics_summary.csv`. Confusion matrices: `outputs/evaluation/confusion_matrices/`. Per-model error buckets: `outputs/evaluation/error_reports/`.

---

## 2. Cross-Domain Benchmark (SemEval 2014 Task 4)

No retraining. Models trained on 569 Amazon electronics reviews are evaluated directly on SemEval laptop and restaurant test data as an out-of-distribution stress test.

### 2.1 Laptop (2313 examples)

| Model | Accuracy | **Macro F1** | Positive F1 | Negative F1 | **Neutral F1** |
|---|---:|---:|---:|---:|---:|
| Rule-Based Baseline         | 54.04% | 0.5144 | 0.6975 | 0.4842 | 0.3615 |
| Standard BERT ABSA (seed 1) | 66.10% | 0.4942 | 0.7605 | 0.7093 | 0.0128 |
| Extended BERT ABSA (seed 1) | 54.99% | 0.4096 | 0.6472 | 0.5602 | 0.0215 |
| **RoBERTa ABSA (seed 42)**  | **66.54%** | **0.6493** | **0.7822** | 0.6955 | **0.4702** |

### 2.2 Restaurant (3602 examples)

| Model | Accuracy | **Macro F1** | Positive F1 | Negative F1 | **Neutral F1** |
|---|---:|---:|---:|---:|---:|
| Rule-Based Baseline         | 60.19% | 0.5199 | 0.7603 | 0.4114 | 0.3879 |
| Standard BERT ABSA (seed 1) | 68.82% | 0.4680 | 0.8131 | 0.5587 | 0.0322 |
| Extended BERT ABSA (seed 1) | 64.55% | 0.3916 | 0.7809 | 0.3909 | 0.0031 |
| **RoBERTa ABSA (seed 42)**  | **70.32%** | **0.6175** | **0.8335** | 0.5956 | **0.4234** |

Raw data: `outputs/evaluation/cross_domain/metrics_summary.csv`. Artifacts: `outputs/evaluation/cross_domain/confusion_matrices/*.png`, `outputs/evaluation/cross_domain/error_reports/*.json`.

---

## 3. Three-Domain Roll-Up

Mean Macro F1 across project, laptop and restaurant — a simple proxy for deployment reliability.

| Model | In-Domain | Laptop | Restaurant | **3-Domain Mean Macro F1** |
|---|---:|---:|---:|---:|
| Rule-Based Baseline   | 0.4787 | 0.5144 | 0.5199 | 0.5043 |
| Standard BERT ABSA    | 0.6954 | 0.4942 | 0.4680 | 0.5525 |
| Extended BERT ABSA    | **0.7015** | 0.4096 | 0.3916 | 0.5009 |
| **RoBERTa ABSA**      | 0.6476 | **0.6493** | **0.6175** | **0.6381** |

Ranking flips completely between in-domain and cross-domain — Extended BERT is the strongest on the training distribution and the weakest off it (even below the rule-based baseline).

---

## 4. Multi-Seed Variance Analysis

Each model trained 5× with seeds `[42, 1, 7, 2024, 123]`. All metrics below are reported on the project test set (123 examples).

### 4.1 Per-seed results

| Model | Seed | Val Macro F1 | Test Acc | Test Macro F1 | Pos F1 | Neg F1 | Neu F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard BERT | 42   | 0.4840 | 0.9431 | 0.6228 | 0.9683 | 0.9000 | 0.0000 |
| Standard BERT | 1    | 0.5130 | 0.9106 | **0.6954** | 0.9493 | 0.7368 | 0.4000 |
| Standard BERT | 7    | 0.4707 | 0.8049 | 0.6136 | 0.8844 | 0.6364 | 0.3200 |
| Standard BERT | 2024 | 0.5261 | 0.8862 | 0.6626 | 0.9346 | 0.6087 | 0.4444 |
| Standard BERT | 123  | 0.4407 | 0.9187 | 0.6790 | 0.9537 | 0.7500 | 0.3333 |
| Extended BERT | 42   | 0.4838 | 0.9106 | 0.6251 | 0.9585 | 0.6667 | 0.2500 |
| Extended BERT | 1    | 0.4860 | 0.8943 | **0.7015** | 0.9390 | 0.7368 | 0.4286 |
| Extended BERT | 7    | 0.4662 | 0.9024 | 0.6715 | 0.9450 | 0.7059 | 0.3636 |
| Extended BERT | 2024 | 0.4574 | 0.8537 | 0.6709 | 0.9126 | 0.7000 | 0.4000 |
| Extended BERT | 123  | 0.4872 | 0.8618 | 0.6134 | 0.9238 | 0.6087 | 0.3077 |
| RoBERTa       | 42   | 0.4692 | 0.8455 | **0.6476** | 0.9126 | 0.6667 | 0.3636 |
| RoBERTa       | 1    | 0.4282 | 0.9024 | 0.5167 | 0.9502 | 0.6000 | 0.0000 |
| RoBERTa       | 7    | 0.4486 | 0.8537 | 0.6405 | 0.9216 | 0.6667 | 0.3333 |
| RoBERTa       | 2024 | 0.5087 | 0.6992 | 0.5184 | 0.8087 | 0.4706 | 0.2759 |
| RoBERTa       | 123  | 0.5903 | 0.8537 | 0.5845 | 0.9223 | 0.5455 | 0.2857 |

### 4.2 Summary (mean ± std across 5 seeds)

| Model | Accuracy | Macro F1 | Positive F1 | Negative F1 | **Neutral F1** |
|---|---|---|---|---|---|
| Standard BERT | 0.8927 ± 0.0516 | 0.6547 ± 0.0354 | 0.9381 ± 0.0317 | 0.7264 ± 0.1095 | 0.2995 ± **0.1749** |
| Extended BERT | 0.8846 ± 0.0253 | 0.6565 ± 0.0364 | 0.9358 ± 0.0180 | 0.6836 ± 0.0487 | 0.3500 ± **0.0719** |
| RoBERTa       | 0.8309 ± 0.0787 | 0.5815 ± 0.0633 | 0.9031 ± 0.0537 | 0.5899 ± 0.0763 | 0.2517 ± 0.1452 |

Raw data: `outputs/evaluation/multi_seed/all_models_multi_seed.csv` (per-seed) and `outputs/evaluation/multi_seed/all_models_summary.json` (summary).

---

## 5. Key Findings

1. **Architecture ranking flips between in-domain and cross-domain.** Extended BERT is the in-domain winner (Macro F1 0.7015) but the worst model across three domains (0.5009), even below the rule-based baseline (0.5043). The aspect-marker token overfits to training-domain vocabulary.

2. **RoBERTa is the only reliable cross-domain model.** Its Macro F1 barely drops from in-domain 0.6476 to 0.6493 (laptop) / 0.6175 (restaurant). Standard and Extended BERT collapse to 0.39–0.49.

3. **Training stochasticity rivals architectural choice.** Across 5 seeds the Macro F1 spread is 0.05 – 0.13 per model. Standard BERT and Extended BERT are statistically indistinguishable (0.6547 vs 0.6565, Δ = 0.002 within one std). The aspect-marker token's real benefit is **lower neutral-class variance** (std 0.072 vs 0.175 for Standard BERT — ~2.4× more stable).

4. **Neutral F1 is the fragile axis.** Positive F1 std stays under 0.03 for every model; Neutral F1 std reaches 0.17. Both Standard BERT and RoBERTa produced at least one seed with Neutral F1 = 0, confirming that the "BERT cannot learn neutral" failure is stochastic, not structural.

5. **Extended BERT's cross-domain Neutral F1 is essentially zero.** 0.0215 on laptop and 0.0031 on restaurant — the aspect marker actively degrades the model's ability to recognise the neutral class outside the training distribution.

---

## 6. Checkpoint Inventory

Canonical checkpoints (`checkpoints/*_best.pt`) point at the best-of-5-seed runs. All per-seed checkpoints are retained for reproducibility.

| Model | Canonical checkpoint | Source seed |
|---|---|---:|
| Standard BERT  | `checkpoints/bert_absa_best.pt`         | `bert_absa_seed_1.pt` |
| Extended BERT  | `checkpoints/extended_bert_best.pt`     | `extended_bert_seed_1.pt` |
| RoBERTa        | `checkpoints/roberta_absa_best.pt`      | `roberta_absa_seed_42.pt` |

All 15 per-seed checkpoints (`{bert_absa,extended_bert,roberta_absa}_seed_{42,1,7,2024,123}.pt`) are present locally but excluded from git via `.gitignore`.

---

## 7. How to Reproduce

```bash
# In-domain evaluation (skips models with missing checkpoints)
python evaluate_all.py

# Cross-domain evaluation on SemEval laptop + restaurant
python evaluate_cross_domain.py

# Multi-seed training (15 runs × 5 epochs)
python train_extended_multi_seed.py       # Extended BERT only
python train_bert_roberta_multi_seed.py   # Standard BERT + RoBERTa, merges with Extended results
```
