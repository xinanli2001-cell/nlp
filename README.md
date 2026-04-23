# COMP6713 ABSA Project

Aspect-Based Sentiment Analysis for English e-commerce electronics reviews.

The repository implements four systems side by side: a rule-based baseline, Standard BERT, an Extended BERT variant with aspect-marker tokens, and RoBERTa. All four are trained or evaluated under the same pipeline and compared on both the project's own labelled data and on SemEval 2014 Task 4 for cross-domain generalisation. Consolidated numbers live in `RESULTS.md`.

## 1. Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional on slow networks: pre-download bert-base-uncased and roberta-base
# into the local HuggingFace cache with retry.
python preload_hf.py
```

## 2. Running Tests

```bash
pytest tests/ -v
```

No additional setup is needed for imports because `conftest.py` at the project root adds the repository root to `sys.path`.

## 3. Data

Primary corpus: 814 review–aspect pairs with three-way sentiment labels (`positive`, `negative`, `neutral`) in `data/reviews_all.csv`. External corpora for cross-domain evaluation: SemEval 2014 Task 4 laptop (2313 rows) and restaurant (3602 rows) in `data/semeval_laptop.csv` and `data/semeval_restaurant.csv`.

```bash
# Generate train/val/test splits from data/reviews_all.csv with a fixed seed
# (group-by-id so no review leaks across splits)
python src/data/split.py

# Inter-annotator agreement. Pass annotator 1 first and annotator 2 second;
# both files must share the id and sentiment columns.
python src/data/iaa.py data/reviews_all.csv data/iaa_annotator2.csv

# Re-download SemEval 2014 Task 4 data (the committed CSVs were produced this way)
python src/data/semeval_loader.py --dataset laptop --output data/semeval_laptop.csv
python src/data/semeval_loader.py --dataset restaurant --output data/semeval_restaurant.csv
```

## 4. Training

Single-seed training scripts (one per neural system):

```bash
python train_bert.py        # Standard BERT ABSA
python train_extended.py    # Extended BERT with [ASPECT] / [/ASPECT] markers
python train_roberta.py     # RoBERTa ABSA
```

Each script saves the best-val-macro-F1 checkpoint to `checkpoints/`:

```text
checkpoints/bert_absa_best.pt
checkpoints/extended_bert_best.pt
checkpoints/roberta_absa_best.pt
```

Full multi-seed benchmark and learning-curve runs (used by the paper and `RESULTS.md`):

```bash
# 20 seeds × 3 models × 3 domains, with crash-safe resume
python train_full_benchmark.py

# Per-epoch learning curves: 5 epochs for all three models
python train_all_models_curves.py

# Per-epoch learning curves: 10 epochs for Extended BERT and RoBERTa
python train_ext_roberta_10ep_curves.py
```

## 5. Evaluation

```bash
# In-domain: rule-based baseline + the three BERT-family models on data/final/test.csv
python evaluate_all.py

# Cross-domain: same four systems on SemEval 2014 laptop and restaurant
python evaluate_cross_domain.py
```

Behaviour:
- the rule-based baseline always runs;
- if a BERT-family checkpoint is missing, that model is skipped with a warning instead of crashing.

Artifacts from `evaluate_all.py`:

```text
outputs/evaluation/
├── metrics_summary.csv
├── metrics_summary.json
├── confusion_matrices/*.png
└── error_reports/*.json
```

Artifacts from `evaluate_cross_domain.py`:

```text
outputs/evaluation/cross_domain/
├── metrics_summary.csv
├── metrics_summary.json
├── confusion_matrices/*.png
└── error_reports/*.json
```

To regenerate every figure that appears in `RESULTS.md`:

```bash
python build_best_strategy_csv.py        # merges curve CSVs into best_strategy.csv
python make_figures.py                   # figures 1–5
python make_all_models_curves.py         # figure 6 (learning curves)
```

## 6. CLI

```bash
python cli.py --review "Battery life is excellent." --aspect battery --model extended
python cli.py --input_file data/final/test.csv --model roberta --output predictions.csv
```

Supported `--model` values:
- `baseline` → spaCy + VADER
- `bert` → Standard BERT ABSA
- `extended` → BERT with `[ASPECT] ... [/ASPECT]` markers
- `roberta` → RoBERTa ABSA

## 7. Demo

```bash
python demo.py
```

The Gradio interface opens locally, usually at `http://127.0.0.1:7860`, with selectors for review text, aspect category, and model.

## 8. Repository Layout

### Top-level scripts

| File | Purpose |
|---|---|
| `cli.py` | Command-line interface: single-input or batch-CSV prediction with any of the four models. |
| `demo.py` | Gradio web demo for interactive review/aspect/model selection. |
| `evaluate_all.py` | In-domain evaluation on the held-out project test set. Writes metrics, confusion matrices, and structured error reports under `outputs/evaluation/`. |
| `evaluate_cross_domain.py` | Cross-domain evaluation on SemEval 2014 laptop and restaurant data for all four systems. |
| `train_bert.py` | Single-seed training of Standard BERT ABSA. |
| `train_extended.py` | Single-seed training of Extended BERT with `[ASPECT]` marker tokens. |
| `train_roberta.py` | Single-seed training of RoBERTa ABSA. |
| `train_full_benchmark.py` | 20-seed × 3-model × 3-domain benchmark with resume support; produces `outputs/evaluation/multi_seed_full_cuda/full_benchmark.csv`. |
| `train_all_models_curves.py` | Per-epoch learning curves at 5 epochs for all three neural systems. |
| `train_ext_roberta_10ep_curves.py` | Per-epoch learning curves at 10 epochs for Extended BERT and RoBERTa. |
| `build_best_strategy_csv.py` | Merges the 5-epoch and 10-epoch curve CSVs into the best-strategy CSV consumed by the figure scripts. |
| `make_figures.py` | Regenerates figures 1–5 (violin, cross-domain scatter, per-class F1, variance, dataset distribution) from the best-strategy CSV. |
| `make_all_models_curves.py` | Regenerates figure 6 (per-epoch learning curves) from the two curve CSVs. |
| `preload_hf.py` | Utility that downloads `bert-base-uncased` and `roberta-base` with retry; useful on networks with flaky HuggingFace access. |
| `conftest.py` | pytest configuration that adds the project root to `sys.path` so local imports resolve during testing. |

### Package modules (`src/`)

| Path | Purpose |
|---|---|
| `src/data/dataset.py` | Standard BERT sequence-pair ABSA dataset. |
| `src/data/extended_dataset.py` | Extended BERT dataset with `[ASPECT] / [/ASPECT]` marker injection. |
| `src/data/roberta_dataset.py` | RoBERTa sequence-pair dataset (no `token_type_ids`). |
| `src/data/split.py` | Group-by-id stratified split into train/val/test (prevents review-level leakage). |
| `src/data/iaa.py` | Cohen's κ inter-annotator agreement calculator. |
| `src/data/semeval_loader.py` | Downloads SemEval 2014 Task 4 XML and converts to the project's four-field CSV schema. |
| `src/models/baseline.py` | Rule-based ABSA using spaCy keyword matching + VADER sentiment scoring. |
| `src/models/aspects.py` | Shared aspect taxonomy and keyword dictionary used by baseline, Extended BERT, and error analysis. |
| `src/models/bert_absa.py` | Standard BERT classification model. |
| `src/models/extended_bert.py` | Extended BERT model with resized token embeddings for `[ASPECT]` / `[/ASPECT]`. |
| `src/models/robertaabsa.py` | RoBERTa classification model. |
| `src/evaluation/metrics.py` | Macro precision/recall/F1 + per-class F1 computation and pretty-printer. |
| `src/evaluation/error_analysis.py` | Heuristic error bucketing (implicit aspect, negation, sarcasm, aspect mismatch, short text, other) and structured JSON reporting. |

### Tests (`tests/`)

Unit tests for the baseline, dataset encoding, error analysis, IAA, metrics, and the data split logic. Run via `pytest tests/ -v`.

### Exploratory scripts (`experiments/`)

Intermediate scripts and an early draft that shaped the final protocol but are not part of the main pipeline. See `experiments/README.md` for a per-file description.

### Data and outputs

| Path | Content |
|---|---|
| `data/reviews_all.csv` | Raw 814 manually annotated review–aspect pairs. |
| `data/iaa_annotator2.csv` | Second annotator's labels for the 180-pair IAA sample. |
| `data/semeval_laptop.csv`, `data/semeval_restaurant.csv` | SemEval 2014 Task 4 splits used for cross-domain evaluation. |
| `data/final/` | `train.csv`, `val.csv`, `test.csv` produced by `src/data/split.py`. |
| `outputs/evaluation/` | Metrics CSV/JSON, confusion matrices, and error reports from the evaluation scripts. |
| `outputs/figures/` | Publication-ready PDF and PNG versions of figures 1–6. |
| `RESULTS.md` | Consolidated benchmark results (in-domain + cross-domain + variance + learning curves), mirroring the numbers reported in the paper. |

Every function inside every file additionally carries a short docstring describing its behaviour; open any script to read the per-function documentation.

## 9. Reproducibility Checklist

1. Install dependencies (`pip install -r requirements.txt`) and the spaCy English model.
2. Optional on slow networks: run `python preload_hf.py` to warm the HuggingFace cache.
3. Regenerate the train/val/test splits with `python src/data/split.py` if `data/final/` is empty.
4. Place or retrain the three BERT-family checkpoints under `checkpoints/` (either via the single-seed `train_*.py` scripts or via `train_full_benchmark.py`).
5. Run `python evaluate_all.py` for in-domain metrics and `python evaluate_cross_domain.py` for SemEval laptop + restaurant.
6. Inspect `outputs/evaluation/` for summary CSV/JSON, confusion matrices, and structured error reports.
7. Run `python demo.py` or `python cli.py ...` for interactive prediction, or open `RESULTS.md` for the consolidated numerical summary.
