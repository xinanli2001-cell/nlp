# COMP6713 ABSA Project

Aspect-Based Sentiment Analysis for English e-commerce electronics reviews.

## 1. Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. Running Tests

```bash
pytest tests/ -v
```

No additional setup is needed for imports because `conftest.py` at the project root adds the repository root to `sys.path`.

## 3. Data

Annotated dataset: 814 review–aspect pairs with three-way sentiment labels (`positive`, `negative`, `neutral`).

```bash
# Generate train/val/test splits
python src/data/split.py

# Inter-annotator agreement
# Requires a second annotator CSV in the same format: id,sentiment
python src/data/iaa.py data/reviews_ok.csv data/iaa_annotator2.csv
```

## 4. Training

```bash
python train_bert.py
python train_extended.py
```

Trained checkpoints are expected at:

```text
checkpoints/bert_absa_best.pt
checkpoints/extended_bert_best.pt
```

## 5. Evaluation

```bash
python evaluate_all.py
```

Behavior:
- the rule-based baseline always runs;
- if a BERT checkpoint is missing, that model is skipped with a warning instead of crashing.

Artifacts are written to:

```text
outputs/evaluation/
├── metrics_summary.csv
├── metrics_summary.json
├── confusion_matrices/
└── error_reports/
```

## 6. CLI

```bash
python cli.py --review "Battery life is excellent." --aspect battery --model extended
python cli.py --input_file data/final/test.csv --model extended --output predictions.csv
```

Models:
- `baseline` → spaCy + VADER
- `bert` → standard BERT ABSA
- `extended` → BERT with `[ASPECT] ... [/ASPECT]` markers

## 7. Demo

```bash
python demo.py
```

The Gradio interface opens locally, usually at `http://127.0.0.1:7860`.

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

1. Install dependencies.
2. Download the spaCy English model.
3. Place any trained checkpoints in `checkpoints/`.
4. Run `python evaluate_all.py`.
5. Inspect `outputs/evaluation/` for metrics, confusion matrices, and structured error reports.
6. Run `python demo.py` or `python cli.py ...` for interactive prediction.
