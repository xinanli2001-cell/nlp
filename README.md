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

## 8. Reproducibility Checklist

1. Install dependencies.
2. Download the spaCy English model.
3. Place any trained checkpoints in `checkpoints/`.
4. Run `python evaluate_all.py`.
5. Inspect `outputs/evaluation/` for metrics, confusion matrices, and structured error reports.
6. Run `python demo.py` or `python cli.py ...` for interactive prediction.
