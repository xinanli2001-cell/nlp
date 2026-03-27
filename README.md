# COMP6713 ABSA Project

Aspect-Based Sentiment Analysis for English e-commerce electronics reviews.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data

Annotated dataset: 789 reviews with aspect-sentiment labels (positive/negative/neutral).

```bash
# Generate train/val/test splits
python src/data/split.py

# Inter-annotator agreement
python src/data/iaa.py data/reviews_ok.csv data/iaa_annotator2.csv
```

## Train

```bash
python train_bert.py       # Standard BERT (20 credits)
python train_extended.py   # Extended BERT with aspect markers (30 credits)
```

## Evaluate

```bash
python evaluate_all.py
```

## CLI

```bash
python cli.py --review "Battery life is excellent." --aspect battery --model extended
python cli.py --input_file data/final/test.csv --model extended --output predictions.csv
```

Models: `baseline` (spaCy+VADER), `bert`, `extended`

## Demo

```bash
python demo.py
# Opens Gradio UI at http://127.0.0.1:7860
```

## Credits

| Part | Item | Credits |
|------|------|---------|
| A | ABSA problem + e-commerce domain | 10 |
| B | Own labeled dataset + IAA | 20 |
| B | VADER lexicon | 10 |
| C | Rule-based baseline | required |
| C | Fine-tuned BERT | 20 |
| C | Extended BERT (aspect-marked input) | 30 |
| D | Macro P/R/F1 | 10 |
| D | Error analysis | 5 |
| D | CLI | 5 |
| D | Gradio demo | 10 |
| **Total** | | **120** |
