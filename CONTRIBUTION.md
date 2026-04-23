# Team Contributions

This file records what each team member contributed, reconstructed from
the git history of this repository (`git log --all`). Each section lists
the member's real name, GitHub identity (the one visible in `git log`),
the branch they primarily worked on, and the files or features
introduced by their commits.

## Branch layout

```
main ────────── initial scaffold and baseline system
 ├── partA ──── data curation (SemEval + second-annotator IAA)
 ├── roberta ── RoBERTa model addition
 ├── feature/c-part-evaluation ── evaluation pipeline and error buckets
 └── partD-integration ── merge of the three feature branches + cross-domain,
                          multi-seed benchmarking, figures, and report
                          (this is the submission branch)
```

## Members

### Xinan Li — GitHub author: 李欣桉 `<z5549727@ad.unsw.edu.au>`

Branch: `main`. 15 commits, roughly +3,657 / −832 lines.

Built the initial four-file pipeline on the `main` branch:

- Project scaffold, `requirements.txt`, `.gitignore`, `conftest.py`.
- `src/data/dataset.py`, `src/data/extended_dataset.py`,
  `src/data/split.py`, `src/data/iaa.py` — PyTorch dataset wrappers, the
  group-by-id train/val/test splitter, and the Cohen's κ calculator.
- `src/models/baseline.py` — spaCy + VADER rule-based ABSA.
- `src/models/bert_absa.py`, `src/models/extended_bert.py` — standard
  and extended BERT ABSA classifiers (the latter with `[ASPECT]` /
  `[/ASPECT]` marker tokens and resized embeddings).
- `src/evaluation/metrics.py`, `src/evaluation/error_analysis.py` —
  macro precision/recall/F1 with per-class breakdown, and the original
  error-bucketing module (implicit aspect, negation, sarcasm, short
  text, etc.).
- `train_bert.py`, `train_extended.py` — single-seed training scripts.
- `cli.py`, `demo.py` — first version of the command-line interface and
  the Gradio demo.
- Initial unit tests under `tests/` for baseline, dataset, IAA,
  metrics, and split logic.
- First draft of `README.md`.
- Initial draft of the project report

### Huijie Cheng — GitHub author: Chhhjnb `<z5527580@ad.unsw.edu.au>`

Branch: `partA`. 1 commit, +7,357 / −0 lines.

Data curation and second-annotator materials:

- `data/iaa_annotator2.csv` — 181-row second-annotator labels used for
  the Cohen's κ agreement study.
- `data/reviews_all_clean.csv` — cleaned 814-row primary corpus.
- `data/semeval_laptop.csv` (2,313 rows) and `data/semeval_restaurant.csv`
  (3,602 rows) — SemEval 2014 Task 4 splits used for cross-domain
  evaluation.
- `src/data/semeval_loader.py` — converts SemEval XML into the project's
  four-field CSV schema.
- `data/aspect_distribution.svg`, `data/sentiment_distribution.svg`,
  `data/semeval_stats.json` — dataset statistics.
- `PARTA_REPORT.md` — data-curation write-up (later archived into
  `experiments/`).

### Guanyu Chen — GitHub author: HarryLester `<z5746041@ad.unsw.edu.au>`

Branch: `roberta`. 1 commit, +175 / −0 lines.

RoBERTa model track:

- `src/models/robertaabsa.py` — RoBERTa-base sequence-pair classifier.
- `src/data/roberta_dataset.py` — RoBERTa tokenisation wrapper (no
  `token_type_ids`).
- `train_roberta.py` — single-seed training entry point.
- Minor edits to `evaluate_all.py` to wire the RoBERTa checkpoint into
  the evaluation loop.

### Yize Shen — GitHub author: Ezwinnnn `<z5459644@ad.unsw.edu.au>`

Branch: `feature/c-part-evaluation`. 1 commit, +637 / −158 lines.

Evaluation pipeline hardening:

- Extended `src/evaluation/error_analysis.py` — added the
  `aspect_mismatch` bucket and enlarged the sarcasm pattern list.
- Tightened `src/models/baseline.py` and added `src/models/aspects.py` —
  a shared aspect taxonomy used by the baseline, Extended BERT, and the
  error-analysis module.
- Overhauled `evaluate_all.py` — writes `metrics_summary.csv/.json`,
  confusion-matrix PNGs, and structured error reports under
  `outputs/evaluation/`, and gracefully skips any missing checkpoint
  with a warning rather than crashing.
- Added `tests/test_error_analysis.py`.
- Miscellaneous README and `cli.py` cleanups.

### Lunshuo Tian — GitHub author: Liiizhen `<z5644811@ad.unsw.edu.au>`

Branch: `partD-integration` (submission branch). 18 commits,
roughly +9,129 / −196 lines.

Integration, multi-seed benchmarking, cross-domain evaluation, and
report:

- **Merging.** Pulled `partA`, `feature/c-part-evaluation`, and
  `roberta` into a single integration branch
  (`integration/d-merge` → `partD-integration`). Unified the RoBERTa
  checkpoint into Yize Shen's checkpoint-safe runner
  (`integrate: unify RoBERTa evaluation into C's checkpoint-safe
  runner`).
- **Cross-domain evaluation.** `evaluate_cross_domain.py` — runs all
  four systems on SemEval 2014 laptop and restaurant, writes metrics,
  confusion matrices, and error reports to
  `outputs/evaluation/cross_domain/`.
- **Multi-seed benchmark.** `train_full_benchmark.py` — 20-seed ×
  3-model × 3-domain run with crash-safe resume, producing
  `outputs/evaluation/multi_seed_full_cuda/full_benchmark.csv`
  (used for the mean ± std tables in the report).
- **Learning curves.** `train_all_models_curves.py` (5 epochs for all
  three) and `train_ext_roberta_10ep_curves.py` (10 epochs for Extended
  BERT and RoBERTa), plus `build_best_strategy_csv.py` and
  `make_all_models_curves.py` to produce
  `outputs/figures/fig6_all_models_learning_curves.*`.
- **Figures.** `make_figures.py` — figures 1–5 (Macro F1 distribution,
  cross-domain scatter, per-class F1, variance, dataset distribution).
- **Report.** Supplementary experiments (cross-domain, multi-seed,
  learning curves) and detail refinements on top of Xinan Li's initial
  report draft.

## How to verify

All of the statements above can be reproduced from the git history:

```bash
git log --all --author="<name>" --stat
git shortlog -sne --all
```
