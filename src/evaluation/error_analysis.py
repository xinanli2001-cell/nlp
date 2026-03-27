# src/evaluation/error_analysis.py
"""
Qualitative error analysis: bucket misclassified examples into error types.

Error types:
- sarcasm:         text contains sarcasm indicators
- implicit_aspect: aspect not mentioned in text
- negation:        text contains negation affecting sentiment
- short_text:      review is very short (< 5 words)
- other:           none of the above patterns matched
"""

import re
from collections import Counter, defaultdict
from src.models.baseline import ASPECT_KEYWORDS

SARCASM_PATTERNS  = [r"\bsure\b", r"\bgreat job\b", r"\bwonderful\b.*!", r"\bthanks a lot\b"]
NEGATION_PATTERNS = [r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bwon't\b", r"\bcan't\b", r"\bdon't\b"]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_error(review: str, aspect: str, gold: str, pred: str) -> str:
    """Return error type string for a single misclassified example."""
    words = review.split()
    # Use full keyword list (same as baseline) for a fair implicit_aspect check
    keywords_for_aspect = ASPECT_KEYWORDS.get(aspect.lower(), [aspect.lower().replace("_", " ")])
    if not keywords_for_aspect:  # "overall" has empty list — treat as always present
        keywords_for_aspect = [aspect.lower()]
    aspect_in_text = any(kw in review.lower() for kw in keywords_for_aspect)

    if not aspect_in_text:
        return "implicit_aspect"
    if _matches_any(review, NEGATION_PATTERNS) and gold != pred:
        return "negation"
    if _matches_any(review, SARCASM_PATTERNS):
        return "sarcasm"
    if len(words) < 5:
        return "short_text"
    return "other"


def analyse_errors(
    reviews: list[str],
    aspects:  list[str],
    golds:    list[str],
    preds:    list[str],
    model_name: str = "",
) -> dict:
    """Return a summary of error types and example cases."""
    error_counts: Counter = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)

    for review, aspect, gold, pred in zip(reviews, aspects, golds, preds):
        if gold == pred:
            continue
        etype = classify_error(review, aspect, gold, pred)
        error_counts[etype] += 1
        if len(examples[etype]) < 3:
            examples[etype].append({
                "review": review[:100],
                "aspect": aspect,
                "gold": gold,
                "pred": pred,
            })

    total_errors = sum(error_counts.values())
    print(f"\n=== Error Analysis: {model_name} ===")
    print(f"Total errors: {total_errors}")
    for etype, count in error_counts.most_common():
        print(f"  {etype}: {count} ({100*count/total_errors:.1f}%)")
        for ex in examples[etype]:
            print(f"    [{ex['gold']}→{ex['pred']}] {ex['aspect']}: \"{ex['review']}\"")

    return {"total_errors": total_errors, "breakdown": dict(error_counts)}
