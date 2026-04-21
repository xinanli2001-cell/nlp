"""Utilities for qualitative ABSA error analysis.

The heuristics here are intentionally lightweight and transparent. They do not
attempt to *prove* why a model made a mistake; instead, they assign each
misclassified example to the most plausible error bucket so we can inspect
trends across models.

Supported buckets
-----------------
- sarcasm:         explicit sarcasm cue phrases appear in the review
- implicit_aspect: the target aspect is not mentioned explicitly in the text
- negation:        local or sentence-level negation likely flips polarity
- aspect_mismatch: sentiment cues appear to belong to a different aspect
- short_text:      review is too short to provide enough aspect context
- other:           none of the above heuristics fired
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.models.aspects import ASPECT_KEYWORDS, get_aspect_keywords

# Expanded sarcasm cue list as requested in the project split.
SARCASM_PATTERNS = [
    r"\byeah right\b",
    r"\boh sure\b",
    r"\bjust perfect\b",
    r"\bperfect\.\b",
    r"\bgreat, just great\b",
    r"\bthanks for nothing\b",
    r"\blove how\b",
    r"\bwhat a surprise\b",
    r"\bamazing how\b",
    r"\bnice job\b",
    r"\bwell done\b",
    r"\bas if\b",
    r"\bcouldn't be better\b",
    r"\bgood luck with that\b",
    r"\bexactly what i needed\b",
    r"\bso helpful\b",
    r"\bbrilliant\b",
    r"\bfantastic\b",
    r"\bwonderful\b",
    r"\bsure\b",
]

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bnothing\b",
    r"\bnowhere\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bwithout\b",
    r"\bwon't\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bdon't\b",
    r"\bdoesn't\b",
    r"\bdidn't\b",
    r"\bisn't\b",
    r"\baren't\b",
    r"\bwasn't\b",
    r"\bweren't\b",
    r"\bhardly\b",
    r"\bscarcely\b",
]

_POSITIVE_CUES = {
    "amazing", "awesome", "best", "clear", "excellent", "fast", "good",
    "great", "impressive", "intuitive", "love", "nice", "perfect",
    "reliable", "responsive", "smooth", "solid", "worth", "easy",
}

_NEGATIVE_CUES = {
    "annoying", "awful", "bad", "broken", "cheap", "confusing", "crash",
    "crashes", "dead", "died", "disappointing", "drains", "expensive",
    "fails", "flimsy", "hate", "lag", "laggy", "poor", "slow", "terrible",
    "unreliable", "useless", "weak", "worse", "worst",
}

_LABELS = ("positive", "negative", "neutral")


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Return ``True`` when any regex pattern matches the text."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _tokenise(text: str) -> list[str]:
    """Tokenise text with a simple regex-based splitter suitable for heuristics."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _find_keyword_hits(tokens: list[str], keywords: list[str]) -> list[int]:
    """Return token indices where any keyword (including multiword ones) appears."""
    hits: list[int] = []
    for keyword in keywords:
        keyword_tokens = _tokenise(keyword)
        if not keyword_tokens:
            continue
        k = len(keyword_tokens)
        for start in range(len(tokens) - k + 1):
            if tokens[start:start + k] == keyword_tokens:
                hits.append(start)
    return hits


def _window(tokens: list[str], center: int, radius: int = 4) -> list[str]:
    """Return a small token window around a matched aspect mention."""
    start = max(0, center - radius)
    end = min(len(tokens), center + radius + 1)
    return tokens[start:end]


def _has_negation(tokens: list[str]) -> bool:
    """Detect whether a local token window contains negation."""
    joined = " ".join(tokens)
    return _matches_any(joined, NEGATION_PATTERNS)


def _label_from_window(tokens: list[str]) -> str:
    """Infer a coarse sentiment label from lexical cues in a token window."""
    pos_hits = sum(token in _POSITIVE_CUES for token in tokens)
    neg_hits = sum(token in _NEGATIVE_CUES for token in tokens)
    if _has_negation(tokens):
        pos_hits, neg_hits = neg_hits, pos_hits
    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


def _mentioned_aspects(review: str) -> dict[str, list[int]]:
    """Return all aspect mentions detected in the review as token indices."""
    tokens = _tokenise(review)
    mentions: dict[str, list[int]] = {}
    for aspect in ASPECT_KEYWORDS:
        keywords = get_aspect_keywords(aspect)
        hits = _find_keyword_hits(tokens, keywords)
        if hits:
            mentions[aspect] = hits
    return mentions


def _is_target_aspect_present(review: str, aspect: str) -> bool:
    """Return whether the target aspect has an explicit surface-form mention."""
    if aspect.lower() == "overall":
        return True
    mentions = _mentioned_aspects(review)
    return aspect.lower() in mentions


def _is_aspect_mismatch(review: str, aspect: str, gold: str, pred: str) -> bool:
    """Heuristically detect wrong sentiment being attributed to a different aspect.

    The heuristic looks for at least two aspect mentions in the same review. If
    the local sentiment window around the *target* aspect aligns with the gold
    label (or is neutral) while the local window around another aspect aligns
    with the predicted label, we bucket the error as ``aspect_mismatch``.
    """
    target_aspect = aspect.lower()
    if target_aspect == "overall":
        return False

    tokens = _tokenise(review)
    mentions = _mentioned_aspects(review)
    if target_aspect not in mentions:
        return False

    other_aspects = {a: hits for a, hits in mentions.items() if a != target_aspect and hits}
    if not other_aspects:
        return False

    target_labels = [_label_from_window(_window(tokens, idx)) for idx in mentions[target_aspect]]
    target_supports_gold = gold in target_labels or all(label == "neutral" for label in target_labels)
    if not target_supports_gold:
        return False

    for hits in other_aspects.values():
        other_labels = [_label_from_window(_window(tokens, idx)) for idx in hits]
        if pred in other_labels:
            return True
    return False


def classify_error(review: str, aspect: str, gold: str, pred: str) -> str:
    """Return the most plausible error bucket for one misclassified example."""
    words = review.split()
    aspect_present = _is_target_aspect_present(review, aspect)

    if not aspect_present:
        return "implicit_aspect"
    if _matches_any(review, SARCASM_PATTERNS):
        return "sarcasm"
    if _matches_any(review, NEGATION_PATTERNS):
        return "negation"
    if _is_aspect_mismatch(review, aspect, gold, pred):
        return "aspect_mismatch"
    if len(words) < 5:
        return "short_text"
    return "other"


def analyse_errors(
    reviews: list[str],
    aspects: list[str],
    golds: list[str],
    preds: list[str],
    model_name: str = "",
    output_path: str | Path | None = None,
    max_examples_per_bucket: int = 3,
) -> dict:
    """Summarise error buckets and optionally persist a structured JSON report."""
    error_counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)

    for review, aspect, gold, pred in zip(reviews, aspects, golds, preds):
        if gold == pred:
            continue
        error_type = classify_error(review, aspect, gold, pred)
        error_counts[error_type] += 1
        if len(examples[error_type]) < max_examples_per_bucket:
            examples[error_type].append(
                {
                    "review": review,
                    "aspect": aspect,
                    "gold": gold,
                    "pred": pred,
                }
            )

    total_errors = sum(error_counts.values())
    breakdown = {}
    for error_type in sorted(error_counts.keys()):
        count = error_counts[error_type]
        breakdown[error_type] = {
            "count": count,
            "percentage": round((100.0 * count / total_errors) if total_errors else 0.0, 1),
            "examples": examples[error_type],
        }

    summary = {
        "model_name": model_name,
        "n_examples": len(golds),
        "total_errors": total_errors,
        "total_correct": len(golds) - total_errors,
        "labels": list(_LABELS),
        "breakdown": breakdown,
    }

    print(f"\n=== Error Analysis: {model_name} ===")
    print(f"Total errors: {total_errors}")
    for error_type, payload in sorted(breakdown.items(), key=lambda item: item[1]["count"], reverse=True):
        print(f"  {error_type}: {payload['count']} ({payload['percentage']:.1f}%)")
        for ex in payload["examples"]:
            snippet = ex["review"][:120].replace("\n", " ")
            print(f"    [{ex['gold']}→{ex['pred']}] {ex['aspect']}: \"{snippet}\"")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved error report to {output_path}")

    return summary
