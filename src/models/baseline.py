# src/models/baseline.py
"""
Rule-based ABSA using spaCy dependency parsing + VADER sentiment.

Strategy:
1. Locate the aspect term (or a synonym) in the sentence using spaCy tokens.
2. Extract the sentence window around the aspect mention.
3. Score that window with VADER compound score.
4. Threshold: compound >= 0.05 → positive, <= -0.05 → negative, else neutral.
"""

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Aspect category → surface form keywords
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "battery":      ["battery", "charge", "charging", "power"],
    "screen":       ["screen", "display", "resolution", "brightness"],
    "sound":        ["sound", "audio", "speaker", "volume", "bass"],
    "performance":  ["performance", "speed", "fast", "slow", "lag", "processor"],
    "price":        ["price", "cost", "value", "expensive", "cheap", "worth"],
    "usability":    ["usability", "easy", "setup", "interface", "use", "intuitive"],
    "design":       ["design", "build", "look", "size", "weight", "compact"],
    "connectivity": ["wifi", "bluetooth", "connection", "connect", "network"],
    "build_quality": ["quality", "durable", "sturdy", "flimsy", "material"],
    "overall":      [],  # no specific keywords — use full sentence
}

_WINDOW = 5  # tokens before/after aspect mention


class RuleBasedABSA:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self._vader = SentimentIntensityAnalyzer()

    def _get_keywords(self, aspect: str) -> list[str]:
        aspect = aspect.lower()
        return ASPECT_KEYWORDS.get(aspect, [aspect])

    def _extract_window(self, doc, aspect: str) -> str:
        """Return a windowed text around the aspect mention, or full text."""
        keywords = self._get_keywords(aspect)
        tokens = list(doc)
        for i, tok in enumerate(tokens):
            if tok.lemma_.lower() in keywords or tok.text.lower() in keywords:
                start = max(0, i - _WINDOW)
                end   = min(len(tokens), i + _WINDOW + 1)
                return " ".join(t.text for t in tokens[start:end])
        return doc.text

    def predict(self, review: str, aspect: str) -> str:
        doc = self._nlp(review)
        window_text = self._extract_window(doc, aspect)
        scores = self._vader.polarity_scores(window_text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"

    def predict_batch(self, reviews: list[str], aspects: list[str]) -> list[str]:
        return [self.predict(r, a) for r, a in zip(reviews, aspects)]
