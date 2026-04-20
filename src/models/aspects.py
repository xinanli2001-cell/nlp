"""Shared aspect taxonomy and keyword lists used across the project.

This module centralises the aspect keyword dictionary so that datasets,
rule-based models, and evaluation utilities can reuse the same mapping without
introducing heavyweight runtime dependencies on each other.
"""

from __future__ import annotations

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "battery": ["battery", "charge", "charging", "power"],
    "screen": ["screen", "display", "resolution", "brightness"],
    "sound": ["sound", "audio", "speaker", "volume", "bass"],
    "performance": ["performance", "speed", "fast", "slow", "lag", "processor"],
    "price": ["price", "cost", "value", "expensive", "cheap", "worth"],
    "usability": ["usability", "easy", "setup", "interface", "use", "intuitive"],
    "design": ["design", "build", "look", "size", "weight", "compact"],
    "connectivity": ["wifi", "bluetooth", "connection", "connect", "network"],
    "build_quality": ["quality", "durable", "sturdy", "flimsy", "material"],
    "overall": [],
}


def get_aspect_keywords(aspect: str) -> list[str]:
    """Return configured keywords for an aspect, falling back to the aspect name itself.

    Parameters
    ----------
    aspect:
        Canonical aspect label such as ``battery`` or ``build_quality``.
    """
    aspect = aspect.lower()
    return ASPECT_KEYWORDS.get(aspect, [aspect.replace("_", " "), aspect])
