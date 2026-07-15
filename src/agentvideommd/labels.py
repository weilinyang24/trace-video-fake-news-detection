from __future__ import annotations

import re

LABEL_ORDER = ("real", "fake")


def normalize_prediction(text: str | None) -> tuple[str, bool]:
    """Match the legacy evaluator: malformed outputs fall back to ``fake``."""
    value = (text or "").strip().lower()
    real_match = re.search(r"\breal\b", value)
    fake_match = re.search(r"\bfake\b", value)
    # Preserve VideoMMD's exact precedence when an output contains both words.
    if real_match:
        return "real", True
    if fake_match:
        return "fake", True
    if "真实" in value or value == "真":
        return "real", True
    if "虚假" in value or "辟谣" in value or value == "假":
        return "fake", True
    return "fake", False


def normalize_gold(value: str) -> str:
    label = value.strip().lower()
    if label not in LABEL_ORDER:
        raise ValueError(f"Unsupported gold label: {value!r}")
    return label
