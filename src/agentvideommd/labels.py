from __future__ import annotations

import re

LABEL_ORDER = ("real", "fake")


def normalize_prediction(text: str | None) -> tuple[str, bool]:
    """Match the legacy evaluator: malformed outputs fall back to ``fake``."""
    value = (text or "").strip().lower()
    if re.search(r"\bnot\s+fake\b|\bnot\s+false\b|\bnot\s+misleading\b", value):
        return "real", True
    if "并非虚假" in value or "不是虚假" in value or "不是谣言" in value:
        return "real", True
    fake_phrase = re.search(
        r"\bnot\s+real\b|\bnot\s+true\b|\bfalse\b|\buntrue\b|\bmisleading\b|"
        r"\bfabricated\b|\bhoax\b|\brumou?r\b|\bnot\s+credible\b|\bunreliable\b",
        value,
    )
    if fake_phrase:
        return "fake", True
    if any(
        phrase in value
        for phrase in [
            "不真实",
            "不可信",
            "不属实",
            "不可靠",
            "不实",
            "造假",
            "伪造",
            "谣言",
            "误导",
        ]
    ):
        return "fake", True
    real_phrase = re.search(r"\btrue\b|\bcredible\b|\bauthentic\b|\breliable\b|\bverified\b", value)
    if real_phrase:
        return "real", True
    if any(phrase in value for phrase in ["可信", "属实", "可靠", "真实可信"]):
        return "real", True
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
