from __future__ import annotations

from datetime import datetime, timezone

FAKE_NEWS_PROMPT_EN = """You are evaluating short-video news veracity.

Task: determine whether the news claim or event conveyed by the current video and its auxiliary observations is factually real or fake.

This is not deepfake detection. Do not judge only whether the footage looks visually authentic. Judge whether the conveyed news claim is true as stated, considering the video and the auxiliary observations together.

Balanced decision rule:
- Choose fake when there is clear evidence that the claim is false, debunked, misleading in context, mismatched with the shown video, or uses old/unrelated footage for a different event.
- Choose real when the claim appears factually true, plausible, or not contradicted by the video and auxiliary observations.
- If the available evidence is insufficient or ambiguous, do not assume the claim is fake just because it is controversial, emotional, political, or unusual. In ambiguous cases, prefer real.

Reply with exactly one word: real or fake."""

FAKE_NEWS_PROMPT_ZH = FAKE_NEWS_PROMPT_EN


def _text(value: object) -> str:
    if value is None:
        return ""
    result = str(value).strip()
    return "" if result.lower() == "null" else result


def _time(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = float(value)
        if timestamp > 1e12:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError):
        return str(value)


def _format_observations(row: dict, fields: tuple[tuple[str, str, bool], ...], time_key: str | None) -> list[str]:
    parts: list[str] = []
    for title, key, claim in fields:
        value = _text(row.get(key))
        if value:
            value = f"<claim>{value}</claim>" if claim else value
            parts.append(f"- {title}: {value}")
    if time_key:
        publish_time = _time(row.get(time_key))
        if publish_time:
            parts.append(f"- publish time metadata: {publish_time}")
    return parts


FAKETT_FIELDS = (
    ("news claim to verify", "event", True),
    ("uploader caption", "description", False),
    ("uploader profile", "user_description", False),
)

FAKESV_FIELDS = (
    ("news claim to verify", "title", True),
    ("event keywords", "keywords", False),
    ("uploader profile", "author_intro", False),
    ("uploader location", "author_place", False),
)


def _format_examples(examples: list[dict] | None, fields: tuple[tuple[str, str, bool], ...], time_key: str | None) -> list[str]:
    if not examples:
        return []
    parts = [
        "",
        "Calibration examples from the training split:",
        "Use these examples only to calibrate the label definition. Do not classify the current sample by topic similarity.",
    ]
    for index, example in enumerate(examples, 1):
        parts.append(f"Example {index} known label: {example['label']}")
        parts.extend(_format_observations(example["source"], fields, time_key))
    return parts


def build_fakett_prompt(row: dict, examples: list[dict] | None = None) -> str:
    parts = [FAKE_NEWS_PROMPT_EN]
    parts.extend(_format_examples(examples, FAKETT_FIELDS, "publish_time"))
    parts.extend(["", "Current sample to classify:", "Auxiliary observations:"])
    parts.extend(_format_observations(row, FAKETT_FIELDS, "publish_time"))
    return "\n".join(parts)


def build_fakesv_prompt(row: dict, examples: list[dict] | None = None, max_comments: int = 3) -> str:
    parts = [FAKE_NEWS_PROMPT_ZH]
    parts.extend(_format_examples(examples, FAKESV_FIELDS, "publish_time_norm"))
    parts.extend(["", "Current sample to classify:", "Auxiliary observations:"])
    parts.extend(_format_observations(row, FAKESV_FIELDS, "publish_time_norm"))
    comments = [_text(item).replace("\n", " ") for item in (row.get("comments") or [])]
    comments = [item for item in comments if item][:max_comments]
    if comments:
        parts.append(f"- selected comments: {' | '.join(comments)}")
    return "\n".join(parts)
