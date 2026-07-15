from __future__ import annotations

from datetime import datetime, timezone

FAKE_NEWS_PROMPT_EN = """Determine whether the news claim or event conveyed by this short video and the auxiliary observations is factually real or fake.

This task is short-video news veracity detection, not deepfake detection. Judge whether the news conveyed by the video and its associated context is true as stated.

Label the sample fake if the conveyed news is false, debunked, misleading in context, old footage presented as a new event, mismatched with the claimed time, place, person, or event, or uses unrelated or repurposed footage to support a false claim.

Label the sample real if the conveyed news content is factually true and the video plus accompanying context do not misleadingly distort the event.

Do not decide based only on uploader identity, political stance, hashtags, emotional tone, or whether the footage merely looks visually authentic. Use them only as auxiliary context.

Reply with exactly one word: real or fake."""

FAKE_NEWS_PROMPT_ZH = """请判断这个短视频及其辅助信息所表达的新闻事件或主张是真实还是虚假。

这是一项短视频新闻真实性判断任务，不是深度伪造检测任务。你需要判断视频及其相关文本上下文所表达的新闻内容是否属实。

如果新闻内容错误、已被辟谣、存在断章取义，或使用旧视频、无关素材，或视频与声称的时间、地点、人物、事件不匹配，则标为 fake。

如果新闻内容事实正确，且视频及上下文没有误导性地歪曲事件，则标为 real。

不要仅根据发布者身份、立场、情绪、标签热度或画面是否逼真来判断；这些只能作为辅助信息。

只回答一个词：real 或 fake。"""


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


def build_fakett_prompt(row: dict) -> str:
    parts = [FAKE_NEWS_PROMPT_EN, "", "Auxiliary observations:"]
    fields = (
        ("news claim to verify", "event", True),
        ("uploader caption", "description", False),
        ("uploader profile", "user_description", False),
    )
    for title, key, claim in fields:
        value = _text(row.get(key))
        if value:
            value = f"<claim>{value}</claim>" if claim else value
            parts.append(f"- {title}: {value}")
    publish_time = _time(row.get("publish_time"))
    if publish_time:
        parts.append(f"- publish time metadata: {publish_time}")
    return "\n".join(parts)


def build_fakesv_prompt(row: dict, max_comments: int = 3) -> str:
    parts = [FAKE_NEWS_PROMPT_ZH, "", "辅助信息："]
    fields = (
        ("待核验新闻主张", "title", True),
        ("事件关键词", "keywords", False),
        ("发布者简介", "author_intro", False),
        ("发布者地区", "author_place", False),
    )
    for title, key, claim in fields:
        value = _text(row.get(key))
        if value:
            value = f"<claim>{value}</claim>" if claim else value
            parts.append(f"- {title}: {value}")
    publish_time = _time(row.get("publish_time_norm"))
    if publish_time:
        parts.append(f"- 发布时间: {publish_time}")
    comments = [_text(item).replace("\n", " ") for item in (row.get("comments") or [])]
    comments = [item for item in comments if item][:max_comments]
    if comments:
        parts.append(f"- 部分评论: {' | '.join(comments)}")
    return "\n".join(parts)

