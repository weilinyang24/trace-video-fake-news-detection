from __future__ import annotations

import json
import random
import re
import hashlib
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import _load_annotations, _load_ids, _normalize_label
from .io import read_jsonl, write_json_atomic
from .labels import normalize_prediction
from .metrics import compute_metrics
from .runner import (
    _load_video_frames,
    _metadata_value,
    _normalize_attn_implementation,
    _torch_dtype,
    load_config,
    load_model,
    resolve_path,
)


def _json_from_text(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _json_or_raw(text: str, fallback_key: str, error_key: str = "parse_error") -> dict[str, Any]:
    try:
        return _json_from_text(text)
    except Exception as exc:
        return {
            fallback_key: text[:4000],
            error_key: f"{type(exc).__name__}: {exc}",
        }


def _judge_from_raw(text: str) -> dict[str, Any]:
    text = _strip_thinking(text)
    judge = _json_or_raw(text, "judge_raw_summary")
    if "prediction" in judge:
        return judge
    label_match = re.search(r"^\s*(?:LABEL|PREDICTION)\s*[:：]\s*(real|fake)\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if label_match:
        prediction, parse_ok = label_match.group(1).lower(), True
    else:
        prediction, parse_ok = normalize_prediction(text)
    score_match = re.search(r"(?:FAKE_SCORE|fake_score|score)\s*[:：]?\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    fake_score = float(score_match.group(1)) if score_match else (0.5 if parse_ok else 0.5)
    confidence_match = re.search(r"(?:CONFIDENCE|confidence)\s*[:：]?\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    confidence = float(confidence_match.group(1)) if confidence_match else (0.5 if parse_ok else 0.49)
    reason_match = re.search(r"(?:REASON|reason)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else (text or "Empty judge output.")[:500]
    return {
        **judge,
        "prediction": prediction if parse_ok else "fake",
        "fake_score": fake_score,
        "confidence": confidence,
        "key_factors": ["parsed compact judge label format"],
        "rationale": reason[:500],
        "parse_fallback": True,
    }


def _strip_thinking(text: str) -> str:
    value = text or ""
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"^\s*<think>.*", "", value, flags=re.DOTALL | re.IGNORECASE)
    return value.strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _video_file_fingerprint(path: Path) -> str:
    if not path.exists():
        return _sha256_text(f"missing:{path}")
    stat = path.stat()
    return _sha256_text(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")


def _visual_cache_key(
    *,
    dataset: str,
    sample_id: str,
    video_path: Path,
    model_path: str,
    prompt: str,
    max_video_frames: int,
    visual_max_new_tokens: int,
    schema_version: int,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": schema_version,
        "dataset": dataset,
        "sample_id": sample_id,
        "video": _video_file_fingerprint(video_path),
        "model_path": model_path,
        "prompt_hash": _sha256_text(prompt),
        "max_video_frames": max_video_frames,
        "visual_max_new_tokens": visual_max_new_tokens,
    }
    return _sha256_text(_canonical_json(identity)), identity


class StageOneVisualCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage1_visual_cache (
                cache_key TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                identity_json TEXT NOT NULL,
                visual_json TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                backend TEXT,
                sampled_frames INTEGER,
                source_total_frames INTEGER,
                elapsed_seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_stage1_visual_cache_sample "
            "ON stage1_visual_cache(dataset, sample_id)"
        )
        self.connection.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT visual_json, raw_text, backend, sampled_frames, source_total_frames, elapsed_seconds
            FROM stage1_visual_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "visual": json.loads(row[0]),
            "raw_text": row[1],
            "backend": row[2],
            "sampled_frames": row[3],
            "source_total_frames": row[4],
            "elapsed_seconds": row[5],
        }

    def put(
        self,
        *,
        cache_key: str,
        identity: dict[str, Any],
        visual: dict[str, Any],
        raw_text: str,
        backend: str,
        sampled_frames: int,
        source_total_frames: int | None,
        elapsed_seconds: float,
    ) -> None:
        if "parse_error" in visual:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO stage1_visual_cache(
                    cache_key, dataset, sample_id, identity_json, visual_json, raw_text,
                    backend, sampled_frames, source_total_frames, elapsed_seconds,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    identity_json=excluded.identity_json,
                    visual_json=excluded.visual_json,
                    raw_text=excluded.raw_text,
                    backend=excluded.backend,
                    sampled_frames=excluded.sampled_frames,
                    source_total_frames=excluded.source_total_frames,
                    elapsed_seconds=excluded.elapsed_seconds,
                    updated_at=excluded.updated_at
                """,
                (
                    cache_key,
                    str(identity["dataset"]),
                    str(identity["sample_id"]),
                    _canonical_json(identity),
                    _canonical_json(visual),
                    raw_text,
                    backend,
                    sampled_frames,
                    source_total_frames,
                    elapsed_seconds,
                    now,
                    now,
                ),
            )


def _calibrate_prediction(judge: dict[str, Any]) -> dict[str, Any]:
    prediction = str(judge.get("prediction", "")).lower()
    rationale = str(judge.get("rationale", ""))
    factors = " ".join(str(item) for item in judge.get("key_factors", []))
    text = f"{rationale} {factors}".lower()
    weak_visual_only = any(
        phrase in text
        for phrase in [
            "does not show",
            "not show",
            "not visible",
            "not supported by visible",
            "no direct visual",
            "not directly shown",
            "incomplete visual",
        ]
    )
    strong_fake = any(
        phrase in text
        for phrase in [
            "conspiracy",
            "microchip",
            "microchips",
            "false context",
            "unrelated footage",
            "old footage",
            "contradict",
            "mismatch",
            "fabricated",
            "hoax",
            "rumor",
        ]
    )
    if prediction == "fake" and weak_visual_only and not strong_fake:
        calibrated = dict(judge)
        calibrated["prediction"] = "real"
        calibrated["fake_score"] = min(float(judge.get("fake_score") or 0.5), 0.45)
        calibrated["confidence"] = min(float(judge.get("confidence") or 0.5), 0.62)
        calibrated["key_factors"] = [
            *list(judge.get("key_factors", [])),
            "calibrated: weak visual non-proof alone is insufficient for fake",
        ]
        calibrated["rationale"] = (
            f"{rationale} Calibration changed the label to real because lack of direct visual proof alone "
            "is not sufficient evidence of fake news."
        ).strip()
        return calibrated
    return judge


def _generate_text_with_video(
    model: Any,
    processor: Any,
    video_frames: Any,
    video_metadata: Any,
    prompt: str,
    fps: float,
    max_new_tokens: int,
    do_sample: bool = False,
) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "processor_kwargs": {"videos_kwargs": {"fps": fps, "video_metadata": [video_metadata]}},
    }
    try:
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)
    except (TypeError, ValueError) as exc:
        if "metadata" not in str(exc).lower():
            raise
        template_kwargs["processor_kwargs"] = {"videos_kwargs": {"fps": fps}}
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)
    generation = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation["pad_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation)
    generated_ids = []
    for input_ids, output in zip(inputs.input_ids, output_ids):
        input_length = input_ids.shape[0]
        has_prefix = output.shape[0] >= input_length and torch.equal(
            output[:input_length].to(input_ids.device), input_ids
        )
        generated_ids.append(output[input_length:] if has_prefix else output)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _uniform_sample_video(video_frames: Any, video_metadata: Any, max_frames: int, fps: float) -> tuple[Any, dict[str, Any]]:
    total = len(video_frames)
    if max_frames <= 0 or total <= max_frames:
        return video_frames, {
            "total_num_frames": total,
            "fps": fps,
            "duration": float(total) / fps if fps > 0 else None,
        }
    indices = [min(total - 1, int(index * total / max_frames)) for index in range(max_frames)]
    try:
        sampled = video_frames[indices]
    except Exception:
        sampled = [video_frames[index] for index in indices]
    original_duration = _metadata_value(video_metadata, "duration")
    if original_duration is None:
        original_fps = float(_metadata_value(video_metadata, "fps", 0) or 0)
        original_duration = float(total) / original_fps if original_fps > 0 else float(max_frames) / fps if fps > 0 else None
    return sampled, {
        "total_num_frames": len(sampled),
        "fps": fps,
        "duration": original_duration,
        "source_total_num_frames": total,
    }


def _load_text_model(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.get("text_model", {})
    model_path = str(model_config.get("path", "/data2/573ops_ser/models/Qwen3-8B"))
    attn_implementation = _normalize_attn_implementation(str(model_config.get("attn_implementation", "sdpa")))
    print(
        json.dumps(
            {
                "event": "load_text_model",
                "model_path": model_path,
                "attn_implementation": attn_implementation,
                "dtype": model_config.get("dtype", "bfloat16"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=_torch_dtype(torch, model_config.get("dtype", "bfloat16")),
        attn_implementation=attn_implementation,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    return model, tokenizer


def _generate_text_only(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    generation = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation["pad_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation)
    generated = output_ids[:, inputs.input_ids.shape[-1] :]
    return _strip_thinking(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_#@]+", lowered))
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    for item in cjk:
        words.update(item[index : index + 2] for index in range(max(0, len(item) - 1)))
    return words


def _current_context(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    marker = "Current sample to classify:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt


def _source_text(dataset: str, source: dict[str, Any]) -> str:
    if dataset == "fakett":
        fields = ["event", "description", "user_description", "publish_time"]
    else:
        fields = ["title", "keywords", "author_intro", "author_place", "publish_time_norm"]
    parts = [str(source.get(field, "") or "") for field in fields]
    if dataset == "fakesv":
        parts.extend(str(item) for item in (source.get("comments") or [])[:5])
    return "\n".join(part for part in parts if part.strip())


def _source_claim(dataset: str, source: dict[str, Any]) -> str:
    key = "event" if dataset == "fakett" else "title"
    value = str(source.get(key, "") or "").strip()
    if value:
        return value
    text = _source_text(dataset, source)
    return text.splitlines()[0].strip() if text.splitlines() else ""


def _load_dynamic_examples(
    repo_root: Path,
    dataset: str,
    current_text: str,
    per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    annotations = _load_annotations(repo_root / "data" / "annotations" / f"{dataset}.jsonl")
    train_ids = _load_ids(repo_root / "data" / "splits" / dataset / "train.txt")
    current_tokens = _tokens(current_text)
    buckets: dict[str, list[tuple[float, str, dict[str, Any]]]] = {"real": [], "fake": []}
    rng = random.Random(seed)
    for sample_id in train_ids:
        source = annotations.get(sample_id)
        if not source:
            continue
        label = _normalize_label(dataset, source, sample_id)
        if label not in buckets:
            continue
        text = _source_text(dataset, source)
        example_tokens = _tokens(text)
        union = current_tokens | example_tokens
        score = len(current_tokens & example_tokens) / len(union) if union else 0.0
        # deterministic tie breaker
        score += rng.random() * 1e-6
        buckets[label].append((score, sample_id, source))
    examples: list[dict[str, Any]] = []
    for label in ("real", "fake"):
        ranked = sorted(buckets[label], key=lambda item: item[0], reverse=True)[:per_label]
        for score, sample_id, source in ranked:
            examples.append(
                {
                    "id": sample_id,
                    "label": label,
                    "similarity": round(score, 4),
                    "claim": _source_claim(dataset, source),
                    "text": _source_text(dataset, source)[:1200],
                }
            )
    return examples


def _visual_prompt(row: dict[str, Any], dataset: str) -> str:
    current_context = _current_context(row)
    if dataset.lower() == "fakesv":
        return (
            "你是中文短视频虚假新闻检测中的视觉证据提取器。\n"
            "不要直接判断 real/fake，只提取当前视频和辅助文本中可以观察到的证据。\n"
            "FakeSV 中 title 是主要待核查新闻声称；keywords、发布者信息、地点和评论只是辅助上下文，不是独立事实证明。\n"
            "虚假新闻也可能使用真实画面，所以要区分：画面直接证明、字幕/旁白/评论重复声称、弱相关画面、以及画面与声称不匹配。\n\n"
            "每个列表最多3项。不要输出Markdown、代码块或额外解释。只返回合法JSON，key必须使用英文：\n"
            "{\n"
            '  "claim_in_auxiliary_text": "string",\n'
            '  "visible_scene_summary": "string",\n'
            '  "visible_text_or_captions": ["string"],\n'
            '  "visible_entities": ["string"],\n'
            '  "video_claim_relation": "directly_shows_claim | narrates_or_repeats_claim | weakly_related | unrelated_or_mismatch | unclear",\n'
            '  "possible_mismatch_cues": ["string"],\n'
            '  "rumor_or_sensational_cues": ["string"],\n'
            '  "ordinary_news_cues": ["string"]\n'
            "}\n\n"
            f"当前样本辅助文本：\n{current_context}"
        )
    return (
        "You are a visual evidence extractor for short-video fake-news detection.\n"
        "Do NOT decide real/fake. Extract only what the current video and the provided auxiliary text show.\n"
        "Fake-news videos can repeat or dramatize their own claim, so distinguish direct visual proof from narration/caption conveyance.\n\n"
        "Keep every string short. Use at most 3 items per list. Do not include markdown fences, comments, or extra text.\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "claim_in_auxiliary_text": "string",\n'
        '  "visible_scene_summary": "string",\n'
        '  "visible_text_or_captions": ["string"],\n'
        '  "visible_entities": ["string"],\n'
        '  "video_claim_relation": "directly_shows_claim | narrates_or_repeats_claim | weakly_related | unrelated_or_mismatch | unclear",\n'
        '  "possible_mismatch_cues": ["string"],\n'
        '  "rumor_or_sensational_cues": ["string"],\n'
        '  "ordinary_news_cues": ["string"]\n'
        "}\n\n"
        f"Auxiliary text for the current sample:\n{current_context}"
    )


def _judge_prompt(row: dict[str, Any], visual: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    current_context = _current_context(row)
    return (
        "You are a calibrated judge for short-video fake-news detection.\n"
        "Classify whether the CURRENT video news item is real or fake for dataset evaluation.\n\n"
        "Important principles:\n"
        "1. This is not deepfake detection. A visually authentic clip can still be fake news by false context or overclaim.\n"
        "2. Do NOT mark fake merely because the video does not fully prove the claim. Lack of direct visual proof is only a weak signal.\n"
        "3. Many real samples are ordinary news, science, experiment, explainer, or public-affairs posts with limited internal proof. "
        "If such a claim is plausible and there is no clear contradiction, choose real.\n"
        "4. Do not mark real merely because the video/caption repeats the claim. Fake videos can self-support their own story.\n"
        "5. Prefer fake only for strong misinformation patterns: explicit claim-video contradiction, unrelated/old-footage false context, "
        "clear conspiracy/rumor framing, fabricated extraordinary claim, or title/caption making a stronger claim than the video/text can support.\n"
        "6. Marketing language, emotional wording, or incomplete visual demonstration alone is not enough for fake.\n"
        "7. Use external common knowledge only for widely known false conspiracy claims (for example vaccine microchips), not for every uncertain claim.\n\n"
        "Decision calibration:\n"
        "- If evidence is weak/ambiguous and the claim is ordinary or plausible, output LABEL: real with moderate confidence.\n"
        "- If the main reason is only 'not shown in the sampled frames', output LABEL: real unless there is another strong fake-news cue.\n"
        "- If the claim is extraordinary, conspiratorial, or clearly mismatched with the video, output LABEL: fake.\n\n"
        "Use the retrieved training examples only to calibrate the dataset boundary, not as factual evidence.\n"
        f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"current_auxiliary_prompt: {current_context}\n\n"
        f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
        "Do not output hidden reasoning or <think> blocks. Return exactly 4 lines in this format:\n"
        "LABEL: real or fake\n"
        "FAKE_SCORE: number_between_0_and_1\n"
        "CONFIDENCE: number_between_0_and_1\n"
        "REASON: one concise sentence"
    )


def _judge_review_prompt(
    row: dict[str, Any],
    visual: dict[str, Any],
    examples: list[dict[str, Any]],
    initial_judge: dict[str, Any],
    retrieval_prior: dict[str, Any],
    dataset: str,
) -> str:
    current_context = _current_context(row)
    if dataset.lower() == "fakesv":
        return (
            "你是二阶段中文虚假新闻裁决的复核 Agent。\n"
            "请检查初判是否犯了这些错误：过度相信画面/字幕重复、因为缺少完整证明而误判 fake、忽略训练集检索边界、"
            "或没有发现当前视频中的画面-声称不匹配。\n\n"
            "修正规则：\n"
            "- 只有存在当前视频内部矛盾、无关/旧画面、伪造/谣言结构，或 fake 检索先验与当前谣言化线索一致时，才建议 real->fake。\n"
            "- 如果 fake 的理由只是证据不完整、情绪化表达、评论质疑或不确定性，尤其检索偏 real 时，建议 fake->real。\n"
            "- 如果证据混合，保持初判并降低信心。\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"initial_judge_json: {json.dumps(initial_judge, ensure_ascii=False)}\n\n"
            "只返回合法 JSON，key 必须使用英文：\n"
            "{\n"
            '  "needs_revision": true,\n'
            '  "suggested_label": "real | fake | keep",\n'
            '  "confidence_delta": -0.1,\n'
            '  "review_reason": "一句中文理由",\n'
            '  "risk_type": "overtrust_visual_repetition | weak_nonproof | retrieval_ignored | current_mismatch | none"\n'
            "}"
        )
    return (
        "You are a verifier agent for the second-stage fake-news judge.\n"
        "Review the initial decision for dataset-calibrated short-video fake-news detection. "
        "Do not use hidden reasoning. Focus on whether the initial judge over-trusted visual repetition, "
        "over-penalized missing proof, or ignored retrieved training examples.\n\n"
        "Correction policy:\n"
        "- Recommend changing real->fake only for strong current evidence: claim-video contradiction, old/unrelated footage, hoax/fabrication, "
        "or fake-leaning retrieval prior plus current rumor/sensational cues.\n"
        "- Recommend changing fake->real when the only fake reason is weak visual non-proof, emotional wording, or uncertainty, "
        "especially if retrieved examples favor real.\n"
        "- If evidence is genuinely mixed, keep the initial label and lower confidence.\n\n"
        f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
        f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"current_auxiliary_prompt: {current_context}\n\n"
        f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
        f"initial_judge_json: {json.dumps(initial_judge, ensure_ascii=False)}\n\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "needs_revision": true,\n'
        '  "suggested_label": "real | fake | keep",\n'
        '  "confidence_delta": -0.1,\n'
        '  "review_reason": "one concise sentence",\n'
        '  "risk_type": "overtrust_visual_repetition | weak_nonproof | retrieval_ignored | current_mismatch | none"\n'
        "}"
    )


def _judge_final_prompt(
    row: dict[str, Any],
    visual: dict[str, Any],
    examples: list[dict[str, Any]],
    initial_judge: dict[str, Any],
    review: dict[str, Any],
    retrieval_prior: dict[str, Any],
    dataset: str,
) -> str:
    current_context = _current_context(row)
    if dataset.lower() == "fakesv":
        return (
            "你是中文短视频虚假新闻检测二阶段 loop 的最终裁决器。\n"
            "请综合初判和复核意见，输出最终 real/fake。检索样本用于校准数据集边界，但不是当前事件的事实证明。\n\n"
            "最终规则：\n"
            "- 只有复核指出具体当前视频证据或明确检索边界问题时，才改变初判。\n"
            "- 不要因为抽样视频缺少完整证明就判 fake。\n"
            "- 不要因为字幕、评论或画面重复异常声称就判 real。\n"
            "- 证据混合时使用中等置信度。\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"initial_judge_json: {json.dumps(initial_judge, ensure_ascii=False)}\n\n"
            f"verifier_review_json: {json.dumps(review, ensure_ascii=False)}\n\n"
            "不要输出思考过程或 <think>。严格输出4行，key必须使用英文：\n"
            "LABEL: real or fake\n"
            "FAKE_SCORE: number_between_0_and_1\n"
            "CONFIDENCE: number_between_0_and_1\n"
            "REASON: 用一句中文简要说明最终依据"
        )
    return (
        "You are the final arbiter in a lightweight judge loop for short-video fake-news detection.\n"
        "Use the initial decision and verifier review, but output your own final label. "
        "Respect dataset-calibrated retrieved examples without treating them as factual proof.\n\n"
        "Final arbitration rules:\n"
        "- Change the initial label only when the verifier names a concrete current-video cue or a clear retrieval-boundary issue.\n"
        "- Do not label fake merely because the sampled video lacks complete proof.\n"
        "- Do not label real merely because captions or visuals repeat an extraordinary claim.\n"
        "- Prefer moderate confidence when evidence is mixed.\n\n"
        f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
        f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"current_auxiliary_prompt: {current_context}\n\n"
        f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
        f"initial_judge_json: {json.dumps(initial_judge, ensure_ascii=False)}\n\n"
        f"verifier_review_json: {json.dumps(review, ensure_ascii=False)}\n\n"
        "Do not output hidden reasoning or <think> blocks. Return exactly 4 lines in this format:\n"
        "LABEL: real or fake\n"
        "FAKE_SCORE: number_between_0_and_1\n"
        "CONFIDENCE: number_between_0_and_1\n"
        "REASON: one concise sentence"
    )


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        if not row.get("error") and row.get("parse_ok", False):
            completed.add(str(row["id"]))
    return completed


def evaluate_prediction_file(prediction_path: Path) -> dict[str, Any]:
    latest = {str(row["id"]): row for row in read_jsonl(prediction_path)}
    failed = sum(bool(row.get("error")) for row in latest.values())
    parse_failures = sum(not row.get("parse_ok", False) for row in latest.values())
    if failed or parse_failures:
        invalid = [sample_id for sample_id, row in latest.items() if row.get("error") or not row.get("parse_ok", False)]
        raise RuntimeError(
            f"Strict metrics are undefined: failed={failed}, parse_failures={parse_failures}, "
            f"total={len(latest)}, invalid_id_examples={invalid[:10]}"
        )
    metrics = compute_metrics(
        [str(row["label"]) for row in latest.values()],
        [str(row["prediction"]) for row in latest.values()],
    )
    metrics["attempted"] = len(latest)
    metrics["failed"] = failed
    metrics["parse_failures"] = parse_failures
    return metrics


def _current_claim_text(row: dict[str, Any]) -> str:
    context = _current_context(row)
    claim_match = re.search(r"<claim>(.*?)</claim>", context, flags=re.IGNORECASE | re.DOTALL)
    if claim_match:
        return " ".join(claim_match.group(1).split())
    for line in context.splitlines():
        if "claim" in line.lower() and ":" in line:
            return line.split(":", 1)[1].strip()
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _normalize_claim_text(text: str) -> str:
    value = text.lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return " ".join(value.split())


def _retrieval_prior(examples: list[dict[str, Any]], current_claim: str, margin: float = 0.025) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {"real": [], "fake": []}
    current_norm = _normalize_claim_text(current_claim)
    for example in examples:
        label = str(example.get("label", "")).lower()
        if label not in by_label:
            continue
        text_lines = str(example.get("text") or "").splitlines()
        claim = str(example.get("claim") or (text_lines[0] if text_lines else "")).strip()
        same_claim = bool(current_norm and _normalize_claim_text(claim) == current_norm)
        by_label[label].append(
            {
                "id": example.get("id"),
                "similarity": float(example.get("similarity") or 0.0),
                "same_claim": same_claim,
                "claim": claim,
            }
        )
    top_real = max((item["similarity"] for item in by_label["real"]), default=0.0)
    top_fake = max((item["similarity"] for item in by_label["fake"]), default=0.0)
    same_real = sum(bool(item["same_claim"]) for item in by_label["real"])
    same_fake = sum(bool(item["same_claim"]) for item in by_label["fake"])
    if same_real > same_fake:
        prior = "real"
        strength = 0.9
        reason = "same-claim retrieved examples favor real"
    elif same_fake > same_real:
        prior = "fake"
        strength = 0.9
        reason = "same-claim retrieved examples favor fake"
    elif top_real >= top_fake + margin:
        prior = "real"
        strength = min(0.85, 0.5 + (top_real - top_fake) * 3)
        reason = "nearest retrieved examples favor real"
    elif top_fake >= top_real + margin:
        prior = "fake"
        strength = min(0.85, 0.5 + (top_fake - top_real) * 3)
        reason = "nearest retrieved examples favor fake"
    else:
        prior = "neutral"
        strength = 0.0
        reason = "retrieved examples are balanced"
    return {
        "prior": prior,
        "strength": round(strength, 4),
        "reason": reason,
        "top_real_similarity": round(top_real, 4),
        "top_fake_similarity": round(top_fake, 4),
        "same_claim_real": same_real,
        "same_claim_fake": same_fake,
        "margin": margin,
    }


def _judge_from_raw(text: str) -> dict[str, Any]:
    text = _strip_thinking(text)
    judge = _json_or_raw(text, "judge_raw_summary")
    if "prediction" in judge:
        return judge
    label_match = re.search(r"^\s*(?:LABEL|PREDICTION)\s*[:：]\s*(real|fake)\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if label_match:
        prediction, parse_ok = label_match.group(1).lower(), True
    else:
        prediction, parse_ok = normalize_prediction(text)
    score_match = re.search(r"(?:FAKE_SCORE|fake_score|score)\s*[:：]?\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    fake_score = float(score_match.group(1)) if score_match else (0.5 if parse_ok else 0.5)
    confidence_match = re.search(r"(?:CONFIDENCE|confidence)\s*[:：]?\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    confidence = float(confidence_match.group(1)) if confidence_match else (0.5 if parse_ok else 0.49)
    reason_match = re.search(r"(?:REASON|reason)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else (text or "Empty judge output.")[:500]
    return {
        **judge,
        "prediction": prediction if parse_ok else "fake",
        "fake_score": fake_score,
        "confidence": confidence,
        "key_factors": ["parsed compact judge label format"],
        "rationale": reason[:500],
        "parse_fallback": True,
    }


def _calibrate_prediction(judge: dict[str, Any], retrieval_prior: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction = str(judge.get("prediction", "")).lower()
    rationale = str(judge.get("rationale", ""))
    factors = " ".join(str(item) for item in judge.get("key_factors", []))
    text = f"{rationale} {factors}".lower()
    weak_visual_only = any(
        phrase in text
        for phrase in [
            "does not show",
            "not show",
            "not visible",
            "not supported by visible",
            "no direct visual",
            "not directly shown",
            "incomplete visual",
            "without evidence in the visual",
        ]
    )
    strong_fake = any(
        phrase in text
        for phrase in [
            "false context",
            "unrelated footage",
            "old footage",
            "fabricated",
            "hoax",
        ]
    )
    prior = retrieval_prior or {}
    prior_label = str(prior.get("prior", "neutral")).lower()
    prior_strength = float(prior.get("strength", 0.0) or 0.0)
    if prediction == "fake" and prior_label == "real" and prior_strength >= 0.5 and not strong_fake:
        calibrated = dict(judge)
        calibrated["prediction"] = "real"
        calibrated["fake_score"] = min(float(judge.get("fake_score") or 0.5), 0.42)
        calibrated["confidence"] = min(float(judge.get("confidence") or 0.5), 0.68)
        calibrated["key_factors"] = [
            *list(judge.get("key_factors", [])),
            "calibrated: retrieved train-set prior favors real",
        ]
        calibrated["rationale"] = (
            f"{rationale} Calibration changed the label to real because highly similar retrieved "
            "training examples favor the dataset's real label."
        ).strip()
        return calibrated
    if prediction == "fake" and weak_visual_only and not strong_fake:
        calibrated = dict(judge)
        calibrated["prediction"] = "real"
        calibrated["fake_score"] = min(float(judge.get("fake_score") or 0.5), 0.45)
        calibrated["confidence"] = min(float(judge.get("confidence") or 0.5), 0.62)
        calibrated["key_factors"] = [
            *list(judge.get("key_factors", [])),
            "calibrated: weak visual non-proof alone is insufficient for fake",
        ]
        calibrated["rationale"] = (
            f"{rationale} Calibration changed the label to real because lack of direct visual proof alone "
            "is not sufficient evidence of fake news."
        ).strip()
        return calibrated
    return judge


def _judge_prompt(row: dict[str, Any], visual: dict[str, Any], examples: list[dict[str, Any]], dataset: str) -> str:
    current_context = _current_context(row)
    retrieval_prior = _retrieval_prior(examples, _current_claim_text(row))
    if dataset.lower() == "fakesv":
        return (
            "你是中文短视频虚假新闻检测的校准裁决器。请判断当前样本在数据集中的标签：real 或 fake。\n\n"
            "核心原则：\n"
            "1. 这不是深度伪造检测。画面看起来真实，不代表新闻声称真实；真实旧视频配错标题也属于虚假新闻。\n"
            "2. 在 FakeSV 中，title 是当前待核查新闻声称。keywords、发布者简介、地点和评论是辅助上下文，不是独立事实证明。\n"
            "3. 不要因为抽样画面没有完整证明声称就直接判 fake；缺少直接视觉证明只是弱信号。\n"
            "4. 字幕、旁白、标题或评论重复声称，不是独立证明。\n"
            "5. 普通新闻、科普、实验、历史或公共事件内容，在没有当前视频内强反证时可以判 real。\n"
            "6. 优先判 fake 的强证据包括：当前视频内部明确矛盾、画面与声称高置信不匹配、无关/旧画面冒充当前事件、伪造/谣言结构，"
            "或声称中的关键身份、地点、原因、数量、速度等细节明显超出画面支持范围。\n"
            "7. 营销语言、情绪化表达、争议性、评论区质疑或不完整演示本身不足以判 fake。\n\n"
            "检索先验规则：\n"
            "- retrieval_prior 来自训练集相似样本，是数据集边界校准线索，不是当前事件的外部事实证明。\n"
            "- 如果检索和当前证据冲突且没有决定性观察，降低 CONFIDENCE，而不是编造证据。\n"
            "- 不要按主题相似度机械复制训练样本标签。\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            "不要输出思考过程或 <think>。严格输出4行，key必须使用英文：\n"
            "LABEL: real or fake\n"
            "FAKE_SCORE: number_between_0_and_1\n"
            "CONFIDENCE: number_between_0_and_1\n"
            "REASON: 用一句中文简要说明决定性当前视频证据"
        )
    return (
        "You are a calibrated judge for short-video fake-news detection.\n"
        "Classify whether the CURRENT video news item is real or fake for dataset evaluation.\n\n"
        "Important principles:\n"
        "1. This is not deepfake detection. A visually authentic clip can still be fake news by false context or overclaim.\n"
        "2. Do NOT mark fake merely because the video does not fully prove the claim. Lack of direct visual proof is only a weak signal.\n"
        "3. Many real samples are ordinary news, science, experiment, explainer, history, or public-affairs posts with limited internal proof. "
        "If such a claim is plausible and there is no direct contradiction, choose real.\n"
        "4. Do not use external world knowledge to override the dataset-calibrated retrieved examples. "
        "This benchmark's labels are calibrated by the copied training split, not by open-world fact checking alone.\n"
        "5. Prefer fake only for strong misinformation patterns: explicit claim-video contradiction, unrelated/old-footage false context, "
        "fabricated/hoax evidence, or current-post rumor framing that is stronger than the retrieval prior.\n"
        "6. Marketing language, emotional wording, or incomplete visual demonstration alone is not enough for fake.\n\n"
        "Retrieval prior rules:\n"
        "- retrieval_prior is computed from train-set examples and is a strong dataset-boundary cue.\n"
        "- If retrieval_prior is real, predict fake only for old/unrelated footage, explicit contradiction, fabricated/hoax evidence, or strong current-post rumor framing.\n"
        "- If retrieval_prior is fake, predict real only when current video/text clearly presents ordinary coherent news or direct support.\n"
        "- If retrieval_prior is neutral, rely on visual evidence and current text.\n\n"
        "Decision calibration:\n"
        "- If evidence is weak/ambiguous and the claim is ordinary or plausible, output LABEL: real with moderate confidence.\n"
        "- If the main reason is only 'not shown in the sampled frames', output LABEL: real unless there is another strong fake-news cue.\n"
        "- If retrieved real examples have the same claim as the current sample, treat LABEL: real as the default unless the current video uses old/unrelated footage or a hoax format.\n\n"
        f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
        f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"current_auxiliary_prompt: {current_context}\n\n"
        f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
        "Do not output hidden reasoning or <think> blocks. Return exactly 4 lines in this format:\n"
        "LABEL: real or fake\n"
        "FAKE_SCORE: number_between_0_and_1\n"
        "CONFIDENCE: number_between_0_and_1\n"
        "REASON: one concise sentence"
    )


def run_two_stage(
    config_path: Path,
    repo_root: Path,
    video_root_override: Path | None = None,
    text_model_path_override: str | None = None,
    limit: int | None = None,
    resume: bool = False,
    prediction_path_override: Path | None = None,
    metrics_path_override: Path | None = None,
    refresh_visual_cache_override: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if text_model_path_override:
        config.setdefault("text_model", {})["path"] = text_model_path_override
    two_stage = config.get("two_stage", {})
    dataset_name = str(config["dataset"]["name"]).lower()
    manifest_path = resolve_path(repo_root, config["dataset"]["manifest"])
    video_root = video_root_override or Path(config["dataset"]["video_root"])
    prediction_value = two_stage.get("predictions")
    if prediction_value is None:
        prediction_value = str(config["output"]["predictions"]).replace(".jsonl", "_twostage.jsonl")
    metrics_value = two_stage.get("metrics")
    if metrics_value is None:
        metrics_value = str(config["output"]["metrics"]).replace(".json", "_twostage.json")
    prediction_path = prediction_path_override or resolve_path(repo_root, prediction_value)
    metrics_path = metrics_path_override or resolve_path(repo_root, metrics_value)
    rows = list(read_jsonl(manifest_path))
    if limit is not None:
        rows = rows[:limit]

    import torch

    seed = int(config["generation"].get("seed", 2025))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    vlm, processor = load_model(config)
    text_model, tokenizer = _load_text_model(config)
    completed = _completed_ids(prediction_path) if resume else set()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(config["video"]["fps"])
    decoder_backend = str(config["video"].get("decoder_backend", "decord"))
    fallback_num_frames = int(config["video"].get("fallback_num_frames", 16))
    visual_tokens = int(two_stage.get("visual_max_new_tokens", 512))
    judge_tokens = int(two_stage.get("judge_max_new_tokens", 384))
    per_label = int(two_stage.get("retrieval_per_label", 2))
    max_video_frames = int(two_stage.get("max_video_frames", 64))
    cache_config = two_stage.get("visual_cache", {})
    visual_cache = (
        StageOneVisualCache(resolve_path(repo_root, cache_config.get("path", "outputs/cache/stage1_visual.sqlite3")))
        if bool(cache_config.get("enabled", True))
        else None
    )
    cache_schema_version = int(cache_config.get("schema_version", 1))
    refresh_visual_cache = bool(cache_config.get("refresh", False)) or refresh_visual_cache_override
    loop_config = two_stage.get("judge_loop", {})
    loop_enabled = bool(loop_config.get("enabled", True))
    review_tokens = int(loop_config.get("review_max_new_tokens", 256))
    final_tokens = int(loop_config.get("final_max_new_tokens", judge_tokens))
    visual_cache_hits = 0
    visual_cache_misses = 0

    with prediction_path.open("a", encoding="utf-8", newline="\n") as output:
        for index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            if sample_id in completed:
                continue
            started = time.perf_counter()
            result = {"id": sample_id, "label": row["label"], "video": row["video"]}
            try:
                video_path = video_root / row["video"]
                visual_prompt = _visual_prompt(row, dataset_name)
                cache_key, cache_identity = _visual_cache_key(
                    dataset=dataset_name,
                    sample_id=sample_id,
                    video_path=video_path,
                    model_path=str(config.get("model", {}).get("path", "")),
                    prompt=visual_prompt,
                    max_video_frames=max_video_frames,
                    visual_max_new_tokens=visual_tokens,
                    schema_version=cache_schema_version,
                )
                cached_visual = None if refresh_visual_cache or visual_cache is None else visual_cache.get(cache_key)
                if cached_visual is not None:
                    visual_cache_hits += 1
                    visual = cached_visual["visual"]
                    visual_raw = cached_visual["raw_text"]
                    backend = cached_visual.get("backend")
                    print(
                        json.dumps(
                            {
                                "event": "stage1_visual_cache_hit",
                                "id": sample_id,
                                "video": row["video"],
                                "cache_key": cache_key,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    visual_cache_misses += 1
                    visual_started = time.perf_counter()
                    video_frames, video_metadata, backend, decoder_errors = _load_video_frames(
                        video_path,
                        fps,
                        decoder_backend,
                        fallback_num_frames,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "decode_video",
                                "video": row["video"],
                                "backend": backend,
                                "sampled_frames": len(video_frames),
                                "total_frames": _metadata_value(video_metadata, "total_num_frames"),
                                "fallback_errors": decoder_errors,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    video_frames, model_video_metadata = _uniform_sample_video(video_frames, video_metadata, max_video_frames, fps)
                    print(
                        json.dumps(
                            {
                                "event": "sample_video_for_model",
                                "video": row["video"],
                                "model_frames": len(video_frames),
                                "source_total_frames": _metadata_value(model_video_metadata, "source_total_num_frames"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    visual_raw = _generate_text_with_video(
                        vlm,
                        processor,
                        video_frames,
                        model_video_metadata,
                        visual_prompt,
                        fps,
                        max_new_tokens=visual_tokens,
                        do_sample=False,
                    )
                    visual = _json_or_raw(visual_raw, "visual_raw_summary")
                    if visual_cache is not None:
                        visual_cache.put(
                            cache_key=cache_key,
                            identity=cache_identity,
                            visual=visual,
                            raw_text=visual_raw,
                            backend=str(backend),
                            sampled_frames=len(video_frames),
                            source_total_frames=_metadata_value(model_video_metadata, "source_total_num_frames"),
                            elapsed_seconds=round(time.perf_counter() - visual_started, 4),
                        )
                examples = _load_dynamic_examples(
                    repo_root,
                    dataset_name,
                    _current_context(row),
                    per_label=per_label,
                    seed=seed,
                )
                retrieval_prior = _retrieval_prior(examples, _current_claim_text(row))
                judge_raw = _generate_text_only(
                    text_model,
                    tokenizer,
                    _judge_prompt(row, visual, examples, dataset_name),
                    max_new_tokens=judge_tokens,
                    do_sample=False,
                )
                initial_judge = _calibrate_prediction(_judge_from_raw(judge_raw), retrieval_prior)
                review_raw = ""
                review = {}
                final_raw = ""
                judge = initial_judge
                if loop_enabled:
                    review_raw = _generate_text_only(
                        text_model,
                        tokenizer,
                        _judge_review_prompt(row, visual, examples, initial_judge, retrieval_prior, dataset_name),
                        max_new_tokens=review_tokens,
                        do_sample=False,
                    )
                    review = _json_or_raw(review_raw, "review_raw_summary")
                    final_raw = _generate_text_only(
                        text_model,
                        tokenizer,
                        _judge_final_prompt(row, visual, examples, initial_judge, review, retrieval_prior, dataset_name),
                        max_new_tokens=final_tokens,
                        do_sample=False,
                    )
                    judge = _calibrate_prediction(_judge_from_raw(final_raw), retrieval_prior)
                prediction, parse_ok = normalize_prediction(str(judge.get("prediction", "")))
                result.update(
                    {
                        "prediction": prediction,
                        "parse_ok": parse_ok,
                        "fake_score": judge.get("fake_score"),
                        "confidence": judge.get("confidence"),
                        "key_factors": judge.get("key_factors", []),
                        "rationale": judge.get("rationale", ""),
                        "visual_evidence": visual,
                        "retrieved_examples": examples,
                        "retrieval_prior": retrieval_prior,
                        "visual_raw": visual_raw,
                        "judge_raw": judge_raw,
                        "judge_loop_enabled": loop_enabled,
                        "initial_judge": initial_judge,
                        "review": review,
                        "review_raw": review_raw,
                        "final_judge_raw": final_raw,
                        "visual_cache_hit": cached_visual is not None,
                        "visual_cache_key": cache_key,
                        "error": None,
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "prediction": "fake",
                        "parse_ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({"index": index, **{k: v for k, v in result.items() if k not in {"visual_evidence", "visual_raw", "judge_raw", "retrieved_examples"}}}, ensure_ascii=False), flush=True)

    if limit is not None:
        return {
            "mode": "smoke_test",
            "requested": len(rows),
            "predictions": str(prediction_path),
            "metrics_written": False,
            "visual_cache_hits": visual_cache_hits,
            "visual_cache_misses": visual_cache_misses,
            "judge_loop_enabled": loop_enabled,
        }
    metrics = evaluate_prediction_file(prediction_path)
    metrics["visual_cache_hits"] = visual_cache_hits
    metrics["visual_cache_misses"] = visual_cache_misses
    metrics["judge_loop_enabled"] = loop_enabled
    write_json_atomic(metrics_path, metrics)
    return metrics
