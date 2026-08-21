from __future__ import annotations

import json
import random
import re
import hashlib
import sqlite3
import shutil
import subprocess
import tempfile
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


def _audio_cache_key(
    *,
    dataset: str,
    sample_id: str,
    video_path: Path,
    model_path: str,
    prompt: str,
    audio_max_new_tokens: int,
    schema_version: int,
    sample_rate: int,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": schema_version,
        "dataset": dataset,
        "sample_id": sample_id,
        "video": _video_file_fingerprint(video_path),
        "model_path": model_path,
        "prompt_hash": _sha256_text(prompt),
        "audio_max_new_tokens": audio_max_new_tokens,
        "sample_rate": sample_rate,
    }
    return _sha256_text(_canonical_json(identity)), identity


class AudioEvidenceCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_evidence_cache (
                cache_key TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                identity_json TEXT NOT NULL,
                audio_json TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                backend TEXT,
                elapsed_seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_audio_evidence_cache_sample "
            "ON audio_evidence_cache(dataset, sample_id)"
        )
        self.connection.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT audio_json, raw_text, backend, elapsed_seconds
            FROM audio_evidence_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "audio": json.loads(row[0]),
            "raw_text": row[1],
            "backend": row[2],
            "elapsed_seconds": row[3],
        }

    def put(
        self,
        *,
        cache_key: str,
        identity: dict[str, Any],
        audio: dict[str, Any],
        raw_text: str,
        backend: str,
        elapsed_seconds: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO audio_evidence_cache(
                    cache_key, dataset, sample_id, identity_json, audio_json, raw_text,
                    backend, elapsed_seconds, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    identity_json=excluded.identity_json,
                    audio_json=excluded.audio_json,
                    raw_text=excluded.raw_text,
                    backend=excluded.backend,
                    elapsed_seconds=excluded.elapsed_seconds,
                    updated_at=excluded.updated_at
                """,
                (
                    cache_key,
                    str(identity["dataset"]),
                    str(identity["sample_id"]),
                    _canonical_json(identity),
                    _canonical_json(audio),
                    raw_text,
                    backend,
                    elapsed_seconds,
                    now,
                    now,
                ),
            )


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


def _load_audio_model(config: dict[str, Any]):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2AudioForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Current transformers does not expose Qwen2AudioForConditionalGeneration. "
            "Please upgrade transformers or install a version that supports Qwen2-Audio."
        ) from exc

    audio_config = config.get("audio_model", {})
    model_path = str(audio_config.get("path", "/data2/573ops_ser/models/Qwen2-Audio-7B-Instruct"))
    attn_implementation = _normalize_attn_implementation(str(audio_config.get("attn_implementation", "sdpa")))
    print(
        json.dumps(
            {
                "event": "load_audio_model",
                "model_path": model_path,
                "attn_implementation": attn_implementation,
                "dtype": audio_config.get("dtype", "bfloat16"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model_kwargs = {
        "attn_implementation": attn_implementation,
        "device_map": audio_config.get("device_map", "auto"),
        "trust_remote_code": bool(audio_config.get("trust_remote_code", True)),
    }
    dtype = _torch_dtype(torch, audio_config.get("dtype", "bfloat16"))
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            **model_kwargs,
        ).eval()
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **model_kwargs,
        ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=bool(audio_config.get("trust_remote_code", True)),
    )
    return model, processor


def _extract_audio_wav(video_path: Path, output_path: Path, sample_rate: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable is not available; cannot extract audio")
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg audio extraction failed for {video_path}")
    if not output_path.exists() or output_path.stat().st_size <= 44:
        raise RuntimeError(f"ffmpeg produced empty audio for {video_path}")


def _generate_text_with_audio(
    model: Any,
    processor: Any,
    wav_path: Path,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
) -> str:
    import torch

    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("librosa is required for Qwen2-Audio inference; install librosa soundfile") from exc

    sample_rate = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000) or 16000)
    audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(wav_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=rendered, audios=[audio], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    generation = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation["pad_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation)
    generated = output_ids[:, inputs.input_ids.shape[-1] :]
    return _strip_thinking(processor.batch_decode(generated, skip_special_tokens=True)[0].strip())


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


def _audio_prompt(row: dict[str, Any], dataset: str) -> str:
    current_context = _current_context(row)
    if dataset.lower() == "fakesv":
        return (
            "你是中文短视频虚假新闻检测中的音频证据提取器。\n"
            "请只分析音频中的旁白、说话内容、语气、背景声和可能的煽动性表达，不要直接判断 real/fake。\n"
            "注意：旁白或字幕朗读 title 只是重复声称，不等于证明声称真实；情绪化语气只能作为弱线索。\n\n"
            "只返回合法 JSON，key 必须使用英文：\n"
            "{\n"
            '  "audio_available": true,\n'
            '  "speech_transcript": "string",\n'
            '  "spoken_claims": ["string"],\n'
            '  "speaker_style": "news_report | narration | interview | casual | dramatic | unknown",\n'
            '  "emotion_tone": "neutral | urgent | angry | fearful | excited | sad | unknown",\n'
            '  "rumor_or_sensational_audio_cues": ["string"],\n'
            '  "audio_visual_consistency": "consistent | mismatch | audio_only_claim | unclear",\n'
            '  "audio_title_consistency": "supports_title | contradicts_title | repeats_title_only | unrelated | unclear",\n'
            '  "decision_relevance": "high | medium | low",\n'
            '  "summary": "一句中文摘要"\n'
            "}\n\n"
            f"当前样本辅助文本：\n{current_context}"
        )
    return (
        "You are an audio evidence extractor for short-video fake-news detection.\n"
        "Analyze only speech, narration, tone, background sound, and sensational audio cues. Do NOT decide real/fake.\n"
        "Important: narration repeating the title is not independent proof; emotional tone is only a weak cue.\n\n"
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "audio_available": true,\n'
        '  "speech_transcript": "string",\n'
        '  "spoken_claims": ["string"],\n'
        '  "speaker_style": "news_report | narration | interview | casual | dramatic | unknown",\n'
        '  "emotion_tone": "neutral | urgent | angry | fearful | excited | sad | unknown",\n'
        '  "rumor_or_sensational_audio_cues": ["string"],\n'
        '  "audio_visual_consistency": "consistent | mismatch | audio_only_claim | unclear",\n'
        '  "audio_title_consistency": "supports_title | contradicts_title | repeats_title_only | unrelated | unclear",\n'
        '  "decision_relevance": "high | medium | low",\n'
        '  "summary": "one concise sentence"\n'
        "}\n\n"
        f"Auxiliary text for the current sample:\n{current_context}"
    )


def _judge_review_prompt(
    row: dict[str, Any],
    visual: dict[str, Any],
    audio: dict[str, Any],
    examples: list[dict[str, Any]],
    initial_judge: dict[str, Any],
    retrieval_prior: dict[str, Any],
    dataset: str,
) -> str:
    current_context = _current_context(row)
    if dataset.lower() == "fakesv":
        return (
            "你是二阶段中文虚假新闻裁决的复核 Agent。\n"
            "你的默认任务是验证初判，而不是重新自由裁决。除非发现高置信错误，否则必须保持初判。\n\n"
            "修正规则：\n"
            "- 默认 suggested_label 必须是 keep。\n"
            "- 当初判与 retrieval_prior 强冲突，且当前视觉/音频没有支持初判的强证据时，可以建议改判。\n"
            "- 如果 initial=real 但 retrieval_prior=fake 且 strength>=0.75，并且当前视频主要只是字幕/旁白重复声称、标题党/谣言化表达、无直接真实新闻证据，可以建议 real->fake。\n"
            "- 如果 initial=fake 但 retrieval_prior=real 且 strength>=0.75，并且当前视频/音频直接呈现普通新闻、可信来源或完整事件链，可以建议 fake->real。\n"
            "- initial=fake 且 retrieval_prior=fake 时，禁止建议 fake->real，除非视频/音频明确直接证明新闻声称为真。\n"
            "- initial=real 且 retrieval_prior=real 时，禁止建议 real->fake，除非存在明确当前视频内部矛盾、无关/旧画面、伪造/谣言结构。\n"
            "- 不要因为“缺少完整证明”“情绪化表达”“评论质疑”“音频重复标题”建议改判。\n"
            "- 如果证据混合、不确定、或只是弱线索，保持初判并降低信心。\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
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
        "Your default job is to validate the initial decision, not to freely re-decide the label. "
        "Recommend a label change only for high-confidence initial-judge errors.\n\n"
        "Correction policy:\n"
        "- Default suggested_label must be keep.\n"
        "- You may suggest a change when the initial label strongly conflicts with retrieval_prior AND current visual/audio evidence does not support it.\n"
        "- If initial=real but retrieval_prior=fake with strength>=0.75, and the current video mainly repeats the claim through text/audio, uses sensational/rumor framing, or lacks direct ordinary-news verification, you may suggest real->fake.\n"
        "- If initial=fake but retrieval_prior=real with strength>=0.75, and the current video/audio directly presents ordinary news, credible source material, or a coherent event chain, you may suggest fake->real.\n"
        "- If initial=fake and retrieval_prior=fake, do NOT recommend fake->real unless the current video/audio directly verifies the claim as ordinary true news.\n"
        "- If initial=real and retrieval_prior=real, do NOT recommend real->fake unless there is explicit contradiction, old/unrelated footage, fabrication, or hoax structure.\n"
        "- Never change a label merely because proof is incomplete, wording is emotional, comments are skeptical, or audio repeats the title.\n"
        "- If evidence is mixed or uncertain, keep the initial label and lower confidence.\n\n"
        f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
        f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"current_auxiliary_prompt: {current_context}\n\n"
        f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
        f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
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
    audio: dict[str, Any],
    examples: list[dict[str, Any]],
    initial_judge: dict[str, Any],
    review: dict[str, Any],
    retrieval_prior: dict[str, Any],
    dataset: str,
    freeform_label_space: bool = False,
) -> str:
    current_context = _current_context(row)
    if freeform_label_space:
        if dataset.lower() == "fakesv":
            return (
                "你是中文短视频虚假新闻检测二阶段 loop 的最终裁决器。\n"
                "这是标签空间控制消融：所有证据组件都可用，但不要按固定 `LABEL: real/fake` 格式输出，也不要为了标签平衡人为偏向任一类别。\n"
                "请自然语言判断当前短视频新闻整体更可信/真实，还是更不可信/虚假/误导；即使证据不完全，也必须给出一个倾向，方便后续评估解析。\n\n"
                f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
                f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
                f"current_auxiliary_prompt: {current_context}\n\n"
                f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
                f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
                f"initial_judge_json: {json.dumps(initial_judge, ensure_ascii=False)}\n\n"
                f"verifier_review_json: {json.dumps(review, ensure_ascii=False)}\n\n"
                "请用一小段中文回答，不要输出 JSON、Markdown、LABEL 字段或 <think>。"
            )
        return (
            "You are the final arbiter in a lightweight judge loop for short-video fake-news detection.\n"
            "This is a label-space-control ablation: all evidence components are available, but do not use the fixed `LABEL: real/fake` output format "
            "and do not artificially bias toward either class for label balancing.\n"
            "Answer freely in natural language whether the current video news is more credible/true or more false/misleading. "
            "Even if evidence is incomplete, state a clear leaning so the evaluator can parse it.\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
            f"initial_judge_json: {json.dumps(initial_judge, ensure_ascii=False)}\n\n"
            f"verifier_review_json: {json.dumps(review, ensure_ascii=False)}\n\n"
            "Answer in one concise paragraph. Do not output JSON, markdown, LABEL fields, or <think>."
        )
    if dataset.lower() == "fakesv":
        return (
            "你是中文短视频虚假新闻检测二阶段 loop 的最终裁决器。\n"
            "请以初判为默认最终结果；复核意见只用于发现高置信初判错误。检索样本用于校准数据集边界，但不是当前事件的事实证明。\n\n"
            "最终规则：\n"
            "- 默认输出 initial_judge 的 LABEL。\n"
            "- 只有 verifier_review 明确 suggested_label 不是 keep，且给出高置信当前视频/音频证据时，才允许改判。\n"
            "- initial=fake 且 retrieval_prior=fake 时，极难改成 real；除非当前视频/音频直接证明标题声称是真实普通新闻。\n"
            "- initial=real 且 retrieval_prior=real 时，极难改成 fake；除非存在明确矛盾、旧/无关画面、伪造或谣言结构。\n"
            "- 不要因为缺少完整证明、情绪化语气、音频重复标题、或复核中的泛泛不确定性改判。\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
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
        "Treat the initial decision as the default final label. The verifier review is only for catching high-confidence initial-judge errors. "
        "Respect dataset-calibrated retrieved examples without treating them as factual proof.\n\n"
        "Final arbitration rules:\n"
        "- Default to the initial_judge LABEL.\n"
        "- Change the label only when verifier_review suggests a non-keep label AND names high-confidence current visual/audio evidence.\n"
        "- If initial=fake and retrieval_prior=fake, almost never change to real unless current video/audio directly verifies the claim as ordinary true news.\n"
        "- If initial=real and retrieval_prior=real, almost never change to fake unless there is explicit contradiction, old/unrelated footage, fabrication, or hoax structure.\n"
        "- Do not change because proof is incomplete, tone is emotional, audio repeats the title, or the review is merely uncertain.\n"
        "- Prefer keeping initial with adjusted confidence when evidence is mixed.\n\n"
        f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
        f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
        f"current_auxiliary_prompt: {current_context}\n\n"
        f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
        f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
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


def _evidence_text(*items: Any) -> str:
    return " ".join(json.dumps(item, ensure_ascii=False).lower() for item in items)


def _apply_conservative_loop_gate(
    *,
    initial_judge: dict[str, Any],
    review: dict[str, Any],
    final_judge: dict[str, Any],
    retrieval_prior: dict[str, Any],
    visual: dict[str, Any],
    audio: dict[str, Any],
) -> dict[str, Any]:
    initial_label = str(initial_judge.get("prediction", "")).lower()
    final_label = str(final_judge.get("prediction", "")).lower()
    if initial_label not in {"real", "fake"} or final_label not in {"real", "fake"}:
        return initial_judge
    if final_label == initial_label:
        kept = dict(final_judge)
        kept["loop_gate"] = "kept_final_same_as_initial"
        return kept

    suggested = str(review.get("suggested_label", "keep")).lower()
    review_reason = str(review.get("review_reason", ""))
    review_text = _evidence_text(review, review_reason)
    evidence_text = _evidence_text(visual, audio, final_judge)
    support_text = f"{review_text} {evidence_text}"
    prior_label = str(retrieval_prior.get("prior", "neutral")).lower()
    prior_strength = float(retrieval_prior.get("strength", 0.0) or 0.0)
    final_confidence = float(final_judge.get("confidence", 0.0) or 0.0)

    strong_fake_cue = any(
        phrase in evidence_text
        for phrase in [
            "contradiction",
            "contradicts",
            "mismatch",
            "unrelated_or_mismatch",
            "unrelated footage",
            "old footage",
            "fabricated",
            "hoax",
            "false context",
            "misleading",
            "sensational",
            "sensationalized",
            "clickbait",
            "text overlay",
            "text overlays",
            "narrates_or_repeats_claim",
            "rumor",
            "谣言",
            "伪造",
            "矛盾",
            "不匹配",
            "无关",
            "旧画面",
            "误导",
            "标题党",
            "夸张",
        ]
    )
    strong_real_cue = any(
        phrase in evidence_text
        for phrase in [
            "ordinary coherent news",
            "ordinary news",
            "ordinary news elements",
            "directly verifies",
            "directly supports",
            "directly shows",
            "credible source",
            "credible context",
            "support real news",
            "supports real news",
            "similar claims support real",
            "does not clearly contradict",
            "not clearly contradict",
            "普通新闻",
            "普通新闻元素",
            "频道报道",
            "客服介入",
            "正在介入",
            "乡村频道",
            "未发现明确伪造",
            "未发现明确伪造或矛盾",
            "直接证明",
            "直接支持",
            "可信来源",
        ]
    )
    hard_fake_cue = any(
        phrase in evidence_text
        for phrase in [
            "explicit contradiction",
            "direct contradiction",
            "contradicts the claim",
            "old footage",
            "unrelated footage",
            "fabricated",
            "hoax",
            "false context",
            "明确矛盾",
            "直接矛盾",
            "旧画面",
            "无关画面",
            "伪造",
            "谣言结构",
        ]
    )
    negated_hard_fake_cue = any(
        phrase in evidence_text
        for phrase in [
            "no clear evidence of fabrication",
            "no clear evidence of contradiction",
            "does not clearly contradict",
            "not clearly contradict",
            "未发现明确伪造",
            "未发现明确伪造或矛盾",
            "没有明确伪造",
            "没有明确矛盾",
            "无明确伪造",
            "无明确矛盾",
        ]
    )
    if negated_hard_fake_cue:
        hard_fake_cue = False
    review_or_final_supports_fake = suggested == "fake" or any(
        phrase in support_text
        for phrase in [
            "strongly favors fake",
            "favors fake",
            "supports fake",
            "supporting the fake",
            "fake label",
            "fake examples",
            "fake prior",
            "aligning with the retrieval prior",
            "aligns with the retrieval prior",
            "supporting a fake label",
            "支持 fake",
            "支持虚假",
            "偏向 fake",
            "强烈偏向 fake",
        ]
    )
    review_or_final_supports_real = suggested == "real" or any(
        phrase in support_text
        for phrase in [
            "strongly favors real",
            "favors real",
            "supports real",
            "support real news",
            "supports real news",
            "real label",
            "real examples",
            "real prior",
            "ordinary news",
            "ordinary news elements",
            "credible source",
            "credible context",
            "符合 real",
            "支持 real",
            "支持真实",
            "普通新闻",
            "普通新闻元素",
            "可信来源",
        ]
    )
    weak_change_reason = any(
        phrase in review_text
        for phrase in [
            "lack of direct visual proof",
            "lacks direct visual proof",
            "insufficient evidence",
            "not enough evidence",
            "uncertain",
            "weak_nonproof",
            "缺少完整证明",
            "证据不完整",
            "不确定",
        ]
    )

    allow = False
    reason = "blocked_by_conservative_loop_gate"
    if suggested == final_label and not weak_change_reason:
        if final_label == "fake":
            allow = (
                (prior_label == "fake" and prior_strength >= 0.75 and strong_fake_cue)
                or (hard_fake_cue and final_confidence >= 0.8 and not strong_real_cue)
            )
            reason = "allowed_real_to_fake_strong_fake_cue" if allow else reason
        elif final_label == "real":
            allow = strong_real_cue or (prior_label == "real" and prior_strength >= 0.65 and strong_real_cue)
            reason = "allowed_fake_to_real_strong_real_cue" if allow else reason

    if initial_label == "real" and final_label == "fake" and prior_label == "fake" and prior_strength >= 0.75:
        prior_aligned_fake_cue = any(
            phrase in evidence_text
            for phrase in [
                "text overlay",
                "text overlays",
                "misleading caption",
                "meme",
                "no credible news cue",
                "no credible news cues",
                "does not support the claim",
                "explicitly states the claim is false",
                "hoax",
                "fabricated",
                "false context",
            ]
        )
        allow = bool(review_or_final_supports_fake and prior_aligned_fake_cue)
        reason = "allowed_real_to_fake_strong_fake_prior_with_repetition_or_misleading_cue" if allow else reason

    if initial_label == "fake" and final_label == "real" and prior_label == "real" and prior_strength >= 0.75:
        prior_aligned_real_cue = any(
            phrase in evidence_text
            for phrase in [
                "directly_shows_claim",
                "directly supports",
                "directly verifies",
                "ordinary coherent news",
                "ordinary news",
                "ordinary news elements",
                "credible source",
                "credible context",
                "support real news",
                "supports real news",
                "similar claims support real",
                "not clearly contradict",
                "does not clearly contradict",
                "news_report",
                "普通新闻",
                "普通新闻元素",
                "频道报道",
                "客服介入",
                "正在介入",
                "乡村频道",
                "未发现明确伪造",
                "未发现明确伪造或矛盾",
                "可信来源",
            ]
        )
        allow = bool(review_or_final_supports_real and prior_aligned_real_cue and not hard_fake_cue)
        reason = "allowed_fake_to_real_strong_real_prior_with_direct_support" if allow else reason

    if initial_label == "fake" and final_label == "real" and prior_label == "neutral":
        allow = bool(review_or_final_supports_real and strong_real_cue and not hard_fake_cue)
        reason = "allowed_fake_to_real_neutral_prior_with_ordinary_news_evidence" if allow else reason

    if initial_label == "fake" and final_label == "real" and prior_label == "fake" and prior_strength >= 0.5:
        allow = bool(strong_real_cue and suggested == "real" and not weak_change_reason)
        reason = "allowed_fake_to_real_against_fake_prior_with_direct_real_evidence" if allow else "blocked_fake_to_real_against_fake_prior"
    if initial_label == "real" and final_label == "fake" and prior_label == "real" and prior_strength >= 0.5:
        allow = bool(strong_fake_cue and suggested == "fake" and not weak_change_reason)
        reason = "allowed_real_to_fake_against_real_prior_with_strong_fake_evidence" if allow else "blocked_real_to_fake_against_real_prior"

    if allow:
        accepted = dict(final_judge)
        accepted["loop_gate"] = reason
        return accepted

    kept = dict(initial_judge)
    kept["loop_gate"] = reason
    kept["blocked_final_prediction"] = final_label
    kept["blocked_final_rationale"] = final_judge.get("rationale", "")
    kept["key_factors"] = [
        *list(initial_judge.get("key_factors", [])),
        reason,
    ]
    return kept


def _judge_prompt(
    row: dict[str, Any],
    visual: dict[str, Any],
    audio: dict[str, Any],
    examples: list[dict[str, Any]],
    dataset: str,
    freeform_label_space: bool = False,
) -> str:
    current_context = _current_context(row)
    retrieval_prior = _retrieval_prior(examples, _current_claim_text(row))
    if freeform_label_space:
        if dataset.lower() == "fakesv":
            return (
                "你是中文短视频虚假新闻检测的校准裁决器。\n"
                "这是标签空间控制消融：视觉、音频、RAG 与检索先验都可使用，但不要按固定 `LABEL: real/fake` 格式输出，"
                "也不要为了标签平衡人为偏向任一类别。\n"
                "请自然语言判断当前短视频新闻整体更可信/真实，还是更不可信/虚假/误导；即使证据不完全，也必须给出一个倾向，方便后续评估解析。\n\n"
                f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
                f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
                f"current_auxiliary_prompt: {current_context}\n\n"
                f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
                f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
                "请用一小段中文回答，不要输出 JSON、Markdown、LABEL 字段或 <think>。"
            )
        return (
            "You are a calibrated judge for short-video fake-news detection.\n"
            "This is a label-space-control ablation: visual evidence, audio evidence, RAG examples, and retrieval prior are available, "
            "but do not use the fixed `LABEL: real/fake` output format and do not artificially bias toward either class for label balancing.\n"
            "Answer freely in natural language whether the current video news is more credible/true or more false/misleading. "
            "Even if evidence is incomplete, state a clear leaning so the evaluator can parse it.\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
            "Answer in one concise paragraph. Do not output JSON, markdown, LABEL fields, or <think>."
        )
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
            "音频证据规则：\n"
            "- audio_evidence_json 是旁白/语气/背景声辅助证据。\n"
            "- 旁白重复 title 或字幕，不等于独立证明。\n"
            "- 情绪化、急迫、夸张语气只能作为弱 fake cue，不能单独决定 fake。\n"
            "- 如果音频中的关键声称与画面/title 明显矛盾，或只有音频在制造关键误导声称，这是强 fake cue。\n\n"
            "检索先验规则：\n"
            "- retrieval_prior 来自训练集相似样本，是数据集边界校准线索，不是当前事件的外部事实证明。\n"
            "- 如果检索和当前证据冲突且没有决定性观察，降低 CONFIDENCE，而不是编造证据。\n"
            "- 不要按主题相似度机械复制训练样本标签。\n\n"
            f"retrieval_prior: {json.dumps(retrieval_prior, ensure_ascii=False)}\n\n"
            f"retrieved_training_examples: {json.dumps(examples, ensure_ascii=False)}\n\n"
            f"current_auxiliary_prompt: {current_context}\n\n"
            f"visual_evidence_json: {json.dumps(visual, ensure_ascii=False)}\n\n"
            f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
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
        "Audio evidence rules:\n"
        "- audio_evidence_json contains speech, tone, and background sound cues.\n"
        "- Narration repeating the title/caption is not independent proof.\n"
        "- Emotional, urgent, or dramatic tone is only a weak fake cue and cannot decide fake by itself.\n"
        "- If audio makes a key claim that contradicts the video/title or introduces the decisive misleading context, treat it as strong fake evidence.\n\n"
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
        f"audio_evidence_json: {json.dumps(audio, ensure_ascii=False)}\n\n"
        "Do not output hidden reasoning or <think> blocks. Return exactly 4 lines in this format:\n"
        "LABEL: real or fake\n"
        "FAKE_SCORE: number_between_0_and_1\n"
        "CONFIDENCE: number_between_0_and_1\n"
        "REASON: one concise sentence"
    )


def _direct_vlm_rag_prompt(
    row: dict[str, Any],
    examples: list[dict[str, Any]],
    dataset: str,
    use_rag: bool = True,
    freeform_label_space: bool = False,
) -> str:
    current_context = _current_context(row)
    if freeform_label_space:
        if dataset.lower() == "fakesv":
            return (
                "你是中文短视频新闻真实性分析助手。请直接观看当前视频，并阅读当前样本文本。\n"
                "不要使用检索样例、先验、文本 Judge、review loop 或音频证据。\n"
                "请用自然语言回答：这个短视频新闻整体是否可信、是否存在虚假或误导风险，并说明主要依据。\n"
                "不要被要求在固定标签集合中选择，也不要输出固定格式。\n\n"
                f"当前样本文本：\n{current_context}"
            )
        return (
            "You are a short-video news credibility analyst. Watch the current video and read the current sample text.\n"
            "Do not use retrieved examples, priors, a separate text Judge, a review loop, or audio evidence.\n"
            "Answer freely in natural language: is this short-video news credible, or does it appear false/misleading? "
            "Explain the main basis for your assessment.\n"
            "Do not choose from a predefined label set and do not use a fixed output format.\n\n"
            f"Current sample text:\n{current_context}"
        )
    rag_block_zh = (
        f"retrieved_train_examples_json: {json.dumps(examples, ensure_ascii=False)}\n\n"
        if use_rag
        else ""
    )
    rag_block_en = (
        f"retrieved_train_examples_json: {json.dumps(examples, ensure_ascii=False)}\n\n"
        if use_rag
        else ""
    )
    if dataset.lower() == "fakesv":
        mode_line = (
            "这是 E5 消融实验：只允许使用 Qwen3-VL + RAG examples。不要调用文本 Judge、不要做多轮 review、不要使用音频证据。\n\n"
            if use_rag
            else "这是 E6 消融实验：只允许使用 Qwen3-VL 直接观看视频和当前样本文本。不要使用 RAG、retrieval prior、文本 Judge、review loop 或音频证据。\n\n"
        )
        rag_rule = (
            "2. RAG 样例来自训练集相似样本，用于理解数据集标签边界；它不是外部事实证明，也不要把样例标签简单投票当成最终答案。\n"
            if use_rag
            else "2. 当前实验不提供任何检索样例或先验；只能依据当前视频与当前样本文本判断。\n"
        )
        return (
            "你是中文短视频虚假新闻检测模型。请直接观看当前视频，并结合检索到的训练集相似样本，判断当前样本标签：real 或 fake。\n\n"
            f"{mode_line}"
            "判断原则：\n"
            "1. title 是主要待核查新闻声称；keywords、发布者信息、地点和评论只是辅助上下文。\n"
            f"{rag_rule}"
            "3. 不要因为画面缺少完整证明就直接判 fake；短视频常常只呈现事件片段。\n"
            "4. 优先判 fake 的强线索包括：画面与标题关键事实明显不匹配、旧/无关画面冒充当前事件、字幕/旁白只是重复谣言、标题党/夸张营销、伪造或谣言结构。\n"
            "5. 如果视频呈现普通新闻、现场记录、科普实验或完整事件链，且没有明确矛盾，倾向 real。\n\n"
            f"当前样本辅助文本：\n{current_context}\n\n"
            f"{rag_block_zh}"
            "不要输出 JSON、Markdown 或思维过程。严格输出 4 行：\n"
            "LABEL: real or fake\n"
            "FAKE_SCORE: number_between_0_and_1\n"
            "CONFIDENCE: number_between_0_and_1\n"
            "REASON: 一句中文理由"
        )
    mode_line = (
        "This is the E5 ablation: Qwen3-VL + RAG examples only. Do not use a separate text Judge, review loop, or audio evidence.\n\n"
        if use_rag
        else "This is the E6 ablation: Qwen3-VL direct inference only. Do not use RAG, retrieval prior, a separate text Judge, review loop, or audio evidence.\n\n"
    )
    rag_rule = (
        "2. Retrieved examples are training-set neighbors that help calibrate dataset label boundaries. They are not external fact proof, and you must not simply vote by example labels.\n"
        if use_rag
        else "2. No retrieved examples or priors are provided in this experiment; rely only on the current video and current sample text.\n"
    )
    return (
        "You are a short-video fake-news detector. Watch the current video directly and classify the current sample as real or fake.\n\n"
        f"{mode_line}"
        "Decision principles:\n"
        "1. The event/description is the claim to verify; hashtags and user text are auxiliary context.\n"
        f"{rag_rule}"
        "3. Do not classify fake merely because the sampled video lacks full visual proof; short videos often show partial evidence.\n"
        "4. Strong fake cues include mismatch between video and claim, old/unrelated footage, captions/narration merely repeating a rumor, sensational framing, hoax/fabrication structure, or false context.\n"
        "5. If the video shows ordinary news/reporting/science/event footage with no clear contradiction, prefer real.\n\n"
        f"Current sample auxiliary text:\n{current_context}\n\n"
        f"{rag_block_en}"
        "Do not output JSON, markdown, or hidden reasoning. Return exactly 4 lines:\n"
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

    ablation_config = two_stage.get("ablation", {})
    direct_vlm_rag = bool(ablation_config.get("direct_vlm_rag", False))
    freeform_label_space = bool(ablation_config.get("freeform_label_space", False))
    text_model = None
    tokenizer = None
    if not direct_vlm_rag:
        text_model, tokenizer = _load_text_model(config)
    vlm = None
    processor = None
    audio_model = None
    audio_processor = None
    completed = _completed_ids(prediction_path) if resume else set()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(config["video"]["fps"])
    decoder_backend = str(config["video"].get("decoder_backend", "decord"))
    fallback_num_frames = int(config["video"].get("fallback_num_frames", 16))
    visual_tokens = int(two_stage.get("visual_max_new_tokens", 512))
    judge_tokens = int(two_stage.get("judge_max_new_tokens", 384))
    direct_vlm_tokens = int(two_stage.get("direct_vlm_max_new_tokens", judge_tokens))
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
    use_audio_evidence = bool(ablation_config.get("use_audio", True))
    use_rag = bool(ablation_config.get("use_rag", True))
    use_retrieval_prior = bool(ablation_config.get("use_retrieval_prior", use_rag))
    use_loop = bool(ablation_config.get("use_loop", True))
    active_ablation = {
        "use_audio": use_audio_evidence,
        "use_rag": use_rag,
        "use_retrieval_prior": use_retrieval_prior,
        "use_loop": use_loop,
        "direct_vlm_rag": direct_vlm_rag,
        "freeform_label_space": freeform_label_space,
    }
    loop_config = two_stage.get("judge_loop", {})
    loop_enabled = bool(loop_config.get("enabled", True)) and use_loop
    review_tokens = int(loop_config.get("review_max_new_tokens", 256))
    final_tokens = int(loop_config.get("final_max_new_tokens", judge_tokens))
    visual_cache_hits = 0
    visual_cache_misses = 0
    audio_config = two_stage.get("audio_cache", {})
    audio_enabled = bool(audio_config.get("enabled", False)) and use_audio_evidence
    audio_cache = (
        AudioEvidenceCache(resolve_path(repo_root, audio_config.get("path", "outputs/cache/audio_evidence.sqlite3")))
        if audio_enabled
        else None
    )
    audio_schema_version = int(audio_config.get("schema_version", 1))
    audio_tokens = int(audio_config.get("max_new_tokens", 512))
    audio_sample_rate = int(audio_config.get("sample_rate", 16000))
    refresh_audio_cache = bool(audio_config.get("refresh", False))
    live_audio_on_miss = bool(audio_config.get("live_on_miss", False))
    audio_cache_hits = 0
    audio_cache_misses = 0

    with prediction_path.open("a", encoding="utf-8", newline="\n") as output:
        for index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            if sample_id in completed:
                continue
            started = time.perf_counter()
            result = {"id": sample_id, "label": row["label"], "video": row["video"]}
            try:
                video_path = video_root / row["video"]
                if direct_vlm_rag:
                    if use_rag:
                        examples = _load_dynamic_examples(
                            repo_root,
                            dataset_name,
                            _current_context(row),
                            per_label=per_label,
                            seed=seed,
                        )
                    else:
                        examples = []
                    if vlm is None or processor is None:
                        vlm, processor = load_model(config)
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
                                "mode": "direct_vlm_rag",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    direct_raw = _generate_text_with_video(
                        vlm,
                        processor,
                        video_frames,
                        model_video_metadata,
                        _direct_vlm_rag_prompt(
                            row,
                            examples,
                            dataset_name,
                            use_rag=use_rag,
                            freeform_label_space=freeform_label_space,
                        ),
                        fps,
                        max_new_tokens=direct_vlm_tokens,
                        do_sample=False,
                    )
                    judge = _judge_from_raw(direct_raw)
                    prediction, parse_ok = normalize_prediction(str(judge.get("prediction", "")))
                    result.update(
                        {
                            "prediction": prediction,
                            "parse_ok": parse_ok,
                            "fake_score": judge.get("fake_score"),
                            "confidence": judge.get("confidence"),
                            "key_factors": judge.get("key_factors", []),
                            "rationale": judge.get("rationale", ""),
                            "retrieved_examples": examples,
                            "ablation": active_ablation,
                            "judge_raw": direct_raw,
                            "direct_vlm_rag_raw": direct_raw,
                            "judge_loop_enabled": False,
                            "visual_cache_hit": False,
                            "audio_cache_hit": False,
                            "error": None,
                        }
                    )
                    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "index": index,
                                **{
                                    k: v
                                    for k, v in result.items()
                                    if k not in {"judge_raw", "retrieved_examples", "direct_vlm_rag_raw"}
                                },
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
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
                    if vlm is None or processor is None:
                        vlm, processor = load_model(config)
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
                audio_prompt = _audio_prompt(row, dataset_name)
                audio_key, audio_identity = _audio_cache_key(
                    dataset=dataset_name,
                    sample_id=sample_id,
                    video_path=video_path,
                    model_path=str(config.get("audio_model", {}).get("path", "")),
                    prompt=audio_prompt,
                    audio_max_new_tokens=audio_tokens,
                    schema_version=audio_schema_version,
                    sample_rate=audio_sample_rate,
                )
                cached_audio = None if refresh_audio_cache or audio_cache is None else audio_cache.get(audio_key)
                if not use_audio_evidence:
                    audio_raw = ""
                    audio = {
                        "audio_available": False,
                        "audio_disabled_by_ablation": True,
                        "decision_relevance": "low",
                        "summary": "Audio evidence was disabled by the ablation configuration.",
                    }
                elif cached_audio is not None:
                    audio_cache_hits += 1
                    audio = cached_audio["audio"]
                    audio_raw = cached_audio["raw_text"]
                    print(
                        json.dumps(
                            {
                                "event": "audio_evidence_cache_hit",
                                "id": sample_id,
                                "video": row["video"],
                                "cache_key": audio_key,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                elif audio_enabled and live_audio_on_miss:
                    audio_cache_misses += 1
                    if audio_model is None or audio_processor is None:
                        audio_model, audio_processor = _load_audio_model(config)
                    audio_started = time.perf_counter()
                    try:
                        with tempfile.TemporaryDirectory(prefix="agentvideommd_audio_") as tmpdir:
                            wav_path = Path(tmpdir) / f"{sample_id}.wav"
                            _extract_audio_wav(video_path, wav_path, audio_sample_rate)
                            audio_raw = _generate_text_with_audio(
                                audio_model,
                                audio_processor,
                                wav_path,
                                audio_prompt,
                                max_new_tokens=audio_tokens,
                                do_sample=False,
                            )
                        audio = _json_or_raw(audio_raw, "audio_raw_summary")
                    except Exception as audio_exc:
                        audio_raw = ""
                        audio = {
                            "audio_available": False,
                            "audio_error": f"{type(audio_exc).__name__}: {audio_exc}",
                            "decision_relevance": "low",
                            "summary": "Audio extraction or audio-model inference failed.",
                        }
                    if audio_cache is not None:
                        audio_cache.put(
                            cache_key=audio_key,
                            identity=audio_identity,
                            audio=audio,
                            raw_text=audio_raw,
                            backend="qwen2_audio_live",
                            elapsed_seconds=round(time.perf_counter() - audio_started, 4),
                        )
                else:
                    audio_cache_misses += 1 if audio_enabled else 0
                    audio_raw = ""
                    audio = {
                        "audio_available": False,
                        "audio_error": "audio cache miss; live extraction disabled",
                        "decision_relevance": "low",
                        "summary": "Audio evidence was not available for this judge run.",
                    }
                if use_rag:
                    examples = _load_dynamic_examples(
                        repo_root,
                        dataset_name,
                        _current_context(row),
                        per_label=per_label,
                        seed=seed,
                    )
                    retrieval_prior = _retrieval_prior(examples, _current_claim_text(row))
                else:
                    examples = []
                    retrieval_prior = {
                        "prior": "neutral",
                        "strength": 0.0,
                        "reason": "rag_disabled_by_ablation",
                        "top_real_similarity": 0.0,
                        "top_fake_similarity": 0.0,
                        "same_claim_real": 0,
                        "same_claim_fake": 0,
                        "margin": 0.0,
                        "hard_override_allowed": False,
                    }
                calibration_prior = retrieval_prior if use_rag else None
                judge_raw = _generate_text_only(
                    text_model,
                    tokenizer,
                    _judge_prompt(
                        row,
                        visual,
                        audio,
                        examples,
                        dataset_name,
                        freeform_label_space=freeform_label_space,
                    ),
                    max_new_tokens=judge_tokens,
                    do_sample=False,
                )
                initial_judge = _calibrate_prediction(_judge_from_raw(judge_raw), calibration_prior)
                review_raw = ""
                review = {}
                final_raw = ""
                final_judge = {}
                judge = initial_judge
                if loop_enabled:
                    review_raw = _generate_text_only(
                        text_model,
                        tokenizer,
                        _judge_review_prompt(row, visual, audio, examples, initial_judge, retrieval_prior, dataset_name),
                        max_new_tokens=review_tokens,
                        do_sample=False,
                    )
                    review = _json_or_raw(review_raw, "review_raw_summary")
                    final_raw = _generate_text_only(
                        text_model,
                        tokenizer,
                        _judge_final_prompt(
                            row,
                            visual,
                            audio,
                            examples,
                            initial_judge,
                            review,
                            retrieval_prior,
                            dataset_name,
                            freeform_label_space=freeform_label_space,
                        ),
                        max_new_tokens=final_tokens,
                        do_sample=False,
                    )
                    final_judge = _calibrate_prediction(_judge_from_raw(final_raw), calibration_prior)
                    judge = _apply_conservative_loop_gate(
                        initial_judge=initial_judge,
                        review=review,
                        final_judge=final_judge,
                        retrieval_prior=retrieval_prior,
                        visual=visual,
                        audio=audio,
                    )
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
                        "audio_evidence": audio,
                        "retrieved_examples": examples,
                        "retrieval_prior": retrieval_prior,
                        "ablation": active_ablation,
                        "visual_raw": visual_raw,
                        "judge_raw": judge_raw,
                        "judge_loop_enabled": loop_enabled,
                        "initial_judge": initial_judge,
                        "review": review,
                        "review_raw": review_raw,
                        "final_judge": final_judge,
                        "final_judge_raw": final_raw,
                        "loop_gate": judge.get("loop_gate"),
                        "visual_cache_hit": cached_visual is not None,
                        "visual_cache_key": cache_key,
                        "audio_raw": audio_raw,
                        "audio_cache_hit": cached_audio is not None,
                        "audio_cache_key": audio_key,
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
            print(json.dumps({"index": index, **{k: v for k, v in result.items() if k not in {"visual_evidence", "visual_raw", "audio_evidence", "audio_raw", "judge_raw", "retrieved_examples"}}}, ensure_ascii=False), flush=True)

    if limit is not None:
        return {
            "mode": "smoke_test",
            "requested": len(rows),
            "predictions": str(prediction_path),
            "metrics_written": False,
            "visual_cache_hits": visual_cache_hits,
            "visual_cache_misses": visual_cache_misses,
            "audio_cache_hits": audio_cache_hits,
            "audio_cache_misses": audio_cache_misses,
            "judge_loop_enabled": loop_enabled,
            "ablation": active_ablation,
        }
    metrics = evaluate_prediction_file(prediction_path)
    metrics["visual_cache_hits"] = visual_cache_hits
    metrics["visual_cache_misses"] = visual_cache_misses
    metrics["audio_cache_hits"] = audio_cache_hits
    metrics["audio_cache_misses"] = audio_cache_misses
    metrics["judge_loop_enabled"] = loop_enabled
    metrics["ablation"] = active_ablation
    write_json_atomic(metrics_path, metrics)
    return metrics


def build_evidence_cache(
    config_path: Path,
    repo_root: Path,
    video_root_override: Path | None = None,
    audio_model_path_override: str | None = None,
    limit: int | None = None,
    include_visual: bool = True,
    include_audio: bool = True,
    refresh: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if audio_model_path_override:
        config.setdefault("audio_model", {})["path"] = audio_model_path_override
    two_stage = config.get("two_stage", {})
    dataset_name = str(config["dataset"]["name"]).lower()
    manifest_path = resolve_path(repo_root, config["dataset"]["manifest"])
    video_root = video_root_override or Path(config["dataset"]["video_root"])
    rows = list(read_jsonl(manifest_path))
    if limit is not None:
        rows = rows[:limit]

    fps = float(config["video"]["fps"])
    decoder_backend = str(config["video"].get("decoder_backend", "decord"))
    fallback_num_frames = int(config["video"].get("fallback_num_frames", 16))
    visual_tokens = int(two_stage.get("visual_max_new_tokens", 512))
    max_video_frames = int(two_stage.get("max_video_frames", 64))
    visual_cache_config = two_stage.get("visual_cache", {})
    visual_cache = StageOneVisualCache(
        resolve_path(repo_root, visual_cache_config.get("path", "outputs/cache/stage1_visual.sqlite3"))
    )
    visual_schema_version = int(visual_cache_config.get("schema_version", 1))

    audio_config = two_stage.get("audio_cache", {})
    audio_cache = AudioEvidenceCache(resolve_path(repo_root, audio_config.get("path", "outputs/cache/audio_evidence.sqlite3")))
    audio_schema_version = int(audio_config.get("schema_version", 1))
    audio_tokens = int(audio_config.get("max_new_tokens", 512))
    audio_sample_rate = int(audio_config.get("sample_rate", 16000))

    stats: dict[str, Any] = {
        "dataset": dataset_name,
        "requested": len(rows),
        "visual": {"enabled": include_visual, "hits": 0, "misses": 0, "written": 0, "failed": 0},
        "audio": {"enabled": include_audio, "hits": 0, "misses": 0, "written": 0, "failed": 0},
        "visual_cache": str(visual_cache.path),
        "audio_cache": str(audio_cache.path),
    }

    if include_visual:
        vlm, processor = load_model(config)
        for index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            video_path = video_root / row["video"]
            prompt = _visual_prompt(row, dataset_name)
            cache_key, identity = _visual_cache_key(
                dataset=dataset_name,
                sample_id=sample_id,
                video_path=video_path,
                model_path=str(config.get("model", {}).get("path", "")),
                prompt=prompt,
                max_video_frames=max_video_frames,
                visual_max_new_tokens=visual_tokens,
                schema_version=visual_schema_version,
            )
            if not refresh and visual_cache.get(cache_key) is not None:
                stats["visual"]["hits"] += 1
                print(json.dumps({"event": "visual_cache_hit", "index": index, "id": sample_id}, ensure_ascii=False), flush=True)
                continue
            stats["visual"]["misses"] += 1
            started = time.perf_counter()
            try:
                video_frames, video_metadata, backend, decoder_errors = _load_video_frames(
                    video_path,
                    fps,
                    decoder_backend,
                    fallback_num_frames,
                )
                video_frames, model_video_metadata = _uniform_sample_video(video_frames, video_metadata, max_video_frames, fps)
                raw = _generate_text_with_video(
                    vlm,
                    processor,
                    video_frames,
                    model_video_metadata,
                    prompt,
                    fps,
                    max_new_tokens=visual_tokens,
                    do_sample=False,
                )
                visual = _json_or_raw(raw, "visual_raw_summary")
                visual_cache.put(
                    cache_key=cache_key,
                    identity=identity,
                    visual=visual,
                    raw_text=raw,
                    backend=str(backend),
                    sampled_frames=len(video_frames),
                    source_total_frames=_metadata_value(model_video_metadata, "source_total_num_frames"),
                    elapsed_seconds=round(time.perf_counter() - started, 4),
                )
                stats["visual"]["written"] += 1
                print(
                    json.dumps(
                        {
                            "event": "visual_cache_write",
                            "index": index,
                            "id": sample_id,
                            "video": row["video"],
                            "backend": backend,
                            "frames": len(video_frames),
                            "decoder_errors": decoder_errors,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                stats["visual"]["failed"] += 1
                print(
                    json.dumps(
                        {
                            "event": "visual_cache_error",
                            "index": index,
                            "id": sample_id,
                            "video": row["video"],
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        try:
            import torch

            del vlm, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if include_audio:
        audio_model, audio_processor = _load_audio_model(config)
        for index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            video_path = video_root / row["video"]
            prompt = _audio_prompt(row, dataset_name)
            cache_key, identity = _audio_cache_key(
                dataset=dataset_name,
                sample_id=sample_id,
                video_path=video_path,
                model_path=str(config.get("audio_model", {}).get("path", "")),
                prompt=prompt,
                audio_max_new_tokens=audio_tokens,
                schema_version=audio_schema_version,
                sample_rate=audio_sample_rate,
            )
            if not refresh and audio_cache.get(cache_key) is not None:
                stats["audio"]["hits"] += 1
                print(json.dumps({"event": "audio_cache_hit", "index": index, "id": sample_id}, ensure_ascii=False), flush=True)
                continue
            stats["audio"]["misses"] += 1
            started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix="agentvideommd_audio_") as tmpdir:
                    wav_path = Path(tmpdir) / f"{sample_id}.wav"
                    _extract_audio_wav(video_path, wav_path, audio_sample_rate)
                    raw = _generate_text_with_audio(
                        audio_model,
                        audio_processor,
                        wav_path,
                        prompt,
                        max_new_tokens=audio_tokens,
                        do_sample=False,
                    )
                audio = _json_or_raw(raw, "audio_raw_summary")
                backend = "qwen2_audio"
            except Exception as exc:
                raw = ""
                audio = {
                    "audio_available": False,
                    "audio_error": f"{type(exc).__name__}: {exc}",
                    "decision_relevance": "low",
                    "summary": "Audio extraction or audio-model inference failed.",
                }
                backend = "audio_error"
                stats["audio"]["failed"] += 1
            audio_cache.put(
                cache_key=cache_key,
                identity=identity,
                audio=audio,
                raw_text=raw,
                backend=backend,
                elapsed_seconds=round(time.perf_counter() - started, 4),
            )
            stats["audio"]["written"] += 1
            print(
                json.dumps(
                    {
                        "event": "audio_cache_write",
                        "index": index,
                        "id": sample_id,
                        "video": row["video"],
                        "backend": backend,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return stats
