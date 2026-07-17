from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from .io import read_jsonl, write_json_atomic
from .labels import normalize_prediction
from .metrics import compute_metrics


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = ("model", "dataset", "video", "generation", "output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    return config


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _completed_ids(path: Path, retry_errors: bool = True) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate an interrupted final append
            latest[str(row["id"])] = row
    # Keep this consistent with evaluate_prediction_file(), which also uses
    # the latest appended record for each id.
    return {
        sample_id
        for sample_id, row in latest.items()
        if not row.get("error") and row.get("parse_ok", False)
    }


def _torch_dtype(torch_module: Any, value: str):
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]


def _normalize_attn_implementation(value: str) -> str:
    value = value.strip().lower()
    aliases = {
        "flash_attn": "flash_attention_2",
        "flash-attn": "flash_attention_2",
        "flash_attention": "flash_attention_2",
        "flash attention 2": "flash_attention_2",
        "flash-attention-2": "flash_attention_2",
    }
    return aliases.get(value, value)


def load_model(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_config = config["model"]
    model_path = str(model_config["path"])
    attn_implementation = _normalize_attn_implementation(
        str(model_config.get("attn_implementation", "sdpa"))
    )
    print(
        json.dumps(
            {
                "event": "load_model",
                "model_path": model_path,
                "attn_implementation": attn_implementation,
                "dtype": model_config.get("dtype", "bfloat16"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=_torch_dtype(torch, model_config.get("dtype", "bfloat16")),
        attn_implementation=attn_implementation,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        raise RuntimeError(
            f"{type(processor).__name__} has no video_processor. Model path {model_path!r} "
            "is not a Qwen3-VL raw-video checkpoint."
        )
    video_config = config["video"]
    min_visual_tokens = int(video_config.get("min_visual_tokens", 256))
    max_visual_tokens = int(video_config.get("max_visual_tokens", 16384))
    spatial_compression = int(video_config.get("spatial_compression", 32))
    temporal_compression = int(video_config.get("temporal_compression", 2))
    pixels_per_visual_token = spatial_compression * spatial_compression * temporal_compression
    video_processor.size = {
        "longest_edge": max_visual_tokens * pixels_per_visual_token,
        "shortest_edge": min_visual_tokens * pixels_per_visual_token,
    }
    print(json.dumps({
        "event": "initialize_video_processor",
        "processor_class": type(processor).__name__,
        "video_processor_class": type(video_processor).__name__,
        "fps": video_config["fps"],
        "min_visual_tokens": min_visual_tokens,
        "max_visual_tokens": max_visual_tokens,
        "size": video_processor.size,
    }, ensure_ascii=False), flush=True)
    return model, processor


def _metadata_value(metadata: Any, key: str, default: Any = None) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _safe_video_sample_indices(metadata: Any, fps: float | None = None, num_frames: int | None = None, **_: Any):
    total_num_frames = _metadata_value(metadata, "total_num_frames")
    if total_num_frames is None:
        total_num_frames = _metadata_value(metadata, "num_frames")
    total_num_frames = int(total_num_frames or 0)
    if total_num_frames <= 0:
        return []

    desired_frames = int(num_frames or 0)
    if desired_frames <= 0 and fps:
        video_fps = float(_metadata_value(metadata, "fps", 0) or 0)
        duration = total_num_frames / video_fps if video_fps > 0 else _metadata_value(metadata, "duration")
        if duration is not None:
            desired_frames = int(math.ceil(float(duration) * float(fps)))
    if desired_frames <= 0:
        desired_frames = total_num_frames

    sampled_frames = max(1, min(desired_frames, total_num_frames))
    if sampled_frames == total_num_frames:
        return list(range(total_num_frames))
    return [min(total_num_frames - 1, int(index * total_num_frames / sampled_frames)) for index in range(sampled_frames)]


def _placeholder_video_frames(num_frames: int):
    import numpy as np

    return np.zeros((max(1, num_frames), 224, 224, 3), dtype=np.uint8)


def _load_video_with_ffmpeg(video_path: Path, num_frames: int, fps: float):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable is not available")
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="videommd_ffmpeg_") as tmpdir:
        output_pattern = str(Path(tmpdir) / "frame_%03d.jpg")
        command = [
            ffmpeg,
            "-v",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            str(video_path),
            "-vf",
            "thumbnail",
            "-frames:v",
            str(max(1, num_frames)),
            output_pattern,
        ]
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
        if completed.returncode != 0 and not frame_paths:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(stderr or f"ffmpeg fallback failed for {video_path}")
        frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
        if not frames:
            raise RuntimeError(f"ffmpeg fallback produced no frames for {video_path}")
        while len(frames) < max(1, num_frames):
            frames.append(frames[-1].copy())
        video_frames = np.stack(frames[:max(1, num_frames)], axis=0)
    metadata = {
        "total_num_frames": int(video_frames.shape[0]),
        "fps": fps,
        "duration": float(video_frames.shape[0]) / fps if fps > 0 else None,
    }
    return video_frames, metadata


def _load_video_frames(video_path: Path, fps: float, primary_backend: str, fallback_num_frames: int):
    from transformers.video_utils import load_video

    try:
        video_frames, metadata = load_video(
            str(video_path),
            fps=fps,
            backend=primary_backend,
            sample_indices_fn=_safe_video_sample_indices,
        )
        if len(video_frames) == 0:
            raise RuntimeError("decoded zero frames")
        return video_frames, metadata, primary_backend, []
    except Exception as primary_error:
        try:
            video_frames, metadata = _load_video_with_ffmpeg(
                video_path,
                num_frames=fallback_num_frames,
                fps=fps,
            )
            print(
                f"[videommd] ffmpeg fallback video loader used for {video_path.name}: {primary_error}",
                flush=True,
            )
            return video_frames, metadata, "ffmpeg", [f"{primary_backend}: {type(primary_error).__name__}: {primary_error}"]
        except Exception as fallback_error:
            print(
                f"[videommd] placeholder frames used for {video_path.name}: "
                f"primary={primary_error}; fallback={fallback_error}",
                flush=True,
            )
            video_frames = _placeholder_video_frames(fallback_num_frames)
            metadata = {
                "total_num_frames": int(video_frames.shape[0]),
                "fps": fps,
                "duration": float(video_frames.shape[0]) / fps if fps > 0 else None,
            }
            return video_frames, metadata, "placeholder", [
                f"{primary_backend}: {type(primary_error).__name__}: {primary_error}",
                f"ffmpeg: {type(fallback_error).__name__}: {fallback_error}",
            ]


def infer_one(model: Any, processor: Any, row: dict, video_path: Path, config: dict[str, Any]) -> str:
    import torch

    video = config["video"]
    fps = float(video["fps"])
    decoder_backend = str(video.get("decoder_backend", "decord"))
    fallback_num_frames = int(video.get("fallback_num_frames", 16))
    try:
        # Transformers currently falls back to torchvision when torchcodec is
        # absent. Some torchvision builds no longer expose io.read_video, so
        # select the installed decoder explicitly and pass an in-memory video
        # through the official multimodal processor interface. Match the old
        # VideoMMD behavior for bad videos: primary loader, ffmpeg fallback,
        # then placeholder frames.
        video_frames, _video_metadata, decoder_backend, decoder_errors = _load_video_frames(
            video_path,
            fps,
            decoder_backend,
            fallback_num_frames,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Video decoder backend {decoder_backend!r} is unavailable. "
            f"Install it in the inference environment (for example: pip install decord)."
        ) from exc
    if len(video_frames) == 0:
        raise RuntimeError(f"Decoded zero frames from {video_path}")
    print(
        json.dumps(
            {
                "event": "decode_video",
                "video": video_path.name,
                "backend": decoder_backend,
                "fps": fps,
                "sampled_frames": len(video_frames),
                "total_frames": _metadata_value(_video_metadata, "total_num_frames"),
                "frame_shape": list(video_frames.shape[1:]) if hasattr(video_frames, "shape") else None,
                "fallback_errors": decoder_errors,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_frames,
            },
            {"type": "text", "text": row["prompt"]},
        ],
    }]
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        # Newer Transformers versions require kwargs for processor.__call__
        # to be nested here instead of being mixed with template kwargs.
        "processor_kwargs": {"videos_kwargs": {"fps": fps, "video_metadata": [_video_metadata]}},
    }
    try:
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)
    except (TypeError, ValueError) as exc:
        if "metadata" not in str(exc).lower():
            raise
        template_kwargs["processor_kwargs"] = {"videos_kwargs": {"fps": fps}}
        inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)
    visual_keys = {
        "pixel_values",
        "pixel_values_videos",
        "video_pixel_values",
        "video_grid_thw",
    }
    present_visual_keys = sorted(key for key in visual_keys if key in inputs and inputs[key] is not None)
    if not present_visual_keys:
        raise RuntimeError(
            f"{type(processor).__name__} produced no video tensor for {video_path.name}; "
            f"input keys={sorted(inputs.keys())}. This checkpoint does not expose the raw-video "
            "AutoProcessor interface. Use a complete Qwen3-VL-Instruct snapshot."
        )
    generation = dict(config["generation"])
    generation.pop("seed", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation.setdefault("pad_token_id", eos_token_id)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation)
    generated_ids = []
    for input_ids, output in zip(inputs.input_ids, output_ids):
        input_length = input_ids.shape[0]
        has_input_prefix = (
            output.shape[0] >= input_length
            and torch.equal(output[:input_length].to(input_ids.device), input_ids)
        )
        generated_ids.append(output[input_length:] if has_input_prefix else output)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if not response:
        raise RuntimeError(
            f"Model generated {output_ids.shape[-1]} token(s), but decoding produced an empty response."
        )
    return response


def evaluate_prediction_file(prediction_path: Path) -> dict:
    latest: dict[str, dict] = {}
    for row in read_jsonl(prediction_path):
        latest[str(row["id"])] = row
    failed = sum(bool(row.get("error")) for row in latest.values())
    parse_failures = sum(not row.get("parse_ok", False) for row in latest.values())
    if failed or parse_failures:
        invalid_ids = [
            sample_id
            for sample_id, row in latest.items()
            if row.get("error") or not row.get("parse_ok", False)
        ]
        raise RuntimeError(
            f"Strict metrics are undefined: failed={failed}, parse_failures={parse_failures}, "
            f"total={len(latest)}, invalid_id_examples={invalid_ids[:10]}. "
            "Fix inference/decoding and rerun every invalid ID."
        )
    metrics = compute_metrics(
        [str(row["label"]) for row in latest.values()],
        [str(row.get("prediction", "fake")) for row in latest.values()],
    )
    metrics["attempted"] = len(latest)
    metrics["failed"] = failed
    metrics["parse_failures"] = parse_failures
    return metrics


def run(
    config_path: Path,
    repo_root: Path,
    video_root_override: Path | None = None,
    model_path_override: str | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
    validate_only: bool = False,
    skip_video_check: bool = False,
) -> dict:
    config = load_config(config_path)
    if model_path_override:
        config["model"]["path"] = model_path_override
    manifest_path = resolve_path(repo_root, config["dataset"]["manifest"])
    prediction_path = resolve_path(repo_root, config["output"]["predictions"])
    metrics_path = resolve_path(repo_root, config["output"]["metrics"])
    video_root = video_root_override or Path(config["dataset"]["video_root"])
    rows = list(read_jsonl(manifest_path))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No test samples in {manifest_path}")
    required = {"id", "video", "prompt", "label"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Manifest row {index} misses fields: {sorted(missing)}")
    missing_videos = [str(video_root / row["video"]) for row in rows if not (video_root / row["video"]).is_file()]
    if missing_videos and not skip_video_check:
        raise FileNotFoundError(
            f"{len(missing_videos)} videos are missing under {video_root}; examples: {missing_videos[:3]}"
        )
    if validate_only:
        return {
            "dataset": config["dataset"]["name"],
            "samples": len(rows),
            "video_root": str(video_root),
            "missing_videos": len(missing_videos),
            "model": config["model"]["path"],
        }

    import torch

    seed = int(config["generation"].get("seed", 2025))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, processor = load_model(config)
    completed = _completed_ids(prediction_path, retry_errors=True)
    consecutive_errors = 0
    max_consecutive_errors = int(config.get("runtime", {}).get("max_consecutive_errors", 3))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("a", encoding="utf-8", newline="\n") as output:
        for index, row in enumerate(rows, 1):
            sample_id = str(row["id"])
            if sample_id in completed:
                continue
            started = time.perf_counter()
            result = {"id": sample_id, "label": row["label"], "video": row["video"]}
            try:
                response = infer_one(model, processor, row, video_root / row["video"], config)
                prediction, parse_ok = normalize_prediction(response)
                result.update({"response": response, "prediction": prediction, "parse_ok": parse_ok, "error": None})
                consecutive_errors = 0
            except Exception as exc:  # persist per-sample failures so long runs can resume
                consecutive_errors += 1
                result.update({
                    "response": "",
                    "prediction": "fake",
                    "parse_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({"index": index, **result}, ensure_ascii=False), flush=True)
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"Stopped after {consecutive_errors} consecutive inference errors. "
                    f"Inspect the traceback in {prediction_path} and rerun with --retry-errors after fixing it."
                )

    if limit is not None:
        latest = {str(row["id"]): row for row in read_jsonl(prediction_path)}
        selected = [latest.get(str(row["id"])) for row in rows]
        return {
            "mode": "smoke_test",
            "requested": len(rows),
            "successful": sum(bool(row) and not row.get("error") for row in selected),
            "failed": sum(bool(row) and bool(row.get("error")) for row in selected),
            "metrics_written": False,
        }

    metrics = evaluate_prediction_file(prediction_path)
    write_json_atomic(metrics_path, metrics)
    return metrics
