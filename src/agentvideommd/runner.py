from __future__ import annotations

import json
import random
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
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate an interrupted final append
            # Failed attempts are never considered completed. Otherwise a
            # resumed run silently reuses fallback `fake` predictions.
            if row.get("error"):
                continue
            completed.add(str(row["id"]))
    return completed


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
    from transformers import AutoModelForCausalLM, AutoProcessor

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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # InternVL remote code used by the local checkpoint expects the
        # Transformers v4-compatible keyword rather than the newer `dtype`.
        torch_dtype=_torch_dtype(torch, model_config.get("dtype", "bfloat16")),
        attn_implementation=attn_implementation,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(model, "img_context_token_id") and model.img_context_token_id is None:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            raise TypeError(
                f"{type(processor).__name__} does not expose a tokenizer capable of initializing InternVL."
            )
        image_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        unknown_token_id = getattr(tokenizer, "unk_token_id", None)
        if image_context_token_id is None or image_context_token_id == unknown_token_id:
            raise ValueError("The model tokenizer does not contain the required <IMG_CONTEXT> token.")
        model.img_context_token_id = int(image_context_token_id)
        print(
            json.dumps(
                {
                    "event": "initialize_internvl",
                    "img_context_token_id": model.img_context_token_id,
                    "processor_class": type(processor).__name__,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return model, processor


def infer_one(model: Any, processor: Any, row: dict, video_path: Path, config: dict[str, Any]) -> str:
    import torch

    video = config["video"]
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": str(video_path),
                "fps": video["fps"],
                "min_pixels": video["min_pixels"],
                "max_pixels": video["max_pixels"],
            },
            {"type": "text", "text": row["prompt"]},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        fps=video["fps"],
        return_tensors="pt",
    ).to(model.device)
    generation = dict(config["generation"])
    generation.pop("seed", None)
    # InternVLChatModel.generate forwards use_cache=True internally; passing
    # it again reaches the language model as a duplicate keyword argument.
    generation.pop("use_cache", None)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation)
    generated_ids = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def evaluate_prediction_file(prediction_path: Path) -> dict:
    latest: dict[str, dict] = {}
    for row in read_jsonl(prediction_path):
        latest[str(row["id"])] = row
    failed = sum(bool(row.get("error")) for row in latest.values())
    if failed:
        raise RuntimeError(
            f"{failed} of {len(latest)} predictions failed; strict metrics are undefined. "
            "Fix inference errors and rerun so every failed ID is retried."
        )
    metrics = compute_metrics(
        [str(row["label"]) for row in latest.values()],
        [str(row.get("prediction", "fake")) for row in latest.values()],
    )
    metrics["attempted"] = len(latest)
    metrics["failed"] = failed
    metrics["parse_failures"] = sum(not row.get("parse_ok", False) for row in latest.values())
    return metrics


def run(
    config_path: Path,
    repo_root: Path,
    video_root_override: Path | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
    validate_only: bool = False,
    skip_video_check: bool = False,
) -> dict:
    config = load_config(config_path)
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
