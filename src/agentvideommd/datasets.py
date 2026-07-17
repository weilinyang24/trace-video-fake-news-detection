from __future__ import annotations

import random
from pathlib import Path

from .io import read_jsonl
from .prompts import build_fakesv_prompt, build_fakett_prompt


def _load_annotations(path: Path) -> dict[str, dict]:
    rows = {str(row["video_id"]): row for row in read_jsonl(path)}
    if not rows:
        raise ValueError(f"No annotations found in {path}")
    return rows


def _load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return ids


def _normalize_label(dataset: str, source: dict, sample_id: str) -> str | None:
    raw_label = str(source.get("annotation", "")).strip().lower()
    if dataset == "fakett":
        if raw_label not in {"real", "fake"}:
            raise ValueError(f"Unsupported FakeTT label {raw_label!r} for {sample_id}")
        return raw_label
    if dataset == "fakesv":
        label_map = {"真": "real", "假": "fake", "辟谣": None}
        if raw_label not in label_map:
            raise ValueError(f"Unsupported FakeSV label {raw_label!r} for {sample_id}")
        return label_map[raw_label]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_few_shot_examples(
    dataset: str,
    annotations: dict[str, dict],
    train_split_path: Path | None,
    seed: int,
) -> list[dict]:
    if train_split_path is None:
        return []
    train_ids = _load_ids(train_split_path)
    buckets: dict[str, list[dict]] = {"real": [], "fake": []}
    for sample_id in train_ids:
        if sample_id not in annotations:
            continue
        source = annotations[sample_id]
        label = _normalize_label(dataset, source, sample_id)
        if label in buckets:
            buckets[label].append({"id": sample_id, "label": label, "source": source})
    missing = [label for label, examples in buckets.items() if not examples]
    if missing:
        raise ValueError(f"Train split lacks examples for labels: {missing}")
    sampler = random.Random(seed)
    return [sampler.choice(buckets["real"]), sampler.choice(buckets["fake"])]


def build_test_manifest(
    dataset: str,
    annotation_path: Path,
    split_path: Path,
    train_split_path: Path | None = None,
    few_shot_seed: int = 2025,
) -> tuple[list[dict], dict]:
    annotations = _load_annotations(annotation_path)
    sample_ids = _load_ids(split_path)
    missing = [sample_id for sample_id in sample_ids if sample_id not in annotations]
    if missing:
        raise ValueError(f"{len(missing)} split IDs lack annotations; examples: {missing[:5]}")

    few_shot_examples = _build_few_shot_examples(dataset, annotations, train_split_path, few_shot_seed)
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for sample_id in sample_ids:
        source = annotations[sample_id]
        label = _normalize_label(dataset, source, sample_id)
        if dataset == "fakett":
            prompt = build_fakett_prompt(source, examples=few_shot_examples)
        elif dataset == "fakesv":
            if label is None:
                raw_label = str(source.get("annotation", "")).strip().lower()
                dropped[raw_label] = dropped.get(raw_label, 0) + 1
                continue
            prompt = build_fakesv_prompt(source, examples=few_shot_examples)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        rows.append({"id": sample_id, "video": f"{sample_id}.mp4", "prompt": prompt, "label": label})

    stats = {
        "dataset": dataset,
        "split": "test",
        "split_size": len(sample_ids),
        "exported": len(rows),
        "dropped": dropped,
        "real": sum(row["label"] == "real" for row in rows),
        "fake": sum(row["label"] == "fake" for row in rows),
        "few_shot_seed": few_shot_seed if few_shot_examples else None,
        "few_shot_examples": [
            {"id": example["id"], "label": example["label"]} for example in few_shot_examples
        ],
    }
    return rows, stats
