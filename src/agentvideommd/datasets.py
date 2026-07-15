from __future__ import annotations

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


def build_test_manifest(dataset: str, annotation_path: Path, split_path: Path) -> tuple[list[dict], dict]:
    annotations = _load_annotations(annotation_path)
    sample_ids = _load_ids(split_path)
    missing = [sample_id for sample_id in sample_ids if sample_id not in annotations]
    if missing:
        raise ValueError(f"{len(missing)} split IDs lack annotations; examples: {missing[:5]}")

    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for sample_id in sample_ids:
        source = annotations[sample_id]
        raw_label = str(source.get("annotation", "")).strip().lower()
        if dataset == "fakett":
            if raw_label not in {"real", "fake"}:
                raise ValueError(f"Unsupported FakeTT label {raw_label!r} for {sample_id}")
            label = raw_label
            prompt = build_fakett_prompt(source)
        elif dataset == "fakesv":
            label_map = {"真": "real", "假": "fake", "辟谣": None}
            if raw_label not in label_map:
                raise ValueError(f"Unsupported FakeSV label {raw_label!r} for {sample_id}")
            label = label_map[raw_label]
            if label is None:
                dropped[raw_label] = dropped.get(raw_label, 0) + 1
                continue
            prompt = build_fakesv_prompt(source)
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
    }
    return rows, stats

