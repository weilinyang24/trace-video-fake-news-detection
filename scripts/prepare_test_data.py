from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentvideommd.datasets import build_test_manifest
from agentvideommd.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build test-only manifests from copied VideoMMD splits.")
    parser.add_argument("--dataset", choices=("fakett", "fakesv", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ("fakett", "fakesv") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        rows, stats = build_test_manifest(
            dataset=dataset,
            annotation_path=REPO_ROOT / "data" / "annotations" / f"{dataset}.jsonl",
            split_path=REPO_ROOT / "data" / "splits" / dataset / "test.txt",
        )
        output = REPO_ROOT / "data" / "manifests" / f"{dataset}_test.jsonl"
        count = write_jsonl(output, rows)
        assert count == stats["exported"]
        print(json.dumps({**stats, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

