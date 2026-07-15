from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentvideommd.io import write_json_atomic
from agentvideommd.runner import evaluate_prediction_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute VideoMMD-compatible metrics from predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metrics = evaluate_prediction_file(args.predictions)
    if args.output:
        write_json_atomic(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

