from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentvideommd.runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct test-set inference with InternVideo3-8B-Instruct.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-video-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        config_path=args.config,
        repo_root=REPO_ROOT,
        video_root_override=args.video_root,
        limit=args.limit,
        retry_errors=args.retry_errors,
        validate_only=args.validate_only,
        skip_video_check=args.skip_video_check,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

