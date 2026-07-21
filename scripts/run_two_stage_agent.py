from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentvideommd.two_stage import run_two_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage lightweight Qwen3-VL + Qwen3 text agent inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--text-model-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_two_stage(
        config_path=args.config,
        repo_root=REPO_ROOT,
        video_root_override=args.video_root,
        text_model_path_override=args.text_model_path,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
