from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentvideommd.two_stage import build_evidence_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached visual and audio evidence before running the lightweight judge loop."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--audio-model-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_visual = not args.audio_only
    include_audio = not args.visual_only
    result = build_evidence_cache(
        config_path=args.config,
        repo_root=REPO_ROOT,
        video_root_override=args.video_root,
        audio_model_path_override=args.audio_model_path,
        limit=args.limit,
        include_visual=include_visual,
        include_audio=include_audio,
        refresh=args.refresh,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
