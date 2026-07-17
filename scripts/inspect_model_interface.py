from __future__ import annotations

import argparse
import json

import torch  # Load libtorch before Transformers/NumPy integrations to avoid MKL loader conflicts.
from transformers import AutoConfig, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local checkpoint's official Transformers interface.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    result = {
        "torch_version": torch.__version__,
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "auto_map": getattr(config, "auto_map", None),
        "processor_class": type(processor).__name__,
        "has_apply_chat_template": hasattr(processor, "apply_chat_template"),
        "has_video_processor": hasattr(processor, "video_processor"),
        "chat_template_has_video": "video" in str(getattr(processor, "chat_template", "")).lower(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
