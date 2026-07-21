from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether FlashAttention2 is usable in the current Python env.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status when flash-attn cannot be imported.",
    )
    args = parser.parse_args()

    report: dict[str, object] = {}
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import flash_attn
        from flash_attn import flash_attn_func  # noqa: F401

        report["flash_attn"] = {
            "available": True,
            "version": getattr(flash_attn, "__version__", None),
        }
    except Exception as exc:
        report["flash_attn"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    ok = bool(report.get("flash_attn", {}).get("available"))  # type: ignore[union-attr]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
