# AgentVideoMMD

AgentVideoMMD is a lightweight Qwen-based pipeline for short-video fake-news detection on the bundled FakeTT and FakeSV splits.

## Overview

- `src/agentvideommd/runner.py` provides direct VLM inference.
- `src/agentvideommd/two_stage.py` provides the two-stage pipeline with visual evidence extraction, optional audio evidence, and a text judge.
- `data/annotations/`, `data/splits/`, and `data/manifests/` contain the copied benchmark metadata used by this repo.

## Installation

Use Python 3.10 or newer:

```bash
pip install -e .
```

## Configuration

The tracked configs intentionally do not contain machine-local paths.

Before running inference, set these environment variables to your own local model and video directories:

```powershell
$env:AGENTVIDEOMMD_VLM_PATH = "D:\\path\\to\\Qwen3-VL-30B-Instruct"
$env:AGENTVIDEOMMD_TEXT_MODEL_PATH = "D:\\path\\to\\Qwen3-8B"
$env:AGENTVIDEOMMD_AUDIO_MODEL_PATH = "D:\\path\\to\\Qwen2-Audio-7B-Instruct"
$env:AGENTVIDEOMMD_FAKETT_VIDEO_ROOT = "D:\\path\\to\\FakeTT\\video"
$env:AGENTVIDEOMMD_FAKESV_VIDEO_ROOT = "D:\\path\\to\\FakeSV\\video\\videos"
```

The public config files are:

- `configs/qwen3vl/fakett.yaml`
- `configs/qwen3vl/fakesv.yaml`

## Usage

Build a test manifest:

```python
from pathlib import Path
from agentvideommd.datasets import build_test_manifest

rows, stats = build_test_manifest(
    "fakett",
    Path("data/annotations/fakett.jsonl"),
    Path("data/splits/fakett/test.txt"),
    train_split_path=Path("data/splits/fakett/train.txt"),
)
```

Run direct inference:

```python
from pathlib import Path
from agentvideommd.runner import run

metrics = run(
    config_path=Path("configs/qwen3vl/fakett.yaml"),
    repo_root=Path(".").resolve(),
)
```

Run the two-stage pipeline:

```python
from pathlib import Path
from agentvideommd.two_stage import run_two_stage

metrics = run_two_stage(
    config_path=Path("configs/qwen3vl/fakett.yaml"),
    repo_root=Path(".").resolve(),
)
```

## Notes

- Video binaries are intentionally not tracked in this repository.
- Config paths are resolved from environment variables at runtime; do not commit local absolute paths back into tracked YAML files.
- Strict metrics are only written when all latest predictions are valid and parseable.

## Case Study

[Open the full PDF case study](assets/case_study.pdf)

![Case study overview](assets/case_study.png)
