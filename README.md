# AgentVideoMMD

![AgentVideoMMD Case Study](assets/case_study.png)

**AgentVideoMMD** is a lightweight, Qwen-based multimodal pipeline for **short-video fake news detection**, with support for the **FakeTT** and **FakeSV** benchmarks.

The framework supports both direct vision-language model (VLM) inference and a two-stage evidence-driven pipeline that integrates visual, audio, and textual information for fake news detection.

## Overview

AgentVideoMMD provides two inference pipelines:

* **Direct VLM Inference** — `src/agentvideommd/runner.py` performs end-to-end fake news detection using a vision-language model.
* **Two-Stage Evidence Reasoning** — `src/agentvideommd/two_stage.py` first extracts multimodal evidence from the video and then performs final prediction with a text-based judge.

Benchmark metadata and dataset splits used by the repository are provided under:

* `data/annotations/` — dataset annotations
* `data/splits/` — official train/test splits
* `data/manifests/` — generated inference manifests

## Installation

AgentVideoMMD requires **Python 3.10+**.

Install the package in editable mode:

```bash
pip install -e .
```

## Configuration

To keep the repository portable, tracked configuration files do **not** contain machine-specific absolute paths.

Before running inference, configure the paths to your local models and datasets using environment variables:

```powershell
$env:AGENTVIDEOMMD_VLM_PATH = "D:\\path\\to\\Qwen3-VL-30B-Instruct"
$env:AGENTVIDEOMMD_TEXT_MODEL_PATH = "D:\\path\\to\\Qwen3-8B"
$env:AGENTVIDEOMMD_AUDIO_MODEL_PATH = "D:\\path\\to\\Qwen2-Audio-7B-Instruct"

$env:AGENTVIDEOMMD_FAKETT_VIDEO_ROOT = "D:\\path\\to\\FakeTT\\video"
$env:AGENTVIDEOMMD_FAKESV_VIDEO_ROOT = "D:\\path\\to\\FakeSV\\video\\videos"
```

The default public configuration files are:

```text
configs/qwen3vl/fakett.yaml
configs/qwen3vl/fakesv.yaml
```

All machine-local paths are resolved from the corresponding environment variables at runtime.

## Usage

### 1. Build a Test Manifest

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

### 2. Run Direct VLM Inference

```python
from pathlib import Path
from agentvideommd.runner import run

metrics = run(
    config_path=Path("configs/qwen3vl/fakett.yaml"),
    repo_root=Path(".").resolve(),
)
```

### 3. Run the Two-Stage Pipeline

```python
from pathlib import Path
from agentvideommd.two_stage import run_two_stage

metrics = run_two_stage(
    config_path=Path("configs/qwen3vl/fakett.yaml"),
    repo_root=Path(".").resolve(),
)
```

The same workflow can be applied to **FakeSV** by switching to the corresponding configuration and dataset files.

## Case Study

A complete qualitative case study is included in the repository:

**[View the full case study (PDF)](assets/case_study.pdf)**

The case study illustrates the multimodal evidence and reasoning process used by AgentVideoMMD for short-video fake news detection.

## Notes

* Video files are intentionally **not tracked** in this repository.
* Local model and dataset paths should be provided through environment variables.
* Do **not** commit machine-specific absolute paths into the tracked YAML configuration files.
* Strict evaluation metrics are written only when all latest predictions are valid and successfully parsed.
