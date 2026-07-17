# AgentVideoMMD

Direct Qwen3-VL-30B-Instruct inference for short-video fake-news detection on the copied VideoMMD FakeTT and FakeSV splits.

The project does not use Swift and does not link back to the old `videommd` project. Annotations, splits, and generated manifests are regular local copies under `data/`.

## Environment

```bash
pip install -e .
```

The default configs load `/data2/573ops_ser/models/Qwen3-VL-30B-Instruct` with `bfloat16`, `flash_attention_2`, and `device_map: auto`.

## Data

- `data/annotations/` contains copied dataset annotations.
- `data/splits/` contains copied train/val/test split files.
- `data/manifests/` contains regenerated test manifests used by inference.
- FakeTT test: 299 samples, real 200, fake 99.
- FakeSV test: 542 binary samples after dropping the debunked class, real 238, fake 304.

Regenerate manifests with fixed train-set few-shot examples:

```bash
python scripts/prepare_test_data.py --dataset all --few-shot-seed 2025
```

Current fixed examples:

- FakeTT real: `7217585276510620970`
- FakeTT fake: `6885520369231334662`
- FakeSV real: `6958356118821244190`
- FakeSV fake: `6882551291101465856`

Use `--no-few-shot` only for ablations.

## Prompt

Each test prompt contains:

- A balanced veracity instruction.
- One real and one fake text-only calibration example sampled from the train split.
- The current sample auxiliary observations.

The calibration examples include their known labels and metadata text only. Their videos are not added to the model input, so each inference item still contains exactly one current video.

## FakeTT

Smoke test:

```bash
python scripts/run_direct_inference.py \
  --config configs/qwen3vl/fakett.yaml \
  --video-root /data2/573ops_ser/data/FakeTT/FakeTT/video \
  --limit 1
```

Full test:

```bash
python scripts/run_direct_inference.py \
  --config configs/qwen3vl/fakett.yaml \
  --video-root /data2/573ops_ser/data/FakeTT/FakeTT/video
```

Outputs:

- `outputs/qwen3_vl_30b_fewshot/fakett_test_predictions.jsonl`
- `outputs/qwen3_vl_30b_fewshot/fakett_test_metrics.json`

## FakeSV

```bash
python scripts/run_direct_inference.py \
  --config configs/qwen3vl/fakesv.yaml \
  --video-root /data2/573ops_ser/data/FakeSV/video
```

Outputs are written under `outputs/qwen3_vl_30b_fewshot/`.

## Video Input

Videos are decoded with Transformers `load_video(..., backend="decord")`, using a safe fps sampler that caps requested frames at the actual frame count for very short videos. Bad videos follow the same fallback style used in the previous VideoMMD project: primary loader, `ffmpeg` thumbnail fallback, then 224x224 black placeholder frames if both loaders fail. The decoded video object is then passed through the Qwen3-VL `AutoProcessor.apply_chat_template` video path.

The runner checks that video tensors are actually produced. It also uses strict metrics: if any latest prediction record has an inference error or cannot be parsed as `real`/`fake`, metrics are refused until the invalid IDs are rerun.
