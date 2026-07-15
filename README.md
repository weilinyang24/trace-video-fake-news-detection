# AgentVideoMMD

InternVideo3-8B-Instruct 在短视频虚假新闻检测任务上的直接推理基线。当前流程只评测固定测试集，不训练、不读取验证集调参，也不启用搜索、ASR 等 agent 工具。

## 评测口径

- 数据集与 `D:\MMdetection\videommd` 相同：FakeTT、FakeSV。
- `data/splits/` 是原项目 train/val/test 时间划分的独立副本，不是链接。
- FakeSV 延续原项目二分类策略：`真 -> real`、`假 -> fake`、`辟谣 -> 丢弃`。
- 指标与原项目一致：accuracy、macro precision、macro recall、macro F1；无法解析的回答按 `fake` 计。
- 提示词延续原项目的新闻真实性定义，强调本任务不是 deepfake 检测，并要求只输出 `real` 或 `fake`。

## 准备数据

仓库已经包含原始标注与划分的独立副本。重新生成测试 manifest：

```bash
python scripts/prepare_test_data.py --dataset all
```

预期规模：FakeTT 测试集 299 条；FakeSV 原测试划分 720 条，丢弃“辟谣”类后得到二分类测试子集。

## 环境与检查

```bash
pip install -e .
python scripts/run_direct_inference.py \
  --config configs/internvideo3/fakett.yaml \
  --video-root /your/FakeTT/video \
  --validate-only
```

官方 InternVideo3 代码要求 `transformers>=4.57.3`。当前实验使用服务器本地模型目录 `/data2/573ops_ser/models/InternVL-8B`，默认采用 bfloat16、FlashAttention 2 和 `device_map=auto`。

Transformers 中 FlashAttention 2 的合法名称是 `flash_attention_2`，不是 `flash_attn`。如果环境没有安装 FlashAttention，可将 YAML 中的 `attn_implementation` 改为 `eager`，以较慢速度兼容运行。

本地模型包含 InternVL 自定义 remote code，因此加载权重时使用兼容 Transformers v4 的 `torch_dtype` 参数；不要直接改成新版 `dtype` 参数。

InternVL 自定义 `generate()` 会在内部启用 KV cache，因此推理配置不再额外传递 `use_cache`，以免向底层语言模型重复传参。

## 只测试测试集

FakeTT：

```bash
python scripts/run_direct_inference.py \
  --config configs/internvideo3/fakett.yaml \
  --video-root /data2/573ops_ser/data/FakeTT/FakeTT/video
```

FakeSV：

```bash
python scripts/run_direct_inference.py \
  --config configs/internvideo3/fakesv.yaml \
  --video-root /data2/573ops_ser/data/FakeSV/video
```

推理结果逐条追加写入 `outputs/internvideo3/*_predictions.jsonl`，中断后重复命令会按样本 ID 续跑；失败样本默认自动重试，不会被当作有效预测。若全部样本失败，程序会拒绝生成指标。指标写入对应的 `*_metrics.json`。

只重新计算指标：

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/internvideo3/fakett_test_predictions.jsonl \
  --output outputs/internvideo3/fakett_test_metrics.json
```

首次正式全量运行前，建议先加 `--limit 2` 做显存和视频解码冒烟测试。`fps`、像素范围、生成长度和随机种子均固定在 YAML 中，实验报告应保存所用配置和模型 revision。
