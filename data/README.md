# Dataset provenance

These files are independent byte-for-byte copies made from `D:\MMdetection\videommd`; they are regular files, not symbolic or hard links.

- `annotations/fakett.jsonl` <- `data/fakett/data.json`
- `annotations/fakesv.jsonl` <- `data/fakesv/data_complete.json`
- `splits/{dataset}/train.txt` <- `external/ExMRD/data/{dataset}/vids/vid_time3_train.txt`
- `splits/{dataset}/val.txt` <- `external/ExMRD/data/{dataset}/vids/vid_time3_valid.txt`
- `splits/{dataset}/test.txt` <- `external/ExMRD/data/{dataset}/vids/vid_time3_test.txt`

`manifests/` is deterministically regenerated from the copied annotations and test split by `scripts/prepare_test_data.py`. Video binaries are intentionally external and selected through `--video-root`.

