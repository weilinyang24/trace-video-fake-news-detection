import unittest
import json
from pathlib import Path

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy = None

from agentvideommd.labels import normalize_prediction
from agentvideommd.metrics import compute_metrics
from agentvideommd.runner import _normalize_attn_implementation
from agentvideommd.runner import _completed_ids, evaluate_prediction_file
from agentvideommd.runner import _safe_video_sample_indices
from agentvideommd.runner import _placeholder_video_frames


class MetricsTest(unittest.TestCase):
    def test_metrics_match_hand_calculation(self):
        metrics = compute_metrics(["real", "real", "fake", "fake"], ["real", "fake", "fake", "fake"])
        self.assertEqual(metrics["acc"], 0.75)
        self.assertEqual(metrics["per_class"]["real"]["precision"], 1.0)
        self.assertEqual(metrics["per_class"]["real"]["recall"], 0.5)
        self.assertEqual(metrics["per_class"]["fake"]["precision"], 2 / 3)
        self.assertEqual(metrics["per_class"]["fake"]["recall"], 1.0)

    def test_malformed_prediction_falls_back_to_fake(self):
        self.assertEqual(normalize_prediction("uncertain"), ("fake", False))
        self.assertEqual(normalize_prediction("The answer is real."), ("real", True))

    def test_flash_attn_alias_uses_transformers_name(self):
        self.assertEqual(_normalize_attn_implementation("flash_attn"), "flash_attention_2")
        self.assertEqual(_normalize_attn_implementation("flash attention 2"), "flash_attention_2")
        self.assertEqual(_normalize_attn_implementation("sdpa"), "sdpa")

    def test_failed_attempt_is_not_completed_and_has_no_metrics(self):
        path = Path(__file__).parent / "fixtures" / "all_failed_predictions.jsonl"
        self.assertEqual(_completed_ids(path), set())
        with self.assertRaises(RuntimeError):
            evaluate_prediction_file(path)

    def test_latest_invalid_attempt_is_not_completed(self):
        rows = [
            {"id": "a", "label": "real", "prediction": "real", "parse_ok": True, "error": None},
            {"id": "a", "label": "real", "prediction": "fake", "parse_ok": False, "error": None},
            {"id": "b", "label": "fake", "prediction": "fake", "parse_ok": True, "error": None},
        ]
        path = Path(__file__).parent / "fixtures" / "_tmp_predictions.jsonl"
        try:
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            self.assertEqual(_completed_ids(path), {"b"})
            with self.assertRaisesRegex(RuntimeError, "invalid_id_examples=\\['a'\\]"):
                evaluate_prediction_file(path)
        finally:
            path.unlink(missing_ok=True)

    def test_safe_video_sampling_caps_at_total_frames(self):
        metadata = {"total_num_frames": 15, "duration": 15.0}
        self.assertEqual(_safe_video_sample_indices(metadata, fps=2.0), list(range(15)))

    def test_safe_video_sampling_uses_requested_fps(self):
        metadata = {"total_num_frames": 100, "duration": 10.0}
        indices = _safe_video_sample_indices(metadata, fps=2.0)
        self.assertEqual(len(indices), 20)
        self.assertEqual(indices[0], 0)
        self.assertLess(indices[-1], 100)

    def test_safe_video_sampling_prefers_source_fps_over_bad_duration(self):
        metadata = {"total_num_frames": 1521, "fps": 25.0, "duration": 760.5}
        indices = _safe_video_sample_indices(metadata, fps=2.0)
        self.assertEqual(len(indices), 122)
        self.assertEqual(indices[0], 0)
        self.assertLess(indices[-1], 1521)

    def test_placeholder_video_frames_match_videommd_fallback_shape(self):
        if numpy is None:
            self.skipTest("numpy is not installed in this local test environment")
        frames = _placeholder_video_frames(16)
        self.assertEqual(frames.shape, (16, 224, 224, 3))
        self.assertEqual(int(frames.max()), 0)
