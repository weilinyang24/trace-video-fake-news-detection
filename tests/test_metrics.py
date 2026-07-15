import unittest
from pathlib import Path

from agentvideommd.labels import normalize_prediction
from agentvideommd.metrics import compute_metrics
from agentvideommd.runner import _normalize_attn_implementation
from agentvideommd.runner import _completed_ids, evaluate_prediction_file


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
