import unittest
from pathlib import Path

from agentvideommd.datasets import build_test_manifest

class DatasetTest(unittest.TestCase):
    def test_fakesv_drops_debunked(self):
        fixture_root = Path(__file__).parent / "fixtures"
        rows, stats = build_test_manifest(
            "fakesv",
            fixture_root / "fakesv_annotations.jsonl",
            fixture_root / "fakesv_test.txt",
        )
        self.assertEqual([row["label"] for row in rows], ["real", "fake"])
        self.assertEqual(stats["dropped"], {"辟谣": 1})

    def test_fakett_prompt_includes_train_few_shot_examples(self):
        fixture_root = Path(__file__).parent / "fixtures"
        rows, stats = build_test_manifest(
            "fakett",
            fixture_root / "fakett_annotations.jsonl",
            fixture_root / "fakett_test.txt",
            train_split_path=fixture_root / "fakett_train.txt",
            few_shot_seed=2025,
        )
        prompt = rows[0]["prompt"]
        self.assertIn("Calibration examples from the training split", prompt)
        self.assertIn("Example 1 known label: real", prompt)
        self.assertIn("Real training claim", prompt)
        self.assertIn("Example 2 known label: fake", prompt)
        self.assertIn("Fake training claim", prompt)
        self.assertEqual(
            stats["few_shot_examples"],
            [{"id": "train_real", "label": "real"}, {"id": "train_fake", "label": "fake"}],
        )
