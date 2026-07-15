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
