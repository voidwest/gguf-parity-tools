from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parity_tools.io import load_layers, load_logits
from parity_tools.metrics import LogitThresholds, compare_layers, compare_logits, top_k_ids
from parity_tools.token_audit import audit_tokens


class MetricTests(unittest.TestCase):
    def test_top_k_ids_are_descending(self) -> None:
        self.assertEqual(top_k_ids(np.array([0.1, 3.0, 2.0], dtype=np.float32), 2), [1, 2])

    def test_top_k_larger_than_vocab_is_clamped(self) -> None:
        self.assertEqual(top_k_ids(np.array([0.1, 3.0, 2.0], dtype=np.float32), 99), [1, 2, 0])

    def test_compare_logits_passes_close_top1_match(self) -> None:
        candidate = np.array([[0.0, 3.0, 2.0], [4.0, 1.0, 0.0]], dtype=np.float32)
        reference = candidate + np.array([[0.0, 0.01, -0.01], [0.0, 0.0, 0.0]], dtype=np.float32)
        report = compare_logits(candidate, reference, top_k=2, thresholds=LogitThresholds(min_topk_overlap=1.0))
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["summary"]["all_top1_match"])

    def test_compare_logits_fails_token_order_mismatch_when_required(self) -> None:
        candidate = np.array([[0.0, 3.0, 2.0]], dtype=np.float32)
        reference = np.array([[0.0, 2.0, 3.0]], dtype=np.float32)
        report = compare_logits(candidate, reference, top_k=1, thresholds=LogitThresholds(require_top1=True))
        self.assertEqual(report["status"], "fail")

    def test_compare_logits_shape_mismatch_fails_cleanly(self) -> None:
        candidate = np.zeros((1, 3), dtype=np.float32)
        reference = np.zeros((2, 3), dtype=np.float32)
        report = compare_logits(candidate, reference)
        self.assertEqual(report["status"], "fail")
        self.assertIn("shape mismatch", report["notes"])

    def test_compare_logits_empty_input_fails_cleanly(self) -> None:
        candidate = np.zeros((0, 3), dtype=np.float32)
        reference = np.zeros((0, 3), dtype=np.float32)
        report = compare_logits(candidate, reference)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["summary"]["num_prompts"], 0)

    def test_compare_logits_non_finite_fails_cleanly(self) -> None:
        candidate = np.array([[0.0, np.nan, 1.0]], dtype=np.float32)
        reference = np.array([[0.0, 2.0, 1.0]], dtype=np.float32)
        report = compare_logits(candidate, reference)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["finite_check"]["candidate_finite"])

    def test_compare_layers_reports_per_layer_cosine(self) -> None:
        candidate = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        reference = np.array([[1.0, 0.0], [1.0, -1.0]], dtype=np.float32)
        report = compare_layers(candidate, reference)
        self.assertEqual(report["status"], "pass")
        self.assertAlmostEqual(report["layers"][0]["cosine_similarity"], 1.0)
        self.assertAlmostEqual(report["layers"][1]["cosine_similarity"], 0.0)

    def test_compare_layers_empty_input_fails_cleanly(self) -> None:
        report = compare_layers(np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32))
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["summary"]["layers"], 0)

    def test_compare_layers_non_finite_fails_cleanly(self) -> None:
        candidate = np.array([[1.0, np.inf]], dtype=np.float32)
        reference = np.array([[1.0, 2.0]], dtype=np.float32)
        report = compare_layers(candidate, reference)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["finite_check"]["candidate_finite"])

    def test_token_audit_detects_mismatch(self) -> None:
        candidate = {"prompts": [{"prompt": "Hello", "token_ids": [1, 2]}]}
        reference = {"prompts": [{"prompt": "Hello", "token_ids": [1, 3]}]}
        report = audit_tokens(candidate, reference)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["rows"][0]["token_ids_match"])


class LoaderTests(unittest.TestCase):
    def test_load_logits_accepts_single_key_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logits.npz"
            np.savez(path, scores=np.array([1.0, 2.0], dtype=np.float32))
            arr = load_logits(path)
            self.assertEqual(arr.shape, (1, 2))

    def test_load_logits_rejects_ambiguous_npz_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logits.npz"
            np.savez(path, a=np.array([1.0], dtype=np.float32), b=np.array([2.0], dtype=np.float32))
            with self.assertRaises(ValueError):
                load_logits(path)

    def test_load_layers_rejects_non_positive_dims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layers.bin"
            path.write_bytes(b"")
            with self.assertRaises(ValueError):
                load_layers(path, 0, 3)


class CliTests(unittest.TestCase):
    def test_compare_logits_cli_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.npy"
            reference = root / "reference.npy"
            out = root / "report"
            np.save(candidate, np.array([[0.0, 2.0, 1.0]], dtype=np.float32))
            np.save(reference, np.array([[0.0, 2.1, 0.9]], dtype=np.float32))

            cmd = [
                sys.executable,
                "-m",
                "parity_tools",
                "compare-logits",
                "--candidate",
                str(candidate),
                "--reference",
                str(reference),
                "--out",
                str(out),
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
            self.assertTrue((out / "report.json").exists())
            self.assertTrue((out / "summary.md").exists())
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
