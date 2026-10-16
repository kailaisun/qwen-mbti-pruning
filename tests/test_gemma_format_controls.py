"""Prefix controls must score exactly the same raw targets."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_gemma_formats import evaluate_prefix


class GemmaFormatControlTests(unittest.TestCase):
    def test_prefix_does_not_change_target_budget_or_alignment(self):
        class OracleModel:
            def __call__(self, inputs, use_cache=False):
                logits = torch.zeros(*inputs.shape, 16)
                for index in range(inputs.shape[1] - 1):
                    logits[0, index, inputs[0, index + 1]] = 2.0
                return SimpleNamespace(logits=logits)

        blocks = np.array([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=np.int32)
        results = [evaluate_prefix(OracleModel(), blocks, prefix, device="cpu")
                   for prefix in [[], [2], [2, 11, 12, 13]]]
        self.assertTrue(all(result["predicted_tokens"] == 6 for result in results))
        self.assertTrue(all(result["block_nll"] == results[0]["block_nll"] for result in results))
        self.assertEqual([result["input_sequence_length"] for result in results], [4, 5, 8])


if __name__ == "__main__":
    unittest.main()
