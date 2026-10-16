"""No-GPU safety checks for the Gemma experiment runner."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_gemma_sweep import save_result
from run_pruning_sweep import experiment_matrix


class GemmaRunnerTests(unittest.TestCase):
    def test_complete_atomic_result_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            z = np.zeros((60, 49, 7), dtype=np.float32)
            save_result(out, 'example', {'tag': 'example'}, z)
            self.assertEqual(json.loads((out / 'example/summary.json').read_text()), {'tag': 'example'})
            with np.load(out / 'example/results.npz') as saved:
                np.testing.assert_array_equal(saved['opt_logits'], z)
            with self.assertRaises(FileExistsError):
                save_result(out, 'example', {'changed': True}, z + 1)
            self.assertEqual(len(list(out.iterdir())), 1)

    def test_eight_worker_partition_covers_matrix_once(self):
        configs = experiment_matrix()
        all_tags = []
        for worker in range(8):
            filtered = [c for c in configs if (c['method'] == 'sparsegpt') == (worker < 4)]
            tags = [c['tag'] for index, c in enumerate(filtered) if index % 4 == worker % 4]
            self.assertEqual(len(tags), 3 if worker < 4 else 13)
            all_tags.extend(tags)
        self.assertEqual(len(all_tags), 64)
        self.assertEqual(len(set(all_tags)), 64)
        self.assertEqual(set(all_tags), {c['tag'] for c in configs})


if __name__ == '__main__':
    unittest.main()
