"""Small CPU-only synthetic checks; no model weights or benchmark files needed."""
import copy
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_gemma_sweep import (MODEL, PROTOCOL, REVISION, derive_rows, read_run,
                                plot_results, write_cross_model, write_csv, write_json, write_markdown)
from mbti_common import OPTIONS, layer_metrics, load_items, score_mbti
from run_pruning_sweep import experiment_matrix


class GemmaAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gemma-analysis-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.items = load_items()
        self.configs = {cfg["tag"]: cfg for cfg in experiment_matrix()}

    def fixture(self, tag="dense-s00", change_first=False):
        cfg = self.configs[tag]
        logits = np.zeros((60, 3, 7), dtype=np.float32)
        logits[:, :, 1] = 2
        if change_first:
            logits[0, -1, 6] = 3
        choices = [OPTIONS[index] for index in logits[:, -1].argmax(-1)]
        label, dimensions = score_mbti(self.items, choices)
        metrics = layer_metrics(logits)
        hashes = {str(item["id"]): hashlib.sha256(item["text"].encode()).hexdigest()
                  for item in self.items}
        prefix = "synthetic-answer-boundary"
        assistant_prefix = "synthetic-assistant-continuation-prefix"
        probes = [
            {"item_id": self.items[index]["id"], "response": "B", "raw_response": "B<EOS>",
             "generated_token_ids": [10, 2], "strict_single_letter": True,
             "generated_tokens": 2, "hit_length_limit": False, "stopped_on_eos": True,
             "nonterminal_special_token_ids": [], "prompt_sha256": hashes[str(self.items[index]["id"])],
             "answer_prefix": prefix}
            for index in (0, 9, 19, 29, 39, 49)
        ]
        stats = [] if tag == "dense-s00" else [
            {"name": "synthetic.linear", "numel": 100, "masked": 50,
             "mask_sha256": "synthetic", "weight_energy": 1., "weight_delta_energy": .1}
        ]
        summary = {
            "tag": tag, "model": MODEL, "dtype": "bfloat16", "protocol_version": PROTOCOL,
            "sweep_version": "3.3-gemma", "sweep_config": cfg, "smoke": False, "n_items": 60,
            "predicted_mbti": label, "dichotomies": dimensions,
            "per_item_choices": {str(item["id"]): choice for item, choice in zip(self.items, choices)},
            "layer_entropy_mean": metrics["entropy_mean"].tolist(),
            "layer_gap_mean": metrics["gap_mean"].tolist(),
            "native_projection_max_abs_error": 0.,
            "diagnostics": [
                {"option_probability_mass": .4, "unconstrained_top_is_option": True,
                 "unconstrained_top_token": "B", "prompt_sha256": hashes[str(item["id"])],
                 "answer_prefix": prefix}
                for item in self.items
            ],
            "mean_option_probability_mass": .4, "unconstrained_option_rate": 1.,
            "language_eval": {"predicted_tokens": 8160, "block_nll": [2.] * 32,
                              "mean_nll": 2., "perplexity": math.exp(2),
                              "prefix": "explicit_bos", "bos_token_id": 2,
                              "input_sequence_length": 257, "scored_targets_per_sequence": 255},
            "assistant_language_eval": {
                "condition": "native_assistant_continuation", "enable_thinking": False,
                "prefix_text": assistant_prefix, "prefix_token_ids": [2, 10, 11], "prefix_length": 3,
                "prefix_sha256": hashlib.sha256(assistant_prefix.encode()).hexdigest(),
                "input_sequence_length": 259, "scored_targets_per_sequence": 255,
                "predicted_tokens": 8160, "block_nll": [1.] * 32,
                "mean_nll": 1., "perplexity": math.exp(1.),
            },
            "generation_probe": probes,
            "prune": {"per_layer": stats, "block_reconstruction": [], "seconds": 0.,
                      "total_model_parameters": 200, "eligible_parameters": 100,
                      "targeted_parameters": sum(row["numel"] for row in stats),
                      "masked_parameters": sum(row["masked"] for row in stats),
                      "actual_zeros": 50 if stats else 0, "eligible_zero_fraction": .5 if stats else 0.},
            "elapsed_seconds": 0.,
            "provenance": {"model_revision": REVISION, "layer_count": 2,
                           "source_sha256": {"synthetic": "hash"}, "corpus": {"synthetic": True},
                           "calibration_input": {"prefix": "explicit_bos", "bos_token_id": 2,
                                                 "raw_sequence_length": 256, "input_sequence_length": 257,
                                                 "sequences": 64},
                           "prompt": {"prompt_sha256": hashes, "answer_prefix": prefix}},
        }
        return summary, logits

    def publish(self, summary, logits):
        folder = self.root / summary["tag"]
        folder.mkdir(exist_ok=True)
        write_json(folder / "summary.json", summary)
        np.savez_compressed(folder / "results.npz", opt_logits=logits)
        return folder

    def test_dynamic_layers_and_score_reconstruction(self):
        summary, logits = self.fixture()
        folder = self.publish(summary, logits)
        actual, values, choices = read_run(folder, self.items)
        self.assertEqual(values.shape, (60, 3, 7))
        self.assertEqual(choices, ["B"] * 60)
        self.assertEqual(actual["predicted_mbti"], "ENFJ")
        summary["dichotomies"]["EI"]["score"] += 1
        self.publish(summary, logits)
        with self.assertRaisesRegex(ValueError, "score reconstruction"):
            read_run(folder, self.items)

    def test_bad_native_and_language_diagnostics_rejected(self):
        for key in ("native", "ppl", "tokens", "sparsity"):
            with self.subTest(key=key):
                summary, logits = self.fixture()
                if key == "native":
                    summary["native_projection_max_abs_error"] = .1
                elif key == "ppl":
                    summary["language_eval"]["perplexity"] += 1
                elif key == "tokens":
                    summary["language_eval"]["predicted_tokens"] = 1
                else:
                    summary["prune"]["eligible_zero_fraction"] = .2
                with self.assertRaises(ValueError):
                    read_run(self.publish(summary, logits), self.items)

    def test_bos_prefix_and_original_target_alignment_required(self):
        for field, invalid in (("prefix", "none"), ("bos_token_id", 1),
                               ("input_sequence_length", 256), ("scored_targets_per_sequence", 256)):
            with self.subTest(field=field):
                summary, logits = self.fixture()
                summary["language_eval"][field] = invalid
                with self.assertRaisesRegex(ValueError, "BOS"):
                    read_run(self.publish(summary, logits), self.items)
        summary, logits = self.fixture()
        del summary["language_eval"]["prefix"]
        with self.assertRaisesRegex(ValueError, "BOS"):
            read_run(self.publish(summary, logits), self.items)
        summary, logits = self.fixture()
        summary["provenance"]["calibration_input"]["input_sequence_length"] = 256
        with self.assertRaisesRegex(ValueError, "calibration.*BOS"):
            read_run(self.publish(summary, logits), self.items)

    def test_assistant_condition_required_and_validated(self):
        summary, logits = self.fixture()
        del summary["assistant_language_eval"]
        with self.assertRaisesRegex(ValueError, "missing assistant_language_eval"):
            read_run(self.publish(summary, logits), self.items)
        for field, invalid in (("condition", "raw"), ("enable_thinking", True),
                               ("prefix_sha256", "incorrect"), ("prefix_token_ids", [1, 10, 11]),
                               ("prefix_length", 4), ("input_sequence_length", 258),
                               ("scored_targets_per_sequence", 256), ("predicted_tokens", 8192),
                               ("perplexity", 100.)):
            with self.subTest(field=field):
                summary, logits = self.fixture()
                summary["assistant_language_eval"][field] = invalid
                with self.assertRaises(ValueError):
                    read_run(self.publish(summary, logits), self.items)

    def test_assistant_ratios_use_separate_dense_condition(self):
        runs = {}
        for tag in ("dense-s00", "wanda-s50"):
            summary, logits = self.fixture(tag)
            if tag != "dense-s00":
                summary["assistant_language_eval"].update(
                    block_nll=[3.] * 32, mean_nll=3., perplexity=math.exp(3.))
            runs[tag] = read_run(self.publish(summary, logits), self.items)
        configs = [self.configs[tag] for tag in runs]
        rows, _, _ = derive_rows(configs, runs, self.items)
        self.assertEqual(rows[1]["ppl_ratio"], 1.)
        self.assertAlmostEqual(rows[1]["assistant_ppl_ratio"], math.exp(2.))
        self.assertEqual(rows[1]["assistant_nll_delta"], 2.)
        runs["wanda-s50"][0]["assistant_language_eval"]["prefix_text"] = "changed"
        with self.assertRaisesRegex(ValueError, "assistant continuation condition differs"):
            derive_rows(configs, runs, self.items)

    def test_dual_ppl_plots_and_markdown(self):
        runs = {}
        for tag in ("dense-s00", "wanda-s50"):
            summary, logits = self.fixture(tag)
            runs[tag] = read_run(self.publish(summary, logits), self.items)
        configs = [self.configs[tag] for tag in runs]
        rows, _, details = derive_rows(configs, runs, self.items)
        tags = plot_results(self.root, rows, details, 2)
        write_markdown(self.root, rows, ["unfinished"], 2, tags)
        for name in ("sparsity_tradeoffs.png", "layer_jsd_50.png", "dimension_scores_50.png"):
            self.assertGreater((self.root / name).stat().st_size, 1000)
        for name in ("RESULTS.md", "ABLATIONS.md"):
            document = (self.root / name).read_text()
            self.assertIn("Assistant PPL", document)
            self.assertIn("原文 PPL", document)
        self.assertIn("尚未完成", (self.root / "RESULTS.md").read_text())

    def test_special_tokens_make_letter_generation_invalid(self):
        summary, logits = self.fixture()
        probe = summary["generation_probe"][0]
        probe.update(generated_token_ids=[99, 10, 2], generated_tokens=3,
                     raw_response="<SPECIAL>B<EOS>", nonterminal_special_token_ids=[99],
                     strict_single_letter=False)
        folder = self.publish(summary, logits)
        read_run(folder, self.items)
        probe["strict_single_letter"] = True
        self.publish(summary, logits)
        with self.assertRaisesRegex(ValueError, "strict-generation"):
            read_run(folder, self.items)

    def test_eos_at_token_limit_is_not_truncation(self):
        summary, logits = self.fixture()
        probe = summary["generation_probe"][0]
        probe.update(generated_token_ids=[10] * 31 + [2], generated_tokens=32,
                     hit_length_limit=False)
        folder = self.publish(summary, logits)
        read_run(folder, self.items)
        probe["hit_length_limit"] = True
        self.publish(summary, logits)
        with self.assertRaisesRegex(ValueError, "length-limit/EOS"):
            read_run(folder, self.items)

    def test_score_delta_conservation_and_source_consistency(self):
        runs = {}
        for tag in ("dense-s00", "wanda-s50"):
            summary, logits = self.fixture(tag, change_first=tag != "dense-s00")
            runs[tag] = read_run(self.publish(summary, logits), self.items)
        configs = [self.configs[tag] for tag in runs]
        rows, deltas, details = derive_rows(configs, runs, self.items)
        self.assertAlmostEqual(rows[1]["agreement_60"], 59 / 60)
        self.assertEqual(sum(row["changed"] for row in deltas), 1)
        changed_item = self.items[0]
        signed_delta = changed_item["dir"] * (-3 - 2)
        contribution = details["wanda-s50"]["top_score_contributions"][0]
        self.assertEqual(contribution["signed_score_delta"], signed_delta)
        self.assertEqual(rows[1][changed_item["trait"]] - rows[0][changed_item["trait"]], signed_delta)
        runs["wanda-s50"][0]["provenance"]["source_sha256"] = {"synthetic": "changed"}
        with self.assertRaisesRegex(ValueError, "source provenance"):
            derive_rows(configs, runs, self.items)

    def test_cross_model_uses_own_baselines_not_absolute_ppl(self):
        summary, logits = self.fixture()
        runs = {"dense-s00": read_run(self.publish(summary, logits), self.items)}
        rows, _, _ = derive_rows([self.configs["dense-s00"]], runs, self.items)
        gemma = copy.deepcopy(rows[0])
        gemma.update(ppl=100., ppl_ratio=2., agreement_60=.75)
        qwen = copy.deepcopy(rows[0])
        qwen.update(ppl=10., ppl_ratio=3., agreement_60=.5)
        qwen_dir = self.root / "qwen"
        qwen_dir.mkdir()
        write_csv(qwen_dir / "comparison.csv", [qwen])
        matched = write_cross_model(self.root, [gemma], qwen_dir)
        self.assertEqual(matched[0]["gemma_ppl_relative_to_own_dense"], 2.)
        self.assertEqual(matched[0]["qwen_ppl_relative_to_own_dense"], 3.)
        self.assertEqual(matched[0]["gemma_answer_flip_rate"], .25)
        self.assertEqual(matched[0]["qwen_answer_flip_rate"], .5)
        self.assertNotIn("gemma_ppl", matched[0])
        self.assertNotIn("qwen_ppl", matched[0])
        self.assertFalse(any("assistant" in key for key in matched[0]))


if __name__ == "__main__":
    unittest.main()
