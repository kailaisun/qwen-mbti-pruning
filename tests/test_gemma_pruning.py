"""CPU Gemma 4 pruning checks, including real mixed-attention KV sharing."""
from collections import UserDict
from pathlib import Path
import sys
import unittest

import torch
from transformers import (Gemma4UnifiedAudioConfig, Gemma4UnifiedConfig,
                          Gemma4UnifiedForConditionalGeneration,
                          Gemma4UnifiedTextConfig, Gemma4UnifiedVisionConfig)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gemma_pruning import (PREFIX, _replay_block, capture_block_inputs,
                           linear_targets, prune_calibrated, prune_datafree,
                           selected, text_model, verify_block_replay)
from static_pruning import make_mask


def tiny_model(shared_layers=2, k_eq_v=True, dtype=torch.float32):
    torch.manual_seed(39)
    text = Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, global_head_dim=8, num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"] * 3,
        num_kv_shared_layers=shared_layers, attention_k_eq_v=k_eq_v,
        sliding_window=4, max_position_embeddings=64, attention_bias=True,
        use_double_wide_mlp=True, final_logit_softcapping=30.0,
    )
    config = Gemma4UnifiedConfig(
        text_config=text,
        vision_config=Gemma4UnifiedVisionConfig(
            patch_size=2, pooling_kernel_size=2, mm_embed_dim=16,
            mm_posemb_size=8, output_proj_dims=16),
        audio_config=Gemma4UnifiedAudioConfig(audio_embed_dim=8),
    )
    model = Gemma4UnifiedForConditionalGeneration(config).eval().to(dtype=dtype)
    model.set_attn_implementation("eager")
    return model


def tokens():
    return torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
                         [43, 47, 53, 59, 61, 11, 13, 17, 19, 23, 29, 31]])


class GemmaPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_mixed_attention_capture_and_replay(self):
        for shared, k_eq_v in ((0, False), (0, True), (2, True)):
            with self.subTest(shared_layers=shared, k_eq_v=k_eq_v):
                model = tiny_model(shared, k_eq_v)
                base = text_model(model)
                hidden, arguments = capture_block_inputs(model, tokens())
                torch.testing.assert_close(hidden, base.embed_tokens(tokens()), rtol=0, atol=0)
                self.assertEqual(arguments[0].keyword["position_embeddings"][0].shape[-1], 4)
                self.assertEqual(arguments[1].keyword["position_embeddings"][0].shape[-1], 8)
                # The sequence exceeds the four-token window; masks really differ.
                self.assertFalse(torch.equal(arguments[0].keyword["attention_mask"],
                                             arguments[1].keyword["attention_mask"]))
                self.assertTrue(all("shared_kv_states" not in arg.keyword for arg in arguments))
                result = verify_block_replay(model, tokens())
                self.assertTrue(result["passed"])
                self.assertEqual(result["max_abs_error"], 0.0)
                self.assertEqual(result["kv_shared_layers"], [4, 5] if shared else [])
                if shared:
                    self.assertEqual(result["kv_source_layers"], [2, 3])
                    self.assertFalse(hasattr(base.layers[4].self_attn, "k_proj"))
                if k_eq_v:
                    self.assertIsNone(base.layers[1].self_attn.v_proj)

    def test_bfloat16_native_replay(self):
        result = verify_block_replay(tiny_model(dtype=torch.bfloat16), tokens())
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_target_boundary_and_datafree_preservation(self):
        for method, drop_n in (("magnitude", 0), ("row_magnitude", 0),
                               ("random", 0), ("magnitude", 2)):
            with self.subTest(method=method, drop_n=drop_n):
                model = tiny_model()
                original = {name: value.detach().clone() for name, value in model.named_parameters()}
                targets = {name for name, module in linear_targets(model)}
                # Two independent sliding layers (7 each), two independent
                # k_eq_v full layers (6 each), and two KV-shared layers (5 each).
                self.assertEqual(len(targets), 36)
                self.assertTrue(all(name.startswith(PREFIX) for name in targets))
                self.assertTrue(any("embed_vision" in name for name in original))
                self.assertTrue(any("embed_audio" in name for name in original))
                stats, errors = prune_datafree(model, method, .5, drop_n=drop_n)
                self.assertEqual({item["name"] for item in stats}, targets)
                self.assertFalse(errors)
                self.assertTrue(all(item["masked"] == item["numel"] // 2 for item in stats))
                for name, current in model.named_parameters():
                    target = name.removesuffix(".weight")
                    if name.endswith(".weight") and target in targets:
                        if method != "random":
                            mask = make_mask(original[name].abs(), .5,
                                             "row" if method == "row_magnitude" else "matrix", drop_n)
                            torch.testing.assert_close(current, original[name].masked_fill(mask, 0),
                                                       rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(current, original[name], rtol=0, atol=0)

    def test_scope_partition_and_scoped_changes(self):
        for index in range(48):
            name = f"{PREFIX}{index}.mlp.up_proj"
            self.assertEqual(sum(selected(name, scope, 48)
                                 for scope in ("early", "middle", "late")), 1)
            self.assertTrue(selected(name, ("early", "middle", "late")[index // 16], 48))
        for scope in ("attention", "mlp", "early", "middle", "late"):
            model = tiny_model()
            original = {name: module.weight.detach().clone() for name, module in linear_targets(model)}
            stats, _ = prune_datafree(model, "magnitude", .5, scope=scope)
            expected = {name for name in original if selected(name, scope, 6)}
            self.assertEqual({item["name"] for item in stats}, expected)
            for name, module in linear_targets(model):
                if name not in expected:
                    torch.testing.assert_close(module.weight, original[name], rtol=0, atol=0)

    def test_zero_calibrated_pruning_preserves_native_model(self):
        for method in ("wanda", "sparsegpt"):
            with self.subTest(method=method):
                model = tiny_model()
                original = {name: value.detach().clone() for name, value in model.named_parameters()}
                with torch.no_grad():
                    logits = model(tokens(), use_cache=False).logits
                stats, errors = prune_calibrated(model, tokens(), method, 0)
                self.assertEqual(len(stats), 36)
                self.assertEqual(len(errors), 6)
                self.assertTrue(all(item["masked"] == 0 for item in stats))
                self.assertTrue(all(item["relative_output_mse"] == 0 for item in errors))
                for name, current in model.named_parameters():
                    torch.testing.assert_close(current, original[name], rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(model(tokens(), use_cache=False).logits,
                                               logits, rtol=0, atol=0)
                self.assertTrue(verify_block_replay(model, tokens())["passed"])

    def test_before_pass_does_not_commit_kv(self):
        model = tiny_model()
        hidden, arguments = capture_block_inputs(model, tokens())
        states = [UserDict(), UserDict()]
        with torch.no_grad():
            for index in range(2):
                hidden = _replay_block(text_model(model).layers[index], hidden,
                                       arguments[index], states, commit=True)
            self.assertTrue(all(not state for state in states))
            before = _replay_block(text_model(model).layers[2], hidden, arguments[2], states, commit=False)
            self.assertTrue(all(not state for state in states))
            after = _replay_block(text_model(model).layers[2], hidden, arguments[2], states, commit=True)
            self.assertTrue(all("sliding_attention" in state for state in states))
            self.assertFalse(torch.equal(states[0]["sliding_attention"][0],
                                         states[1]["sliding_attention"][0]))
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_calibrated_pruning_commits_pruned_kv_to_each_sample(self):
        for method in ("wanda", "sparsegpt"):
            with self.subTest(method=method):
                model = tiny_model()
                original = {name: value.detach().clone() for name, value in model.named_parameters()}
                target_parameters = {name + ".weight" for name, module in linear_targets(model)}
                captured = {4: [], 5: []}
                handles = []

                def capture_for(index):
                    def capture(module, args, kwargs):
                        pair = kwargs["shared_kv_states"][module.layer_type]
                        captured[index].append(tuple(value.detach().clone() for value in pair))
                    return capture

                for index in captured:
                    attention = text_model(model).layers[index].self_attn
                    handles.append(attention.register_forward_pre_hook(capture_for(index), with_kwargs=True))
                try:
                    stats, errors = prune_calibrated(model, tokens(), method, .5)
                    # Last two entries are the post-pruning pass for samples 0/1.
                    replay_kv = {index: values[-2:] for index, values in captured.items()}
                    for values in captured.values():
                        values.clear()
                    with torch.no_grad():
                        for sample in tokens():
                            model.model(input_ids=sample.unsqueeze(0), use_cache=False, return_dict=True)
                    for index in captured:
                        self.assertEqual(len(captured[index]), 2)
                        for actual, expected in zip(replay_kv[index], captured[index]):
                            for value, reference in zip(actual, expected):
                                torch.testing.assert_close(value, reference, rtol=0, atol=0)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(stats), 36)
                self.assertEqual(len(errors), 6)
                self.assertTrue(all(item["masked"] == item["numel"] // 2 for item in stats))
                for name, current in model.named_parameters():
                    if name not in target_parameters:
                        torch.testing.assert_close(current, original[name], rtol=0, atol=0)
                self.assertTrue(verify_block_replay(model, tokens())["passed"])

    def test_invalid_model_mode_method_and_multimodal_input_rejected(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            prune_datafree(model, "not-a-method", .5)
        with self.assertRaises(ValueError):
            prune_calibrated(model, tokens(), "not-a-method", .5)
        with self.assertRaises(ValueError):
            capture_block_inputs(model.train(), tokens())
        model.eval()
        with self.assertRaises(ValueError):
            capture_block_inputs(model, torch.tensor([[model.config.image_token_id]]))


if __name__ == "__main__":
    unittest.main()
