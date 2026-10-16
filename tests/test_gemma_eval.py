"""CPU-only native Gemma projection and answer-boundary regression tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import (BatchEncoding, Gemma4UnifiedConfig, Gemma4UnifiedForCausalLM,
                          Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedTextConfig)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gemma_eval import (ANSWER_PREFIX, PROTOCOL_VERSION, apply_final_softcap, generation_probe,
                       language_eval, prepend_bos, project_outputs, prompt_provenance,
                       prompt_text, validate_option_tokens)
from mbti_common import OPTIONS, build_prompt, collapse_options, option_token_ids


class FakeTokenizer:
    """A reversible tiny tokenizer with the same canonical answer boundary."""
    def get_chat_template(self):
        return "canonical-test-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "<bos><|turn>user\n" + messages[0]["content"] + "<turn|>\n" + ANSWER_PREFIX

    def encode(self, text, add_special_tokens=False):
        if text in OPTIONS:
            return [OPTIONS.index(text)]
        if text in [" " + c for c in OPTIONS]:
            return [7 + OPTIONS.index(text[1:])]
        for spelling in [" " + c for c in OPTIONS] + OPTIONS:
            if text.endswith(ANSWER_PREFIX + spelling):
                return self.encode(text[:-len(spelling)]) + self.encode(spelling)
        return [100 + ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        token = ids[0]
        return OPTIONS[token] if token < 7 else " " + OPTIONS[token - 7]


class GemmaEvalTests(unittest.TestCase):
    def model(self, full=True, dtype=torch.float32, cap=3.0):
        torch.manual_seed(113)
        text = Gemma4UnifiedTextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, global_head_dim=8, num_global_key_value_heads=1,
            attention_k_eq_v=True, layer_types=["sliding_attention", "full_attention"],
            final_logit_softcapping=cap, sliding_window=16,
        )
        config = Gemma4UnifiedConfig(text_config=text) if full else text
        cls = Gemma4UnifiedForConditionalGeneration if full else Gemma4UnifiedForCausalLM
        model = cls(config).to(dtype=dtype).eval()
        decoder = model.model.language_model if full else model.model
        with torch.no_grad():
            decoder.norm.weight.copy_(torch.linspace(0.1, 2.0, 16))
        return model

    def test_full_native_softcap_and_dynamic_layer_count(self):
        for full in [True, False]:
            with self.subTest(full=full), torch.inference_mode():
                model = self.model(full=full)
                out = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True,
                            use_cache=False, logits_to_keep=1)
                ids = {c: [i, i + 7] for i, c in enumerate(OPTIONS)}
                z, diagnostics = project_outputs(model, out, ids)
                self.assertEqual(z.shape, (3, 7))
                self.assertLess(diagnostics["native_projection_max_abs_error"], 1e-6)
                torch.testing.assert_close(torch.from_numpy(z[-1]), collapse_options(out.logits[0, -1], ids))
                decoder = model.model.language_model if full else model.model
                for index, state in enumerate(out.hidden_states[:-1]):
                    raw = model.lm_head(decoder.norm(state[:, -1]))
                    reference = collapse_options(torch.tanh(raw / 3.0) * 3.0, ids)[0]
                    torch.testing.assert_close(torch.from_numpy(z[index]), reference, rtol=1e-5, atol=1e-6)
                double = model.lm_head(decoder.norm(out.hidden_states[-1][:, -1]))
                double = torch.tanh(double / 3.0) * 3.0
                self.assertGreater(float((double - out.logits[:, -1]).abs().max()), 1e-3)

    def test_softcap_preserves_bfloat16_native_operation_order(self):
        config = SimpleNamespace(final_logit_softcapping=30.0)
        raw = torch.linspace(-70, 70, 1000).to(torch.bfloat16)
        got = apply_final_softcap(raw, config)
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(got, torch.tanh(raw / 30.0) * 30.0, rtol=0, atol=0)
        wrong = (torch.tanh(raw.float() / 30.0) * 30.0).to(torch.bfloat16)
        self.assertGreater(int((got != wrong).sum()), 0)
        with torch.inference_mode():
            model = self.model(dtype=torch.bfloat16, cap=30.0)
            out = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True,
                        use_cache=False, logits_to_keep=1)
            ids = {c: [i, i + 7] for i, c in enumerate(OPTIONS)}
            z, diag = project_outputs(model, out, ids)
            self.assertEqual(diag["native_projection_max_abs_error"], 0.0)
            self.assertEqual(diag["native_logits_dtype"], "torch.bfloat16")
            torch.testing.assert_close(torch.from_numpy(z[-1]), collapse_options(out.logits[0, -1].float(), ids))

    def test_prompt_has_closed_thought_and_contextual_options(self):
        tokenizer = FakeTokenizer()
        prompt = prompt_text(tokenizer, "A test question.")
        self.assertTrue(prompt.endswith(ANSWER_PREFIX))
        self.assertIn(build_prompt("A test question."), prompt)
        ids = option_token_ids(tokenizer)
        self.assertEqual(validate_option_tokens(tokenizer, ids, prompt), ids)
        metadata = prompt_provenance(tokenizer, [{"id": 1, "text": "A test question."}], ids)
        self.assertEqual(metadata["answer_prefix"], ANSWER_PREFIX)
        self.assertEqual(len(metadata["chat_template_sha256"]), 64)
        self.assertEqual(len(metadata["prompt_sha256"]["1"]), 64)
        self.assertFalse(metadata["enable_thinking"])

    def test_reject_open_thought_wrong_tokens_and_invalid_softcap(self):
        tokenizer = FakeTokenizer()
        tokenizer.apply_chat_template = lambda *args, **kwargs: "<|turn>model\n<|channel>thought\n"
        with self.assertRaisesRegex(ValueError, "closed-thought"):
            prompt_text(tokenizer, "Question")
        tokenizer = FakeTokenizer()
        ids = option_token_ids(tokenizer)
        ids["A"] = [31]
        with self.assertRaisesRegex(ValueError, "current tokenizer"):
            validate_option_tokens(tokenizer, ids, prompt_text(tokenizer, "Question"))
        for cap in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                apply_final_softcap(torch.ones(3), SimpleNamespace(final_logit_softcapping=cap))

    def test_native_projection_mismatch_is_not_silently_accepted(self):
        with torch.inference_mode():
            model = self.model()
            out = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True,
                        use_cache=False, logits_to_keep=1)
            out.logits = out.logits + 0.25
            ids = {c: [i, i + 7] for i, c in enumerate(OPTIONS)}
            with self.assertRaises(AssertionError):
                project_outputs(model, out, ids)

    def test_probe_records_raw_special_tokens_and_distinguishes_eos_from_limit(self):
        class ProbeTokenizer(FakeTokenizer):
            all_special_ids = [98, 99]

            def __call__(self, *args, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[20, 21]])})

            def decode(self, ids, skip_special_tokens=False):
                def text(token):
                    if token in self.all_special_ids:
                        return "" if skip_special_tokens else {98: "<|channel>", 99: "<eos>"}[token]
                    return OPTIONS[token] if token < 7 else " " + OPTIONS[token - 7]
                return "".join(text(token) for token in ids)

        class ProbeModel:
            generation_config = SimpleNamespace(eos_token_id=[99])

            def __init__(self):
                self.embedding = torch.nn.Embedding(32, 2)
                self.sequences = iter([
                    [0, 99], [98, 0, 99], [0] * 32,
                    [0] * 31 + [99], [0, 1, 99], [7, 99],
                ])

            def get_input_embeddings(self):
                return self.embedding

            def generate(self, input_ids, **kwargs):
                assert kwargs == dict(do_sample=False, max_new_tokens=32,
                                      temperature=None, top_k=None, top_p=None)
                return torch.cat([input_ids, torch.tensor([next(self.sequences)])], dim=1)

        items = [{"id": i + 1, "text": "Question"} for i in range(60)]
        results = generation_probe(ProbeModel(), ProbeTokenizer(), items)
        self.assertEqual([r["item_id"] for r in results], [1, 10, 20, 30, 40, 50])
        self.assertEqual([r["strict_single_letter"] for r in results], [True, False, False, False, False, True])
        self.assertEqual(results[1]["response"], "A")
        self.assertEqual(results[1]["raw_response"], "<|channel>A<eos>")
        self.assertTrue(results[2]["hit_length_limit"])
        self.assertFalse(results[3]["hit_length_limit"])
        self.assertTrue(results[3]["stopped_on_eos"])

    def test_explicit_bos_preserves_raw_tokens_shape_dtype_and_device(self):
        raw = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.int32)
        original = raw.clone()
        prefixed = prepend_bos(raw)
        self.assertEqual(prefixed.tolist(), [[2, 3, 4, 5], [2, 6, 7, 8]])
        self.assertEqual(prefixed.dtype, raw.dtype)
        self.assertEqual(prefixed.device, raw.device)
        torch.testing.assert_close(raw, original, rtol=0, atol=0)
        self.assertEqual(prepend_bos(raw[0]).tolist(), [2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "double prefix"):
            prepend_bos(prefixed)
        with self.assertRaises(ValueError):
            prepend_bos(raw.float())

    def test_language_eval_keeps_original_targets_and_ignores_bos_prediction(self):
        raw = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])

        class FixedLogitModel:
            def __init__(self):
                self.inputs = []
                self.logits = []

            def __call__(self, inputs, use_cache=False):
                assert not use_cache
                self.inputs.append(inputs.clone())
                logits = torch.zeros(1, 5, 16)
                # If BOS->first-text or the last position is mistakenly scored,
                # these large unrelated logits make the regression fail.
                logits[:, 0, 15] = 40
                logits[:, -1, 15] = 40
                for position in range(1, 4):
                    logits[0, position, inputs[0, position + 1]] = position
                self.logits.append(logits)
                return SimpleNamespace(logits=logits)

        model = FixedLogitModel()
        result = language_eval(model, raw)
        expected = [float(torch.nn.functional.cross_entropy(z[0, 1:-1], r[1:]))
                    for z, r in zip(model.logits, raw)]
        self.assertEqual(result["block_nll"], expected)
        self.assertEqual(result["mean_nll"], sum(expected) / len(expected))
        self.assertEqual(result["predicted_tokens"], 6)
        self.assertEqual(result["input_sequence_length"], 5)
        self.assertEqual(result["scored_targets_per_sequence"], 3)
        self.assertEqual(result["prefix"], "explicit_bos")
        self.assertEqual(result["bos_token_id"], 2)
        self.assertEqual(PROTOCOL_VERSION, "3.2-gemma-bos-native-answer-logits")
        for inputs, original in zip(model.inputs, raw):
            torch.testing.assert_close(inputs[0, 1:], original, rtol=0, atol=0)
            self.assertEqual(int(inputs[0, 0]), 2)
        with self.assertRaises(ValueError):
            language_eval(model, raw[:0])

    def test_language_eval_full_target_budget(self):
        class UniformModel:
            def __call__(self, inputs, use_cache=False):
                return SimpleNamespace(logits=torch.zeros(*inputs.shape, 8))

        result = language_eval(UniformModel(), torch.full((32, 256), 3, dtype=torch.long))
        self.assertEqual(result["predicted_tokens"], 8160)
        self.assertEqual(result["input_sequence_length"], 257)
        self.assertEqual(result["scored_targets_per_sequence"], 255)
        self.assertAlmostEqual(result["perplexity"], 8.0, places=5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
