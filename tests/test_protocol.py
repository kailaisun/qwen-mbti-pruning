"""Regression checks for native logits, unbiased ties, and exact pruning."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbti_common import OPTIONS, collapse_options, project_outputs, score_mbti
from run_experiment import prune_static_


class ProtocolTests(unittest.TestCase):
    def test_final_matches_native_not_double_norm(self):
        torch.manual_seed(19)
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        model = Qwen2ForCausalLM(cfg).eval()
        with torch.no_grad():
            model.model.norm.weight.copy_(torch.linspace(0.2, 3.0, 16))
            out = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True, use_cache=False)
            ids = {c: [i, i+7] for i,c in enumerate(OPTIONS)}
            z, legacy, diagnostics = project_outputs(model, out, ids)
            native = collapse_options(out.logits[:, -1], ids)[0]
            torch.testing.assert_close(torch.from_numpy(z[-1]), native)
            self.assertGreater(float((torch.from_numpy(legacy)-native).abs().max()), 1e-3)
            for i,h in enumerate(out.hidden_states[:-1]):
                reference = collapse_options(model.lm_head(model.model.norm(h[:, -1])),ids)[0]
                torch.testing.assert_close(torch.from_numpy(z[i]),reference,atol=1e-6,rtol=1e-5)
            self.assertLess(diagnostics['native_projection_max_abs_error'],1e-6)

    def test_ties_do_not_default_to_enfj(self):
        items=[{'trait':d,'dir':1} for d in ['EI','NS','FT','JP']]
        label, detail=score_mbti(items,['D']*4)
        self.assertEqual(label,'????')
        self.assertTrue(all(v['pole'] is None and v['tie'] for v in detail.values()))
        self.assertEqual(score_mbti(items,['A','G','A','G'])[0],'ESFP')

    def model(self):
        m=torch.nn.Module()
        m.config=SimpleNamespace()
        m.proj=torch.nn.Linear(8,2,bias=False)
        m.lm_head=torch.nn.Linear(8,2,bias=False)
        with torch.no_grad(): m.proj.weight.fill_(1)
        return m

    def test_exact_unstructured_with_ties(self):
        m=self.model(); untouched=m.lm_head.weight.detach().clone()
        stats=prune_static_(m,'unstructured',0.5)
        self.assertEqual(stats[0]['masked'],8)
        self.assertEqual(int((m.proj.weight==0).sum()),8)
        torch.testing.assert_close(m.lm_head.weight,untouched)
        self.assertTrue((m.proj.weight[0]==0).all())

    def test_two_of_four_and_invalid_ratio(self):
        m=self.model(); prune_static_(m,'2to4',0.5)
        self.assertTrue(((m.proj.weight.reshape(-1,4)==0).sum(-1)==2).all())
        with self.assertRaises(ValueError): prune_static_(m,'2to4',0.3)
        with self.assertRaises(ValueError): prune_static_(m,'unstructured',-0.1)

    def test_pruning_endpoints(self):
        m=self.model(); old=m.proj.weight.detach().clone()
        prune_static_(m,'unstructured',0)
        torch.testing.assert_close(m.proj.weight,old)
        prune_static_(m,'unstructured',1)
        self.assertTrue((m.proj.weight==0).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
