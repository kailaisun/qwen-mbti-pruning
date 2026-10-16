"""Fast CPU checks for masks, second-order compensation, and design coverage."""
import sys
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from static_pruning import (exact_mask, make_mask, sparsegpt_weight, capture_block_inputs,
                            block_forward, prune_datafree, prune_calibrated, selected)
from run_pruning_sweep import experiment_matrix


def reference_magnitude_weight(weight,ratio,drop_n=0):
    """Independent Python-sort oracle, including flat-index tie breaking."""
    groups=weight.reshape(-1,4) if drop_n else weight.reshape(1,-1)
    mask=torch.zeros_like(groups,dtype=torch.bool)
    for row,values in enumerate(groups.abs().tolist()):
        order=sorted(range(len(values)),key=lambda index:(values[index],index))
        count=drop_n if drop_n else int(len(values)*ratio)
        mask[row,order[:count]]=True
    return weight.masked_fill(mask.reshape_as(weight),0)


class StaticPruningTests(unittest.TestCase):
    def test_exact_ties(self):
        x=torch.ones(3,8)
        mask=exact_mask(x,7)
        self.assertEqual(int(mask.sum()),7)
        self.assertTrue(mask.flatten()[:7].all())
        self.assertFalse(mask.flatten()[7:].any())

    def test_row_and_group_budgets(self):
        x=torch.ones(3,12)
        mask=make_mask(x,.3,'row')
        self.assertTrue((mask.sum(1)==3).all())
        for drop in [1,2,3]:
            mask=make_mask(x,drop/4,drop_n=drop)
            self.assertTrue((mask.reshape(-1,4).sum(-1)==drop).all())

    def test_sparsegpt_diagonal_matches_magnitude(self):
        torch.manual_seed(0)
        w=torch.randn(5,16)
        h=torch.eye(16)
        new,mask,extra=sparsegpt_weight(w,h,.5,blocksize=16)
        reference=exact_mask(w.square(),40)
        self.assertTrue(torch.equal(mask,reference))
        torch.testing.assert_close(new,w.masked_fill(reference,0))
        new,mask,_=sparsegpt_weight(w,h,.5,drop_n=2,blocksize=16)
        self.assertTrue((mask.reshape(-1,4).sum(-1)==2).all())
        self.assertTrue((new[mask]==0).all())

    def test_sparsegpt_zero_and_compensation(self):
        torch.manual_seed(9)
        w=torch.randn(4,16)
        x=torch.randn(16,128); x[1]=x[0]+.1*x[1]
        h=2*x@x.T/128
        unchanged,mask,_=sparsegpt_weight(w,h,0,blocksize=8)
        torch.testing.assert_close(unchanged,w)
        new,mask,_=sparsegpt_weight(w,h,.5,blocksize=8)
        self.assertEqual(int(mask.sum()),32)
        self.assertTrue((new[mask]==0).all())
        self.assertGreater(float((new[~mask]-w[~mask]).abs().max()),1e-4)
        naive=w.masked_fill(mask,0)
        self.assertLess(float(((new-w)@x).square().sum()),float(((naive-w)@x).square().sum()))

    def test_capture_matches_full_model(self):
        torch.manual_seed(3)
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
        model=Qwen2ForCausalLM(cfg).eval()
        tokens=torch.randint(0,32,(2,12))
        with torch.no_grad():
            hidden,kwargs=capture_block_inputs(model,tokens)
            for block in model.model.layers:
                hidden=torch.cat([block_forward(block,x.unsqueeze(0),kwargs) for x in hidden],0)
            expected=model.model(tokens,use_cache=False).last_hidden_state
            torch.testing.assert_close(model.model.norm(hidden),expected,rtol=1e-4,atol=1e-5)

    def test_matrix(self):
        matrix=experiment_matrix()
        self.assertEqual(len(matrix),64)
        self.assertEqual(len({c['tag'] for c in matrix}),64)

    def test_datafree_matches_reference_and_preserves_other_parameters(self):
        torch.manual_seed(21)
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
        for tied in [False,True]:
            for drop in [0,1,2,3]:
                with self.subTest(tied=tied,drop_n=drop):
                    model=Qwen2ForCausalLM(cfg).eval()
                    eligible={name for name,p in model.named_parameters()
                              if name.endswith('.weight') and p.ndim==2
                              and ('.self_attn.' in name or '.mlp.' in name)}
                    if tied:
                        with torch.no_grad():
                            for name,p in model.named_parameters():
                                if name in eligible:
                                    p.copy_((torch.arange(p.numel()).reshape_as(p)%5-2).to(p.dtype))
                    original={name:p.detach().clone() for name,p in model.named_parameters()}
                    ratio=drop/4 if drop else .5
                    stats,_=prune_datafree(model,'magnitude',ratio,drop_n=drop)
                    self.assertEqual(len(stats),14)
                    self.assertEqual({x['name']+'.weight' for x in stats},eligible)
                    for name,actual in model.named_parameters():
                        expected=(reference_magnitude_weight(original[name],ratio,drop)
                                  if name in eligible else original[name])
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_calibrated_model_masks_and_scope_partition(self):
        torch.manual_seed(25)
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
        for method in ['wanda','sparsegpt']:
            model=Qwen2ForCausalLM(cfg).eval()
            embed=model.model.embed_tokens.weight.detach().clone()
            head=model.lm_head.weight.detach().clone()
            stats,errors=prune_calibrated(model,torch.randint(0,32,(2,12)),method,.5)
            self.assertEqual(len(stats),14);self.assertEqual(len(errors),2)
            self.assertTrue(all(x['masked']==x['numel']//2 for x in stats))
            torch.testing.assert_close(model.model.embed_tokens.weight,embed,rtol=0,atol=0)
            torch.testing.assert_close(model.lm_head.weight,head,rtol=0,atol=0)
        for index in range(28):
            name=f'model.layers.{index}.mlp.up_proj'
            self.assertEqual(sum(selected(name,s,28) for s in ['early','middle','late']),1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
