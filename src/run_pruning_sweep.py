"""Run the preregistered static-pruning matrix; save each result independently."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mbti_common import (OPTIONS, PROTOCOL_VERSION, build_prompt, dump_results,
                         layer_metrics, load_items, option_token_ids, run_items_eval, score_mbti)
from static_pruning import linear_targets, prune_calibrated, prune_datafree

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def experiment_matrix():
    configs = []
    def add(method, ratio=0., seed=0, scope='all', drop_n=0, calibration_seed=0):
        tag = f'{method}-s{round(ratio*100):02d}'
        if method == 'random': tag += f'-seed{seed}'
        if drop_n: tag += f'-drop{drop_n}of4'
        if scope != 'all': tag += f'-{scope}'
        if calibration_seed: tag += f'-calib{calibration_seed}'
        configs.append(dict(tag=tag,method=method,ratio=ratio,seed=seed,scope=scope,
                            drop_n=drop_n,calibration_seed=calibration_seed))
    add('dense')
    for method in ['magnitude','row_magnitude','wanda','sparsegpt']:
        for ratio in [.1,.2,.3,.4,.5,.6,.7]: add(method, ratio)
    for ratio in [.1,.3,.5,.7]:
        for seed in [0,1,2]: add('random',ratio,seed=seed)
    for method in ['magnitude','wanda','sparsegpt']:
        for n in [1,2,3]: add(method,n/4,drop_n=n)
    for method in ['magnitude','wanda']:
        for scope in ['attention','mlp','early','middle','late']: add(method,.5,scope=scope)
    for method in ['wanda','sparsegpt']:
        for seed in [1,2]: add(method,.5,calibration_seed=seed)
    return configs


@torch.inference_mode()
def language_eval(model, test_tokens):
    losses = []
    for tokens in test_tokens:
        out = model(tokens.unsqueeze(0), use_cache=False)
        loss = torch.nn.functional.cross_entropy(out.logits[0,:-1].float(), tokens[1:], reduction='mean')
        losses.append(float(loss))
    nll = sum(losses)/len(losses)
    if not math.isfinite(nll): raise ValueError('non-finite test loss')
    return {'mean_nll': nll, 'perplexity': math.exp(nll), 'block_nll': losses,
            'predicted_tokens': test_tokens.shape[0]*(test_tokens.shape[1]-1),
            'description': 'WikiText-2 raw test, first 32 non-overlapping 256-token blocks; not full benchmark PPL'}


@torch.inference_mode()
def generation_probe(model, tokenizer, items):
    results = []
    for index in [0,9,19,29,39,49]:
        text = tokenizer.apply_chat_template([{'role':'user','content':build_prompt(items[index]['text'])}],
                                              tokenize=False,add_generation_prompt=True)
        enc = tokenizer(text,return_tensors='pt',add_special_tokens=False).to(model.device)
        seq = model.generate(**enc,do_sample=False,max_new_tokens=32,
                             pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(seq[0,enc.input_ids.shape[1]:],skip_special_tokens=True)
        results.append({'item_id':items[index]['id'],'response':response,
                        'strict_single_letter':response.strip() in OPTIONS,
                        'generated_tokens':int(seq.shape[1]-enc.input_ids.shape[1]),
                        'hit_length_limit':seq.shape[1]-enc.input_ids.shape[1] == 32})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',default=str(ROOT/'outputs/pruning_v3'))
    ap.add_argument('--tags',nargs='*',help='optional subset; all 64 by default')
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='2 calibration blocks, 2 test blocks; separate output only')
    args = ap.parse_args()
    configs = experiment_matrix()
    if args.list:
        print(json.dumps(configs,indent=2)); return
    if args.tags:
        unknown = set(args.tags)-{c['tag'] for c in configs}
        if unknown: ap.error(f'unknown tags: {unknown}')
        configs = [c for c in configs if c['tag'] in args.tags]
    outdir = Path(args.outdir)
    if args.smoke and 'smoke' not in outdir.name: ap.error('smoke requires smoke output directory')
    outdir.mkdir(parents=True,exist_ok=True)
    sources = [ROOT/x for x in ['src/run_pruning_sweep.py','src/static_pruning.py','src/mbti_common.py',
                                'data/mbti_items.json','data/pruning_corpus_manifest.json']]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    pending = []
    for cfg in configs:
        p = outdir/cfg['tag']/'summary.json'
        if p.exists():
            previous = json.loads(p.read_text())
            if previous['sweep_config'] != cfg or previous['provenance']['source_sha256'] != hashes:
                raise RuntimeError(f'incompatible previous result: {p}; use a new output directory')
            with np.load(p.parent/'results.npz') as saved:
                assert saved['opt_logits'].shape == (60,29,7) and np.isfinite(saved['opt_logits']).all()
            print('[reuse]',cfg['tag'],flush=True)
        else: pending.append(cfg)
    if not pending: return
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
                                               dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    # Independent fresh start for every treatment. Only Linear weights are changed.
    targets = linear_targets(model)
    original = {name:module.weight.detach().cpu().clone() for name,module in targets}
    target_count = sum(w.numel() for w in original.values())
    items = load_items(); option_ids = option_token_ids(tokenizer)
    corpus = np.load(ROOT/'data/pruning_corpus.npz')
    corpus_manifest = json.loads((ROOT/'data/pruning_corpus_manifest.json').read_text())
    for key in corpus.files:
        assert hashlib.sha256(corpus[key].tobytes()).hexdigest() == corpus_manifest['array_sha256'][key]
    for cfg in pending:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for name,module in targets: module.weight.copy_(original[name])
        print('[start]',json.dumps(cfg),flush=True)
        calibration = torch.tensor(corpus[f"calibration_seed{cfg['calibration_seed']}"],dtype=torch.long,device='cuda')
        test = torch.tensor(corpus['test'],dtype=torch.long,device='cuda')
        if args.smoke: calibration, test = calibration[:2], test[:2]
        tp = time.time()
        if cfg['method'] == 'dense': stats, errors = [], []
        elif cfg['method'] in ['wanda','sparsegpt']:
            stats, errors = prune_calibrated(model,calibration,cfg['method'],cfg['ratio'],cfg['scope'],cfg['drop_n'])
        else:
            stats, errors = prune_datafree(model,cfg['method'],cfg['ratio'],cfg['seed'],cfg['scope'],cfg['drop_n'])
        prune_seconds = time.time()-tp
        res = run_items_eval(model,tokenizer,items,option_ids,desc=cfg['tag'])
        label, dimensions = score_mbti(items,res['choices'])
        metrics = layer_metrics(res['opt_logits'])
        quality = language_eval(model,test)
        generations = generation_probe(model,tokenizer,items)
        full_zeros = sum(int((module.weight == 0).sum()) for _,module in targets)
        summary = {'protocol_version':PROTOCOL_VERSION,'sweep_version':'3.0','sweep_config':cfg,
                   'tag':cfg['tag'],'model':MODEL,'dtype':'float16','smoke':args.smoke,'n_items':60,
                   'predicted_mbti':label,'dichotomies':dimensions,
                   'per_item_choices':{str(it['id']):ch for it,ch in zip(items,res['choices'])},
                   'layer_entropy_mean':metrics['entropy_mean'].tolist(),
                   'layer_gap_mean':metrics['gap_mean'].tolist(),'diagnostics':res['diagnostics'],
                   'mean_option_probability_mass':float(np.mean([d['option_probability_mass'] for d in res['diagnostics']])),
                   'unconstrained_option_rate':float(np.mean([d['unconstrained_top_is_option'] for d in res['diagnostics']])),
                   'native_projection_max_abs_error':max(d['native_projection_max_abs_error'] for d in res['diagnostics']),
                   'language_eval':quality,'generation_probe':generations,
                   'prune':{'per_layer':stats,'block_reconstruction':errors,'seconds':prune_seconds,
                            'total_model_parameters':sum(p.numel() for p in model.parameters()),
                            'eligible_parameters':target_count,'targeted_parameters':sum(s['numel'] for s in stats),
                            'masked_parameters':sum(s['masked'] for s in stats),'actual_zeros':full_zeros,
                            'eligible_zero_fraction':full_zeros/target_count},
                   'eval_seconds':res['elapsed_sec'],'elapsed_seconds':time.time()-t0,
                   'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'provenance':{'created_utc':datetime.now(timezone.utc).isoformat(),
                                 'source_sha256':hashes,'model_revision':REVISION,
                                 'corpus':corpus_manifest,'gpu':torch.cuda.get_device_name(),
                                 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'packages':{p:importlib.metadata.version(p) for p in
                                             ['torch','transformers','numpy','datasets','safetensors','accelerate']},
                                 'tf32':False,'attention_implementation':'sdpa'}}
        dump_results(outdir,cfg['tag'],summary,res['opt_logits'],legacy_final_logits=res['legacy_final_logits'])
        print(f"[complete] {cfg['tag']} {label} PPL={quality['perplexity']:.4f} sec={time.time()-t0:.1f}",flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print('SWEEP_SUBSET_COMPLETE',datetime.now(timezone.utc).isoformat(),flush=True)


if __name__ == '__main__': main()
