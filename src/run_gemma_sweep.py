"""Pinned Gemma 4-12B text-only extension of the 64-treatment pruning matrix."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration

from mbti_common import layer_metrics, load_items, option_token_ids, score_mbti
from run_pruning_sweep import experiment_matrix
from gemma_pruning import (linear_targets, prune_calibrated, prune_datafree,
                           text_model, verify_block_replay)
from gemma_eval import (assistant_language_eval, generation_probe, language_eval, prepend_bos,
                        prompt_provenance, run_items_eval)

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'google/gemma-4-12B-it'
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
PROTOCOL = '3.3-gemma-dual-ppl-native-answer-logits'


def save_result(outdir, tag, summary, logits):
    """Publish a complete result directory atomically, never overwrite one."""
    target = outdir / tag
    if target.exists():
        raise FileExistsError(target)
    staging = Path(tempfile.mkdtemp(prefix=f'.{tag}.', dir=outdir))
    np.savez_compressed(staging / 'results.npz', opt_logits=logits.astype(np.float32))
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    if target.exists():
        raise FileExistsError(target)
    staging.rename(target)


def source_hashes():
    files = ['src/run_gemma_sweep.py', 'src/gemma_pruning.py', 'src/gemma_eval.py',
             'src/static_pruning.py', 'src/mbti_common.py', 'src/run_pruning_sweep.py',
             'data/mbti_items.json', 'data/gemma4_12b_corpus_manifest.json']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default=str(ROOT / 'outputs/gemma4_12b_v3'))
    ap.add_argument('--tags', nargs='*')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='2 calibration and test blocks; separate output required')
    ap.add_argument('--verify-only', action='store_true', help='validate original full model without a sweep')
    args = ap.parse_args()
    configs = experiment_matrix()
    if args.list:
        print(json.dumps(configs, indent=2)); return
    if args.tags:
        unknown = set(args.tags) - {c['tag'] for c in configs}
        if unknown:
            ap.error(f'unknown tags: {unknown}')
        configs = [c for c in configs if c['tag'] in args.tags]
    outdir = Path(args.outdir)
    if args.smoke and 'smoke' not in outdir.name:
        ap.error('smoke requires a separate smoke output directory')
    outdir.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    pending = []
    for cfg in configs:
        previous_path = outdir / cfg['tag'] / 'summary.json'
        if not previous_path.exists():
            if previous_path.parent.exists():
                raise RuntimeError(f'incomplete result directory: {previous_path.parent}')
            pending.append(cfg); continue
        previous = json.loads(previous_path.read_text())
        if (previous['sweep_config'] != cfg or previous['provenance']['source_sha256'] != hashes
                or previous['smoke'] != args.smoke or previous['protocol_version'] != PROTOCOL):
            raise RuntimeError(f'incompatible result: {previous_path}; choose a new output directory')
        with np.load(previous_path.parent / 'results.npz') as saved:
            z = saved['opt_logits']
            assert z.shape == (60, previous['provenance']['layer_count'] + 1, 7)
            assert z.dtype == np.float32 and np.isfinite(z).all()
        print('[reuse]', cfg['tag'], flush=True)
    if not pending and not args.verify_only:
        return
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('[load]', MODEL, REVISION, flush=True)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    items = load_items()
    option_ids = option_token_ids(tokenizer)
    manifest = json.loads((ROOT / 'data/gemma4_12b_corpus_manifest.json').read_text())
    with np.load(ROOT / 'data/gemma4_12b_corpus.npz') as stored:
        corpus = {key: stored[key].copy() for key in stored.files}
    for key, values in corpus.items():
        assert hashlib.sha256(values.tobytes()).hexdigest() == manifest['array_sha256'][key]
    verification = verify_block_replay(model, prepend_bos(
        torch.as_tensor(corpus['test'][:2, :32], device='cuda', dtype=torch.long), tokenizer.bos_token_id))
    print('[block-replay]', json.dumps(verification), flush=True)
    targets = linear_targets(model)
    layer_count = len(text_model(model).layers)
    target_count = sum(module.weight.numel() for _, module in targets)
    total_count = sum(p.numel() for p in model.parameters())
    prompt_info = prompt_provenance(tokenizer, items, option_ids)
    print('[targets]', len(targets), target_count, 'total', total_count, 'layers', layer_count, flush=True)
    if args.verify_only:
        check = run_items_eval(model, tokenizer, items, option_ids, limit=2, desc='verification')
        print('[native-verification]', json.dumps(check['diagnostics']), flush=True)
        print('GEMMA_VERIFICATION_COMPLETE', flush=True)
        return
    original = {name: module.weight.detach().cpu().clone() for name, module in targets}
    for cfg in pending:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for name, module in targets:
                module.weight.copy_(original[name])
        print('[start]', json.dumps(cfg), flush=True)
        calibration = torch.as_tensor(corpus[f"calibration_seed{cfg['calibration_seed']}"], dtype=torch.long, device='cuda')
        test = torch.as_tensor(corpus['test'], dtype=torch.long, device='cuda')
        if args.smoke:
            calibration, test = calibration[:2], test[:2]
        # Every independent Gemma sequence needs an explicit BOS. Preserve raw
        # corpus hashes and identical 255 scored PPL targets per 256-token block.
        calibration = prepend_bos(calibration, tokenizer.bos_token_id)
        tp = time.time()
        if cfg['method'] == 'dense':
            stats, errors = [], []
        elif cfg['method'] in ('wanda', 'sparsegpt'):
            stats, errors = prune_calibrated(model, calibration, cfg['method'], cfg['ratio'], cfg['scope'], cfg['drop_n'])
        else:
            stats, errors = prune_datafree(model, cfg['method'], cfg['ratio'], cfg['seed'], cfg['scope'], cfg['drop_n'])
        pruning_seconds = time.time() - tp
        replay = verify_block_replay(model, prepend_bos(test[:2, :32], tokenizer.bos_token_id)) if args.smoke else None
        res = run_items_eval(model, tokenizer, items, option_ids, desc=cfg['tag'])
        label, dimensions = score_mbti(items, res['choices'])
        metrics = layer_metrics(res['opt_logits'])
        quality = language_eval(model, test, tokenizer.bos_token_id)
        assistant_quality = assistant_language_eval(model, tokenizer, test)
        generations = generation_probe(model, tokenizer, items)
        full_zeros = sum(int((module.weight == 0).sum()) for _, module in targets)
        summary = {
            'protocol_version': PROTOCOL, 'sweep_version': '3.3-gemma', 'sweep_config': cfg,
            'tag': cfg['tag'], 'model': MODEL, 'dtype': 'bfloat16', 'smoke': args.smoke, 'n_items': 60,
            'predicted_mbti': label, 'dichotomies': dimensions,
            'per_item_choices': {str(it['id']): c for it, c in zip(items, res['choices'])},
            'layer_entropy_mean': metrics['entropy_mean'].tolist(),
            'layer_gap_mean': metrics['gap_mean'].tolist(), 'diagnostics': res['diagnostics'],
            'mean_option_probability_mass': float(np.mean([d['option_probability_mass'] for d in res['diagnostics']])),
            'unconstrained_option_rate': float(np.mean([d['unconstrained_top_is_option'] for d in res['diagnostics']])),
            'native_projection_max_abs_error': max(d['native_projection_max_abs_error'] for d in res['diagnostics']),
            'language_eval': quality, 'assistant_language_eval': assistant_quality,
            'generation_probe': generations,
            'prune': {'per_layer': stats, 'block_reconstruction': errors, 'seconds': pruning_seconds,
                      'total_model_parameters': total_count, 'eligible_parameters': target_count,
                      'targeted_parameters': sum(s['numel'] for s in stats),
                      'masked_parameters': sum(s['masked'] for s in stats), 'actual_zeros': full_zeros,
                      'eligible_zero_fraction': full_zeros / target_count},
            'eval_seconds': res['elapsed_sec'], 'elapsed_seconds': time.time() - t0,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'provenance': {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashes,
                           'model_revision': REVISION, 'corpus': manifest, 'gpu': torch.cuda.get_device_name(),
                           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                           'packages': {p: importlib.metadata.version(p) for p in
                                        ['torch', 'transformers', 'numpy', 'datasets', 'safetensors', 'accelerate']},
                           'layer_count': layer_count, 'tf32': False, 'attention_implementation': 'sdpa',
                           'calibration_input': {'prefix': 'explicit_bos', 'bos_token_id': tokenizer.bos_token_id,
                                                 'raw_sequence_length': 256, 'input_sequence_length': 257,
                                                 'sequences': int(calibration.shape[0])},
                           'prompt': prompt_info, 'unpruned_block_replay': verification,
                           'smoke_pruned_block_replay': replay}}
        save_result(outdir, cfg['tag'], summary, res['opt_logits'])
        print(f"[complete] {cfg['tag']} {label} raw_PPL={quality['perplexity']:.4f} assistant_PPL={assistant_quality['perplexity']:.4f} sec={time.time()-t0:.1f}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print('GEMMA_SWEEP_SUBSET_COMPLETE', datetime.now(timezone.utc).isoformat(), flush=True)


if __name__ == '__main__':
    main()
