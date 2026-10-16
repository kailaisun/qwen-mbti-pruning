"""Static pruning algorithms for dense Qwen2 Linear weights.

SparseGPT's error-compensated update is adapted from IST-DASLab/sparsegpt
(Apache-2.0); see THIRD_PARTY_NOTICES.md and licenses/SparseGPT-APACHE-2.0.txt.
Changes: Linear-only, exact-count stable masks, explicit diagnostics, Qwen2
block calibration, no quantization. Wanda follows locuslab/wanda (MIT).
"""
from __future__ import annotations

import hashlib
import time

import torch

def linear_targets(model):
    """All Linear layers except the LM head (embeddings/norms untouched)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


def selected(name, scope, layer_count):
    if scope == 'all':
        return True
    if scope in ('attention', 'mlp'):
        return ('.self_attn.' if scope == 'attention' else '.mlp.') in name
    index = int(name.split('.')[2])
    # 28 layers partitioned as [0,9), [9,18), [18,28).
    bounds = {'early': (0, layer_count//3),
              'middle': (layer_count//3, 2*layer_count//3),
              'late': (2*layer_count//3, layer_count)}
    lo, hi = bounds[scope]
    return lo <= index < hi


def exact_mask(scores, count):
    """Exactly count entries; break score ties by ascending flat index."""
    flat = scores.flatten()
    if not 0 <= count <= flat.numel():
        raise ValueError('invalid mask count')
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count == flat.numel():
        mask.fill_(True)
    elif count:
        cutoff = flat.kthvalue(count).values
        mask = flat < cutoff
        left = count - int(mask.sum())
        if left:
            ties = torch.nonzero(flat == cutoff, as_tuple=False).flatten()
            mask[ties[:left]] = True
    return mask.reshape_as(scores)


def make_mask(scores, ratio, granularity='matrix', drop_n=0, group_m=4):
    if not 0 <= ratio <= 1:
        raise ValueError('ratio outside [0,1]')
    if drop_n:
        if not 0 < drop_n < group_m or scores.shape[1] % group_m:
            raise ValueError('invalid N:M grouping')
        grouped = scores.reshape(scores.shape[0], -1, group_m)
        order = grouped.argsort(dim=-1, stable=True)
        mask = torch.zeros_like(grouped, dtype=torch.bool)
        mask.scatter_(-1, order[..., :drop_n], True)
        return mask.reshape_as(scores)
    if granularity == 'matrix':
        return exact_mask(scores, int(scores.numel()*ratio))
    if granularity != 'row':
        raise ValueError(granularity)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    count = int(scores.shape[1]*ratio)
    for start in range(0, scores.shape[0], 128):
        order = scores[start:start+128].argsort(dim=-1, stable=True)
        mask[start:start+128].scatter_(1, order[:, :count], True)
    return mask


@torch.no_grad()
def sparsegpt_weight(weight, hessian, ratio, drop_n=0, group_m=4,
                     blocksize=128, damping=0.01):
    """Second-order one-shot pruning; unstructured masks are block-global.

    As upstream, choices within a block are made before that block's updates.
    Unlike upstream's <= quantile, masks contain floor(block_numel*ratio)
    entries exactly. Dead columns may contribute additional actual zeros.
    """
    w = weight.float().clone()
    h = hessian.float().clone()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damping*h.diagonal().mean())
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    del h
    mask_all = torch.zeros_like(w, dtype=torch.bool)
    estimated_loss = 0.0
    for start in range(0, w.shape[1], blocksize):
        end = min(start+blocksize, w.shape[1])
        current = w[:, start:end].clone()
        quantized = torch.zeros_like(current)
        errors = torch.zeros_like(current)
        hinv = upper[start:end, start:end]
        scores = current.square()/hinv.diagonal().square().unsqueeze(0)
        mask = (torch.zeros_like(current, dtype=torch.bool) if drop_n else
                exact_mask(scores, int(scores.numel()*ratio)))
        for col in range(end-start):
            if drop_n and col % group_m == 0:
                score = current[:, col:col+group_m].square()/hinv.diagonal()[col:col+group_m].square()
                mask[:, col:col+group_m] = make_mask(score, ratio, drop_n=drop_n, group_m=group_m)
            original = current[:, col].clone()
            pruned = original.masked_fill(mask[:, col], 0)
            quantized[:, col] = pruned
            error = (original-pruned)/hinv[col, col]
            current[:, col:].sub_(error.unsqueeze(1)*hinv[col, col:].unsqueeze(0))
            errors[:, col] = error
        w[:, start:end] = quantized
        w[:, end:].sub_(errors @ upper[start:end, end:])
        mask_all[:, start:end] = mask
        estimated_loss += float(errors.square().sum())/2
    if not torch.isfinite(w).all():
        raise ValueError('SparseGPT generated non-finite weights')
    return w.to(weight.dtype), mask_all, {'estimated_reconstruction_loss': estimated_loss,
                                        'dead_input_channels': int(dead.sum())}


def weight_stats(name, old, new, mask, extra=None):
    before = old.float()
    delta = new.float()-before
    packed = __import__('numpy').packbits(mask.cpu().numpy().reshape(-1))
    stats = {'name': name, 'numel': old.numel(), 'masked': int(mask.sum()),
             'zeros_before': int((old == 0).sum()), 'zeros_after': int((new == 0).sum()),
             'weight_energy': float(before.square().sum()),
             'weight_delta_energy': float(delta.square().sum()),
             'mask_sha256': hashlib.sha256(packed.tobytes()).hexdigest()}
    stats.update(extra or {})
    return stats


@torch.no_grad()
def prune_datafree(model, method, ratio, seed=0, scope='all', drop_n=0):
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    stats = []
    for name, module in linear_targets(model):
        if not selected(name, scope, model.config.num_hidden_layers):
            continue
        old = module.weight.detach().clone()
        scores = (torch.rand(old.shape, device=old.device, generator=generator) if method == 'random'
                  else old.float().abs())
        mask = make_mask(scores, ratio, 'row' if method == 'row_magnitude' else 'matrix', drop_n)
        module.weight.masked_fill_(mask, 0)
        stats.append(weight_stats(name, old, module.weight, mask))
    return stats, []


class InputCaptured(Exception):
    pass


@torch.no_grad()
def capture_block_inputs(model, calibration):
    captured = {}
    def catch(module, args, kwargs):
        captured.update(kwargs)
        raise InputCaptured()
    handle = model.model.layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        model(calibration[:1], use_cache=False, logits_to_keep=1)
    except InputCaptured:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError('failed to capture Qwen2 block arguments')
    return model.model.embed_tokens(calibration), captured


def block_forward(block, x, kwargs):
    y = block(x, **kwargs)
    return y[0] if isinstance(y, tuple) else y


@torch.no_grad()
def prune_calibrated(model, calibration, method, ratio, scope='all', drop_n=0):
    """Calibrate each block on already-pruned preceding blocks; no MBTI input."""
    hidden, kwargs = capture_block_inputs(model, calibration)
    statistics, block_errors = [], []
    for index, block in enumerate(model.model.layers):
        t0 = time.time()
        targets = {name: mod for name, mod in block.named_modules()
                   if isinstance(mod, torch.nn.Linear)
                   and selected(f'model.layers.{index}.{name}', scope, len(model.model.layers))}
        if not targets:
            hidden = torch.cat([block_forward(block, x.unsqueeze(0), kwargs) for x in hidden], 0)
            continue
        # Q/K/V and gate/up share inputs, so compute each Gram only once.
        shared = {'self_attn.k_proj': 'self_attn.q_proj',
                  'self_attn.v_proj': 'self_attn.q_proj', 'mlp.up_proj': 'mlp.gate_proj'}
        keys = {name: shared.get(name, name) for name in targets}
        collectors = {}
        handles = []
        def hook_for(key):
            def hook(module, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                if method == 'wanda':
                    collectors[key].add_(x.square().sum(0))
                else:
                    collectors[key].addmm_(x.T, x, alpha=2/calibration.shape[0])
            return hook
        for key in sorted(set(keys.values())):
            module = block.get_submodule(key)
            n = module.weight.shape[1]
            shape = (n,) if method == 'wanda' else (n,n)
            collectors[key] = torch.zeros(shape, device=hidden.device, dtype=torch.float32)
            handles.append(module.register_forward_hook(hook_for(key)))
        before = torch.cat([block_forward(block, x.unsqueeze(0), kwargs) for x in hidden], 0)
        for handle in handles:
            handle.remove()
        for name, module in targets.items():
            old = module.weight.detach().clone()
            data = collectors[keys[name]]
            extra = {}
            if method == 'wanda':
                score = old.float().abs()*data.sqrt().unsqueeze(0)
                mask = make_mask(score, ratio, 'row', drop_n)
                module.weight.masked_fill_(mask, 0)
            elif method == 'sparsegpt':
                new, mask, extra = sparsegpt_weight(old, data, ratio, drop_n=drop_n)
                module.weight.copy_(new)
            else:
                raise ValueError(method)
            statistics.append(weight_stats(f'model.layers.{index}.{name}', old, module.weight, mask, extra))
        del collectors, data, old, mask
        after = torch.cat([block_forward(block, x.unsqueeze(0), kwargs) for x in hidden], 0)
        relative = float((after.float()-before.float()).square().sum()/before.float().square().sum())
        block_errors.append({'layer': index, 'relative_output_mse': relative})
        hidden = after
        del before
        print(f'[prune] {method} layer={index} relative_mse={relative:.6g} sec={time.time()-t0:.1f}', flush=True)
    return statistics, block_errors
