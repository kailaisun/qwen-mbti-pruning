"""Text-decoder-only static pruning for Gemma 4 Unified.

The full conditional-generation model is retained: modality projectors, tied
embeddings/head, norms, and biases are never pruning targets. Decoder replay
captures the native model's masks and per-layer rotary embeddings rather than
assuming Qwen's uniform attention layout. Shared KV state is sample-local and
only the post-pruning pass is allowed to commit new KV tensors.

Mask and SparseGPT implementations are reused unchanged from static_pruning;
see THIRD_PARTY_NOTICES.md for the upstream algorithm licenses.
"""
from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
import time

import torch

from static_pruning import make_mask, sparsegpt_weight, weight_stats


PREFIX = "model.language_model.layers."


def text_model(model):
    """Return the text backbone of a full Gemma4Unified conditional model."""
    base = getattr(getattr(model, "model", None), "language_model", None)
    if base is None or not hasattr(base, "layers"):
        raise TypeError("Expected full Gemma4UnifiedForConditionalGeneration model")
    return base


def linear_targets(model):
    """Only decoder attention/MLP Linear weights; no multimodal projections."""
    targets = []
    for index, block in enumerate(text_model(model).layers):
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear) and name.startswith(("self_attn.", "mlp.")):
                targets.append((f"{PREFIX}{index}.{name}", module))
    return targets


def selected(name, scope, layer_count):
    if not name.startswith(PREFIX):
        raise ValueError(f"Not a Gemma text-decoder target: {name}")
    if scope == "all":
        return True
    if scope in ("attention", "mlp"):
        return (".self_attn." if scope == "attention" else ".mlp.") in name
    bounds = {"early": (0, layer_count // 3),
              "middle": (layer_count // 3, 2 * layer_count // 3),
              "late": (2 * layer_count // 3, layer_count)}
    if scope not in bounds:
        raise ValueError(f"Unknown scope: {scope}")
    index = int(name.removeprefix(PREFIX).split(".", 1)[0])
    low, high = bounds[scope]
    return low <= index < high


@torch.no_grad()
def prune_datafree(model, method, ratio, seed=0, scope="all", drop_n=0):
    if method not in ("magnitude", "row_magnitude", "random"):
        raise ValueError(f"Unknown data-free method: {method}")
    base = text_model(model)
    generator = torch.Generator(device=next(base.parameters()).device).manual_seed(seed)
    statistics = []
    for name, module in linear_targets(model):
        if not selected(name, scope, len(base.layers)):
            continue
        old = module.weight.detach().clone()
        scores = (torch.rand(old.shape, device=old.device, generator=generator)
                  if method == "random" else old.float().abs())
        mask = make_mask(scores, ratio, "row" if method == "row_magnitude" else "matrix", drop_n)
        module.weight.masked_fill_(mask, 0)
        statistics.append(weight_stats(name, old, module.weight, mask))
    return statistics, []


class _InputCaptured(Exception):
    pass


@dataclass
class BlockArguments:
    """Immutable native layer arguments except for intentionally excluded KV."""

    positional: tuple
    keyword: dict


def _check_text_tokens(model, tokens):
    if model.training:
        raise ValueError("Calibration/replay requires model.eval()")
    if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
        raise ValueError("Expected a nonempty [samples, sequence] token tensor")
    # No multimodal data or placeholder substitution is part of this experiment.
    for name in ("image_token_id", "video_token_id", "audio_token_id"):
        value = getattr(model.config, name, None)
        if value is not None and bool((tokens == value).any()):
            raise ValueError(f"Text-only calibration contains multimodal placeholder {name}")


@torch.no_grad()
def capture_block_inputs(model, calibration):
    """Capture actual scaled inputs and native arguments for every decoder layer.

    The first, intercepted batched pass stops before layer zero. A separate
    single-sample native pass records layer-specific masks/RoPE for sample-wise
    replay. All calibration sequences must have the same length, no padding
    mask, and positions starting at zero (the fixed corpus protocol).
    """
    _check_text_tokens(model, calibration)
    base = text_model(model)
    captured = {}

    def catch_first(module, args, kwargs):
        captured["hidden"] = (args[0] if args else kwargs["hidden_states"]).detach()
        raise _InputCaptured()

    handle = base.layers[0].register_forward_pre_hook(catch_first, with_kwargs=True)
    try:
        model.model(input_ids=calibration, use_cache=False, return_dict=True)
    except _InputCaptured:
        pass
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Failed to capture native Gemma decoder input")

    arguments = [None] * len(base.layers)

    def capture_for(index):
        def capture(module, args, kwargs):
            keyword = {name: value for name, value in kwargs.items()
                       if name not in ("hidden_states", "shared_kv_states")}
            if keyword.get("past_key_values") is not None:
                raise ValueError("Calibration must not use a generation cache")
            arguments[index] = BlockArguments(tuple(args[1:]), keyword)
        return capture

    handles = [block.register_forward_pre_hook(capture_for(index), with_kwargs=True)
               for index, block in enumerate(base.layers)]
    try:
        model.model(input_ids=calibration[:1], use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in arguments):
        raise RuntimeError("Failed to capture all Gemma layer arguments")
    return captured["hidden"], arguments


def block_forward(block, hidden, arguments, shared_kv_states):
    """One native block call; caller owns the mutable sample-local KV dict."""
    output = block(hidden, *arguments.positional,
                   shared_kv_states=shared_kv_states, **arguments.keyword)
    return output[0] if isinstance(output, tuple) else output


def _replay_block(block, hidden, arguments, states, commit):
    outputs = []
    for index, sample in enumerate(hidden):
        # Native attention only assigns dict entries, never mutates retained
        # KV tensors with use_cache=False, so a shallow copy is sufficient.
        current = UserDict(states[index])
        outputs.append(block_forward(block, sample.unsqueeze(0), arguments, current))
        if commit:
            states[index] = current
    return torch.cat(outputs, dim=0)


def _collector_keys(block, targets):
    """Share statistics only for verified identical-input Gemma projections."""
    aliases = {"self_attn.k_proj": "self_attn.q_proj",
               "self_attn.v_proj": "self_attn.q_proj",
               "mlp.up_proj": "mlp.gate_proj"}
    keys = {}
    for name, module in targets.items():
        key = aliases.get(name, name)
        source = targets.get(key)
        # Shared-KV layers have no k/v module; full-attention k_eq_v layers
        # have no v module. Never invent either as a collector.
        if source is None or source.in_features != module.in_features:
            key = name
        keys[name] = key
    return keys


@torch.no_grad()
def prune_calibrated(model, calibration, method, ratio, scope="all", drop_n=0):
    """Block-sequential Wanda/SparseGPT with already-pruned upstream KV state."""
    if method not in ("wanda", "sparsegpt"):
        raise ValueError(f"Unknown calibrated method: {method}")
    if not 0 <= ratio <= 1:
        raise ValueError("ratio outside [0,1]")
    base = text_model(model)
    hidden, arguments = capture_block_inputs(model, calibration)
    states = [UserDict() for _ in range(calibration.shape[0])]
    statistics, block_errors = [], []
    for index, block in enumerate(base.layers):
        started = time.time()
        targets = {name: module for name, module in block.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and name.startswith(("self_attn.", "mlp."))
                   and selected(f"{PREFIX}{index}.{name}", scope, len(base.layers))}
        if not targets:
            hidden = _replay_block(block, hidden, arguments[index], states, commit=True)
            continue
        keys = _collector_keys(block, targets)
        collectors, handles = {}, []

        def hook_for(key):
            def hook(module, inputs, output):
                values = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                if method == "wanda":
                    collectors[key].add_(values.square().sum(0))
                else:
                    collectors[key].addmm_(values.T, values, alpha=2 / calibration.shape[0])
            return hook

        for key in sorted(set(keys.values())):
            module = block.get_submodule(key)
            width = module.in_features
            shape = (width,) if method == "wanda" else (width, width)
            collectors[key] = torch.zeros(shape, device=hidden.device, dtype=torch.float32)
            handles.append(module.register_forward_hook(hook_for(key)))
        try:
            # This pass is deliberately not allowed to overwrite the KV state
            # that the post-pruning pass starts from.
            before = _replay_block(block, hidden, arguments[index], states, commit=False)
        finally:
            for handle in handles:
                handle.remove()
        for name, module in targets.items():
            old = module.weight.detach().clone()
            data = collectors[keys[name]]
            extra = {}
            if method == "wanda":
                score = old.float().abs() * data.sqrt().unsqueeze(0)
                mask = make_mask(score, ratio, "row", drop_n)
                module.weight.masked_fill_(mask, 0)
            else:
                new, mask, extra = sparsegpt_weight(old, data, ratio, drop_n=drop_n)
                module.weight.copy_(new)
            statistics.append(weight_stats(f"{PREFIX}{index}.{name}", old, module.weight, mask, extra))
        del collectors, data, old, mask
        after = _replay_block(block, hidden, arguments[index], states, commit=True)
        denominator = before.float().square().sum().clamp_min(torch.finfo(torch.float32).tiny)
        relative = float((after.float() - before.float()).square().sum() / denominator)
        block_errors.append({"layer": index, "relative_output_mse": relative})
        hidden = after
        del before
        print(f"[prune] {method} layer={index} relative_mse={relative:.6g} "
              f"sec={time.time() - started:.1f}", flush=True)
    return statistics, block_errors


@torch.no_grad()
def verify_block_replay(model, tokens):
    """Check decoder replay against native full-model hidden states, or raise.

    Native and replay use single-sample calls to avoid conflating batch-kernel
    numerical differences with architectural correctness. Explicit tolerances
    and measured errors are returned for the experiment provenance.
    """
    base = text_model(model)
    hidden, arguments = capture_block_inputs(model, tokens)
    states = [UserDict() for _ in range(tokens.shape[0])]
    for index, block in enumerate(base.layers):
        hidden = _replay_block(block, hidden, arguments[index], states, commit=True)
    actual = torch.cat([base.norm(sample.unsqueeze(0)) for sample in hidden], dim=0)
    expected = torch.cat([model.model(input_ids=sample.unsqueeze(0), use_cache=False,
                                     return_dict=True).last_hidden_state for sample in tokens], dim=0)
    if actual.dtype == torch.bfloat16:
        rtol, atol = 5e-3, 3.125e-2
    elif actual.dtype == torch.float16:
        rtol, atol = 2e-3, 2e-3
    else:
        rtol, atol = 1e-4, 1e-5
    error = (actual.float() - expected.float()).abs()
    relative_mse = float(error.square().sum() /
                         expected.float().square().sum().clamp_min(torch.finfo(torch.float32).tiny))
    diagnostics = {
        "passed": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_abs_error": float(error.max()),
        "per_sample_max_abs_error": error.flatten(1).max(1).values.cpu().tolist(),
        "relative_output_mse": relative_mse,
        "rtol": rtol, "atol": atol, "dtype": str(actual.dtype),
        "samples": tokens.shape[0], "sequence_length": tokens.shape[1],
        "layer_count": len(base.layers),
        "attention_types": sorted(set(base.config.layer_types)),
        "kv_shared_layers": [index for index, block in enumerate(base.layers)
                             if block.self_attn.is_kv_shared_layer],
        "kv_source_layers": [index for index, block in enumerate(base.layers)
                             if block.self_attn.store_full_length_kv],
    }
    if not diagnostics["passed"]:
        raise RuntimeError(f"Gemma native/replay mismatch: {diagnostics}")
    return diagnostics
