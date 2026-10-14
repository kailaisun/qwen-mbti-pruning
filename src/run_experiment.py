#!/usr/bin/env python
"""Run the paper's MBTI evaluation (arXiv:2608.25977) on a HF causal LM.

Optionally applies one-shot *static* weight pruning (no retraining) first,
so pruned and unpruned variants go through exactly the same pipeline.

Examples:
  python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen25-7b-fp16
  python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen25-7b-prune-u50 \
      --prune-pattern unstructured --prune-ratio 0.5
  python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen25-7b-prune-2to4 \
      --prune-pattern 2to4
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mbti_common import (  # noqa: E402
    OPTIONS,
    PROTOCOL_VERSION,
    dump_results,
    layer_metrics,
    load_items,
    option_token_ids,
    run_items_eval,
    score_mbti,
)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def linear_targets(model):
    """All Linear layers except the LM head (embeddings/norms untouched)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


@torch.no_grad()
def prune_static_(model, pattern, ratio):
    """Exact per-matrix magnitude masks; deterministic flat-index tie breaking."""
    if not 0 <= ratio <= 1:
        raise ValueError("prune ratio must lie in [0, 1]")
    if pattern == "2to4" and ratio != 0.5:
        raise ValueError("2:4 pruning has a fixed ratio of 0.5")
    if getattr(model.config, "quantization_config", None):
        raise ValueError("prune the dense baseline, not packed quantized weights")
    statistics = []
    for name, mod in linear_targets(model):
        w = mod.weight
        before = int((w == 0).sum())
        if pattern == "unstructured":
            a = w.detach().float().abs().flatten()
            k = int(a.numel() * ratio)
            mask = torch.zeros_like(a, dtype=torch.bool)
            if k == a.numel():
                mask.fill_(True)
            elif k:
                threshold = a.kthvalue(k).values
                mask = a < threshold
                remaining = k - int(mask.sum())
                ties = torch.nonzero(a == threshold, as_tuple=False).flatten()
                mask[ties[:remaining]] = True
            w.masked_fill_(mask.reshape_as(w), 0)
            masked = int(mask.sum())
        elif pattern == "2to4":
            assert w.shape[1] % 4 == 0, f"{name}: in_features not divisible by 4"
            g = w.detach().float().abs().reshape(w.shape[0], w.shape[1] // 4, 4)
            idx = g.argsort(dim=-1, stable=True)
            mask = torch.zeros_like(g, dtype=torch.bool)
            mask.scatter_(2, idx[..., :2], True)
            w.masked_fill_(mask.reshape_as(w), 0)
            masked = int(mask.sum())
        else:
            raise ValueError(f"unknown pattern {pattern}")
        statistics.append({"name": name, "numel": w.numel(), "masked": masked,
                           "zeros_before": before, "zeros_after": int((w == 0).sum())})
    if not statistics:
        raise ValueError("no eligible linear layers")
    return statistics


def weight_sparsity(model):
    total = zeros = 0
    for _, mod in linear_targets(model):
        total += mod.weight.numel()
        zeros += int((mod.weight == 0).sum())
    return zeros / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--tag", required=True, help="result tag (output dir name)")
    ap.add_argument("--outdir", default=str(HERE.parent / "outputs" / "corrected_v2"))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--save-pruned-dir", default=None, help="save and reload-verify pruned checkpoint")
    ap.add_argument("--items", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES))
    ap.add_argument("--prune-pattern", default="none",
                    choices=["none", "unstructured", "2to4"])
    ap.add_argument("--prune-ratio", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None,
                    help="only evaluate the first N items (smoke test)")
    args = ap.parse_args()
    if not 0 <= args.prune_ratio <= 1 or (args.prune_pattern == "2to4" and args.prune_ratio != 0.5):
        ap.error("invalid pruning ratio (2to4 requires 0.5)")
    if args.limit is not None and args.limit < 1:
        ap.error("limit must be positive")
    result_dir = Path(args.outdir) / args.tag
    if result_dir.exists():
        ap.error(f"result directory already exists: {result_dir}; use a new tag/outdir")
    if args.save_pruned_dir and Path(args.save_pruned_dir).exists():
        ap.error("checkpoint directory already exists; use a new path")
    if args.save_pruned_dir and args.prune_pattern == "none":
        ap.error("save-pruned-dir requires a pruning pattern")

    torch.manual_seed(0)
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(args.device)

    t0 = time.time()
    load_kwargs = {"revision": args.revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], low_cpu_mem_usage=True, **load_kwargs
    )
    model.to(args.device)
    model.eval()
    print(f"[load] {args.model} ({args.dtype}) in {time.time() - t0:.1f}s", flush=True)

    prune_info = None
    if args.prune_pattern != "none":
        t1 = time.time()
        per_layer = prune_static_(model, args.prune_pattern, args.prune_ratio)
        prune_info = {
            "pattern": args.prune_pattern,
            "ratio": args.prune_ratio,
            "linear_weight_sparsity": weight_sparsity(model),
            "seconds": time.time() - t1,
            "algorithm": "per_matrix_exact_magnitude_stable_ties",
            "per_layer": per_layer,
            "total_model_parameters": sum(p.numel() for p in model.parameters()),
            "target_parameters": sum(x["numel"] for x in per_layer),
            "masked_parameters": sum(x["masked"] for x in per_layer),
        }
        print(f"[prune] {args.prune_pattern}: sparsity={prune_info['linear_weight_sparsity']:.6f}", flush=True)

    items = load_items(args.items)
    option_ids = option_token_ids(tokenizer)
    res = run_items_eval(model, tokenizer, items, option_ids,
                         device=args.device, limit=args.limit, desc=args.tag)

    sub_items = items[: len(res["choices"])]
    mbti, detail = score_mbti(sub_items, res["choices"])
    metrics = layer_metrics(res["opt_logits"])
    legacy_choices = [OPTIONS[i] for i in res["legacy_final_logits"].argmax(-1)]
    source_files = [*HERE.glob("*.py"), HERE.parent / "data" / "mbti_items.json"]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "items_sha256": hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest(),
        "model_commit": getattr(model.config, "_commit_hash", None),
        "requested_revision": args.revision,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "seed": 0,
        "layer_count": model.config.num_hidden_layers,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        "attention_implementation": getattr(model.config,"_attn_implementation",None),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
    }
    checkpoint_info = None
    if args.save_pruned_dir:
        path = Path(args.save_pruned_dir).resolve()
        print(f"[checkpoint] saving {path}", flush=True)
        model.save_pretrained(path, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(path)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("[checkpoint] reloading and verifying all evaluated items", flush=True)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPES[args.dtype], local_files_only=True)
        model.to(args.device).eval()
        reloaded = run_items_eval(model, tokenizer, items, option_ids, device=args.device,
                                 limit=args.limit, desc=args.tag+"-reload")
        np.testing.assert_allclose(reloaded["opt_logits"], res["opt_logits"], rtol=1e-4, atol=1e-4)
        if reloaded["choices"] != res["choices"]:
            raise AssertionError("checkpoint reload changed choices")
        checkpoint_info = {"path": str(path), "reload_verified_items": len(sub_items),
                           "reload_max_abs_error": float(np.abs(reloaded["opt_logits"]-res["opt_logits"]).max()),
                           "weight_bytes": sum(p.stat().st_size for p in path.glob("*.safetensors"))}
        (path / "pruning_manifest.json").write_text(json.dumps({"prune":prune_info,"provenance":provenance},indent=2))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": args.tag,
        "model": args.model,
        "dtype": args.dtype,
        "prune": prune_info,
        "n_items": len(res["choices"]),
        "predicted_mbti": mbti,
        "dichotomies": detail,
        "per_item_choices": {str(it["id"]): ch for it, ch in zip(sub_items, res["choices"])},
        "layer_entropy_mean": metrics["entropy_mean"].tolist(),
        "layer_gap_mean": metrics["gap_mean"].tolist(),
        "eval_seconds": res["elapsed_sec"],
        "option_token_ids": option_ids,
        "diagnostics": res["diagnostics"],
        "native_projection_max_abs_error": max(d["native_projection_max_abs_error"] for d in res["diagnostics"]),
        "mean_option_probability_mass": float(np.mean([d["option_probability_mass"] for d in res["diagnostics"]])),
        "unconstrained_option_rate": float(np.mean([d["unconstrained_top_is_option"] for d in res["diagnostics"]])),
        "legacy_double_norm_on_same_weights": {"predicted_mbti": score_mbti(sub_items,legacy_choices)[0],
                                               "changed_choices":sum(a!=b for a,b in zip(legacy_choices,res["choices"]))},
        "checkpoint": checkpoint_info,
        "provenance": provenance,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
    }
    out = dump_results(args.outdir, args.tag, summary, res["opt_logits"],
                       legacy_final_logits=res["legacy_final_logits"])
    print(f"[result] MBTI={mbti}  dichotomies=" +
          " ".join(f"{d}:{v['score']}({v['pole']})" for d, v in detail.items()), flush=True)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
