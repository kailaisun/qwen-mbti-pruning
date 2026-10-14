"""Shared utilities for the MBTI x quantization reproduction (arXiv:2608.25977).

Protocol (paper appendices C/D/E):
- 60 items from 16personalities; 7-point options A..G mapped to scores +3..-3.
- Dichotomy score S_d = sum_i y_i * s(o_i); sign(S_d) decides the pole.
- Layer-wise option logits: for every hidden-state entry (embeddings + each
  transformer layer), project the final-position hidden state through the
  model's final norm + lm_head (intermediate entries only), then take the max logit over the token ids
  spelled by each option letter ("A" / " A").
- Metrics: Shannon entropy H and top1-top2 confidence gap G over the 7
  options, averaged over items, per layer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = "2.0-native-final-logits"

OPTIONS = list("ABCDEFG")
OPTION_SCORES = {"A": 3, "B": 2, "C": 1, "D": 0, "E": -1, "F": -2, "G": -3}

# (key, first pole, second pole). The paper lists dichotomies as {E,I},{N,S},
# {F,T},{J,P}; positive S_d selects the first pole.
DICHOTOMIES = [("EI", "E", "I"), ("NS", "N", "S"), ("FT", "F", "T"), ("JP", "J", "P")]

_PROMPT_BODY = """You are completing a personality traits assessment.
You will be given one statement, Q. Your task is to indicate how well the statement generally describes you.
Select exactly one option:
A = Agree
B = Generally Agree
C = Partially Agree
D = Neither Agree nor Disagree
E = Partially Disagree
F = Generally Disagree
G = Disagree
Respond based on your overall tendency. Do not overthink.
Q: {question}
Return ONLY one letter from: A, B, C, D, E, F, G."""

_PROMPT_COND_HEADER = """Context: You are a human with personality type {mbti_type}.
Traits:
General: {general}
Strengths: {strengths}
Development areas: {development}
Characteristics: {characteristics}
Careers: {careers}
Under stress: {stress}
Relationships: {relationships}

Task:
""" + _PROMPT_BODY


def load_items(path=None):
    """Load the 60 MBTI items (dicts with id/text/trait/dir)."""
    p = Path(path) if path else REPO_ROOT / "data" / "mbti_items.json"
    if not p.is_absolute():
        p = REPO_ROOT / p
    return json.loads(p.read_text())


def build_prompt(question, personality=None):
    """Unconditional prompt, or the personality-conditional variant (Appendix A)."""
    if personality is None:
        return _PROMPT_BODY.format(question=question)
    return _PROMPT_COND_HEADER.format(question=question, **personality)


def option_token_ids(tokenizer):
    """Token-id sets spelling each option letter, as single tokens."""
    ids = {}
    for c in OPTIONS:
        cand = set()
        for text in (c, " " + c):
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1:
                cand.add(enc[0])
        if not cand:
            raise RuntimeError(f"no single-token id found for option {c!r}")
        ids[c] = sorted(cand)
    return ids


def resolve_final_norm(model):
    base = getattr(model, "model", model)
    for attr in ("norm", "final_layernorm", "ln_f"):
        mod = getattr(base, attr, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    raise AttributeError("could not locate the final layer norm")


def collapse_options(logits, option_ids):
    return torch.stack([logits[..., option_ids[c]].amax(-1) for c in OPTIONS], dim=-1)


@torch.inference_mode()
def project_outputs(model, out, option_ids):
    """Logit lens for intermediates; actual model logits for the final entry.

    Qwen2 returns the *already normalized* final hidden state. Applying norm
    there again is incorrect. Only the intermediate entries receive norm.
    Select LM-head rows before projecting intermediate states to avoid L full
    vocabulary projections. The final entry always uses out.logits directly.
    """
    norm = resolve_final_norm(model)
    head = model.get_output_embeddings()
    token_ids = sorted({i for ids in option_ids.values() for i in ids})
    ids = torch.tensor(token_ids, device=head.weight.device)
    lookup = {t: i for i, t in enumerate(token_ids)}
    local_ids = {c: [lookup[t] for t in option_ids[c]] for c in OPTIONS}
    h = torch.stack([x[:, -1, :] for x in out.hidden_states[:-1]], dim=1)
    h = norm(h.to(norm.weight.dtype))
    bias = head.bias.index_select(0, ids) if getattr(head, "bias", None) is not None else None
    intermediate = F.linear(h, head.weight.index_select(0, ids), bias).float()
    final = out.logits[:, -1, :].float()
    z = torch.cat([collapse_options(intermediate, local_ids),
                   collapse_options(final, option_ids).unsqueeze(1)], dim=1)

    # Independent regression check: no-norm LM-head projection must match the
    # native logits on this actual model/backend, including AWQ and GPTQ.
    projected = head(out.hidden_states[-1][:, -1, :]).float()
    torch.testing.assert_close(projected, final, rtol=1e-4, atol=1e-4)
    if not torch.isfinite(z).all():
        raise ValueError("non-finite option logits")
    legacy = collapse_options(head(norm(out.hidden_states[-1][:, -1, :])).float(), option_ids)
    probs = final.softmax(-1)
    diagnostics = {
        "native_projection_max_abs_error": float((projected-final).abs().max()),
        "option_probability_mass": float(probs[0, ids].sum()),
        "unconstrained_top_token_id": int(final[0].argmax()),
        "unconstrained_top_is_option": int(final[0].argmax()) in token_ids,
    }
    return z[0].cpu().numpy(), legacy[0].cpu().numpy(), diagnostics


@torch.inference_mode()
def layer_option_logits(model, tokenizer, question, option_ids, device="cuda:0", personality=None):
    """One question, no sampling; entry 0=embedding, entries 1..L=layers."""
    text = build_prompt(question, personality)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    z, legacy, diagnostics = project_outputs(model, out, option_ids)
    diagnostics["unconstrained_top_token"] = tokenizer.decode([diagnostics["unconstrained_top_token_id"]])
    return z, legacy, diagnostics


def run_items_eval(model, tokenizer, items, option_ids, device="cuda:0", limit=None,
                   personality=None, desc="eval"):
    """Evaluate all items; returns option logits [n, L+1, 7] and final choices."""
    from tqdm import tqdm

    n = len(items) if limit is None else min(limit, len(items))
    all_z, legacy_z, diagnostics = [], [], []
    t0 = time.time()
    for item in tqdm(items[:n], desc=desc):
        z, legacy, diag = layer_option_logits(model, tokenizer, item["text"], option_ids,
                                              device=device, personality=personality)
        all_z.append(z)
        legacy_z.append(legacy)
        diagnostics.append(diag)
    opt_logits = np.stack(all_z)  # [n, L+1, 7]
    choices = [OPTIONS[int(np.argmax(z[-1]))] for z in opt_logits]
    return {"opt_logits": opt_logits, "choices": choices, "diagnostics": diagnostics,
            "legacy_final_logits": np.stack(legacy_z), "elapsed_sec": time.time() - t0}


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_metrics(opt_logits):
    """Layer-wise mean entropy and mean top1-top2 gap over items."""
    p = softmax(opt_logits.astype(np.float32))  # [n, L+1, 7]
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)  # [n, L+1]
    srt = np.sort(p, axis=-1)
    gap = srt[..., -1] - srt[..., -2]
    return {"entropy_mean": entropy.mean(0), "gap_mean": gap.mean(0)}


def score_mbti(items, choices):
    """Appendix C scoring: per-dichotomy signed sums -> 4-letter type."""
    if len(items) != len(choices) or any(c not in OPTIONS for c in choices):
        raise ValueError("items and valid choices must align")
    totals = {d: 0 for d, _, _ in DICHOTOMIES}
    for item, choice in zip(items, choices):
        if item["trait"] in totals and item["dir"] is not None:
            totals[item["trait"]] += int(item["dir"]) * OPTION_SCORES[choice]
    detail = {}
    letters = []
    for d, p1, p2 in DICHOTOMIES:
        s = totals[d]
        pole = p1 if s > 0 else (p2 if s < 0 else None)
        count = sum(it["trait"] == d for it in items)
        detail[d] = {"score": int(s), "pole": pole, "tie": s == 0, "n_items": count,
                     "max_abs_score": 3 * count, "normalized_score": s/(3*count) if count else None}
        letters.append(pole or "?")
    return "".join(letters), detail


def dump_results(outdir, tag, summary, opt_logits, **arrays):
    d = Path(outdir) / tag
    d.mkdir(parents=True, exist_ok=True)
    if (d / "summary.json").exists() or (d / "results.npz").exists():
        raise FileExistsError(f"refusing to overwrite results: {d}")
    np.savez_compressed(d / "results.npz", opt_logits=opt_logits.astype(np.float32), **arrays)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return d
