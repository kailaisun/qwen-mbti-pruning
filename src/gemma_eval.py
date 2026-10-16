"""Gemma 4 MBTI evaluation with an explicit non-thinking answer boundary.

The questionnaire and option aggregation are identical to Qwen v3. Gemma's
native chat format and final-logit softcap are not: the pinned canonical
template already closes an empty thought channel when thinking is disabled,
and the model applies its softcap in the LM-head output dtype. Both details
must be preserved before comparing candidate letters or projecting a lens.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from mbti_common import OPTIONS, build_prompt, collapse_options, option_token_ids


PROTOCOL_VERSION = "3.3-gemma-dual-ppl-native-answer-logits"
ANSWER_PREFIX = "<|turn>model\n<|channel>thought\n<channel|>"
PROBE_INDICES = (0, 9, 19, 29, 39, 49)
CONTINUATION_INSTRUCTION = "Continue the following encyclopedia passage. Output only the passage."


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_text(tokenizer, question):
    """Render the native template and stop after its closed empty thought.

    Fail closed if a different template places us in a thought channel or
    removes the expected answer boundary. We never silently append a custom
    answer cue or compare letter logits before Gemma's formatting tokens.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt.endswith(ANSWER_PREFIX):
        raise ValueError("Gemma non-thinking template lacks the expected closed-thought answer prefix")
    return prompt


def validate_option_tokens(tokenizer, option_ids, prompt):
    """Check all seven options, both spellings, and the contextual boundary."""
    if set(option_ids) != set(OPTIONS):
        raise ValueError("option_ids must contain exactly A through G")
    expected = option_token_ids(tokenizer)
    if any(sorted(set(option_ids[c])) != expected[c] for c in OPTIONS):
        raise ValueError("option_ids do not match the current tokenizer")
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    seen = set()
    for option in OPTIONS:
        for spelling in (option, " " + option):
            ids = tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) != 1 or ids[0] not in option_ids[option]:
                raise ValueError(f"Gemma option {spelling!r} is not a matched single token")
            if tokenizer.encode(prompt + spelling, add_special_tokens=False) != prefix_ids + ids:
                raise ValueError(f"Gemma option {spelling!r} changes tokenization at the answer boundary")
            if tokenizer.decode(ids, skip_special_tokens=False) != spelling:
                raise ValueError(f"Gemma option {spelling!r} does not round-trip exactly")
        current = set(option_ids[option])
        if seen.intersection(current):
            raise ValueError("option token sets must be disjoint")
        seen.update(current)
    return expected


def prompt_provenance(tokenizer, items, option_ids=None):
    """Serializable exact template/prompt fingerprint for experiment summaries."""
    if not items:
        raise ValueError("at least one questionnaire item is required")
    option_ids = option_token_ids(tokenizer) if option_ids is None else option_ids
    template = tokenizer.get_chat_template()
    prompts = [prompt_text(tokenizer, item["text"]) for item in items]
    # Validate every prompt, not just a representative boundary.
    for prompt in prompts:
        validate_option_tokens(tokenizer, option_ids, prompt)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "enable_thinking": False,
        "add_generation_prompt": True,
        "add_special_tokens": False,
        "answer_prefix": ANSWER_PREFIX,
        "answer_prefix_token_ids": tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False),
        "chat_template_sha256": _sha256(template),
        "prompt_sha256": {str(item["id"]): _sha256(prompt) for item, prompt in zip(items, prompts)},
        "option_token_ids": option_ids,
        "option_spellings": ["letter", "space+letter"],
        "option_aggregation": "maximum native logit over the two single-token spellings",
        "boundary_single_token_validation": "all items, all seven options, both spellings",
    }


def resolve_text_model(model):
    """Support the full multimodal wrapper and the text-only HF model."""
    base = getattr(model, "model", model)
    text = getattr(base, "language_model", base)
    if not isinstance(getattr(text, "norm", None), torch.nn.Module) or not hasattr(text, "layers"):
        raise TypeError("could not locate Gemma's text decoder and final norm")
    return text


def resolve_text_config(model):
    config = model.config
    if callable(getattr(config, "get_text_config", None)):
        config = config.get_text_config()
    else:
        config = getattr(config, "text_config", config)
    if not hasattr(config, "num_hidden_layers"):
        config = resolve_text_model(model).config
    return config


def apply_final_softcap(logits, config):
    """Match native operation order/dtype; do not upcast before tanh."""
    cap = getattr(config, "final_logit_softcapping", None)
    if cap is None:
        return logits
    if not math.isfinite(float(cap)) or float(cap) <= 0:
        raise ValueError("final_logit_softcapping must be positive and finite")
    return torch.tanh(logits / cap) * cap


def prepend_bos(tokens, bos_token_id=2):
    """Prefix each independently reset raw-text sequence, without truncation.

    Gemma's pinned tokenizer does not itself add BOS, even when its general
    ``add_special_tokens`` switch is enabled. The official Gemma 4 report,
    Table 11, requires explicit BOS after tokenization. Use this same helper
    for calibration, block replay, and the raw-continuation diagnostic.
    """
    if not isinstance(tokens, torch.Tensor) or tokens.ndim not in (1, 2):
        raise ValueError("raw tokens must be a [sequence] or [batch, sequence] tensor")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("raw tokens must use an integer token-ID dtype")
    if tokens.shape[-1] == 0 or tokens.numel() == 0:
        raise ValueError("raw token sequences must be nonempty")
    if not isinstance(bos_token_id, int) or isinstance(bos_token_id, bool) or bos_token_id < 0:
        raise ValueError("bos_token_id must be a nonnegative integer")
    if bool((tokens[..., 0] == bos_token_id).any()):
        raise ValueError("raw token sequence already starts with BOS; refusing a double prefix")
    bos = tokens.new_full((*tokens.shape[:-1], 1), bos_token_id)
    return torch.cat((bos, tokens), dim=-1)


@torch.inference_mode()
def language_eval(model, test_tokens, bos_token_id=2):
    """Raw continuation PPL with explicit BOS and unchanged scored targets.

    A raw 256-token block becomes 257 input tokens. We deliberately exclude
    the BOS-to-first-text prediction and retain exactly the original 255
    next-token targets, raw[1:]. This is a raw-continuation diagnostic, not a
    chat-conditioned benchmark; Gemma IT can have a high baseline here even
    with healthy native chat behavior. No chat prefix or metric selection is
    used to improve the score.
    """
    if not isinstance(test_tokens, torch.Tensor) or test_tokens.ndim != 2:
        raise ValueError("test tokens must be a [blocks, raw sequence] tensor")
    if test_tokens.shape[0] == 0 or test_tokens.shape[1] < 2:
        raise ValueError("PPL requires nonempty blocks with at least two raw tokens")
    losses = []
    for raw in test_tokens:
        inputs = prepend_bos(raw.unsqueeze(0), bos_token_id=bos_token_id)
        out = model(inputs, use_cache=False)
        if out.logits.ndim != 3 or out.logits.shape[:2] != inputs.shape:
            raise ValueError("PPL requires native logits at every BOS-prefixed input position")
        loss = F.cross_entropy(out.logits[0, 1:-1].float(), raw[1:].long(), reduction="mean")
        losses.append(float(loss))
    nll = sum(losses) / len(losses)
    if not math.isfinite(nll):
        raise ValueError("non-finite Gemma raw-continuation loss")
    targets = int(test_tokens.shape[1]) - 1
    return {
        "mean_nll": nll, "perplexity": math.exp(nll), "block_nll": losses,
        "predicted_tokens": len(losses) * targets,
        "bos_token_id": bos_token_id,
        "input_sequence_length": int(test_tokens.shape[1]) + 1,
        "scored_targets_per_sequence": targets,
        "prefix": "explicit_bos",
        "description": (
            "Gemma-tokenized WikiText-2 raw continuation; explicit BOS per reset block; "
            "score only original raw[1:] targets, excluding BOS-to-first-text prediction. "
            "Short-context subset, not a full benchmark or chat-quality score; this "
            "instruction-tuned model has a known high raw-continuation baseline. "
            "Compare pruning only with this model's identically evaluated dense baseline."
        ),
    }


def continuation_prefix(tokenizer):
    """The fixed auxiliary native-assistant condition, not an MBTI prompt."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": CONTINUATION_INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prefix, str) or not prefix.endswith(ANSWER_PREFIX):
        raise ValueError("continuation template lacks the closed-thought answer prefix")
    if not tokenizer.bos_token or not prefix.startswith(tokenizer.bos_token):
        raise ValueError("native continuation template must include its own BOS")
    return prefix


@torch.inference_mode()
def assistant_language_eval(model, tokenizer, test_tokens):
    """Uniform auxiliary continuation diagnostic; raw PPL remains unchanged.

    This condition was fixed after the input-format preflight and is applied
    to all 64 treatments, not selected per method. Raw tokens are appended to
    the native nonthinking assistant prefix without re-tokenizing a string
    boundary. Only the same original raw[1:] targets are scored. Calibration
    remains explicit-BOS raw text and never uses this auxiliary chat prefix.
    """
    if not isinstance(test_tokens, torch.Tensor) or test_tokens.ndim != 2:
        raise ValueError("test tokens must be a [blocks, raw sequence] tensor")
    if test_tokens.shape[0] == 0 or test_tokens.shape[1] < 2:
        raise ValueError("PPL requires nonempty blocks with at least two raw tokens")
    if test_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("raw tokens must use an integer token-ID dtype")
    if bool((test_tokens[:, 0] == tokenizer.bos_token_id).any()):
        raise ValueError("auxiliary continuation expects raw blocks without a BOS prefix")
    prefix_text = continuation_prefix(tokenizer)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if not prefix_ids or prefix_ids[0] != tokenizer.bos_token_id:
        raise ValueError("native continuation token sequence must begin with BOS")
    prefix = test_tokens.new_tensor(prefix_ids).unsqueeze(0)
    prefix_len = len(prefix_ids)
    losses = []
    for raw in test_tokens:
        inputs = torch.cat((prefix, raw.unsqueeze(0)), dim=1)
        output = model(inputs, use_cache=False)
        if output.logits.ndim != 3 or output.logits.shape[:2] != inputs.shape:
            raise ValueError("auxiliary PPL requires logits at every prefixed input position")
        loss = F.cross_entropy(output.logits[0, prefix_len:-1].float(), raw[1:].long(), reduction="mean")
        losses.append(float(loss))
    nll = sum(losses) / len(losses)
    if not math.isfinite(nll):
        raise ValueError("non-finite Gemma native-assistant continuation loss")
    targets = int(test_tokens.shape[1]) - 1
    return {
        "mean_nll": nll, "perplexity": math.exp(nll), "block_nll": losses,
        "predicted_tokens": len(losses) * targets,
        "condition": "native_assistant_continuation",
        "prefix_text": prefix_text, "prefix_token_ids": prefix_ids,
        "prefix_sha256": _sha256(prefix_text), "prefix_length": prefix_len,
        "input_sequence_length": prefix_len + int(test_tokens.shape[1]),
        "scored_targets_per_sequence": targets, "enable_thinking": False,
        "description": (
            "Uniform auxiliary WikiText-2 continuation under a fixed native nonthinking assistant "
            "prefix, chosen after format preflight and applied to every treatment. Score identical "
            "original raw[1:] targets; this changes conditioning and supplements, never replaces, "
            "the frozen explicit-BOS raw-continuation metric. Calibration remains raw text with BOS. "
            "Short-context subset, not a full benchmark or an overall chat-capability score."
        ),
    }


@torch.inference_mode()
def project_outputs(model, out, option_ids):
    """Return [L+1,7] logits and diagnostics for one item.

    Intermediate states receive final norm exactly once, then head + native
    softcap. The final entry comes directly from out.logits. An independent
    projection of the already-normalized final state verifies the native path.
    """
    text = resolve_text_model(model)
    config = resolve_text_config(model)
    head = model.get_output_embeddings()
    expected_layers = int(config.num_hidden_layers) + 1
    if out.hidden_states is None or len(out.hidden_states) != expected_layers:
        raise ValueError(f"expected {expected_layers} Gemma hidden-state entries")
    if out.logits.ndim != 3 or out.logits.shape[0] != 1:
        raise ValueError("project_outputs requires one item with [1, sequence, vocabulary] logits")
    if set(option_ids) != set(OPTIONS) or any(not option_ids[c] for c in OPTIONS):
        raise ValueError("nonempty token sets for A through G are required")
    token_ids = sorted({int(i) for values in option_ids.values() for i in values})
    if token_ids[0] < 0 or token_ids[-1] >= head.weight.shape[0]:
        raise ValueError("option token ID outside the LM vocabulary")
    if sum(len(set(option_ids[c])) for c in OPTIONS) != len(token_ids):
        raise ValueError("option token sets must be disjoint")
    ids = torch.tensor(token_ids, device=head.weight.device, dtype=torch.long)
    lookup = {token: i for i, token in enumerate(token_ids)}
    local_ids = {option: [lookup[token] for token in option_ids[option]] for option in OPTIONS}
    h = torch.stack([state[:, -1, :] for state in out.hidden_states[:-1]], dim=1)
    h = text.norm(h.to(device=text.norm.weight.device, dtype=text.norm.weight.dtype))
    bias = head.bias.index_select(0, ids) if getattr(head, "bias", None) is not None else None
    intermediate = F.linear(h.to(head.weight.device), head.weight.index_select(0, ids), bias)
    intermediate = apply_final_softcap(intermediate, config).float()
    final = out.logits[:, -1, :].float()
    z = torch.cat([
        collapse_options(intermediate, local_ids),
        collapse_options(final, option_ids).unsqueeze(1),
    ], dim=1)

    # Keep [batch,1,hidden] just like the native model's logits_to_keep=1 path.
    projected = head(out.hidden_states[-1][:, -1:, :])
    projected = apply_final_softcap(projected, config)[:, 0, :].float()
    torch.testing.assert_close(projected, final, rtol=1e-4, atol=1e-4)
    if not torch.isfinite(z).all() or not torch.isfinite(final).all():
        raise ValueError("non-finite Gemma logits")
    top_id = int(final[0].argmax())
    diagnostics = {
        "native_projection_max_abs_error": float((projected - final).abs().max()),
        "option_probability_mass": float(final.softmax(-1)[0, ids].sum()),
        "unconstrained_top_token_id": top_id,
        "unconstrained_top_is_option": top_id in token_ids,
        "hidden_state_entries": expected_layers,
        "final_logit_softcapping": getattr(config, "final_logit_softcapping", None),
        "native_logits_dtype": str(out.logits.dtype),
        "head_dtype": str(head.weight.dtype),
    }
    return z[0].cpu().numpy(), diagnostics


@torch.inference_mode()
def run_items_eval(model, tokenizer, items, option_ids, device="cuda:0", limit=None, desc="eval"):
    """Run the same questionnaire as Qwen with Gemma's native answer format."""
    from tqdm import tqdm

    n = len(items) if limit is None else min(int(limit), len(items))
    if n <= 0:
        raise ValueError("at least one evaluation item is required")
    all_z, diagnostics = [], []
    t0 = time.time()
    for item in tqdm(items[:n], desc=desc):
        prompt = prompt_text(tokenizer, item["text"])
        validate_option_tokens(tokenizer, option_ids, prompt)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        z, diag = project_outputs(model, out, option_ids)
        diag.update({
            "unconstrained_top_token": tokenizer.decode([diag["unconstrained_top_token_id"]]),
            "prompt_sha256": _sha256(prompt),
            "prompt_token_count": int(enc.input_ids.shape[1]),
            "answer_prefix": ANSWER_PREFIX,
        })
        all_z.append(z)
        diagnostics.append(diag)
    opt_logits = np.stack(all_z)
    choices = [OPTIONS[int(np.argmax(z[-1]))] for z in opt_logits]
    return {"opt_logits": opt_logits, "choices": choices, "diagnostics": diagnostics,
            "elapsed_sec": time.time() - t0}


@torch.inference_mode()
def generation_probe(model, tokenizer, items):
    """Six fixed, greedy, 32-token probes; preserve raw tokens and formatting."""
    if len(items) <= max(PROBE_INDICES):
        raise ValueError("the generation probe requires the full questionnaire")
    results = []
    device = model.get_input_embeddings().weight.device
    generation_config = model.generation_config
    eos_ids = generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (tuple, list)) else [eos_ids]) - {None}
    special_ids = set(tokenizer.all_special_ids)
    for index in PROBE_INDICES:
        item = items[index]
        prompt = prompt_text(tokenizer, item["text"])
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        # Preserve the official EOS/suppression policy. Sampling-only controls
        # are disabled explicitly to make these deterministic greedy probes.
        seq = model.generate(**enc, do_sample=False, max_new_tokens=32,
                             temperature=None, top_k=None, top_p=None)
        generated = seq[0, enc.input_ids.shape[1]:].tolist()
        response = tokenizer.decode(generated, skip_special_tokens=True)
        raw_response = tokenizer.decode(generated, skip_special_tokens=False)
        ended_on_eos = bool(generated and generated[-1] in eos_ids)
        body_ids = generated[:-1] if ended_on_eos else generated
        unexpected_specials = [token for token in body_ids if token in special_ids]
        results.append({
            "item_id": item["id"], "response": response, "raw_response": raw_response,
            "generated_token_ids": generated,
            "strict_single_letter": response.strip() in OPTIONS and not unexpected_specials,
            "generated_tokens": len(generated),
            "hit_length_limit": len(generated) == 32 and not ended_on_eos,
            "stopped_on_eos": ended_on_eos,
            "nonterminal_special_token_ids": unexpected_specials,
            "prompt_sha256": _sha256(prompt), "answer_prefix": ANSWER_PREFIX,
        })
    return results
