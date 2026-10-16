"""Post-hoc input-format controls; never replace the frozen pruning metric.

All three conditions score exactly the same original WikiText targets. The
assistant-prefix condition changes conditioning and therefore is a diagnostic
intervention, not a candidate baseline selected for favorable perplexity.
"""
from __future__ import annotations

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
from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
FORMAL_PROTOCOL = "3.2-gemma-bos-native-answer-logits"
CONTROL_PROTOCOL = "posthoc-gemma-input-format-controls-1.0"
CONTINUATION_INSTRUCTION = "Continue the following encyclopedia passage. Output only the passage."
CHAT_QUESTIONS = (
    "What is the capital of France? Answer in one short sentence.",
    "What is 2 + 2? Answer only the number.",
    "Write one short sentence about a cat.",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


@torch.inference_mode()
def evaluate_prefix(model, raw_blocks, prefix_ids, device="cuda"):
    """Score raw[1:] using logits[prefix_len:-1] in every condition."""
    if len(raw_blocks.shape) != 2 or raw_blocks.shape[0] == 0 or raw_blocks.shape[1] < 2:
        raise ValueError("expected nonempty blocks with at least two raw tokens")
    losses = []
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    prefix_len = prefix.shape[1]
    for index, block in enumerate(raw_blocks):
        raw = torch.as_tensor(block, dtype=torch.long, device=device).unsqueeze(0)
        inputs = torch.cat((prefix, raw), dim=1)
        output = model(inputs, use_cache=False)
        if output.logits.shape[:2] != inputs.shape:
            raise ValueError("native logits must cover the complete input")
        scored_logits = output.logits[:, prefix_len:-1].float()
        targets = raw[:, 1:]
        if scored_logits.shape[:2] != targets.shape:
            raise AssertionError("prefix changed the scored target alignment")
        loss = torch.nn.functional.cross_entropy(
            scored_logits.reshape(-1, scored_logits.shape[-1]), targets.reshape(-1), reduction="mean"
        )
        losses.append(float(loss))
        print("[block]", json.dumps({"index": index, "prefix_length": prefix_len, "nll": losses[-1]}), flush=True)
    mean_nll = sum(losses) / len(losses)
    if not math.isfinite(mean_nll):
        raise ValueError("non-finite input-format control NLL")
    return {
        "block_nll": losses,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "n_blocks": len(losses),
        "raw_sequence_length": int(raw_blocks.shape[1]),
        "prefix_length": prefix_len,
        "input_sequence_length": int(raw_blocks.shape[1]) + prefix_len,
        "scored_targets_per_sequence": int(raw_blocks.shape[1]) - 1,
        "predicted_tokens": len(losses) * (int(raw_blocks.shape[1]) - 1),
        "scoring_alignment": "logits[:, prefix_length:-1] against raw_tokens[:, 1:]",
    }


@torch.inference_mode()
def main():
    destination = ROOT / "outputs/gemma4_12b_v3/input_format_controls.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    started = time.time()
    print("[pid]", os.getpid(), "[gpu]", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    corpus_path = ROOT / "data/gemma4_12b_corpus.npz"
    manifest_path = ROOT / "data/gemma4_12b_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(corpus_path) as corpus:
        blocks = corpus["test"].copy()
    raw_hash = sha256_bytes(blocks.tobytes())
    if raw_hash != manifest["array_sha256"]["test"] or blocks.shape != (32, 256):
        raise ValueError("raw test corpus does not match the pinned 32x256 manifest")
    assistant_prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": CONTINUATION_INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not assistant_prefix.endswith("<|turn>model\n<|channel>thought\n<channel|>"):
        raise ValueError("unexpected native nonthinking assistant prefix")
    prefixes = (
        ("no_BOS", [], "", False),
        ("explicit_BOS", [tokenizer.bos_token_id], tokenizer.bos_token, True),
        ("native_nonthinking_assistant_prefix",
         tokenizer.encode(assistant_prefix, add_special_tokens=False), assistant_prefix, False),
    )
    conditions = {}
    for name, ids, text, formal in prefixes:
        print("[start]", name, flush=True)
        metrics = evaluate_prefix(model, blocks, ids)
        metrics.update({
            "prefix_token_ids": ids, "prefix_text": text,
            "prefix_text_sha256": sha256_bytes(text.encode("utf-8")),
            "is_frozen_formal_condition": formal,
            "scored_targets_sha256": sha256_bytes(np.ascontiguousarray(blocks[:, 1:]).tobytes()),
        })
        conditions[name] = metrics
        print("[complete]", name, json.dumps({"nll": metrics["mean_nll"], "ppl": metrics["perplexity"]}), flush=True)
    probes = []
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    for question in CHAT_QUESTIONS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        generated = model.generate(**encoded, do_sample=False, max_new_tokens=48,
                                   temperature=None, top_k=None, top_p=None)
        response_ids = generated[0, encoded.input_ids.shape[1]:].tolist()
        ended_on_eos = bool(response_ids and response_ids[-1] in eos_ids)
        probe = {
            "question": question, "prompt_text": prompt,
            "prompt_token_ids": encoded.input_ids[0].tolist(),
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "raw_response": tokenizer.decode(response_ids, skip_special_tokens=False),
            "response": tokenizer.decode(response_ids, skip_special_tokens=True),
            "generated_token_ids": response_ids, "generated_tokens": len(response_ids),
            "hit_length_limit": len(response_ids) == 48 and not ended_on_eos,
            "stopped_on_eos": ended_on_eos,
        }
        probes.append(probe)
        print("[chat]", json.dumps({"question": question, "raw_response": probe["raw_response"]}), flush=True)
    result = {
        "control_protocol": CONTROL_PROTOCOL, "formal_protocol": FORMAL_PROTOCOL,
        "post_hoc": True, "included_in_formal_64_treatment_matrix": False,
        "interpretation": (
            "Input-distribution control on a pristine model, scoring identical original targets. "
            "The assistant-prefix condition changes conditioning; it must not replace the frozen "
            "explicit-BOS raw-continuation metric or calibration corpus. The no-BOS condition "
            "deliberately omits the documented sequence prefix. These are diagnostics, not "
            "independent quality benchmarks, and three chat probes are not an overall capability test."
        ),
        "model": MODEL, "model_revision": REVISION, "dtype": "bfloat16",
        "attention_implementation": "sdpa", "enable_thinking": False,
        "continuation_instruction": CONTINUATION_INSTRUCTION,
        "conditions": conditions, "ordinary_chat_probes": probes,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256_bytes(Path(__file__).read_bytes())},
            "corpus_file_sha256": sha256_bytes(corpus_path.read_bytes()),
            "corpus_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "raw_test_array_sha256": raw_hash, "corpus": manifest,
            "chat_template_sha256": sha256_bytes(tokenizer.get_chat_template().encode("utf-8")),
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf32": False,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "numpy", "safetensors")},
        },
        "elapsed_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print("GEMMA_INPUT_FORMAT_CONTROLS_COMPLETE", str(destination), flush=True)


if __name__ == "__main__":
    main()
