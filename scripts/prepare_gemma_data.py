"""Retokenize the pinned WikiText corpus with Gemma's pinned tokenizer.

This is the same text/selection policy as the Qwen experiment, not the same
token blocks: tokenizer boundaries differ. Absolute token PPL is therefore
not a cross-model ranking metric. No model weights or GPU are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SEQUENCE_LENGTH = 256


def prepare_arrays(dataset, tokenizer):
    """Pure selection policy shared across seeds; test is never calibration."""
    arrays, selections = {}, {}
    for split in ("train", "test"):
        raw_text = "\n\n".join(dataset[split]["text"])
        tokens = np.asarray(
            tokenizer.encode(raw_text, add_special_tokens=False), dtype=np.int32
        )
        blocks = tokens[: len(tokens) // SEQUENCE_LENGTH * SEQUENCE_LENGTH].reshape(
            -1, SEQUENCE_LENGTH
        )
        required = 64 if split == "train" else 32
        if len(blocks) < required:
            raise ValueError(f"{split}: {len(blocks)} blocks, need {required}")
        if split == "train":
            for seed in (0, 1, 2):
                indices = np.random.default_rng(seed).choice(len(blocks), 64, replace=False)
                key = f"calibration_seed{seed}"
                arrays[key] = blocks[indices]
                selections[key] = indices.tolist()
        else:
            arrays["test"] = blocks[:32]
            selections["test_block_indices"] = list(range(32))
        selections[split] = {
            "tokens": len(tokens),
            "blocks": len(blocks),
            "text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        }
    return arrays, selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/gemma4_12b_corpus.npz")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/gemma4_12b_corpus_manifest.json")
    args = parser.parse_args()
    for target in (args.output, args.manifest):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing corpus artifact: {target}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=DATA_REVISION)
    arrays, selections = prepare_arrays(dataset, tokenizer)
    qwen_manifest = ROOT / "data/pruning_corpus_manifest.json"
    if qwen_manifest.exists():
        qwen = json.loads(qwen_manifest.read_text())
        for split in ("train", "test"):
            if selections[split]["text_sha256"] != qwen["selection"][split]["text_sha256"]:
                raise ValueError(f"{split} raw-text hash differs from pinned Qwen corpus")
    metadata = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "dataset_revision": DATA_REVISION,
        "model": MODEL,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "sequence_length": SEQUENCE_LENGTH,
        "calibration_blocks": 64,
        "test_blocks": 32,
        "test_predicted_tokens": 32 * (SEQUENCE_LENGTH - 1),
        "selection": selections,
        "array_sha256": {
            key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()
        },
        "tokenization": "Join rows with two newlines; add_special_tokens=False; disjoint 256-token blocks",
        "comparison_caveat": (
            "Raw WikiText text and block-selection policy match Qwen, but tokenizer boundaries, "
            "calibration text spans and the first 32 test block spans differ. Compare pruning "
            "PPL relative to each model's own dense baseline, not absolute PPL across models. "
            "Even relative PPL is a within-protocol robustness diagnostic, not a universal ranking."
        ),
        "license": "CC BY-SA 3.0; WikiText based on Wikipedia; token data is not committed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive opens protect existing artifacts even if another process starts
    # after the existence check. An interrupted partial artifact requires review.
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with args.manifest.open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "manifest": str(args.manifest),
                      "arrays": {key: list(value.shape) for key, value in arrays.items()}}))


if __name__ == "__main__":
    main()
