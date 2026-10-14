"""Pinned, disjoint WikiText train calibration / test diagnostic token blocks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def main():
    target = ROOT / 'data/pruning_corpus.npz'
    if target.exists():
        raise FileExistsError(target)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=DATA_REVISION)
    arrays, selections = {}, {}
    for split in ['train', 'test']:
        text = '\n\n'.join(dataset[split]['text'])
        tokens = np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
        blocks = tokens[:len(tokens)//256*256].reshape(-1, 256)
        if split == 'train':
            for seed in [0, 1, 2]:
                idx = np.random.default_rng(seed).choice(len(blocks), 64, replace=False)
                arrays[f'calibration_seed{seed}'] = blocks[idx]
                selections[f'calibration_seed{seed}'] = idx.tolist()
        else:
            arrays['test'] = blocks[:32]
        selections[split] = {'tokens': len(tokens), 'blocks': len(blocks),
                             'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
    np.savez_compressed(target, **arrays)
    metadata = {'dataset': 'Salesforce/wikitext', 'config': 'wikitext-2-raw-v1',
                'dataset_revision': DATA_REVISION, 'model': MODEL, 'model_revision': REVISION,
                'sequence_length': 256, 'calibration_blocks': 64, 'test_blocks': 32,
                'test_predicted_tokens': 32*255, 'selection': selections,
                'array_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
                'license': 'CC BY-SA 3.0; WikiText based on Wikipedia; token data is not committed'}
    (ROOT/'data/pruning_corpus_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k:v.shape for k,v in arrays.items()}))


if __name__ == '__main__':
    main()
