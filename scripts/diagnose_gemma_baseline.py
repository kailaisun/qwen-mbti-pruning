"""Native-model baseline checks before accepting any Gemma pruning result."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_gemma_sweep import MODEL, REVISION


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    corpus = np.load(ROOT / 'data/gemma4_12b_corpus.npz')['test'][:2]
    print('tokenizer', json.dumps({'bos_token_id': tokenizer.bos_token_id,
                                 'default': tokenizer.encode('The capital of France is'),
                                 'no_special': tokenizer.encode('The capital of France is', add_special_tokens=False)}), flush=True)
    for implementation in ['sdpa', 'eager']:
        model.set_attn_implementation(implementation)
        for bos in [False, True]:
            losses = []
            for block in corpus:
                raw = torch.as_tensor(block, dtype=torch.long, device='cuda')[None]
                inputs = torch.cat([raw.new_tensor([[tokenizer.bos_token_id]]), raw], 1) if bos else raw
                out = model(inputs, labels=inputs, use_cache=False)
                shift = 1 if bos else 0
                # Always score the same 255 raw targets, excluding the first raw token.
                loss = torch.nn.functional.cross_entropy(out.logits[:, shift:-1].float().reshape(-1, out.logits.shape[-1]), raw[:, 1:].reshape(-1))
                losses.append(float(loss))
                positions = list(range(8))
                print('first_predictions', json.dumps({'implementation': implementation, 'bos': bos,
                    'targets': tokenizer.convert_ids_to_tokens(raw[0, 1:9].tolist()),
                    'predictions': tokenizer.convert_ids_to_tokens(out.logits[0, shift:shift+8].argmax(-1).tolist()),
                    'native_loss': float(out.loss), 'manual_matched_loss': float(loss)}, ensure_ascii=False), flush=True)
            print('ppl', json.dumps({'implementation': implementation, 'bos': bos,
                                    'block_nll': losses, 'ppl': math.exp(sum(losses) / len(losses))}), flush=True)
    model.set_attn_implementation('sdpa')
    for question in ['What is the capital of France? Answer in one short sentence.',
                     'What is 2 + 2? Answer only the number.',
                     'Write one short sentence about a cat.']:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
        generated = model.generate(**enc, do_sample=False, max_new_tokens=48,
                                   temperature=None, top_k=None, top_p=None)
        answer = tokenizer.decode(generated[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
        print('chat', json.dumps({'question': question, 'answer': answer}, ensure_ascii=False), flush=True)
    print('GEMMA_BASELINE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
