# Third-party attribution

`src/static_pruning.py` contains a Qwen2/Linear-only adaptation of SparseGPT's
second-order pruning update from [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt),
commit `147d2159dc4f3e9f73e47b32c04d7b3708f44436`, licensed under Apache 2.0.
The license is included in `licenses/SparseGPT-APACHE-2.0.txt`.
Changes include exact-count deterministic masks, removal of quantization paths,
new metadata and checks, and integration with Transformers 5 Qwen2 blocks.
`src/gemma_pruning.py` reuses those same mask/weight-update routines and adds
native Gemma 4 mixed-attention decoder replay, K=V handling, and text-only
target selection; it does not redistribute model weights.
Authors: Elias Frantar and Dan Alistarh; [paper](https://arxiv.org/abs/2301.00774).

Wanda's activation-weighted criterion and block-sequential calibration follow
[locuslab/wanda](https://github.com/locuslab/wanda), commit
`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`, MIT, copyright (c) 2023 CMU Locus Lab.
Its license is included in `licenses/Wanda-MIT.txt`.
[Paper](https://arxiv.org/abs/2306.11695).

WikiText-2 raw is obtained from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
and is based on Wikipedia articles under CC BY-SA 3.0. Raw corpus/token blocks
are not committed. The preparation script, pinned revision, sampled block
indices and content hashes make the local calibration/evaluation reproducible.

The MBTI questionnaire and original paper remain the works of their respective
authors. Original paper: [When Personality Meets Quantization](https://arxiv.org/html/2608.25977v2).
No repository-wide redistribution license for these third-party materials is
implied. The original paper full text, model weights and caches are excluded.

The Gemma extension uses [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it),
revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, under its Apache 2.0 model
license. The pinned tokenizer/chat template and Transformers implementation
are loaded from the user's local cache; no downloaded model/tokenizer assets
are committed. Gemma's required BOS format follows Table 11 of its
[technical report](https://arxiv.org/html/2607.02770v1).
