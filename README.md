# Qwen / Gemma 静态剪枝与 MBTI 评测（v3）

论文方法的跨模型扩展实验：[arXiv:2608.25977v2](https://arxiv.org/html/2608.25977v2)。
使用 Qwen2.5-7B-Instruct，并增加 Gemma 4-12B-it；不是原论文模型配置或表格数值的完全复现。
仓库当前版本只收录 v3，包含无条件 60 题评测、逐层选项分布和多方法静态剪枝对照。

## Qwen 多方法剪枝（v3，已完成）

预先确定的 **64 组**实验见 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)：
magnitude、row-magnitude、随机剪枝、Wanda、SparseGPT，多比例 / 4 元组约束 / 模块与层段消融。
增加独立 WikiText test 困惑度诊断、逐题贡献、自由生成和格式合法性分析。
结论和机制分析见 [`PRUNING_REPORT.md`](PRUNING_REPORT.md)，
**64/64 组已完成，15 项测试通过**。完整结果见
[`RESULTS.md`](outputs/pruning_v3/RESULTS.md)，结构/位置/种子比较见
[`ABLATIONS.md`](outputs/pruning_v3/ABLATIONS.md)。结果保存在 `outputs/pruning_v3/`。

主要发现：50% 时 Wanda / SparseGPT 的 test 子集 PPL 约为 17.1，幅值剪枝约为 299–312。
但 ENFJ 标签不代表回答稳定，更不代表模型仍在有效作答；详见报告中的逐题贡献、
常量答案控制及固定掩码模块交互分析。
PPL 使用 WikiText-2 test 的 32 个 256-token 块，共 8,160 个预测目标，
是固定短上下文子集诊断，不是完整标准基准。

## Gemma 4-12B（v3 扩展）

设计见 [`GEMMA_EXPERIMENT_PLAN.md`](GEMMA_EXPERIMENT_PLAN.md)。同样固定 64 组，
独立保存到 `outputs/gemma4_12b_v3/`，不改写 Qwen 原始结果。
目前已完成模型适配与 BOS 修正，正式实验采用统一的 raw/assistant-prefix 双指标。
Gemma 使用原生 BF16、48 层、混合注意力和 K=V；只剪语言 decoder 的 attention/MLP 权重。
原文校准及 PPL 输入显式加 BOS，评分目标仍为每段 255 个、共 8,160 个。
Gemma 原文 PPL 的较高基线与正常聊天回答并存，不能作为跨模型能力排名；
所有 Gemma 配置额外计算固定原生回答前缀下的条件 PPL，仍使用相同预测目标。
两项指标都保留，不择优报告；跨模型仅列出各自原文指标相对 dense 的变化，
并结合逐题答案、选项概率和自由生成判断。输入格式诊断与方案修改时间见实验计划。

```bash
# 使用 requirements-pruning.txt 对应环境；固定 revision，不能下载浮动 main 代替。
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-12B-it', revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7')"
# 首次准备；已有语料时会拒绝覆盖。脚本从本地缓存加载固定 tokenizer。
python scripts/prepare_gemma_data.py
# 先做单独 smoke，不混入正式矩阵：
CUDA_VISIBLE_DEVICES=0 python -B src/run_gemma_sweep.py --smoke \
  --outdir outputs/gemma_smoke_repeat --tags dense-s00 wanda-s50 sparsegpt-s50
# 一张卡顺序运行全部 64 组，或指定 --tags 子集：
CUDA_VISIBLE_DEVICES=0 python -B src/run_gemma_sweep.py --outdir outputs/gemma_repeat
# 四张有充足空闲显存的卡、每卡两个进程（本轮为 H200；需自行检查资源）：
GEMMA_GPUS="0 3 4 7" PYTHON=python OUTDIR=outputs/gemma_repeat \
LOGDIR=logs/gemma_repeat bash scripts/run_gemma_parallel.sh
python src/analyze_gemma_sweep.py --outdir outputs/gemma_repeat
python scripts/check_gemma_status.py --outdir outputs/gemma_repeat --logdir logs/gemma_repeat
```

每组保存 60×49×7 选项 logits、原生校验、提示词/语料/源码/掩码哈希和运行版本。
逐块重放会检查混合注意力与 KV 处理；未使用视觉或音频输入。
未在 Git 中保存稀疏化模型权重，需使用源码、固定语料和配置重建。

## Qwen 运行

```bash
# 环境版本见 requirements-pruning.txt；先缓存固定 revision 的 Qwen 模型。
# 首次下载模型：bash scripts/download_models.sh（仅下载 dense 模型）。
# 仅首次准备校准数据（已有数据时不覆盖）：
python scripts/prepare_pruning_data.py
# 重跑使用新的输出目录，不覆盖已发布结果。
CUDA_VISIBLE_DEVICES=2 python -B src/run_pruning_sweep.py --outdir outputs/pruning_repeat
# 支持 --tags wanda-s50 sparsegpt-s50 等子集；同一代码版本下匹配结果自动复用。
# 或使用双进程（同卡建议至少约 50 GiB 空闲显存，也可配置为两张卡）：
GPU_SPARSE=2 GPU_OTHER=2 PYTHON=python OUTDIR=outputs/pruning_repeat \
LOGDIR=logs/pruning_repeat bash scripts/run_pruning_parallel.sh
# 只在全部 64 组齐全并通过校验后生成最终汇总：
python src/analyze_pruning_sweep.py
# 若重跑到新目录，则使用 --outdir outputs/pruning_repeat 分析新结果。
# 不占 GPU 的进度查询：
python scripts/check_pruning_status.py
# CPU 单元测试，不需要下载模型：
python -B -m unittest discover -s tests -v
```

源码整理仅将共享目标层选择函数移入剪枝模块，未改变剪枝算法或评测协议。
已发布的 64 组原始结果和生成时的源码哈希保持不变；对应源码可从 Git 历史恢复。
当前代码的哈希已变化，重跑必须使用新输出目录，不会覆盖或直接复用已发布结果。
同一代码版本下的兼容结果仍可断点复用。

GitHub 当前文件树不包含旧实验、模型权重、缓存、日志、原论文全文或密钥。
参考实现及许可证见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 文件

- `data/mbti_items.json`：60 题、维度、方向。
- `data/pruning_corpus_manifest.json`：校准/测试数据版本、采样和哈希。
- `src/mbti_common.py`：提示词、逐层 logits、计分和指标。
- `src/static_pruning.py`：目标层选择、静态掩码、Wanda 与 SparseGPT。
- `src/run_pruning_sweep.py`：64 组矩阵、独立权重恢复、评测和溯源记录。
- `src/analyze_pruning_sweep.py`：结果校验、逐题贡献、汇总与图表。
- `outputs/pruning_v3/<tag>/`：每组原始 `summary.json`、`results.npz`。
- `tests/`：计分、掩码、校准、实验设计与分析测试。

条件化提示、UALD、量化未纳入本轮范围。未启用稀疏推理内核，
不将置零比例当作实际压缩或加速收益。
