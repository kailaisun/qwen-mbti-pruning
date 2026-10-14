# Qwen2.5-7B MBTI 评测、量化与静态剪枝

论文方法的单模型复现实验：[arXiv:2608.25977v2](https://arxiv.org/html/2608.25977v2)。
使用 Qwen2.5-7B-Instruct；不是论文中的 Qwen2.5-14B/72B 原表数值复现。
包含无条件 60 题评测、逐层选项分布，以及量化/静态剪枝的对照。

## 多方法剪枝扩展（v3，已完成）

预先确定的 **64 组**实验见 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)：
magnitude、row-magnitude、随机剪枝、Wanda、SparseGPT，多比例 / 4 元组约束 / 模块与层段消融。
增加独立 WikiText test 困惑度诊断、逐题贡献、自由生成和格式合法性分析。
v2 的现有结论见 [`REPORT.md`](REPORT.md)；v3 分析见 [`PRUNING_REPORT.md`](PRUNING_REPORT.md)，
**64/64 组已完成，15 项测试通过**。完整结果见
[`RESULTS.md`](outputs/pruning_v3/RESULTS.md)，结构/位置/种子比较见
[`ABLATIONS.md`](outputs/pruning_v3/ABLATIONS.md)。结果保存在 `outputs/pruning_v3/`。

主要发现：50% 时 Wanda / SparseGPT 的 test 子集 PPL 约为 17.1，幅值剪枝约为 299–312。
但 ENFJ 标签不代表回答稳定，更不代表模型仍在有效作答；详见报告中的逐题贡献、
常量答案控制及固定掩码模块交互分析。

```bash
# 环境版本见 requirements-pruning.txt；先缓存固定 revision 的 Qwen 模型。
# 仅首次准备校准数据（已有数据时不覆盖）：
python scripts/prepare_pruning_data.py
# 真正重跑请使用新的输出目录；默认目录会复用已完成结果。
CUDA_VISIBLE_DEVICES=2 python -B src/run_pruning_sweep.py --outdir outputs/pruning_repeat
# 支持 --tags wanda-s50 sparsegpt-s50 等子集；已有匹配结果自动复用。
# 或使用双进程（同卡建议至少约 50 GiB 空闲显存，也可配置为两张卡）：
GPU_SPARSE=2 GPU_OTHER=2 PYTHON=python OUTDIR=outputs/pruning_repeat \
LOGDIR=logs/pruning_repeat bash scripts/run_pruning_parallel.sh
# 只在全部 64 组齐全并通过校验后生成最终汇总：
python src/analyze_pruning_sweep.py
# 若重跑到新目录，则使用 --outdir outputs/pruning_repeat 分析新结果。
# 不占 GPU 的进度查询：
python scripts/check_pruning_status.py
```

GitHub 不包含模型权重、缓存、日志、原论文全文、旧归档或密钥。
参考实现及许可证见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 上一轮 v2 结果与本地存档

- 修正版结果：`outputs/corrected_v2/`。
- 实验报告：[`REPORT.md`](REPORT.md)。
- 剪枝权重：`checkpoints/corrected_v2/qwen25-7b-prune-u50/` 和 `qwen25-7b-prune-2to4/`，含 tokenizer、config 和剪枝清单；这两个目录链接到本机持久盘 `/home/kailais/mbti_checkpoints/corrected_v2/`。
- 修改前完整备份：`archive/pre_fix_20261014_yoD34B/`。
- `outputs/` 根目录的旧结果含末层重复归一化错误，保留作审计，不作正式结论。

## v2 修正与边界（v3 校准方法另见新报告）

1. 最后一层使用模型原生 `out.logits`，不对已归一化状态再次归一化；每题检查输出头投影与原生 logits 一致。
2. 中间层只投影选项对应的输出头行，避免逐层计算完整词表；模型仅生成最后位置的 logits。
3. 维度零分标为 `?`，不默认选择 E/N/F/J。60 题中 49 题计入四维类型，其余 11 题 AT 不参与四维计分。
4. 非结构化剪枝逐矩阵精确选择 50% 掩码，边界同幅值按展平索引裁决；2:4 每行连续四元素剪两个。
5. 剪枝范围是 transformer 的线性层，排除 lm_head、embedding、norm；不训练、不校准。稀疏率针对被剪线性权重，非全部参数。
6. 权重仍以稠密 FP16 safetensors 保存。50% 置零不等于文件/显存减半，也未启用稀疏加速；4-bit 量化与 50% 稀疏不是等压缩率对照。
7. 结果以 float32 logits 保存；记录模型 revision、题库和源码 SHA-256、软件版本、GPU、逐题诊断。每份剪枝 checkpoint 保存后重新加载并复测全部 60 题。
8. 题目方向为人工标注；中间层 logit lens 是分析约定。选项内熵不是语言能力或真实人格指标。

## 运行上一轮 v2

环境：`/home/kailais/venvs/mbti/bin/python`。运行前使用 `nvidia-smi` 选有空闲显存的卡。
现有结果目录禁止覆盖；一键脚本会复用协议、源码哈希匹配的已完成结果。
要重新计算全部实验，请给新的结果、日志和 checkpoint 目录。

```bash
cd /NFS/T5/kailais/SWM/MBTI

# 回归测试：无需下载模型，使用随机初始化的小型 Qwen2。
/home/kailais/venvs/mbti/bin/python -B -m unittest discover -s tests -v

# 三路进程并发，dense 分支依次跑 FP16 和两种剪枝。
GPU_DENSE=7 GPU_AWQ=1 GPU_GPTQ=3 \
OUTDIR=outputs/repeat_01 LOGDIR=logs/repeat_01 \
CHECKPOINT_DIR=/home/kailais/mbti_checkpoints/repeat_01 bash scripts/run_all_qwen.sh

# 重建当前修正版图表、CSV，验证保存结果。
/home/kailais/venvs/mbti/bin/python -B src/analyze_results.py \
  --outdir outputs/corrected_v2 --legacy-outdir outputs

# 直接使用已保存的剪枝模型评测；无需再次剪枝。
CUDA_VISIBLE_DEVICES=7 /home/kailais/venvs/mbti/bin/python -B src/run_experiment.py \
  --model checkpoints/corrected_v2/qwen25-7b-prune-u50 \
  --local-files-only --tag reloaded-u50 --outdir outputs/reload_example
```

`run_all_qwen.sh` 固定三份官方 checkpoint revision，并使用本地缓存。
模型尚未缓存时，先执行 `scripts/download_models.sh`，其中三个默认模型也固定到相同 revision。
脚本遇到失败会返回非零状态；各变体使用独立日志。

## 文件

- `data/mbti_items.json`：60 题、维度、方向。
- `src/mbti_common.py`：提示词、逐层 logits、计分和指标。
- `src/run_experiment.py`：加载、剪枝、评测、checkpoint 保存/重载验证、溯源记录。
- `src/analyze_results.py`：结果校验、汇总表、CSV、PNG/SVG 曲线和旧新比较。
- `tests/test_protocol.py`：核心协议回归测试。
- `docs/`：论文 HTML/TXT 本地存档。
- `outputs/corrected_v2/<tag>/`：`summary.json`、`results.npz`。
- `outputs/corrected_v2/environment.freeze.txt`：本次软件版本快照。
- `logs/corrected_v2/`：各变体日志；`logs/corrected_v2_driver.log` 为调度日志。

条件化提示、UALD、2-bit 未纳入完成复现范围。v2 没有 PPL，v3 的 PPL 是固定短上下文子集诊断，而非完整标准基准。
