# Qwen2.5-7B 静态剪枝扩展实验（v3）

状态：64/64 组已完成并通过校验（2026-10-14）。最终分析见 `PRUNING_REPORT.md`。
仓库当前版本只收录 v3 的代码、报告与实验结果。

## 预先确定的实验矩阵

- 固定 Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`，FP16。
- dense 基线 1 组。
- magnitude（逐矩阵）、row-magnitude（逐输出行）、Wanda、SparseGPT：各 10/20/30/40/50/60/70% 稀疏率，28 组。
- 随机逐矩阵剪枝：10/30/50/70%，种子 0/1/2，12 组。
- magnitude / Wanda / SparseGPT 的 4 元组约束：每 4 个权重删除 1/2/3 个，9 组。注意这里明确说的是**删除数量**，避免 N:M 保留/删除记法混淆。
- magnitude / Wanda 在 50% 局部稀疏率下，只剪 attention、只剪 MLP、只剪 early/middle/late 层，各 5 组，共 10 组。这些是定位消融，**总删除量不同，不作等预算排名**。
- Wanda / SparseGPT 的 50% 结果追加校准采样种子 1/2；主实验使用种子 0，共 4 组。
- 合计 **64 组**，不是从结果里挑选有利配置。

## 固定协议

所有运行从未剪枝原始权重独立开始。除 SparseGPT 的单次误差补偿外，不训练、不微调；所有掩码在推理前固定，故仍是静态剪枝。

剪枝目标为 28 层的 196 个 attention/MLP Linear 权重；不剪 lm_head、embedding、norm、bias。
Wanda 为逐行 `|W_ij| * ||X_j||_2`，逐 transformer block 顺序校准和剪枝。
SparseGPT 为二阶逆 Hessian 近似、blocksize=128、damping=1%；逐 transformer block 顺序处理。
SparseGPT 用精确个数掩码替代上游阈值的边界多剪；实际 masked/zero 比例逐模块记录。

校准数据：固定 revision 的 WikiText-2 raw **train**，64 个不重叠 256-token 块（16,384 tokens），采样种子 0/1/2。
语言能力诊断：同数据集 **test** 前 32 个 256-token 块，共 8,160 个预测目标；不是完整 WikiText-2 标准 PPL，也不是综合能力基准。
校准不使用 MBTI 题目、答案或测试集。

每组运行全部 60 个 MBTI 条目；49 个参与四维分数，11 个 A/T 条目不参与类型。
使用 native-final-logits 评估，不重复 final norm；零分记 `?`。
额外保存层级概率、逐题答案、合法选项总概率质量、非受限首 token、6 道固定题的自由生成。

## 结论依据

1. 与 dense 的逐题一致率、维度分数、逐题分数贡献，避免只看四字母类型。
2. 逐层 JSD / 层级选项变化定位偏移；这是 logit-lens 诊断，不等同于因果解释。
3. 校准误差、独立 test PPL、首 token 合法率 / 选项概率质量、自由生成区分格式失效和测评偏好变化。
4. attention / MLP / 分段层消融是受控干预，但删除预算不同，解释须保留这个限制。
5. 随机种子和校准种子重复量化部分敏感性；不把 60 道固定题当作独立人格样本，不给虚假人口学置信区间。
6. 稠密张量置零不自动压缩文件或加速，不宣称部署收益。

## 参考和差异

- [原 MBTI 论文](https://arxiv.org/html/2608.25977v2)：原文 Qwen 为 14B/72B，本项目为 7B 扩展，不是原表格逐项复现。
- [Wanda 官方代码](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6)。本项目校准集/长度不同，不宣称复现其论文绝对 PPL。
- [SparseGPT 官方代码](https://github.com/IST-DASLab/sparsegpt/tree/147d2159dc4f3e9f73e47b32c04d7b3708f44436)。针对 Qwen2/Transformers 5 接口适配。
- [WikiText 数据集](https://huggingface.co/datasets/Salesforce/wikitext)，revision `b08601e04326c79dfdd32d625aee71d232d685c3`。

GitHub 仓库为 `kailaisun/qwen-mbti-pruning`（私有）。只提交代码、文档、轻量结果和图表；不提交凭据、模型权重、缓存、原论文全文、运行日志或旧归档。

## 运行期间追加的分析（明确为事后诊断）

观察到高稀疏率回答趋同后，增加 7 种常量答案 A–G 的解析对照，不新增或筛选模型实验。
同时输出选项概率加权分数（`soft_*`），以及正/负向题分别取均值再相减的
计分敏感性分析（`balanced_*`）。这不是原论文指标，也不是经过验证的新人格量表；
全部主表仍保留既定求和计分。追加分析不改变 64 组模型实验及其结果。

对预设的 attention / MLP / all 幅值剪枝消融，追加固定掩码 2×2 交互分析：
先验证两个局部掩码互不相交、并集与 all 掩码逐模块 SHA-256 完全相同，
再在相同 test 上计算 `L(all) - L(attention) - L(MLP) + L(dense)`。
这项分析使用已有四组实验，不新增模型运行；Wanda 因校准输入变化可能导致不同掩码，
不直接套用该固定掩码解释。
