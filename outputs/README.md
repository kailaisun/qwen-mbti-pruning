# 实验结果版本

最新多方法剪枝实验：[`pruning_v3/RESULTS.md`](pruning_v3/RESULTS.md)，64/64 组完成。
机制与结论分析：[`PRUNING_REPORT.md`](../PRUNING_REPORT.md)。

上一轮量化与两种剪枝修正版：[`corrected_v2/`](corrected_v2/)。两版都使用模型原生末层 logits，
维度零分保留为未定，并将 logits 以 float32 保存；v3 增加了 PPL、种子、结构与位置对照。

本目录根下的五个 `qwen25-*` 目录、`summary_table.md`、`layer_curves.png`、
`per_item_choices.csv` 为 2026-10-13 的旧版结果，存在末层重复归一化问题，
仅保留供审计，不应用作修正后的实验结论。

修改前的代码、报告和产物另有完整备份：
`../archive/pre_fix_20261014_yoD34B/`。

这些旧版目录及备份仅保留在本机，未提交 GitHub。
