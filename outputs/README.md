# 实验结果版本

正式修正版入口：[`corrected_v2/`](corrected_v2/)。该目录使用模型原生末层 logits，
维度零分保留为未定，并将 logits 以 float32 保存。

本目录根下的五个 `qwen25-*` 目录、`summary_table.md`、`layer_curves.png`、
`per_item_choices.csv` 为 2026-10-13 的旧版结果，存在末层重复归一化问题，
仅保留供审计，不应用作修正后的实验结论。

修改前的代码、报告和产物另有完整备份：
`../archive/pre_fix_20261014_yoD34B/`。
