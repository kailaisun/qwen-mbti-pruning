# v3 实验结果

Qwen 结果保存在 `pruning_v3/`，64/64 组实验已完成。

- [完整结果表](pruning_v3/RESULTS.md)
- [结构、位置和种子对照](pruning_v3/ABLATIONS.md)
- [结论与机制分析](../PRUNING_REPORT.md)
- [逐配置指标](pruning_v3/comparison.csv)
- [逐题答案变化](pruning_v3/item_deltas.csv)

Gemma 4-12B 扩展独立保存在 `gemma4_12b_v3/`，进度与固定设计见
[Gemma 实验计划](../GEMMA_EXPERIMENT_PLAN.md)。预检 smoke 不属于正式矩阵，不纳入 Git。

每个配置目录保留原始 `summary.json` 和 `results.npz`，未改写测评输出或生成时的溯源哈希。
其他本地历史结果、模型权重和备份不在当前 GitHub 文件树中。
