# Gemma 4-12B 静态剪枝实验（v3 扩展）

状态：2026-10-16 05:45 UTC 在首个剪枝组完成前，根据 dense 输入格式对照统一增加第二项 PPL；正式协议升级为 `3.3-gemma-dual-ppl-native-answer-logits`。模型适配和 BOS 剪枝重放 smoke 均已通过。进度见 `python scripts/check_gemma_status.py`，后台会话为 `gemma_v3_sweep`，日志 `logs/gemma4_12b_v3/`。

## 固定设计

- 模型：`google/gemma-4-12B-it`，revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`。
- 使用原生 BF16 权重、SDPA；仅文本输入，不评测视觉或音频能力。
- 实验配置与已完成 Qwen 实验相同：dense 1 组；magnitude、row_magnitude、Wanda、SparseGPT 各 10–70% 七档共 28 组；random 四档三种子共 12 组；三方法 drop1/2/3-of-4 共 9 组；magnitude/Wanda 50% 的 attention/MLP/early/middle/late 消融共 10 组；Wanda/SparseGPT 50% 校准种子 1/2 共 4 组。共 64 组，不按结果筛选。
- 每组从原始权重独立恢复；只剪 48 个语言 decoder 层中的 attention/MLP 线性权重，排除输入/输出 embedding、norm、bias 与模态投影。报告真实目标权重与全模型参数预算。
- 48 层等分为 early 0–15、middle 16–31、late 32–47；不得复用 Qwen 的层索引或 29 个 hidden-state 假设。
- Gemma 的 global attention 使用 K=V，部分层不存在独立 v_proj；按实际模块清单剪枝，不补造权重。该 checkpoint 的 `num_kv_shared_layers=0`。
- Wanda/SparseGPT 校准遵循 transformer-block 顺序，复用已有精确掩码与误差补偿算法；适配 Gemma 的真实层输入、混合注意力 mask 与 RoPE。

## 数据与测评

- 仍用同一份 60 题、提示词正文和人工方向标注，49 题计入四维，零分记 `?`。
- 使用官方聊天模板 `enable_thinking=False`；模板已经提供空 thought 段，紧接其结束标记评估 A–G。不得在 thought 开始位置评分，也不额外添加诱导答案。
- 最后一层必须使用原生 logits；中间层 final norm 只应用一次，并遵循原生 soft-capping。逐题独立检查投影与原生输出一致。
- 每组保存全 60 题答案、动态层数的选项 logits、合法首 token、全词表选项质量、6 道固定题的 32-token 贪心生成。
- WikiText-2 raw revision `b08601e04326c79dfdd32d625aee71d232d685c3`。使用 Gemma 自己的 tokenizer 重建语料：train 64×256 tokens，抽样种子 0/1/2；test 前 32×256 tokens，8,160 个预测目标。
- 遵循 [Gemma 4 官方报告 Table 11](https://arxiv.org/html/2607.02770v1)，每个独立校准/测试块显式前置 BOS（ID 2）。保存的原始数组不变，实际输入 257 tokens；PPL 仅对原始 tokens[1:] 评分，对齐 logits[1:-1]，仍为每块 255 个目标。聊天模板自身已有 BOS，不重复添加。
- 所有 64 组同时保留两项 PPL：上述 raw+BOS 指标，以及固定原生 non-thinking 回答前缀下的条件 PPL。后者的固定 user 文本为 `Continue the following encyclopedia passage. Output only the passage.`，随后拼接同一份原始 token 块，仍只评分原来的每块 255 个目标。保存完整 prefix token IDs、文本与哈希；不加入测试答案作为上下文，不改变校准数据。该指标仅是条件续写诊断，不是标准聊天能力基准。
- 语料不复用 Qwen token IDs；原始文本版本相同，但 token 切分和块边界不同。仅比较各模型相对自己 dense 基线的 PPL/NLL 变化，不以绝对 PPL 跨 tokenizer 排名。
- Gemma BF16 与历史 Qwen FP16 的数值格式不同；跨模型结果并非仅改变架构的严格控制实验。

## 验证与保存

1. 小型模型单测验证动态层数、混合 attention、K=V、soft-cap、原生 logits 和受保护参数。
2. 实际权重验证逐块重放与原生前向一致，再跑独立 smoke；smoke 不计入 64 组。
3. 正式结果逐组保存到 `outputs/gemma4_12b_v3/`，不得覆盖 `outputs/pruning_v3/` 的 Qwen 结果。
4. 记录 checkpoint、tokenizer、源码、数据哈希和软件版本；不提交权重、token 数据或密钥。
5. 完成后报告全矩阵、逐题变化、相对原始文本 NLL/PPL 变化及失效案例；不把 MBTI 类型稳定直接当作模型能力或真实人格稳定。

## 预检异常与协议修正（在正式矩阵前确定）

原生模型在前两个原始测试块（510 个相同评分目标）上的 SDPA PPL：无 BOS 为 346,512.24；显式 BOS 后为 1,555.57。原生 labels loss 与手动交叉熵一致；eager 注意力对照也呈现同方向问题。官方 tokenizer 的默认调用和 `add_special_tokens=True` 都不会自动添加 BOS，因此不能仅依赖该参数。

原生聊天同时能正确回答巴黎、2+2，并生成正常猫的句子；模型逐块重放与原生 logits 检查误差为 0。BOS 遗漏是已确认的协议错误，但不能解释修正后全部高 PPL。指令微调/原始文本分布不匹配是候选解释，未被因果验证。正式评估固定采用 BOS + raw WikiText，不通过挑选提示词来压低指标；PPL 仅作为该输入条件下的诊断，不能单独代表通用语言能力。

无 BOS 的四组预检、修正后 BOS smoke 及协议 3.2 的首轮尝试整体保存在本地 `outputs/gemma4_12b_protocol32_preflight/`，不进入正式结果、不覆盖、不计入 64 组。预检脚本 `scripts/diagnose_gemma_baseline.py` 可复现原生 BOS/注意力/聊天对照。

额外输入格式对照为 dense 预检后的诊断：保持同一份 32×255 个预测目标，比较无 BOS、显式 BOS、原生 non-thinking assistant prefix，PPL 分别为 237,517.79、1,552.62、39.49；32/32 个块均按此方向改善。这直接显示输入条件的影响，但不独立证明是训练数据或架构造成。对照保存在 `outputs/gemma4_12b_v3/input_format_controls.json`。

2026-10-16 05:45 UTC，首轮 64 组尚无任何剪枝组完成时，停止本次启动，完整归档已有 dense 及预检数据，决定给全部 64 组统一增加 prefix 条件指标；raw 指标保留，不以较好数值替代它。此修改晚于 dense 诊断、早于剪枝效果分析，因此不是事先注册的双指标方案。剪枝矩阵、BOS 校准、MBTI 与目标文本均不变。源码提交 `1a398fc` 保留了修改前协议，正式源码另行提交并记录逐文件哈希。

模型信息依据 [官方模型卡](https://huggingface.co/google/gemma-4-12B-it)；许可证 Apache 2.0。
