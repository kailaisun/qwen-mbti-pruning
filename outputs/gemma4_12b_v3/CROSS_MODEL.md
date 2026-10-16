# Gemma / Qwen：相对自身基线的剪枝变化

当前有 64/64 个匹配配置。

PPL 倍数 = 剪枝模型 PPL / 同模型 dense PPL；回答改变率也分别相对各自 dense 回答。
Gemma 与 Qwen 的 tokenizer、token 边界、实际文本覆盖范围、模型结构与原生 chat 模板不同；因此不比较绝对 PPL，倍数也仅是本协议内的稳健性诊断，不是模型综合能力排名。
Gemma 原文校准与 PPL 输入为显式 BOS（ID 2）+ 256 原始 token；PPL 仍只评分每段原始位置 1–255，排除 BOS→首个原始 token。Qwen 原协议未改动。
本表仅比较原文 PPL：Gemma 另有 assistant 前缀条件下的辅助 PPL，但 Qwen 没有对应数据，故不混入跨模型表。该辅助条件不是公平的通用语言能力基准。
Gemma 在有效 chat 模板下能正常回答，也可能在无 chat 包装的原文续写上出现较高 PPL；不能据此将原文 PPL 或其倍数当作通用能力排名，PPL 下降也不自动意味着人格或对话能力改善。
Gemma 使用 BF16，既有 Qwen 使用 FP16；模型大小也不同。跨家族差异不是控制所有变量后的因果结论。

| 配置 | Gemma PPL 倍数 | Qwen PPL 倍数 | Gemma 回答改变 | Qwen 回答改变 | Gemma / Qwen 类型 |
|---|---:|---:|---:|---:|---|
| dense-s00 | 1.000 | 1.000 | 0.0% | 0.0% | ?NTJ / ENFJ |
| magnitude-s10 | 0.952 | 1.005 | 11.7% | 13.3% | ?NTJ / EN?J |
| magnitude-s20 | 1.524 | 1.093 | 20.0% | 41.7% | ENFJ / ENFJ |
| magnitude-s30 | 8.996 | 1.769 | 40.0% | 31.7% | ENFJ / ENFJ |
| magnitude-s40 | 231515.849 | 11.119 | 91.7% | 56.7% | INFP / ENFJ |
| magnitude-s50 | 168910905.224 | 24.032 | 83.3% | 81.7% | ?STJ / EN?J |
| magnitude-s60 | 53732617.341 | 385.478 | 46.7% | 88.3% | IST? / IST? |
| magnitude-s70 | 24965976.594 | 3843.568 | 93.3% | 83.3% | ENFP / ENTJ |
| row_magnitude-s10 | 0.945 | 1.007 | 13.3% | 8.3% | ?NTJ / ENFJ |
| row_magnitude-s20 | 1.112 | 1.114 | 16.7% | 30.0% | ENFJ / ENFJ |
| row_magnitude-s30 | 3.289 | 1.819 | 38.3% | 46.7% | ENFJ / ENFJ |
| row_magnitude-s40 | 14485.511 | 7.139 | 85.0% | 55.0% | ENFJ / ENFJ |
| row_magnitude-s50 | 1094377905.542 | 23.019 | 85.0% | 76.7% | INTJ / ENFJ |
| row_magnitude-s60 | 449635727.185 | 8671.516 | 48.3% | 75.0% | ESFP / ENTJ |
| row_magnitude-s70 | 60588557.302 | 2460.534 | 81.7% | 95.0% | ENFJ / ISTP |
| wanda-s10 | 1.011 | 1.001 | 5.0% | 1.7% | ?NTJ / ENFJ |
| wanda-s20 | 0.937 | 1.024 | 6.7% | 15.0% | ?NTJ / ENFJ |
| wanda-s30 | 0.998 | 1.080 | 11.7% | 36.7% | ?NTJ / ENFJ |
| wanda-s40 | 1.035 | 1.182 | 28.3% | 46.7% | EN?J / ENFJ |
| wanda-s50 | 1.564 | 1.319 | 45.0% | 53.3% | ENFJ / ENFJ |
| wanda-s60 | 5.625 | 1.864 | 86.7% | 68.3% | ?SF? / ENFJ |
| wanda-s70 | 7245.983 | 8.148 | 100.0% | 71.7% | ENFJ / ENFJ |
| sparsegpt-s10 | 1.039 | 1.002 | 3.3% | 10.0% | ?NTJ / ENFJ |
| sparsegpt-s20 | 1.019 | 1.024 | 11.7% | 30.0% | ?NTJ / ENFJ |
| sparsegpt-s30 | 1.073 | 1.076 | 15.0% | 38.3% | ?NTJ / ENFJ |
| sparsegpt-s40 | 1.350 | 1.140 | 15.0% | 51.7% | ENTJ / ENFJ |
| sparsegpt-s50 | 1.982 | 1.320 | 48.3% | 80.0% | ENFJ / ENFJ |
| sparsegpt-s60 | 2.593 | 1.630 | 83.3% | 68.3% | ENFJ / ENFJ |
| sparsegpt-s70 | 5.988 | 2.645 | 30.0% | 71.7% | ??TP / ENFJ |
| random-s10-seed0 | 458.674 | 2.432 | 86.7% | 83.3% | ENFP / ENFJ |
| random-s10-seed1 | 22907.916 | 2.671 | 83.3% | 66.7% | ENFJ / ENFJ |
| random-s10-seed2 | 99.330 | 2.898 | 83.3% | 60.0% | ESFP / ENFJ |
| random-s30-seed0 | 463864568.222 | 1941.189 | 88.3% | 71.7% | ?S?J / ENFJ |
| random-s30-seed1 | 225022523.546 | 2764.304 | 83.3% | 100.0% | INFJ / ENFJ |
| random-s30-seed2 | 315189877.164 | 5598.959 | 75.0% | 100.0% | ?NFJ / ?NFP |
| random-s50-seed0 | 371374784.775 | 48574.507 | 93.3% | 73.3% | INT? / ENFJ |
| random-s50-seed1 | 301294557.089 | 1598.693 | 95.0% | 100.0% | ESFP / ENFJ |
| random-s50-seed2 | 154555877.682 | 76260.303 | 91.7% | 78.3% | ?SFP / ???? |
| random-s70-seed0 | 109176995.623 | 16807.292 | 55.0% | 71.7% | ENTP / ENFJ |
| random-s70-seed1 | 239355472.921 | 22850.133 | 78.3% | 98.3% | ?NTP / ISTP |
| random-s70-seed2 | 520633537.052 | 10576.792 | 76.7% | 71.7% | EN?P / ENFJ |
| magnitude-s25-drop1of4 | 8.556 | 1.602 | 23.3% | 63.3% | ENFJ / ENFJ |
| magnitude-s50-drop2of4 | 264357585.599 | 121.217 | 60.0% | 68.3% | ENTJ / ENFJ |
| magnitude-s75-drop3of4 | 111635439.123 | 1762355842.227 | 93.3% | 85.0% | ESF? / ISFJ |
| wanda-s25-drop1of4 | 1.349 | 1.154 | 16.7% | 45.0% | ENTJ / ENFJ |
| wanda-s50-drop2of4 | 12.353 | 2.053 | 83.3% | 71.7% | ENFJ / ENFJ |
| wanda-s75-drop3of4 | 4490381689.574 | 454.328 | 96.7% | 71.7% | ISTP / ENFJ |
| sparsegpt-s25-drop1of4 | 1.148 | 1.123 | 11.7% | 68.3% | ?NTJ / ENFJ |
| sparsegpt-s50-drop2of4 | 2.720 | 1.682 | 46.7% | 61.7% | ENFJ / ENFJ |
| sparsegpt-s75-drop3of4 | 13.698 | 12.657 | 96.7% | 71.7% | ISTP / ENFJ |
| magnitude-s50-attention | 80407.130 | 1.234 | 100.0% | 48.3% | ISTP / ENFJ |
| magnitude-s50-mlp | 1174532.618 | 3.708 | 98.3% | 83.3% | ENFJ / ENFJ |
| magnitude-s50-early | 1338.959 | 1.721 | 98.3% | 61.7% | INFP / ENFJ |
| magnitude-s50-middle | 1.500 | 1.124 | 83.3% | 50.0% | ENFJ / ENFJ |
| magnitude-s50-late | 46044452.619 | 18.905 | 93.3% | 45.0% | ?S?P / ENTJ |
| wanda-s50-attention | 1.201 | 1.048 | 15.0% | 41.7% | ENTJ / ENFJ |
| wanda-s50-mlp | 1.089 | 1.297 | 51.7% | 48.3% | ENFJ / ENFJ |
| wanda-s50-early | 1.480 | 1.090 | 28.3% | 35.0% | ENFJ / ENFJ |
| wanda-s50-middle | 0.870 | 1.078 | 38.3% | 53.3% | ENFJ / ENFJ |
| wanda-s50-late | 1.724 | 1.213 | 10.0% | 43.3% | ?NTJ / ENFJ |
| wanda-s50-calib1 | 1.331 | 1.324 | 73.3% | 50.0% | ENFJ / ENFJ |
| wanda-s50-calib2 | 1.239 | 1.328 | 78.3% | 48.3% | ENFJ / ENFJ |
| sparsegpt-s50-calib1 | 1.832 | 1.331 | 41.7% | 63.3% | ENFJ / ENFJ |
| sparsegpt-s50-calib2 | 1.874 | 1.330 | 28.3% | 73.3% | EN?J / ENFJ |

相同四字母类型不等于逐题回答稳定，也不代表语言能力未下降；结合选项概率、自由生成探针与 PPL 倍数判断。
