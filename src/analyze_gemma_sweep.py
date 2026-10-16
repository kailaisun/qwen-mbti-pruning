"""Validate and summarize Gemma's independent 64-treatment pruning extension.

Qwen source/results are read-only. Cross-model diagnostics use relative PPL
and answer changes against each model's own dense baseline; they do not rank
absolute PPL from different tokenizers. Intermediate logit lenses are model-
specific, so layer JSD is plotted within Gemma only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pruning_sweep import AXES, METHODS, balanced_score, jsd, write_csv
from mbti_common import OPTIONS, OPTION_SCORES, layer_metrics, load_items, score_mbti, softmax
from run_pruning_sweep import experiment_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
PROTOCOL = "3.2-gemma-bos-native-answer-logits"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_run(folder, items):
    summary = json.loads((folder / "summary.json").read_text())
    require(not summary["smoke"] and summary["n_items"] == len(items) == 60,
            f"{folder}: full 60-item non-smoke run required")
    require(summary["protocol_version"] == PROTOCOL, f"{folder}: incompatible protocol")
    require(summary["sweep_version"] == "3.2-gemma", f"{folder}: incompatible sweep")
    require(summary["model"] == MODEL, f"{folder}: wrong model")
    require(summary["dtype"] == "bfloat16", f"{folder}: expected bfloat16")
    require(summary["provenance"]["model_revision"] == REVISION, f"{folder}: wrong revision")
    calibration = summary["provenance"].get("calibration_input", {})
    require(calibration == {"prefix": "explicit_bos", "bos_token_id": 2,
                            "raw_sequence_length": 256, "input_sequence_length": 257,
                            "sequences": 64},
            f"{folder}: expected 64 calibration sequences, each explicit BOS + 256 raw tokens")
    with np.load(folder / "results.npz", allow_pickle=False) as saved:
        logits = saved["opt_logits"].copy()
    require(logits.ndim == 3, f"{folder}: expected [items, layers+1, options]")
    layers = summary["provenance"].get("layer_count", logits.shape[1] - 1)
    require(logits.shape == (60, layers + 1, 7) and layers > 0,
            f"{folder}: inconsistent layer count or logit shape")
    require(logits.dtype == np.float32 and np.isfinite(logits).all(),
            f"{folder}: non-finite or non-float32 logits")
    metrics = layer_metrics(logits)
    np.testing.assert_allclose(summary["layer_entropy_mean"], metrics["entropy_mean"],
                               atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(summary["layer_gap_mean"], metrics["gap_mean"],
                               atol=1e-6, rtol=1e-6)
    choices = [OPTIONS[i] for i in logits[:, -1].argmax(-1)]
    label, detail = score_mbti(items, choices)
    require(label == summary["predicted_mbti"] and detail == summary["dichotomies"],
            f"{folder}: score reconstruction failed")
    require(choices == [summary["per_item_choices"][str(item["id"])] for item in items],
            f"{folder}: saved choices differ from native final logits")
    require(summary["native_projection_max_abs_error"] <= 1e-4,
            f"{folder}: native projection mismatch")
    quality = summary["language_eval"]
    require(quality.get("prefix") == "explicit_bos" and quality.get("bos_token_id") == 2,
            f"{folder}: explicit Gemma BOS token 2 is required for raw-text evaluation")
    require(quality.get("input_sequence_length") == 257 and
            quality.get("scored_targets_per_sequence") == 255,
            f"{folder}: expected BOS + 256 raw tokens but only the original 255 prediction targets")
    require(quality["predicted_tokens"] == 8160 and len(quality["block_nll"]) == 32,
            f"{folder}: expected 32 test blocks / 8160 prediction targets")
    require(np.isfinite(quality["block_nll"]).all(), f"{folder}: non-finite test loss")
    require(math.isclose(np.mean(quality["block_nll"]), quality["mean_nll"], rel_tol=1e-7),
            f"{folder}: NLL does not reconstruct from blocks")
    require(math.isclose(math.exp(quality["mean_nll"]), quality["perplexity"], rel_tol=1e-7),
            f"{folder}: PPL does not reconstruct from NLL")
    require(len(summary["generation_probe"]) == 6 and len(summary["diagnostics"]) == 60,
            f"{folder}: incomplete generation/option diagnostics")
    mass = np.asarray([row["option_probability_mass"] for row in summary["diagnostics"]])
    require(np.isfinite(mass).all() and ((mass >= 0) & (mass <= 1.00001)).all(),
            f"{folder}: invalid option probability mass")
    require(math.isclose(float(mass.mean()), summary["mean_option_probability_mass"],
                         rel_tol=1e-7, abs_tol=1e-12), f"{folder}: inconsistent mean option mass")
    require(math.isclose(np.mean([row["unconstrained_top_is_option"] for row in summary["diagnostics"]]),
                         summary["unconstrained_option_rate"], abs_tol=1e-10),
            f"{folder}: inconsistent legal-first-token rate")
    prompt_info = summary["provenance"]["prompt"]
    for item, diagnostic in zip(items, summary["diagnostics"]):
        require(diagnostic["prompt_sha256"] == prompt_info["prompt_sha256"][str(item["id"])],
                f"{folder}: item prompt hash differs from provenance")
        require(diagnostic["answer_prefix"] == prompt_info["answer_prefix"],
                f"{folder}: item answer boundary differs from provenance")
    for probe in summary["generation_probe"]:
        require(probe["strict_single_letter"] == (
            probe["response"].strip() in OPTIONS and not probe["nonterminal_special_token_ids"]),
                f"{folder}: inconsistent strict-generation diagnostic")
        generated = probe["generated_token_ids"]
        require(probe["generated_tokens"] == len(generated) and 0 < len(generated) <= 32,
                f"{folder}: inconsistent generated-token count")
        require(probe["hit_length_limit"] == (len(generated) == 32 and not probe["stopped_on_eos"]),
                f"{folder}: inconsistent length-limit/EOS diagnostic")
        body = generated[:-1] if probe["stopped_on_eos"] else generated
        require(all(token in body for token in probe["nonterminal_special_token_ids"]),
                f"{folder}: recorded nonterminal special is absent from generated body")
        require(probe["prompt_sha256"] == prompt_info["prompt_sha256"][str(probe["item_id"])],
                f"{folder}: generation prompt hash differs from provenance")
        require(probe["answer_prefix"] == prompt_info["answer_prefix"],
                f"{folder}: generation answer boundary differs from provenance")
    require([probe["item_id"] for probe in summary["generation_probe"]] ==
            [items[i]["id"] for i in (0, 9, 19, 29, 39, 49)],
            f"{folder}: unexpected generation probe items")
    stats = summary["prune"]
    require(stats["masked_parameters"] == sum(row["masked"] for row in stats["per_layer"]),
            f"{folder}: mask-count reconstruction failed")
    require(0 <= stats["masked_parameters"] <= stats["actual_zeros"] <= stats["eligible_parameters"],
            f"{folder}: invalid sparsity counts")
    require(stats["masked_parameters"] <= stats["targeted_parameters"] <= stats["eligible_parameters"],
            f"{folder}: invalid targeted-parameter counts")
    require(stats["targeted_parameters"] == sum(row["numel"] for row in stats["per_layer"]),
            f"{folder}: target-count reconstruction failed")
    require(math.isclose(stats["eligible_zero_fraction"],
                         stats["actual_zeros"] / stats["eligible_parameters"], abs_tol=1e-10),
            f"{folder}: inconsistent eligible sparsity")
    return summary, logits, choices


def response_controls(items):
    return {
        "purpose": "Post-hoc response-style sensitivity only; primary scoring unchanged",
        "key_counts": {
            axis: {"positive": sum(it["trait"] == axis and it["dir"] == 1 for it in items),
                   "negative": sum(it["trait"] == axis and it["dir"] == -1 for it in items)}
            for axis in AXES
        },
        "constant_answers": {
            choice: {"original_type": score_mbti(items, [choice] * len(items))[0],
                     "original_scores": score_mbti(items, [choice] * len(items))[1],
                     "balanced_type": balanced_score(items, [choice] * len(items))[0]}
            for choice in OPTIONS
        },
    }


def interaction_control(runs):
    tags = ["dense-s00", "magnitude-s50-attention", "magnitude-s50-mlp", "magnitude-s50"]
    if not all(tag in runs for tag in tags):
        return {"complete": False, "missing": [tag for tag in tags if tag not in runs]}
    summaries = [runs[tag][0] for tag in tags]
    all_masks = {row["name"]: row["mask_sha256"] for row in summaries[3]["prune"]["per_layer"]}
    split_masks = [{row["name"]: row["mask_sha256"] for row in s["prune"]["per_layer"]}
                   for s in summaries[1:3]]
    require(set(split_masks[0]).isdisjoint(split_masks[1]), "Factorial scopes overlap")
    require({**split_masks[0], **split_masks[1]} == all_masks, "Factorial masks differ")
    losses = [s["language_eval"]["mean_nll"] for s in summaries]
    return {
        "complete": True,
        "type": "Post-hoc fixed-mask attention x MLP 2x2 factorial on held-out mean NLL",
        "masks_verified_disjoint_and_union_equal": True,
        "treatments": {tag: {"mean_nll": loss, "ppl": runs[tag][0]["language_eval"]["perplexity"]}
                       for tag, loss in zip(tags, losses)},
        "interaction_nats_per_token": losses[3] - losses[1] - losses[2] + losses[0],
        "interpretation": "L(all)-L(attention)-L(MLP)+L(dense); fixed-model/corpus diagnostic, no significance claim",
        "scope": "Magnitude only; activation-dependent Wanda masks need not match across scopes",
    }


def derive_rows(configs, runs, items):
    baseline, baseline_logits, baseline_choices = runs["dense-s00"]
    baseline_probs = softmax(baseline_logits)
    sorted_probs = np.sort(baseline_probs[:, -1], axis=-1)
    baseline_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
    dense_ordinal = np.array([OPTION_SCORES[c] for c in baseline_choices])
    scored = np.array([item["trait"] in AXES for item in items])
    rows, item_rows, details = [], [], {}
    for cfg in configs:
        tag = cfg["tag"]
        summary, logits, choices = runs[tag]
        require(summary["sweep_config"] == cfg, f"{tag}: configuration differs from fixed matrix")
        require(summary["provenance"]["source_sha256"] == baseline["provenance"]["source_sha256"],
                f"{tag}: source provenance differs from baseline")
        require(summary["provenance"]["corpus"] == baseline["provenance"]["corpus"],
                f"{tag}: corpus provenance differs from baseline")
        require(summary["provenance"].get("prompt") == baseline["provenance"].get("prompt"),
                f"{tag}: prompt provenance differs from baseline")
        require(logits.shape == baseline_logits.shape, f"{tag}: layer shape differs")
        # No embedding/final norm/head pruning: the embedding-only lens is a
        # useful invariant for identical prompt templates and restored weights.
        np.testing.assert_allclose(logits[:, 0], baseline_logits[:, 0], atol=1e-5, rtol=1e-5)
        probabilities = softmax(logits)
        changed = np.array(choices) != np.array(baseline_choices)
        divergence = jsd(probabilities, baseline_probs)
        votes = {choice: choices.count(choice) for choice in OPTIONS}
        contributions = []
        for index, (item, before, after) in enumerate(zip(items, baseline_choices, choices)):
            direction = int(item["dir"] or 0) if item["trait"] in AXES else 0
            delta = direction * (OPTION_SCORES[after] - OPTION_SCORES[before])
            row = {
                "tag": tag, "item_id": item["id"], "trait": item["trait"], "direction": direction,
                "dense_choice": before, "pruned_choice": after, "changed": bool(changed[index]),
                "dense_signed_score": direction * OPTION_SCORES[before],
                "pruned_signed_score": direction * OPTION_SCORES[after], "signed_score_delta": delta,
                "dense_option_gap": float(baseline_gap[index]),
                "argmax_tied_options": int((logits[index, -1] == logits[index, -1].max()).sum()),
                "final_jsd": float(divergence[index, -1]),
                "option_probability_mass": summary["diagnostics"][index]["option_probability_mass"],
                "unconstrained_top_token_json": json.dumps(
                    summary["diagnostics"][index]["unconstrained_top_token"], ensure_ascii=False),
            }
            item_rows.append(row)
            if delta:
                contributions.append(row)
        for axis in AXES:
            require(sum(row["signed_score_delta"] for row in contributions if row["trait"] == axis) ==
                    summary["dichotomies"][axis]["score"] - baseline["dichotomies"][axis]["score"],
                    f"{tag}: {axis} item contributions do not conserve score change")
        stats = summary["prune"]["per_layer"]
        energy = sum(row["weight_energy"] for row in stats)
        expected_item = probabilities[:, -1] @ np.array([OPTION_SCORES[c] for c in OPTIONS])
        expected_dimensions = {
            axis: float(sum(int(item["dir"]) * expected_item[index]
                            for index, item in enumerate(items) if item["trait"] == axis))
            for axis in AXES
        }
        balanced_type, balanced_dimensions = balanced_score(items, choices)
        ordinal = np.array([OPTION_SCORES[c] for c in choices])
        row = {
            "tag": tag, "method": cfg["method"], "requested_ratio": cfg["ratio"], "scope": cfg["scope"],
            "drop_n_of_4": cfg["drop_n"], "seed": cfg["seed"], "calibration_seed": cfg["calibration_seed"],
            "eligible_zero_fraction": summary["prune"]["eligible_zero_fraction"],
            "masked_model_fraction": summary["prune"]["masked_parameters"] / summary["prune"]["total_model_parameters"],
            "mbti": summary["predicted_mbti"],
            **{axis: summary["dichotomies"][axis]["score"] for axis in AXES},
            **{"soft_" + axis: expected_dimensions[axis] for axis in AXES},
            "balanced_mbti": balanced_type,
            **{"balanced_" + axis: balanced_dimensions[axis] for axis in AXES},
            "agreement_60": float((~changed).mean()), "agreement_49": float((~changed[scored]).mean()),
            "mean_absolute_option_step": float(np.abs(ordinal - dense_ordinal).mean()),
            "mean_absolute_option_step_49": float(np.abs(ordinal - dense_ordinal)[scored].mean()),
            "opposite_polarity_items": int((ordinal * dense_ordinal < 0).sum()),
            "neutral_status_changes": int(((ordinal == 0) != (dense_ordinal == 0)).sum()),
            "argmax_tied_items": int(((logits[:, -1] == logits[:, -1].max(-1, keepdims=True)).sum(-1) > 1).sum()),
            "ppl": summary["language_eval"]["perplexity"],
            "ppl_ratio": summary["language_eval"]["perplexity"] / baseline["language_eval"]["perplexity"],
            "test_nll_delta": summary["language_eval"]["mean_nll"] - baseline["language_eval"]["mean_nll"],
            "option_mass": summary["mean_option_probability_mass"],
            "legal_first_token_rate": summary["unconstrained_option_rate"],
            "strict_generation_rate": float(np.mean([p["strict_single_letter"] for p in summary["generation_probe"]])),
            "neutral_D_fraction": votes["D"] / len(items), "final_jsd": float(divergence[:, -1].mean()),
            "most_common_choice": max(votes, key=votes.get), "most_common_fraction": max(votes.values()) / len(items),
            "weight_relative_mse": sum(s["weight_delta_energy"] for s in stats) / energy if energy else 0.,
            "entropy": summary["layer_entropy_mean"][-1], "gap": summary["layer_gap_mean"][-1],
            "flipped_dense_gap_mean": float(baseline_gap[changed].mean()) if changed.any() else None,
            "stable_dense_gap_mean": float(baseline_gap[~changed].mean()) if (~changed).any() else None,
            "prune_seconds": summary["prune"]["seconds"], "elapsed_seconds": summary["elapsed_seconds"],
        }
        rows.append(row)
        details[tag] = {
            "choice_histogram": votes, "expected_dimension_scores": expected_dimensions,
            "balanced_type": balanced_type, "balanced_dimension_scores": balanced_dimensions,
            "top_score_contributions": sorted(contributions, key=lambda r: abs(r["signed_score_delta"]), reverse=True),
            "layer_jsd": divergence.mean(0).tolist(), "generation_probe": summary["generation_probe"],
            "block_reconstruction": summary["prune"]["block_reconstruction"],
        }
    return rows, item_rows, details


def plot_results(outdir, rows, details, layers):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method and r["scope"] == "all"
                  and not r["drop_n_of_4"] and not r["calibration_seed"]]
        ratios = sorted({r["requested_ratio"] for r in subset})
        if not ratios:
            continue
        for ax, key in zip(axes.flat, ("agreement_60", "ppl_ratio", "option_mass", "neutral_D_fraction")):
            values = [[r[key] for r in subset if r["requested_ratio"] == ratio] for ratio in ratios]
            means = np.array([np.mean(value) for value in values])
            low = means - np.array([min(value) for value in values])
            high = np.array([max(value) for value in values]) - means
            ax.errorbar(np.array(ratios) * 100, means, yerr=np.maximum([low, high], 0),
                        marker="o", capsize=3, label=method)
    for ax, key in zip(axes.flat, ("agreement_60", "ppl_ratio", "option_mass", "neutral_D_fraction")):
        ax.set_xlabel("Requested eligible-weight sparsity (%)")
        ax.set_ylabel(key)
        ax.grid(alpha=.25)
    axes[0, 1].set_yscale("log")
    # A severely pruned BF16 model may have numerically zero option mass.
    # Symlog includes those actual zeros without silently clipping the data.
    axes[1, 0].set_yscale("symlog", linthresh=1e-12)
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Gemma 4 12B: within-model pruning diagnostics")
    fig.tight_layout()
    fig.savefig(outdir / "sparsity_tradeoffs.png", dpi=160)
    plt.close(fig)
    tags = [r["tag"] for r in rows if r["scope"] == "all" and r["requested_ratio"] == .5
            and not r["seed"] and not r["calibration_seed"]]
    if not tags:
        return tags
    fig, ax = plt.subplots(figsize=(12, max(3, len(tags) * .45)))
    im = ax.imshow([details[tag]["layer_jsd"] for tag in tags], aspect="auto", vmin=0)
    ax.set_yticks(range(len(tags)), tags)
    ax.set_xlabel(f"Hidden-state entry (0 = embedding; {layers} = native final)")
    fig.colorbar(im, ax=ax, label="Mean option-distribution JSD (nats)")
    fig.tight_layout()
    fig.savefig(outdir / "layer_jsd_50.png", dpi=160)
    plt.close(fig)
    bytag = {r["tag"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7, max(3, len(tags) * .45)))
    values = np.array([[bytag[tag][axis] for axis in AXES] for tag in tags])
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-39, vmax=39)
    ax.set_yticks(range(len(tags)), tags)
    ax.set_xticks(range(4), AXES)
    for i in range(len(tags)):
        for j in range(4):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Signed dimension score")
    fig.tight_layout()
    fig.savefig(outdir / "dimension_scores_50.png", dpi=160)
    plt.close(fig)
    return tags


def write_cross_model(outdir, rows, qwen_dir):
    comparison = qwen_dir / "comparison.csv"
    if not comparison.exists():
        (outdir / "CROSS_MODEL.md").write_text("# 跨模型对照\n\nQwen v3 comparison.csv 不存在；未生成跨模型对照。\n")
        return []
    with comparison.open(newline="") as handle:
        qwen_rows = {row["tag"]: row for row in csv.DictReader(handle)}
    matched = []
    for gemma in rows:
        if gemma["tag"] not in qwen_rows:
            continue
        qwen = qwen_rows[gemma["tag"]]
        for field in ("method", "scope"):
            require(gemma[field] == qwen[field], f"Cross-model {gemma['tag']}: mismatched {field}")
        for field in ("requested_ratio", "drop_n_of_4", "seed", "calibration_seed"):
            require(float(gemma[field]) == float(qwen[field]), f"Cross-model {gemma['tag']}: mismatched {field}")
        matched.append({
            "tag": gemma["tag"], "method": gemma["method"], "scope": gemma["scope"],
            "requested_ratio": gemma["requested_ratio"],
            "gemma_mbti": gemma["mbti"], "qwen_mbti": qwen["mbti"],
            "gemma_ppl_relative_to_own_dense": gemma["ppl_ratio"],
            "qwen_ppl_relative_to_own_dense": float(qwen["ppl_ratio"]),
            "gemma_answer_flip_rate": 1 - gemma["agreement_60"],
            "qwen_answer_flip_rate": 1 - float(qwen["agreement_60"]),
            "gemma_option_step_mae": gemma["mean_absolute_option_step"],
            "qwen_option_step_mae": float(qwen["mean_absolute_option_step"]),
            "gemma_legal_first_token_rate": gemma["legal_first_token_rate"],
            "qwen_legal_first_token_rate": float(qwen["legal_first_token_rate"]),
            "gemma_eligible_zero_fraction": gemma["eligible_zero_fraction"],
            "qwen_eligible_zero_fraction": float(qwen["eligible_zero_fraction"]),
        })
    if matched:
        write_csv(outdir / "cross_model_comparison.csv", matched)
    lines = ["# Gemma / Qwen：相对自身基线的剪枝变化", "", f"当前有 {len(matched)}/64 个匹配配置。", "",
             "PPL 倍数 = 剪枝模型 PPL / 同模型 dense PPL；回答改变率也分别相对各自 dense 回答。",
             "Gemma 与 Qwen 的 tokenizer、token 边界、实际文本覆盖范围、模型结构与原生 chat 模板不同；因此不比较绝对 PPL，倍数也仅是本协议内的稳健性诊断，不是模型综合能力排名。",
             "Gemma 原文校准与 PPL 输入为显式 BOS（ID 2）+ 256 原始 token；PPL 仍只评分每段原始位置 1–255，排除 BOS→首个原始 token。Qwen 原协议未改动。",
             "Gemma 在有效 chat 模板下能正常回答，也可能在无 chat 包装的原文续写上出现较高 PPL；不能据此将原文 PPL 或其倍数当作通用能力排名，PPL 下降也不自动意味着人格或对话能力改善。",
             "Gemma 使用 BF16，既有 Qwen 使用 FP16；模型大小也不同。跨家族差异不是控制所有变量后的因果结论。", "",
             "| 配置 | Gemma PPL 倍数 | Qwen PPL 倍数 | Gemma 回答改变 | Qwen 回答改变 | Gemma / Qwen 类型 |",
             "|---|---:|---:|---:|---:|---|"]
    for row in matched:
        lines.append(f"| {row['tag']} | {row['gemma_ppl_relative_to_own_dense']:.3f} | "
                     f"{row['qwen_ppl_relative_to_own_dense']:.3f} | {row['gemma_answer_flip_rate']:.1%} | "
                     f"{row['qwen_answer_flip_rate']:.1%} | {row['gemma_mbti']} / {row['qwen_mbti']} |")
    lines += ["", "相同四字母类型不等于逐题回答稳定，也不代表语言能力未下降；结合选项概率、自由生成探针与 PPL 倍数判断。", ""]
    (outdir / "CROSS_MODEL.md").write_text("\n".join(lines))
    return matched


def write_markdown(outdir, rows, missing, layers, compare_tags):
    header = "| 配置 | 类型 | EI/NS/FT/JP | 一致题数 | PPL | PPL / dense | 合法首 token | D 比例 |"
    separator = "|---|---|---|---:|---:|---:|---:|---:|"
    def line(row):
        return (f"| {row['tag']} | {row['mbti']} | {'/'.join(str(row[d]) for d in AXES)} | "
                f"{round(row['agreement_60'] * 60)}/60 | {row['ppl']:.3f} | {row['ppl_ratio']:.3f} | "
                f"{row['legal_first_token_rate']:.1%} | {row['neutral_D_fraction']:.1%} |")
    lines = ["# Gemma 4 12B 静态剪枝：完整结果表", "",
             f"已验证 {len(rows)}/64 组。" + ("全部完成。" if not missing else "尚未完成，不作最终排名。"), "",
             "仅文本输入、语言层静态剪枝；模型版本、语料哈希、掩码哈希保存在每组 summary.json。",
             "PPL 为 WikiText-2 test 开头 32 × 256 原始 token 短上下文子集；每段显式前置 BOS（ID 2），模型输入长度 257。",
             "只评分原始 token 的位置 1–255（logits[1:-1] 对齐 raw_tokens[1:]），不评分 BOS→首个原始 token；共 8,160 个预测位置，不是完整基准。原始语料文件未修改，校准输入同样前置 BOS。",
             "一致率相对 Gemma 自身 dense 回答，不是正确率或真实人格保持率；零分 `?` 表示未确定方向。",
             "自由生成只有 6 个固定探针。原生回答位置的首 token 合法率与整段自由生成是不同诊断。", "", header, separator]
    lines.extend(line(row) for row in rows)
    lines += ["", "## 图表", "", "![多比例比较](sparsity_tradeoffs.png)"]
    if compare_tags:
        lines += ["", "![50% 层级偏移](layer_jsd_50.png)", "", "![50% 维度分数](dimension_scores_50.png)"]
    lines += ["", "## 读数限制", "",
              "逐题贡献见 `item_deltas.csv`；逐层散度、回答分布与生成原文见 `analysis.json`。",
              "`soft_*` 为概率加权分数，`balanced_*` 为正/负向题分别均值再相减的事后敏感性诊断，均不替代主要评分。",
              "当前手工方向键下全选 A/B/C 得 ENFJ，全选 E/F/G 得 ISTP；常量回答对照见 `response_controls.json`。",
              "随机曲线为种子均值，误差棒为已完成种子的最小–最大范围，不是置信区间。",
              "局部 scope 的删除预算不相同；稀疏权重保存在 dense 张量中，不声称部署加速。",
              "原文续写与原生 chat 是不同输入分布：即使补足必需 BOS，较高原文 PPL 仍可与正常的简单 chat 回答并存。当前 PPL 仅作固定协议诊断，不作模型综合能力排名；PPL 下降也不必然意味着问卷/对话能力改善。",
              "早期无 BOS 的 smoke/诊断不纳入这 64 组正式结果，也不能混用为 dense 基线。",
              "[结构/位置/种子消融](ABLATIONS.md)；[跨模型相对自身基线对照](CROSS_MODEL.md)。", ""]
    (outdir / "RESULTS.md").write_text("\n".join(lines))
    ablations = ["# Gemma：结构、剪枝位置与种子", "", f"当前 {len(rows)}/64 组；完整性以 RESULTS.md 为准。", ""]
    groups = [
        ("4 元组约束：drop 表示每四个删除的数量", [r for r in rows if r["drop_n_of_4"]]),
        ("局部模块内 50%：不是等全模型删除预算", [r for r in rows if r["scope"] != "all"]),
        ("50% 校准种子重复", [r for r in rows if r["method"] in ("wanda", "sparsegpt")
                              and r["requested_ratio"] == .5 and r["scope"] == "all" and not r["drop_n_of_4"]]),
        ("随机掩码种子重复", [r for r in rows if r["method"] == "random"]),
    ]
    for title, subset in groups:
        ablations += ["", "## " + title, "", header, separator]
        ablations.extend(line(row) for row in subset)
    first, second = layers // 3, layers * 2 // 3
    ablations += ["", f"模型共有 {layers} 层。三段范围（零起始）：early=0–{first - 1}、middle={first}–{second - 1}、late={second}–{layers - 1}。",
                  "层数相同也不保证参数预算相同（例如全局/局部注意力结构不同）；以 comparison.csv 实际 eligible_zero_fraction 为准。",
                  "`eligible` 的分母是全部可剪语言层线性权重，不是局部被选模块；不会剪 embedding、LM head、norm 或多模态编码器。",
                  "幅值掩码 attention×MLP 的固定掩码交互（若四组均完成）见 `interaction_controls.json`；在 NLL 而非 PPL 上计算加法交互。", ""]
    (outdir / "ABLATIONS.md").write_text("\n".join(ablations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/gemma4_12b_v3")
    parser.add_argument("--qwen-dir", type=Path, default=ROOT / "outputs/pruning_v3")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    matrix = experiment_matrix()
    missing = [cfg["tag"] for cfg in matrix if not (args.outdir / cfg["tag"] / "summary.json").exists()]
    if missing and not args.allow_partial:
        raise RuntimeError(f"Missing {len(missing)} treatments: {missing}")
    configs = [cfg for cfg in matrix if cfg["tag"] not in missing]
    items = load_items()
    runs = {cfg["tag"]: read_run(args.outdir / cfg["tag"], items) for cfg in configs}
    require("dense-s00" in runs, "Dense Gemma baseline must be complete before relative analysis")
    rows, item_rows, details = derive_rows(configs, runs, items)
    layers = runs["dense-s00"][1].shape[1] - 1
    write_csv(args.outdir / "comparison.csv", rows)
    write_csv(args.outdir / "item_deltas.csv", item_rows)
    write_json(args.outdir / "response_controls.json", response_controls(items))
    write_json(args.outdir / "interaction_controls.json", interaction_control(runs))
    write_json(args.outdir / "analysis.json", {
        "complete": not missing, "missing": missing, "n_runs": len(rows),
        "model": MODEL, "model_revision": REVISION, "protocol_version": PROTOCOL,
        "baseline": "dense-s00", "layer_count": layers, "runs": details,
    })
    compare_tags = plot_results(args.outdir, rows, details, layers)
    write_cross_model(args.outdir, rows, args.qwen_dir)
    write_markdown(args.outdir, rows, missing, layers, compare_tags)
    print(f"Validated {len(rows)}/64 Gemma runs; layer count {layers}; complete={not missing}. "
          "Wrote comparison.csv, item_deltas.csv, analysis.json, RESULTS.md, ABLATIONS.md, "
          "CROSS_MODEL.md and plots.")


if __name__ == "__main__":
    main()
