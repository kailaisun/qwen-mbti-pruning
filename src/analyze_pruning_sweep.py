"""Validate every treatment and derive auditable item/layer/quality comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mbti_common import OPTIONS, OPTION_SCORES, load_items, score_mbti, softmax
from run_pruning_sweep import experiment_matrix

ROOT=Path(__file__).resolve().parents[1]
AXES=['EI','NS','FT','JP']
METHODS=['magnitude','row_magnitude','wanda','sparsegpt','random']


def jsd(p,q):
    middle=(p+q)/2
    return .5*((p*np.log(np.maximum(p,1e-30)/np.maximum(middle,1e-30))).sum(-1)
               +(q*np.log(np.maximum(q,1e-30)/np.maximum(middle,1e-30))).sum(-1))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def read_run(folder,items):
    summary=json.loads((folder/'summary.json').read_text())
    assert not summary['smoke'] and summary['n_items']==60
    assert summary['protocol_version']=='2.0-native-final-logits'
    z=np.load(folder/'results.npz')['opt_logits']
    assert z.shape==(60,29,7) and z.dtype==np.float32 and np.isfinite(z).all()
    choices=[OPTIONS[i] for i in z[:,-1].argmax(-1)]
    label,detail=score_mbti(items,choices)
    assert label==summary['predicted_mbti'] and detail==summary['dichotomies']
    assert choices==[summary['per_item_choices'][str(it['id'])] for it in items]
    assert summary['native_projection_max_abs_error']<=1e-4
    assert summary['language_eval']['predicted_tokens']==8160
    assert len(summary['language_eval']['block_nll'])==32
    assert len(summary['generation_probe'])==6
    assert summary['prune']['masked_parameters']==sum(x['masked'] for x in summary['prune']['per_layer'])
    assert summary['prune']['masked_parameters']<=summary['prune']['actual_zeros']
    return summary,z,choices


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',default=str(ROOT/'outputs/pruning_v3'))
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args();outdir=Path(args.outdir);items=load_items()
    configs=experiment_matrix();missing=[c['tag'] for c in configs if not (outdir/c['tag']/'summary.json').exists()]
    if missing and not args.allow_partial: raise RuntimeError(f'Missing {len(missing)} treatments: {missing}')
    configs=[c for c in configs if c['tag'] not in missing]
    runs={c['tag']:read_run(outdir/c['tag'],items) for c in configs}
    base,bz,bc=runs['dense-s00'];bp=softmax(bz);bfinal=bp[:,-1]
    baseline_gap=np.sort(bfinal,axis=-1)[:,-1]-np.sort(bfinal,axis=-1)[:,-2]
    rows=[];item_rows=[];details={};layer_js={};curve_data={}
    scored=np.array([it['trait'] in AXES for it in items])
    for cfg in configs:
        tag=cfg['tag'];s,z,choices=runs[tag]
        assert s['sweep_config']==cfg
        assert s['provenance']['source_sha256']==base['provenance']['source_sha256']
        p=softmax(z); changed=np.array(choices)!=np.array(bc)
        divergence=jsd(p,bp);layer_js[tag]=divergence.mean(0).tolist()
        votes={c:choices.count(c) for c in OPTIONS}
        contributions=[]
        for i,(it,a,b) in enumerate(zip(items,bc,choices)):
            direction=int(it['dir'] or 0) if it['trait'] in AXES else 0
            delta=direction*(OPTION_SCORES[b]-OPTION_SCORES[a])
            row={'tag':tag,'item_id':it['id'],'trait':it['trait'],'direction':direction,
                 'dense_choice':a,'pruned_choice':b,'changed':bool(changed[i]),
                 'dense_signed_score':direction*OPTION_SCORES[a],
                 'pruned_signed_score':direction*OPTION_SCORES[b],
                 'signed_score_delta':delta,'dense_option_gap':float(baseline_gap[i]),
                 'final_jsd':float(divergence[i,-1]),
                 'option_probability_mass':s['diagnostics'][i]['option_probability_mass'],
                 'unconstrained_top_token':s['diagnostics'][i]['unconstrained_top_token']}
            item_rows.append(row)
            if delta: contributions.append(row)
        for dimension in AXES:
            assert sum(r['signed_score_delta'] for r in contributions if r['trait']==dimension)==(
                s['dichotomies'][dimension]['score']-base['dichotomies'][dimension]['score'])
        stats=s['prune']['per_layer'];energy=sum(x['weight_energy'] for x in stats)
        row={'tag':tag,'method':cfg['method'],'requested_ratio':cfg['ratio'],'scope':cfg['scope'],
             'drop_n_of_4':cfg['drop_n'],'seed':cfg['seed'],'calibration_seed':cfg['calibration_seed'],
             'eligible_zero_fraction':s['prune']['eligible_zero_fraction'],
             'masked_model_fraction':s['prune']['masked_parameters']/s['prune']['total_model_parameters'],
             'mbti':s['predicted_mbti'],**{d:s['dichotomies'][d]['score'] for d in AXES},
             'agreement_60':float((~changed).mean()),'agreement_49':float((~changed[scored]).mean()),
             'ppl':s['language_eval']['perplexity'],'ppl_ratio':s['language_eval']['perplexity']/base['language_eval']['perplexity'],
             'test_nll_delta':s['language_eval']['mean_nll']-base['language_eval']['mean_nll'],
             'option_mass':s['mean_option_probability_mass'],'legal_first_token_rate':s['unconstrained_option_rate'],
             'strict_generation_rate':float(np.mean([x['strict_single_letter'] for x in s['generation_probe']])),
             'neutral_D_fraction':votes['D']/60,'final_jsd':float(divergence[:,-1].mean()),
             'weight_relative_mse':sum(x['weight_delta_energy'] for x in stats)/energy if energy else 0.,
             'entropy':s['layer_entropy_mean'][-1],'gap':s['layer_gap_mean'][-1],
             'flipped_dense_gap_mean':float(baseline_gap[changed].mean()) if changed.any() else None,
             'stable_dense_gap_mean':float(baseline_gap[~changed].mean()) if (~changed).any() else None,
             'prune_seconds':s['prune']['seconds'],'elapsed_seconds':s['elapsed_seconds']}
        rows.append(row)
        details[tag]={'choice_histogram':votes,'top_score_contributions':sorted(contributions,key=lambda x:abs(x['signed_score_delta']),reverse=True),
                      'layer_jsd':layer_js[tag],'generation_probe':s['generation_probe'],
                      'block_reconstruction':s['prune']['block_reconstruction']}
    write_csv(outdir/'comparison.csv',rows);write_csv(outdir/'item_deltas.csv',item_rows)
    (outdir/'analysis.json').write_text(json.dumps({'complete':not missing,'missing':missing,
        'n_runs':len(rows),'baseline':base['tag'],'runs':details},indent=2,ensure_ascii=False,allow_nan=False))

    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for method in METHODS:
        subset=[r for r in rows if r['method']==method and r['scope']=='all' and not r['drop_n_of_4'] and not r['calibration_seed']]
        ratios=sorted(set(r['requested_ratio'] for r in subset))
        for ax,key in zip(axs.flat,['agreement_60','ppl_ratio','option_mass','neutral_D_fraction']):
            means=[np.mean([r[key] for r in subset if r['requested_ratio']==ratio]) for ratio in ratios]
            stds=[np.std([r[key] for r in subset if r['requested_ratio']==ratio]) for ratio in ratios]
            ax.errorbar(np.array(ratios)*100,means,yerr=stds,marker='o',capsize=3,label=method)
            ax.set_xlabel('Requested eligible-weight sparsity (%)');ax.set_ylabel(key);ax.grid(alpha=.25)
    axs[0,1].set_yscale('log');axs[1,0].set_yscale('log');axs[0,0].legend(fontsize=8)
    fig.suptitle('Static pruning: response agreement, language loss and format diagnostics')
    fig.tight_layout();fig.savefig(outdir/'sparsity_tradeoffs.png',dpi=160);plt.close(fig)
    compare_tags=[r['tag'] for r in rows if r['scope']=='all' and r['requested_ratio']==.5 and not r['seed'] and not r['calibration_seed']]
    if compare_tags:
        fig,ax=plt.subplots(figsize=(12,max(3,len(compare_tags)*.45)))
        im=ax.imshow([layer_js[t] for t in compare_tags],aspect='auto',vmin=0)
        ax.set_yticks(range(len(compare_tags)),compare_tags);ax.set_xlabel('Hidden-state entry (0 = embedding; 28 = native final)')
        fig.colorbar(im,ax=ax,label='Mean option-distribution JSD (nats)');fig.tight_layout()
        fig.savefig(outdir/'layer_jsd_50.png',dpi=160);plt.close(fig)
        bytag={r['tag']:r for r in rows}
        fig,ax=plt.subplots(figsize=(7,max(3,len(compare_tags)*.45)))
        values=np.array([[bytag[t][d] for d in AXES] for t in compare_tags])
        im=ax.imshow(values,aspect='auto',cmap='coolwarm',vmin=-39,vmax=39)
        ax.set_yticks(range(len(compare_tags)),compare_tags);ax.set_xticks(range(4),AXES)
        for i in range(len(compare_tags)):
            for j in range(4): ax.text(j,i,str(values[i,j]),ha='center',va='center')
        fig.colorbar(im,ax=ax,label='Signed dimension score');fig.tight_layout()
        fig.savefig(outdir/'dimension_scores_50.png',dpi=160);plt.close(fig)

    header='| 配置 | 类型 | EI/NS/FT/JP | 一致率(60) | PPL | 合法首 token | 选项总概率 | D 比例 |'
    separator='|---|---|---|---:|---:|---:|---:|---:|'
    def table_line(r):
        return f"| {r['tag']} | {r['mbti']} | {'/'.join(str(r[d]) for d in AXES)} | {r['agreement_60']:.1%} | {r['ppl']:.2f} | {r['legal_first_token_rate']:.1%} | {r['option_mass']:.6g} | {r['neutral_D_fraction']:.1%} |"
    lines=['# 静态剪枝扩展：完整结果表','',f"已验证 {len(rows)}/64 组。{'全部完成。' if not missing else '尚未完成，不作最终排名。'}",'',
           'PPL 为固定 WikiText-2 test 短上下文子集诊断；一致率不是准确率，也不是真实人格保持率。',
           '零分 `?` 表示该维度没有确定方向。首 token 不合法不等同于整段生成一定不合法；自由生成只有 6 个固定探针。','',header,separator]
    lines.extend(table_line(r) for r in rows)
    lines+=['','## 图表','','![多比例比较](sparsity_tradeoffs.png)','','![50% 层级偏移](layer_jsd_50.png)','','![50% 维度分数](dimension_scores_50.png)',
            '','逐题可审计分解见 `item_deltas.csv`，逐层和生成原文见 `analysis.json`。',
            '随机方法误差棒仅为 3 个掩码种子的标准差，不是人群/模型家族置信区间。',
            '局部 scope 消融的总删除参数不同，不能直接当等预算方法排名。','']
    (outdir/'RESULTS.md').write_text('\n'.join(lines))
    print(f'Validated {len(rows)}/64; wrote comparison.csv, item_deltas.csv, analysis.json, RESULTS.md and plots.')


if __name__=='__main__':main()
