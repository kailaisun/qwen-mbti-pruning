#!/usr/bin/env python
"""Aggregate MBTI run results: type table (paper Table 2 style) + layer curves
(paper Figure 1 style: decisional entropy and top1-top2 confidence gap)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from mbti_common import PROTOCOL_VERSION, load_items, score_mbti, layer_metrics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(HERE.parent / "outputs" / "corrected_v2"))
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--legacy-outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tags = args.tags or ['qwen25-7b-fp16','qwen25-7b-awq4','qwen25-7b-gptq4',
                        'qwen25-7b-prune-u50','qwen25-7b-prune-2to4']
    summaries = {t: json.loads((outdir / t / "summary.json").read_text()) for t in tags}
    items = load_items()
    base = summaries['qwen25-7b-fp16']['per_item_choices']
    records=[]
    for tag,s in summaries.items():
        if s.get('protocol_version') != PROTOCOL_VERSION or s['n_items'] != len(items):
            raise ValueError(f'{tag}: wrong protocol or incomplete questionnaire')
        with np.load(outdir/tag/'results.npz',allow_pickle=False) as data:
            z=data['opt_logits']
        if z.shape != (60,29,7) or z.dtype != np.float32 or not np.isfinite(z).all():
            raise ValueError(f'{tag}: invalid logits')
        c=s['per_item_choices']
        choices=['ABCDEFG'[x] for x in z[:,-1].argmax(-1)]
        if choices != [c[str(i['id'])] for i in items]:
            raise ValueError(f'{tag}: inconsistent saved choices')
        label,detail=score_mbti(items,choices)
        if label != s['predicted_mbti'] or detail != s['dichotomies']:
            raise ValueError(f'{tag}: inconsistent scores')
        metrics=layer_metrics(z)
        np.testing.assert_allclose(metrics['entropy_mean'],s['layer_entropy_mean'],atol=1e-6)
        np.testing.assert_allclose(metrics['gap_mean'],s['layer_gap_mean'],atol=1e-6)
        if s['native_projection_max_abs_error'] > 1e-4:
            raise ValueError(f'{tag}: native projection failed')
        scored=[str(it['id']) for it in items if it['trait']!='AT']
        record={'tag':tag,'mbti':label,'scores':{d:v['score'] for d,v in detail.items()},
                'agreement_all':sum(c[k]==base[k] for k in base),
                'agreement_scored':sum(c[k]==base[k] for k in scored),
                'scored_n':len(scored),'entropy':s['layer_entropy_mean'][-1],
                'gap':s['layer_gap_mean'][-1],
                'option_mass':s['mean_option_probability_mass'],
                'unconstrained_option_rate':s['unconstrained_option_rate'],
                'double_norm_changed':s['legacy_double_norm_on_same_weights']['changed_choices']}
        if args.legacy_outdir and (Path(args.legacy_outdir)/tag/'summary.json').is_file():
            old=json.loads((Path(args.legacy_outdir)/tag/'summary.json').read_text())
            record['old_mbti']=old['predicted_mbti']
            record['old_choices_changed']=sum(c[k]!=old['per_item_choices'][k] for k in c)
        records.append(record)

    with (outdir/'per_item_choices.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['id','trait','dir','text',*tags])
        for it in items:
            writer.writerow([it['id'],it['trait'],it['dir'],it['text'],
                             *[summaries[t]['per_item_choices'][str(it['id'])] for t in tags]])
    (outdir/'comparison.json').write_text(json.dumps(records,indent=2))
    packages=summaries['qwen25-7b-fp16']['provenance']['packages']
    (outdir/'environment.freeze.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(packages.items())))

    header = "| tag | MBTI | EI / NS / FT / JP | agreement /60 | agreement /49 | final H | final gap |"
    rows = [header, "|" + "---|" * 7]
    for t,r in zip(tags,records):
        s = summaries[t]
        d = s["dichotomies"]
        rows.append(
            f"| {t} | {s['predicted_mbti']} | "
            f"{d['EI']['score']} / {d['NS']['score']} / {d['FT']['score']} / {d['JP']['score']} | "
            f"{r['agreement_all']}/60 | {r['agreement_scored']}/49 | {r['entropy']:.3f} | {r['gap']:.3f} |"
        )
    table = "\n".join(rows) + "\n"
    (outdir / "summary_table.md").write_text(table)
    print(table)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for t in tags:
        s = summaries[t]
        ent = s["layer_entropy_mean"][1:]  # drop embedding row (layer 0)
        gap = s["layer_gap_mean"][1:]
        xs = range(1, len(ent) + 1)
        axes[0].plot(xs, ent, label=t, lw=1.6)
        axes[1].plot(xs, gap, label=t, lw=1.6)
    axes[0].set_title("Layer-wise decisional entropy")
    axes[1].set_title("Layer-wise confidence gap (top-1 minus top-2)")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    png = outdir / "layer_curves.png"
    fig.savefig(png, dpi=200)
    fig.savefig(outdir/'layer_curves.svg')
    plt.close(fig)
    print(f"[saved] {png}")


if __name__ == "__main__":
    main()
