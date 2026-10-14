"""Read-only status; works without importing torch or having a CUDA context."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
summaries=[]
for path in (root/'outputs/pruning_v3').glob('*/summary.json'):
    s=json.loads(path.read_text())
    if not s.get('smoke'):
        summaries.append(s)
print(f'Completed and saved: {len(summaries)}/64')
for s in sorted(summaries,key=lambda x:x['provenance']['created_utc'])[-8:]:
    print(f"{s['tag']:30s} {s['predicted_mbti']:4s} PPL={s['language_eval']['perplexity']:12.3f} "
          f"legal={s['unconstrained_option_rate']:.1%}")
for name in ['sparse_worker.log','other_worker.log','parallel_driver.log']:
    path=root/'logs/pruning_v3'/name
    if path.exists():
        lines=path.read_text(errors='replace').splitlines()
        useful=[line for line in lines if line.startswith(('[start]','[prune]','[complete]','SWEEP','ALL_','Traceback'))]
        print(name+': '+(useful[-1] if useful else 'initializing'))
