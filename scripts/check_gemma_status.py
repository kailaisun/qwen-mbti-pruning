"""Read-only, standard-library progress report for Gemma experiments."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default=str(ROOT / 'outputs/gemma4_12b_v3'))
    ap.add_argument('--logdir', default=str(ROOT / 'logs/gemma4_12b_v3'))
    args = ap.parse_args()
    complete = []
    for path in Path(args.outdir).glob('*/summary.json'):
        s = json.loads(path.read_text())
        if not s.get('smoke') and (path.parent / 'results.npz').is_file():
            complete.append(s)
    print(f'Gemma complete: {len(complete)}/64')
    for method in ['dense', 'magnitude', 'row_magnitude', 'random', 'wanda', 'sparsegpt']:
        print(method, sum(s['sweep_config']['method'] == method for s in complete))
    for log in sorted(Path(args.logdir).glob('worker_?.log')):
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            tail = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()
        lines = [line for line in tail if line.strip()]
        print(log.name, lines[-1] if lines else '(empty)')


if __name__ == '__main__':
    main()
