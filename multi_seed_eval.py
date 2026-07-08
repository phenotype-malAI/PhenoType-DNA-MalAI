"""
multi_seed_eval.py — NEW

Runs train.py + eval_held_out.py across multiple seeds and reports
mean +/- std for the numbers that go in Table II (closed-world accuracy,
macro F1) and Table IV (open-world rejection rate). Addresses the
"single run, no statistical significance" gap.

Usage:
    python multi_seed_eval.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --seeds 0 1 2 3 4 \
        --epochs 100 --device cuda
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_one_seed(seed, args):
    out_dir = Path(args.out_dir) / f'seed_{seed}'
    print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")

    subprocess.run([
        sys.executable, 'train.py',
        '--csv', args.csv,
        '--out_dir', str(out_dir),
        '--device', args.device,
        '--epochs', str(args.epochs),
        '--seed', str(seed),
        '--threshold', '0.986',
    ], check=True)

    subprocess.run([
        sys.executable, 'eval_held_out.py',
        '--csv', args.held_out_csv,
        '--encoder', str(out_dir / 'behaviour_encoder.pt'),
        '--centroids', str(out_dir / 'family_centroids.pt'),
        '--out_dir', str(out_dir),
        '--device', args.device,
    ], check=True)

    with open(out_dir / 'test_report.json') as f:
        test_report = json.load(f)

    held_out_summary = pd.read_csv(out_dir / 'held_out_summary.csv')
    # eval_held_out.py's save_outputs() writes one row per held-out family
    # and no OVERALL row -- compute the overall rejection rate here instead.
    overall_unknown_rate = (
        held_out_summary['Flagged UNKNOWN'].sum()
        / held_out_summary['Total Samples'].sum() * 100
    )

    return {
        'seed': seed,
        'test_accuracy': test_report['test_accuracy'],
        'macro_f1': float(np.mean(list(test_report['per_family_f1'].values()))),
        'open_world_rejection_pct': float(overall_unknown_rate),
    }


def main(args):
    rows = [run_one_seed(s, args) for s in args.seeds]
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 70}\nSUMMARY across {len(args.seeds)} seeds\n{'=' * 70}")
    print(df.to_string(index=False))

    summary = {
        'n_seeds': len(args.seeds),
        'test_accuracy_mean': float(df['test_accuracy'].mean()),
        'test_accuracy_std': float(df['test_accuracy'].std()),
        'macro_f1_mean': float(df['macro_f1'].mean()),
        'macro_f1_std': float(df['macro_f1'].std()),
        'open_world_rejection_pct_mean': float(df['open_world_rejection_pct'].mean()),
        'open_world_rejection_pct_std': float(df['open_world_rejection_pct'].std()),
    }

    print("\nFor the paper:")
    print(f"  Closed-world accuracy: {summary['test_accuracy_mean'] * 100:.2f}% "
          f"+/- {summary['test_accuracy_std'] * 100:.2f}")
    print(f"  Macro F1:              {summary['macro_f1_mean']:.3f} "
          f"+/- {summary['macro_f1_std']:.3f}")
    print(f"  Open-world rejection:  {summary['open_world_rejection_pct_mean']:.1f}% "
          f"+/- {summary['open_world_rejection_pct_std']:.1f}")

    out_dir = Path(args.out_dir)
    df.to_csv(out_dir / 'multi_seed_results.csv', index=False)
    with open(out_dir / 'multi_seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_dir / 'multi_seed_results.csv'}, "
          f"{out_dir / 'multi_seed_summary.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='final_dna_v2.csv')
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--out_dir', default='outputs/multi_seed')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
