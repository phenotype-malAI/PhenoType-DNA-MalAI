"""
multi_seed_ablation.py — NEW

Runs ablation.py across multiple seeds and reports mean +/- std for
Table IV, the same way multi_seed_eval.py already does for Table II/III.
Addresses the "Table IV is the one number held to a lower rigor bar"
gap: ablation.py is fully seeded (torch.manual_seed + np.random.seed +
a seeded train/val/test split) but GPU training is not bit-reproducible
across sessions even with a fixed seed -- see the CHECKPOINT_RERUN
file this replaces. Averaging across explicit seeds 0-4 is the correct
fix, not chasing bit-exact determinism.

Each seed's ablation.py run writes to its own out_dir (ablation.py
always uses fixed filenames within out_dir, so seeds MUST NOT share a
directory or they'll overwrite each other -- see run_one_seed below).

Usage:
    python multi_seed_ablation.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --seeds 0 1 2 3 4 \
        --epochs 50 --device cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAMES = [
    'Transformer+SupCon\n(Ours)',
    'Transformer+CrossEntropy',
    'Transformer+MeanPool',
    'TF-IDF+LogReg\n(Baseline)',
]
FAMILIES = ['AgentTesla', 'Formbook', 'Lokibot', 'Redline', 'njRAT']


def run_one_seed(seed, args):
    out_dir = Path(args.out_dir) / f'seed_{seed}'
    print(f"\n{'=' * 70}\nABLATION — SEED {seed}\n{'=' * 70}")

    subprocess.run([
        sys.executable, 'ablation.py',
        '--csv', args.csv,
        '--held_out_csv', args.held_out_csv,
        '--out_dir', str(out_dir),
        '--device', args.device,
        '--epochs', str(args.epochs),
        '--seed', str(seed),
    ], check=True)

    df = pd.read_csv(out_dir / 'ablation_results.csv')
    df.insert(0, 'seed', seed)
    return df


def main(args):
    all_rows = [run_one_seed(s, args) for s in args.seeds]
    combined = pd.concat(all_rows, ignore_index=True)

    out_dir = Path(args.out_dir)
    combined.to_csv(out_dir / 'multi_seed_ablation_raw.csv', index=False)
    print(f"\nRaw per-seed rows saved -> {out_dir / 'multi_seed_ablation_raw.csv'}")

    # -- Aggregate mean +/- std per model, for Table IV --
    summary_rows = []
    for model_name in combined['Model'].unique():
        sub = combined[combined['Model'] == model_name]
        row = {'Model': model_name.replace('\n', ' '), 'n_seeds': len(sub)}
        row['Accuracy_mean'] = round(sub['Accuracy'].mean(), 4)
        row['Accuracy_std'] = round(sub['Accuracy'].std(), 4)
        for fam in FAMILIES:
            col = f'F1_{fam}'
            if col in sub.columns:
                row[f'{col}_mean'] = round(sub[col].mean(), 4)
                row[f'{col}_std'] = round(sub[col].std(), 4)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / 'multi_seed_ablation_summary.csv', index=False)

    print(f"\n{'=' * 70}\nSUMMARY across {len(args.seeds)} seeds (for Table IV)\n{'=' * 70}")
    for _, row in summary.iterrows():
        print(f"\n{row['Model']}  (n={row['n_seeds']})")
        print(f"  Accuracy: {row['Accuracy_mean'] * 100:.2f}% +/- {row['Accuracy_std'] * 100:.2f}")
        for fam in FAMILIES:
            mcol, scol = f'F1_{fam}_mean', f'F1_{fam}_std'
            if mcol in row:
                print(f"  F1[{fam}]: {row[mcol]:.3f} +/- {row[scol]:.3f}")

    print(f"\nSaved -> {out_dir / 'multi_seed_ablation_summary.csv'}")
    print("\nUse this table in place of the single-run Table IV.")
    print("Delete/ignore ablation_results_CHECKPOINT_RERUN_do_not_cite.csv now.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='final_dna_v2.csv')
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--out_dir', default='outputs/ablation_multi_seed')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
