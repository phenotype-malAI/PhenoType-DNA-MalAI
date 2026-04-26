"""
PHENO TYPE — eval_held_out.py
Open-world evaluation: run attribution on families the model has never seen.
Reports UNKNOWN rate, false attribution breakdown, and average similarity scores.

Usage (from your project root):
    python eval_held_out.py --csv held_out_families.csv

Outputs:
    - Summary table printed to console
    - outputs/held_out_results.json   (full per-sample results)
    - outputs/held_out_summary.csv    (table for the paper)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from attribute import load_model_and_centroids, attribute, THRESHOLD
from dataset import IDX_TO_FAMILY


def evaluate(csv_path, encoder_path, centroids_path, device_str, threshold):
    df = pd.read_csv(csv_path)
    tok_cols = [f'tok_{i}' for i in range(1200)]

    # Verify columns
    missing = [c for c in tok_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing token columns. First missing: {missing[0]}")

    model, centroids, device = load_model_and_centroids(
        encoder_path, centroids_path, device_str
    )

    held_out_families = sorted(df['family'].unique())
    print(f"\nHeld-out families detected: {held_out_families}")
    print(f"Total samples: {len(df)}")
    print(f"Attribution threshold: {threshold}\n")

    # Per-family accumulators
    stats = {
        fam: {
            'total': 0,
            'unknown': 0,
            'attributed': 0,
            'attributed_to': defaultdict(int),   # which known family it was wrongly sent to
            'sim_scores': [],                     # best cosine sim score for every sample
        }
        for fam in held_out_families
    }

    all_results = []

    for _, row in df.iterrows():
        fam = row['family']
        sha = str(row['sha256'])[:16]
        tokens = torch.tensor(row[tok_cols].values.astype('int64'), dtype=torch.long)

        pred_family, best_score, all_scores, _ = attribute(
            tokens, model, centroids, device, threshold=threshold
        )

        stats[fam]['total'] += 1
        stats[fam]['sim_scores'].append(best_score)

        if pred_family == 'UNKNOWN':
            stats[fam]['unknown'] += 1
        else:
            stats[fam]['attributed'] += 1
            stats[fam]['attributed_to'][pred_family] += 1

        all_results.append({
            'true_family': fam,
            'sha256_prefix': sha,
            'predicted': pred_family,
            'best_score': round(best_score, 6),
            'all_scores': {k: round(v, 6) for k, v in all_scores.items()},
        })

    return stats, all_results, held_out_families


def print_report(stats, held_out_families, threshold):
    sep = '═' * 80

    print(sep)
    print("  OPEN-WORLD EVALUATION — Held-Out Family Rejection Report")
    print(f"  Threshold: {threshold}")
    print(sep)

    # Main summary table
    print(f"\n{'Family':<14} {'Total':>6} {'UNKNOWN':>8} {'UNKNOWN%':>9} "
          f"{'Attributed':>11} {'Avg Score':>10}")
    print('─' * 62)

    overall_total = 0
    overall_unknown = 0

    for fam in held_out_families:
        s = stats[fam]
        n = s['total']
        unk = s['unknown']
        attr = s['attributed']
        avg_score = sum(s['sim_scores']) / len(s['sim_scores']) if s['sim_scores'] else 0
        pct = 100 * unk / n if n > 0 else 0

        print(f"{fam:<14} {n:>6} {unk:>8} {pct:>8.1f}% {attr:>11} {avg_score:>10.4f}")
        overall_total += n
        overall_unknown += unk

    print('─' * 62)
    overall_pct = 100 * overall_unknown / overall_total if overall_total > 0 else 0
    print(f"{'TOTAL':<14} {overall_total:>6} {overall_unknown:>8} "
          f"{overall_pct:>8.1f}% {overall_total-overall_unknown:>11}")

    # Attribution breakdown for samples that were NOT flagged UNKNOWN
    print(f"\n{'─'*62}")
    print("  False Attribution Breakdown (samples that slipped through as known family)")
    print(f"{'─'*62}")

    any_slipped = False
    for fam in held_out_families:
        s = stats[fam]
        if s['attributed'] > 0:
            any_slipped = True
            print(f"\n  {fam} ({s['attributed']} samples attributed to a known family):")
            for known_fam, count in sorted(s['attributed_to'].items(),
                                           key=lambda x: -x[1]):
                pct = 100 * count / s['attributed']
                print(f"    -> {known_fam:<14} {count:>3} samples  ({pct:.1f}%)")

    if not any_slipped:
        print("  None — all held-out samples correctly flagged UNKNOWN.")

    print(f"\n{sep}")
    print(f"  Overall UNKNOWN rate: {overall_pct:.1f}%  "
          f"({overall_unknown}/{overall_total} samples correctly rejected)")
    print(sep)


def save_outputs(stats, all_results, held_out_families, threshold, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full per-sample JSON
    json_path = out_dir / 'held_out_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved -> {json_path}")

    # Summary CSV for the paper table
    rows = []
    for fam in held_out_families:
        s = stats[fam]
        n = s['total']
        unk = s['unknown']
        attr = s['attributed']
        avg_score = round(sum(s['sim_scores']) / len(s['sim_scores']), 4) if s['sim_scores'] else 0
        unk_pct = round(100 * unk / n, 1) if n > 0 else 0

        # Most common false attribution (if any)
        if s['attributed_to']:
            top_false = max(s['attributed_to'], key=s['attributed_to'].get)
            top_false_n = s['attributed_to'][top_false]
            false_attr_str = f"{top_false} ({top_false_n})"
        else:
            false_attr_str = "None"

        rows.append({
            'Held-Out Family': fam,
            'Total Samples': n,
            'Flagged UNKNOWN': unk,
            'UNKNOWN Rate (%)': unk_pct,
            'Incorrectly Attributed': attr,
            'Primary False Attribution': false_attr_str,
            'Avg Cosine Similarity': avg_score,
        })

    csv_path = out_dir / 'held_out_summary.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Summary CSV saved    -> {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',       required=True,
                        help='Path to held-out families CSV')
    parser.add_argument('--encoder',   default='outputs/batch_size64/behaviour_encoder.pt')
    parser.add_argument('--centroids', default='outputs/batch_size64/family_centroids.pt')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override threshold (default: uses THRESHOLD from attribute.py)')
    parser.add_argument('--device',    default='cpu')
    parser.add_argument('--out_dir',   default='outputs/batch_size64')
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else THRESHOLD

    stats, all_results, held_out_families = evaluate(
        args.csv, args.encoder, args.centroids, args.device, threshold
    )

    print_report(stats, held_out_families, threshold)
    save_outputs(stats, all_results, held_out_families, threshold, args.out_dir)