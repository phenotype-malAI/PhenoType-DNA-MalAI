"""
compute_open_world_roc.py — NEW

ROC / AUROC for the open-world known-vs-novel decision. Combines:
  - <out_dir>/test_scores.csv        (known test samples, is_known=1)   <- from train.py
  - <out_dir>/held_out_results.json  (novel samples)                    <- from eval_held_out.py

Addresses the "no ROC/PR curve or AUROC for the open-set task" gap:
you already log a best-centroid cosine score per sample in both files,
this just treats the score as a threshold-independent detector and
scores it properly, then marks where your deployed threshold (0.986)
sits on the curve.

Usage:
    python compute_open_world_roc.py \
        --test_scores outputs/batch_size64/test_scores.csv \
        --held_out_results outputs/batch_size64/held_out_results.json \
        --out_dir outputs/batch_size64
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
    known = pd.read_csv(args.test_scores)
    known_scores = known['best_score'].values
    known_labels = np.ones(len(known_scores))  # 1 = should be attributed (known)

    with open(args.held_out_results) as f:
        novel = json.load(f)
    novel_scores = np.array([r['best_score'] for r in novel])
    novel_labels = np.zeros(len(novel_scores))  # 0 = should be rejected (novel)

    scores = np.concatenate([known_scores, novel_scores])
    labels = np.concatenate([known_labels, novel_labels])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (known vs. novel, by best-centroid cosine score): {auroc:.4f}")

    # Locate where the actually-deployed threshold (0.986) sits on this curve.
    deployed = args.deployed_threshold
    idx = int(np.argmin(np.abs(thresholds - deployed)))
    print(f"At threshold={thresholds[idx]:.4f} (closest to deployed {deployed}): "
          f"TPR(known kept)={tpr[idx]:.4f}  FPR(novel wrongly kept)={fpr[idx]:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(fpr, tpr, color='#2a9d8f', linewidth=2, label=f'ROC (AUROC={auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1, label='Chance')
    ax.scatter([fpr[idx]], [tpr[idx]], color='#e63946', zorder=5,
               label=f'Deployed θ={deployed} (FPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f})')
    ax.set_xlabel('False Positive Rate (novel sample wrongly kept)', color='#aaaaaa')
    ax.set_ylabel('True Positive Rate (known sample correctly kept)', color='#aaaaaa')
    ax.set_title('Open-World Known-vs-Novel ROC', color='white', fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.legend(loc='lower right', fontsize=8, facecolor='#1c1c2e',
              edgecolor='#333333', labelcolor='white')
    plt.tight_layout()
    fig.savefig(out_dir / 'open_world_roc.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"ROC plot saved -> {out_dir / 'open_world_roc.png'}")

    result = {
        'auroc': float(auroc),
        'deployed_threshold': deployed,
        'tpr_at_deployed': float(tpr[idx]),
        'fpr_at_deployed': float(fpr[idx]),
        'n_known': int(len(known_scores)),
        'n_novel': int(len(novel_scores)),
    }
    with open(out_dir / 'open_world_auroc.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result JSON saved -> {out_dir / 'open_world_auroc.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_scores', required=True,
                         help='Path to test_scores.csv produced by the updated train.py')
    parser.add_argument('--held_out_results', required=True,
                         help='Path to held_out_results.json produced by eval_held_out.py')
    parser.add_argument('--out_dir', default='outputs')
    parser.add_argument('--deployed_threshold', type=float, default=0.986)
    main(parser.parse_args())
