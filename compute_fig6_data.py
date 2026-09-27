"""
compute_fig6_data.py — NEW

Extracts per-sample cosine similarity against ALL FIVE training centroids
(not just the best/argmax one) for each held-out novel family, then
regenerates Fig. 6 (score-distribution box plot + centroid-similarity
heatmap) so it matches the actual current run instead of the stale
pre-fix figure.

This is the one piece of data eval_held_out.py never saved: it only
keeps each sample's BEST score for the rejection decision, not its
similarity to every individual centroid.

Usage (run from the repo root, e.g. /content/repo in Colab):
    python compute_fig6_data.py \
        --held_out_csv data/held_out_families.csv \
        --encoder /content/drive/MyDrive/PhenoType/outputs/single_run/behaviour_encoder.pt \
        --centroids /content/drive/MyDrive/PhenoType/outputs/single_run/family_centroids.pt \
        --out_dir /content/drive/MyDrive/PhenoType/outputs/single_run \
        --device cuda

Outputs:
    held_out_full_scores.csv   — per-sample similarity to all 5 centroids
                                  (this is the raw data Fig 6 needs — keep
                                  it, it's also useful for other analyses)
    fig6_threshold_calibration.png  — regenerated Fig 6, both panels
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from model import BehaviourEncoder
from dataset import IDX_TO_FAMILY


@torch.no_grad()
def compute_full_scores(encoder_path, centroids_path, held_out_csv, device, batch_size=16):
    device = torch.device(device)
    model = BehaviourEncoder()
    model.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=False))
    model.to(device).eval()
    centroids = torch.load(centroids_path, map_location='cpu', weights_only=False)
    cent_matrix = torch.stack([centroids[i] for i in range(5)]).to(device)
    family_names = [IDX_TO_FAMILY[i] for i in range(5)]

    df = pd.read_csv(held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    tokens = torch.tensor(df[tok_cols].values.astype('int64'), dtype=torch.long)
    held_out_families = df['family'].values

    loader = DataLoader(TensorDataset(tokens), batch_size=batch_size)
    all_sims = []
    for (batch,) in loader:
        fp = model(batch.to(device))
        sims = (fp @ cent_matrix.T).cpu().numpy()  # (B, 5)
        all_sims.append(sims)
    all_sims = np.concatenate(all_sims, axis=0)  # (N, 5)

    rows = []
    for i in range(len(held_out_families)):
        row = {'held_out_family': held_out_families[i]}
        for j, fname in enumerate(family_names):
            row[f'sim_{fname}'] = round(float(all_sims[i, j]), 6)
        row['best_score'] = round(float(all_sims[i].max()), 6)
        row['best_centroid'] = family_names[int(all_sims[i].argmax())]
        rows.append(row)

    return pd.DataFrame(rows), family_names


def plot_fig6(scores_df, family_names, threshold, out_path):
    held_out_order = ['Amadey', 'Dacic', 'Qakbot', 'Remcos', 'Smokeloader']
    held_out_order = [f for f in held_out_order if f in scores_df['held_out_family'].unique()] \
        or sorted(scores_df['held_out_family'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0d1117')

    # ---- Panel (a): box plot of best-score distribution per held-out family ----
    ax = axes[0]
    ax.set_facecolor('#161b22')
    box_data = [scores_df.loc[scores_df['held_out_family'] == fam, 'best_score'].values
                for fam in held_out_order]
    bp = ax.boxplot(box_data, tick_labels=held_out_order, patch_artist=True,
                     boxprops=dict(facecolor='#2a9d8f', alpha=0.7, color='#2a9d8f'),
                     medianprops=dict(color='white'),
                     whiskerprops=dict(color='#aaaaaa'),
                     capprops=dict(color='#aaaaaa'),
                     flierprops=dict(markeredgecolor='#aaaaaa', markersize=4))
    ax.axhline(threshold, color='#e63946', linestyle='--', linewidth=1.5,
               label=f'θ = {threshold}')
    ax.set_ylabel('Best Cosine Similarity Score', color='#aaaaaa')
    ax.set_title('(a) Score Distribution per Held-Out Family', color='white', fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.legend(facecolor='#1c1c2e', edgecolor='#333', labelcolor='white', fontsize=9)

    for i, fam in enumerate(held_out_order):
        rej_pct = (scores_df.loc[scores_df['held_out_family'] == fam, 'best_score'] < threshold).mean() * 100
        ax.text(i + 1, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'{rej_pct:.0f}%\nrej.', color='#aaaaaa', ha='center', fontsize=8)

    # ---- Panel (b): heatmap of avg similarity vs each of the 5 training centroids ----
    ax2 = axes[1]
    heat = np.zeros((len(held_out_order), len(family_names)))
    for i, fam in enumerate(held_out_order):
        sub = scores_df.loc[scores_df['held_out_family'] == fam]
        for j, tf in enumerate(family_names):
            heat[i, j] = sub[f'sim_{tf}'].mean()

    im = ax2.imshow(heat, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    ax2.set_xticks(range(len(family_names)))
    ax2.set_xticklabels(family_names, color='#eaeaea', rotation=30, ha='right')
    ax2.set_yticks(range(len(held_out_order)))
    ax2.set_yticklabels(held_out_order, color='#eaeaea')
    ax2.set_xlabel('Training Family Centroid', color='#aaaaaa')
    ax2.set_ylabel('Held-Out Family', color='#aaaaaa')
    ax2.set_title('(b) Avg Cosine Similarity vs Training Centroids', color='white', fontweight='bold')
    for i in range(len(held_out_order)):
        for j in range(len(family_names)):
            txt = ax2.text(j, i, f'{heat[i, j]:.3f}', ha='center', va='center',
                            color='white', fontsize=9, fontweight='bold')
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
    cbar = fig.colorbar(im, ax=ax2)
    cbar.ax.yaxis.set_tick_params(color='#aaaaaa')
    plt.setp(cbar.ax.get_yticklabels(), color='#aaaaaa')
    cbar.set_label('Avg Cosine Sim.', color='#aaaaaa')
    fig.patch.set_facecolor('#0d1117')
    ax2.set_facecolor('#0d1117')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Fig 6 saved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--centroids', required=True)
    parser.add_argument('--out_dir', default='outputs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', type=float, default=0.986)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    scores_df, family_names = compute_full_scores(
        args.encoder, args.centroids, args.held_out_csv, device
    )

    csv_path = out_dir / 'held_out_full_scores.csv'
    scores_df.to_csv(csv_path, index=False)
    print(f"Per-sample, per-centroid scores saved -> {csv_path}")
    print(scores_df.groupby('held_out_family')['best_score'].describe())

    plot_fig6(scores_df, family_names, args.threshold,
              out_dir / 'fig6_threshold_calibration.png')
