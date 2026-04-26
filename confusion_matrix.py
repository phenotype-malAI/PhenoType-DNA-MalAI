"""
PHENO TYPE — confusion_matrix.py
Generates a publication-quality confusion matrix from the test set.

Usage:
    python confusion_matrix.py --csv final_dna_v2.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import make_splits, IDX_TO_FAMILY
from model import BehaviourEncoder
from attribute import load_model_and_centroids


@torch.no_grad()
def get_predictions(model, dataset, centroids, device, batch_size=16):
    model.eval()
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cent_matrix = torch.stack([centroids[i] for i in range(5)]).to(device)
    preds, trues = [], []
    for tokens, labels in loader:
        fp   = model(tokens.to(device))
        sims = fp @ cent_matrix.T
        pred = sims.argmax(dim=1).cpu()
        preds.append(pred)
        trues.append(labels)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def plot_confusion_matrix(preds, trues, save_path):
    family_names = [IDX_TO_FAMILY[i] for i in range(5)]
    cm = confusion_matrix(trues, preds, labels=list(range(5)))

    # Normalise rows (true class) to get recall per cell
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(family_names, color='white', fontsize=10, rotation=25, ha='right')
    ax.set_yticklabels(family_names, color='white', fontsize=10)
    ax.set_xlabel('Predicted', color='#aaaaaa', fontsize=12, labelpad=10)
    ax.set_ylabel('True',      color='#aaaaaa', fontsize=12, labelpad=10)
    ax.set_title('PHENO TYPE — Confusion Matrix (Test Set)',
                 color='white', fontsize=13, fontweight='bold', pad=14)

    for i in range(5):
        for j in range(5):
            val   = cm_norm[i, j]
            count = cm[i, j]
            color = 'white' if val < 0.5 else '#0d1117'
            ax.text(j, i, f'{val:.2f}\n({count})',
                    ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold' if i == j else 'normal')

    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Confusion matrix saved → {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',       required=True)
    parser.add_argument('--encoder',   default='outputs/behaviour_encoder.pt')
    parser.add_argument('--centroids', default='outputs/family_centroids.pt')
    parser.add_argument('--out_dir',   default='outputs')
    parser.add_argument('--device',    default='cpu')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    model, centroids, device = load_model_and_centroids(
        args.encoder, args.centroids, args.device
    )

    _, _, test_set = make_splits(args.csv, seed=args.seed)
    preds, trues   = get_predictions(model, test_set, centroids, device)

    plot_confusion_matrix(preds, trues, out_dir / 'confusion_matrix.png')

    # Print per-family accuracy
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(
        trues, preds,
        target_names=[IDX_TO_FAMILY[i] for i in range(5)],
        zero_division=0
    ))