"""
PHENO TYPE — visualise.py
Generates t-SNE cluster plot of test-set fingerprints coloured by family.
Also saves embeddings CSV for the dashboard map view.

Usage:
    python visualise.py --csv final_dna_v2.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import make_splits, IDX_TO_FAMILY
from model import BehaviourEncoder

FAMILY_COLORS = {
    'AgentTesla': '#e63946',
    'Formbook':   '#2a9d8f',
    'Lokibot':    '#e9c46a',
    'Redline':    '#a8dadc',
    'njRAT':      '#f4a261',
}


@torch.no_grad()
def extract_fingerprints(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    fps, lbls = [], []
    for tokens, labels in loader:
        fp = model(tokens.to(device)).cpu()
        fps.append(fp)
        lbls.append(labels)
    return torch.cat(fps).numpy(), torch.cat(lbls).numpy()


def plot_tsne(fingerprints, labels, save_path, perplexity=30, n_iter=1000, seed=42):
    print(f"Running t-SNE on {len(fingerprints)} samples …")
    import sklearn
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init='pca',
        learning_rate='auto',
    )
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 5):
        tsne_kwargs['max_iter'] = n_iter
    else:
        tsne_kwargs['n_iter'] = n_iter
    tsne = TSNE(**tsne_kwargs)
    proj = tsne.fit_transform(fingerprints)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    for fam_name, color in FAMILY_COLORS.items():
        idx = [i for i, l in enumerate(labels) if IDX_TO_FAMILY[l] == fam_name]
        if not idx:
            continue
        ax.scatter(proj[idx, 0], proj[idx, 1], s=18, c=color,
                   alpha=0.75, linewidths=0, label=fam_name)

    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    leg = ax.legend(handles=patches, facecolor='#1c1c2e', edgecolor='#333333',
                    labelcolor='white', fontsize=11, title='Malware Family',
                    title_fontsize=11, markerscale=1.5, framealpha=0.9)
    leg.get_title().set_color('white')

    ax.set_title('PHENO TYPE — Behavioural Fingerprint Clusters (t-SNE)',
                 color='white', fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('t-SNE dim 1', color='#888888', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', color='#888888', fontsize=10)
    ax.tick_params(colors='#444444')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"t-SNE plot saved → {save_path}")
    plt.close(fig)
    return proj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',          required=True)
    parser.add_argument('--encoder',      default='outputs/behaviour_encoder.pt')
    parser.add_argument('--centroids',    default='outputs/family_centroids.pt')
    parser.add_argument('--out_dir',      default='outputs')
    parser.add_argument('--split',        default='test', choices=['train','val','test','all'])
    parser.add_argument('--device',       default='cpu')
    parser.add_argument('--perplexity',   type=int,   default=30)
    parser.add_argument('--max_iter',       type=int,   default=1000)
    parser.add_argument('--seed',         type=int,   default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BehaviourEncoder()
    model.load_state_dict(torch.load(args.encoder, map_location='cpu', weights_only=False))
    model.to(device).eval()

    train_set, val_set, test_set = make_splits(args.csv, seed=args.seed)
    dataset_map = {'train': train_set, 'val': val_set, 'test': test_set}

    if args.split == 'all':
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([train_set, val_set, test_set])
        all_labels = np.concatenate([train_set.labels, val_set.labels, test_set.labels])
    else:
        dataset = dataset_map[args.split]
        all_labels = dataset.labels

    fingerprints, labels = extract_fingerprints(model, dataset, device)

    proj = plot_tsne(fingerprints, labels,
                     save_path=str(out_dir / 'tsne_clusters.png'),
                     perplexity=args.perplexity,
                     n_iter=args.max_iter,
                     seed=args.seed)

    # Save embeddings CSV for dashboard
    rows = []
    for i in range(len(labels)):
        rows.append({
            'tsne_x':  float(proj[i, 0]),
            'tsne_y':  float(proj[i, 1]),
            'label':   int(labels[i]),
            'family':  IDX_TO_FAMILY[int(labels[i])],
        })
    emb_path = out_dir / 'test_embeddings.csv'
    pd.DataFrame(rows).to_csv(emb_path, index=False)
    print(f"Embeddings saved → {emb_path}")