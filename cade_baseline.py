"""
cade_baseline.py — NEW

Implements CADE's (Yang et al., USENIX Security 2021) actual open-set
decision rule on PHENOTYPE's already-trained SupCon embeddings: no
retraining, no autoencoder -- this reuses your trained BehaviourEncoder
and family centroids exactly as attribute.py does, and swaps out the
fixed global cosine threshold for CADE's per-class MAD-normalised
anomaly score.

WHAT THIS IS: a faithful re-implementation of CADE's *thresholding
rule*, applied to PHENOTYPE's own contrastive embedding space, so the
comparison isolates "does CADE's decision rule beat a fixed global
cosine threshold on the SAME embeddings" rather than conflating encoder
architecture differences with thresholding-rule differences.

WHAT THIS IS NOT: a reproduction of CADE's own contrastive autoencoder
(reconstruction + contrastive loss). That is a different encoder
entirely; report this distinction explicitly in the paper.

CADE's rule (MAD-based drift detection):
  For each training family c:
    d_i = Euclidean distance of training sample i's embedding to
          family c's centroid
    median_c = median(d_i for i in family c)
    MAD_c    = median(|d_i - median_c|)
  For a new sample assigned (by nearest centroid) to family c:
    anomaly_score = |d_new - median_c| / (b * MAD_c)
    where b = 1.4826 (standard constant making MAD ~ std under
    normality; Iglewicz & Hoaglin's modified z-score convention)
  If anomaly_score > threshold -> UNKNOWN, else -> attributed to c.

Threshold calibrated on the validation split only (same protocol as
theta=0.986 in the main paper), then applied once to test + held-out.

IMPORTANT — matching the split to the checkpoint:
  make_splits(csv, seed=N) must be called with the SAME seed that was
  passed to train.py when that checkpoint was produced, or the
  train/val/test partition won't match the model that was trained on
  it. In this repo:
    outputs/single_run/          -> was trained with the default seed (42)
    outputs/multi_seed/seed_0/   -> --seed 0
    outputs/multi_seed/seed_1/   -> --seed 1
    outputs/multi_seed/seed_2/   -> --seed 2
    outputs/multi_seed/seed_3/   -> --seed 3
    outputs/multi_seed/seed_4/   -> --seed 4
  Pass --seed accordingly for each run below.

Usage (run from the repo root, e.g. /content/repo in Colab):
    python cade_baseline.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --encoder outputs/multi_seed/seed_0/behaviour_encoder.pt \
        --centroids outputs/multi_seed/seed_0/family_centroids.pt \
        --out_dir outputs/cade_baseline/seed_0 \
        --seed 0 --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import BehaviourEncoder
from dataset import make_splits, IDX_TO_FAMILY

MAD_CONST = 1.4826  # standard constant, MAD ~= std under normality


@torch.no_grad()
def embed_all(model, tokens, device, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size)
    embs = []
    for (batch,) in loader:
        embs.append(model(batch.to(device)).cpu())
    return torch.cat(embs).numpy()


def euclidean_to_centroids(embs, centroid_matrix):
    # embs: (N, D), centroid_matrix: (K, D), both L2-normalised
    diffs = embs[:, None, :] - centroid_matrix[None, :, :]
    return np.linalg.norm(diffs, axis=2)  # (N, K)


def fit_cade_stats(train_embs, train_labels, centroid_matrix, num_classes=5):
    """Per-family median distance and MAD, computed on TRAINING embeddings only."""
    dists = euclidean_to_centroids(train_embs, centroid_matrix)  # (N, K)
    medians = np.zeros(num_classes)
    mads = np.zeros(num_classes)
    for c in range(num_classes):
        mask = (train_labels == c)
        d_c = dists[mask, c]
        med = np.median(d_c)
        mad = np.median(np.abs(d_c - med))
        medians[c] = med
        mads[c] = max(mad, 1e-6)  # avoid divide-by-zero on tight clusters
    return medians, mads


def anomaly_scores(embs, centroid_matrix, medians, mads):
    dists = euclidean_to_centroids(embs, centroid_matrix)  # (N, K)
    assigned = dists.argmin(axis=1)
    d_assigned = dists[np.arange(len(embs)), assigned]
    med_assigned = medians[assigned]
    mad_assigned = mads[assigned]
    scores = np.abs(d_assigned - med_assigned) / (MAD_CONST * mad_assigned)
    return scores, assigned


def calibrate_threshold(val_embs, val_labels, centroid_matrix, medians, mads,
                         sweep=np.arange(1.0, 8.01, 0.25)):
    """Pick the anomaly-score threshold that maximises validation accuracy
    under the same 'reject below threshold -> forced to reassign' logic
    used for theta=0.986 in the main paper -- here 'reject' means
    anomaly_score EXCEEDS threshold."""
    scores, assigned = anomaly_scores(val_embs, centroid_matrix, medians, mads)
    best_thr, best_acc = sweep[0], -1.0
    for thr in sweep:
        kept = scores <= thr
        correct = int(((assigned == val_labels) & kept).sum())
        acc = correct / len(val_labels)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


def evaluate_split(embs, labels, centroid_matrix, medians, mads, threshold):
    scores, assigned = anomaly_scores(embs, centroid_matrix, medians, mads)
    kept = scores <= threshold
    correct = int(((assigned == labels) & kept).sum())
    acc = correct / len(labels)
    return acc, kept, assigned, scores


def evaluate_openworld(embs, families, centroid_matrix, medians, mads, threshold):
    scores, assigned = anomaly_scores(embs, centroid_matrix, medians, mads)
    rejected = scores > threshold
    rows = []
    for fam in sorted(set(families)):
        mask = (families == fam)
        n = int(mask.sum())
        rej = int(rejected[mask].sum())
        rows.append({'Family': fam, 'Total': n, 'Rejected_UNKNOWN': rej,
                     'Rejection_Rate_%': round(100 * rej / n, 1) if n else 0.0})
    overall_rej = int(rejected.sum())
    rows.append({'Family': 'OVERALL', 'Total': len(rejected),
                 'Rejected_UNKNOWN': overall_rej,
                 'Rejection_Rate_%': round(100 * overall_rej / len(rejected), 1)})
    return pd.DataFrame(rows)


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = make_splits(args.csv, seed=args.seed)

    model = BehaviourEncoder()
    model.load_state_dict(torch.load(args.encoder, map_location='cpu', weights_only=False))
    model.to(device).eval()

    centroids = torch.load(args.centroids, map_location='cpu', weights_only=False)
    centroid_matrix = torch.stack([centroids[i] for i in range(5)]).numpy()

    train_embs = embed_all(model, train_set.tokens, device)
    val_embs = embed_all(model, val_set.tokens, device)
    test_embs = embed_all(model, test_set.tokens, device)

    medians, mads = fit_cade_stats(train_embs, train_set.labels, centroid_matrix)
    print(f"Per-family medians: {dict(zip(IDX_TO_FAMILY.values(), medians.round(4)))}")
    print(f"Per-family MADs:    {dict(zip(IDX_TO_FAMILY.values(), mads.round(4)))}")

    threshold, val_acc = calibrate_threshold(val_embs, val_set.labels, centroid_matrix, medians, mads)
    print(f"Calibrated threshold (validation): {threshold:.2f}  (val acc under this rule: {val_acc:.4f})")

    test_acc, _, _, _ = evaluate_split(test_embs, test_set.labels, centroid_matrix, medians, mads, threshold)
    print(f"Closed-world test accuracy under CADE rule: {test_acc:.4f}")

    held_df = pd.read_csv(args.held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    held_tokens = held_df[tok_cols].values.astype('int64')
    held_embs = embed_all(model, held_tokens, device)
    families = held_df['family'].values

    ow_df = evaluate_openworld(held_embs, families, centroid_matrix, medians, mads, threshold)
    print("\nOpen-world rejection under CADE rule:")
    print(ow_df.to_string(index=False))

    pd.DataFrame([{'Model': 'CADE-style (MAD-normalised centroid distance)',
                    'Threshold': threshold, 'Closed_World_Acc': round(test_acc, 4)}]
                 ).to_csv(out_dir / 'cade_closed_world.csv', index=False)
    ow_df.to_csv(out_dir / 'cade_openworld.csv', index=False)
    print(f"\nSaved -> {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='final_dna_v2.csv')
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--centroids', required=True)
    parser.add_argument('--out_dir', default='outputs/cade_baseline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)
