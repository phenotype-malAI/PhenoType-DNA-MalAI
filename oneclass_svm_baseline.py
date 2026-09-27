"""
oneclass_svm_baseline.py — NEW

One-Class SVM per family, fit on frozen SupCon embeddings (no
retraining -- reuses an existing behaviour_encoder.pt + centroids the
same way cade_baseline.py does). Each family gets its own
sklearn.svm.OneClassSVM fit on that family's training embeddings; at
inference, a sample is first assigned to its nearest centroid (exactly
as attribute.py does), then that family's OneClassSVM.decision_function
score is used as the accept/reject signal in place of the fixed
cosine threshold.

nu is fixed at 0.05 (expected training-set outlier fraction; standard
default) rather than swept, since sweeping nu per family risks
overfitting the open-world result to the held-out set itself. The
decision boundary this produces is threshold-free by construction
(sign of decision_function), so there is no separate calibration step
the way there is for the cosine/energy/CADE/OpenMax rules above.

Usage:
    python oneclass_svm_baseline.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --encoder outputs/multi_seed/seed_0/behaviour_encoder.pt \
        --centroids outputs/multi_seed/seed_0/family_centroids.pt \
        --out_dir outputs/oneclass_svm_baseline/seed_0 \
        --seed 0 --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from model import BehaviourEncoder
from dataset import make_splits, IDX_TO_FAMILY

NUM_CLASSES = 5


@torch.no_grad()
def embed_all(model, tokens, device, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size)
    embs = []
    for (batch,) in loader:
        embs.append(model(batch.to(device)).cpu())
    return torch.cat(embs).numpy()


def fit_per_family_svms(train_embs, train_labels, nu=0.05):
    svms = {}
    for c in range(NUM_CLASSES):
        mask = (train_labels == c)
        svm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        svm.fit(train_embs[mask])
        svms[c] = svm
    return svms


def assign_and_score(embs, centroid_matrix, svms):
    """Nearest-centroid assignment, then that family's own SVM decision score."""
    sims = embs @ centroid_matrix.T
    assigned = sims.argmax(axis=1)
    scores = np.zeros(len(embs))
    for c in range(NUM_CLASSES):
        mask = (assigned == c)
        if mask.sum() > 0:
            scores[mask] = svms[c].decision_function(embs[mask])
    return assigned, scores  # score > 0 -> inlier (accept); score <= 0 -> outlier (reject)


def evaluate_split(embs, labels, centroid_matrix, svms):
    assigned, scores = assign_and_score(embs, centroid_matrix, svms)
    kept = scores > 0
    correct = int(((assigned == labels) & kept).sum())
    acc = correct / len(labels)
    return acc


def evaluate_openworld(embs, families, centroid_matrix, svms):
    assigned, scores = assign_and_score(embs, centroid_matrix, svms)
    rejected = scores <= 0
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
    centroid_matrix = torch.stack([centroids[i] for i in range(NUM_CLASSES)]).numpy()

    train_embs = embed_all(model, train_set.tokens, device)
    test_embs = embed_all(model, test_set.tokens, device)

    svms = fit_per_family_svms(train_embs, train_set.labels, nu=args.nu)

    test_acc = evaluate_split(test_embs, test_set.labels, centroid_matrix, svms)
    print(f"Closed-world test accuracy under One-Class SVM rule: {test_acc:.4f}")

    held_df = pd.read_csv(args.held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    held_tokens = held_df[tok_cols].values.astype('int64')
    held_embs = embed_all(model, held_tokens, device)
    families = held_df['family'].values

    ow_df = evaluate_openworld(held_embs, families, centroid_matrix, svms)
    print("\nOpen-world rejection under One-Class SVM rule:")
    print(ow_df.to_string(index=False))

    pd.DataFrame([{'Model': f'One-Class SVM (per-family, nu={args.nu})',
                    'Closed_World_Acc': round(test_acc, 4)}]
                 ).to_csv(out_dir / 'oneclass_svm_closed_world.csv', index=False)
    ow_df.to_csv(out_dir / 'oneclass_svm_openworld.csv', index=False)
    print(f"\nSaved -> {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='final_dna_v2.csv')
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--centroids', required=True)
    parser.add_argument('--out_dir', default='outputs/oneclass_svm_baseline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nu', type=float, default=0.05)
    args = parser.parse_args()
    run(args)
