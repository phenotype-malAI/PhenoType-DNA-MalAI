"""
openmax_baseline.py — NEW

Implements OpenMax (Bendale & Boult, CVPR 2016) on the existing
CrossEntropy classifier's logits -- no retraining needed, this loads
the checkpoint ablation.py already saved (outputs/ablation/crossentropy_model.pt).

WHAT THIS IS: OpenMax's actual algorithm -- per-class Mean Activation
Vector (MAV) in LOGIT space, Weibull tail-fitting on distances from
correctly-classified training samples to their class's MAV, and the
logit-recalibration + K+1-way softmax (K known classes + 1 "unknown"
pseudo-class) described in the paper.

WHAT THIS IS NOT: the original paper's libmr (meta-recognition)
library for Weibull fitting. libmr is effectively unmaintained and
does not build on modern Python, so this uses scipy.stats.weibull_min
fit on the tail distances instead -- a well known, widely used
substitution in re-implementations of OpenMax. Report this explicitly
if it goes in the paper.

Algorithm (per class c, computed once from training data):
  1. MAV_c = mean logit vector over training samples with true label
     == predicted label == c (correctly classified only).
  2. dist_i = Euclidean distance between sample i's logit vector and
     MAV_c, for every correctly-classified training sample in class c.
  3. Fit Weibull (shape, scale; loc fixed at 0) on the largest
     `tailsize` distances (the "tail" of the distance distribution).

At inference, for a new sample's logit vector v = (v_1 ... v_K):
  1. Rank classes by logit magnitude, descending -> ranks r_1..r_K.
  2. For the top `alpha_rank` classes (here alpha_rank = K = 5, all
     of them, since there are only 5 classes), compute
     w_score_c = weibull_cdf(dist(v, MAV_c); shape_c, scale_c)
     omega_c   = ((alpha_rank - rank_c) / alpha_rank) * w_score_c
  3. Revised known-class logit:  v_hat_c = v_c * (1 - omega_c)
  4. Unknown pseudo-logit:       v_hat_0 = sum_c ( v_c * omega_c )
  5. Softmax over [v_hat_0, v_hat_1, ..., v_hat_K]. If argmax is
     v_hat_0 (index 0, "unknown"), reject as UNKNOWN.

Usage (run from the repo root; needs BOTH the encoder is not required
here, only the CE classifier checkpoint from ablation.py):
    python openmax_baseline.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --ce_checkpoint outputs/ablation/crossentropy_model.pt \
        --out_dir outputs/openmax_baseline/seed_0 \
        --seed 0 --device cuda

NOTE: --ce_checkpoint above is trained once, not per-seed (ablation.py
was run once, not across seeds, unless you've since run
multi_seed_ablation.py). If you only have ONE CrossEntropy checkpoint,
that's fine -- run this script 5 times with --seed 0..4 anyway; the
split changes per seed (train/val/test partition), which is what the
MAV/Weibull fitting and threshold calibration below actually depend
on, even though the classifier weights themselves stay fixed. This
gives you 5 data points reflecting split variance, not full retraining
variance. If you'd rather have both sources of variance, wait for
multi_seed_ablation.py to produce per-seed CE checkpoints first.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import weibull_min
from torch.utils.data import DataLoader, TensorDataset

from ablation import BehaviourClassifier
from dataset import make_splits, IDX_TO_FAMILY

NUM_CLASSES = 5


@torch.no_grad()
def get_logits(model, tokens, device, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size)
    out = []
    for (batch,) in loader:
        out.append(model(batch.to(device)).cpu())
    return torch.cat(out).numpy()


def fit_openmax(train_logits, train_labels, tailsize=20):
    """Returns per-class MAV (K, K) and fitted Weibull (shape, scale) per class."""
    preds = train_logits.argmax(axis=1)
    mavs = np.zeros((NUM_CLASSES, NUM_CLASSES))
    shapes = np.zeros(NUM_CLASSES)
    scales = np.zeros(NUM_CLASSES)

    for c in range(NUM_CLASSES):
        correct_mask = (train_labels == c) & (preds == c)
        class_logits = train_logits[correct_mask]
        if len(class_logits) < 3:
            # Not enough correctly-classified samples to fit a tail -- fall back
            # to using all samples of that class regardless of correctness.
            class_logits = train_logits[train_labels == c]
        mav = class_logits.mean(axis=0)
        mavs[c] = mav

        dists = np.linalg.norm(class_logits - mav, axis=1)
        tail = np.sort(dists)[-min(tailsize, len(dists)):]
        tail = tail[tail > 1e-8]  # weibull_min.fit chokes on exact zeros
        if len(tail) < 3:
            shapes[c], scales[c] = 1.0, 1.0  # degenerate fallback
            continue
        shape, _, scale = weibull_min.fit(tail, floc=0)
        shapes[c], scales[c] = shape, scale

    return mavs, shapes, scales


def openmax_recalibrate(logits, mavs, shapes, scales, alpha_rank=NUM_CLASSES):
    """logits: (N, K). Returns (N, K+1) recalibrated probabilities,
    where column 0 is the 'unknown' pseudo-class."""
    N = logits.shape[0]
    dists = np.linalg.norm(logits[:, None, :] - mavs[None, :, :], axis=2)  # (N, K)
    w_scores = np.zeros_like(dists)
    for c in range(NUM_CLASSES):
        w_scores[:, c] = weibull_min.cdf(dists[:, c], shapes[c], loc=0, scale=scales[c])

    ranks = np.argsort(-logits, axis=1)  # descending logit order, per sample
    rank_of = np.zeros_like(ranks)
    for i in range(N):
        for pos, c in enumerate(ranks[i]):
            rank_of[i, c] = pos  # 0 = highest logit

    omega = np.zeros_like(logits)
    for i in range(N):
        for c in range(NUM_CLASSES):
            if rank_of[i, c] < alpha_rank:
                omega[i, c] = ((alpha_rank - rank_of[i, c]) / alpha_rank) * w_scores[i, c]

    v_hat_known = logits * (1 - omega)
    v_hat_unknown = (logits * omega).sum(axis=1, keepdims=True)
    v_hat = np.concatenate([v_hat_unknown, v_hat_known], axis=1)  # (N, K+1), col 0 = unknown

    exp_v = np.exp(v_hat - v_hat.max(axis=1, keepdims=True))
    probs = exp_v / exp_v.sum(axis=1, keepdims=True)
    return probs  # (N, K+1)


def calibrate_alpha_rank_unused():
    pass  # placeholder -- alpha_rank fixed at NUM_CLASSES (5) here, no free threshold to sweep


def evaluate_closed_world(probs, true_labels):
    """probs: (N, K+1), col 0 = unknown. Predicted class = argmax - 1
    (shift back to 0..K-1 label space), 'rejected' if argmax == 0."""
    pred_col = probs.argmax(axis=1)
    rejected = (pred_col == 0)
    pred_label = pred_col - 1  # -1 for rejected rows, but they don't count as correct anyway
    correct = (~rejected) & (pred_label == true_labels)
    acc = correct.sum() / len(true_labels)
    return acc, rejected


def evaluate_openworld(probs, families):
    pred_col = probs.argmax(axis=1)
    rejected = (pred_col == 0)
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

    model = BehaviourClassifier()
    model.load_state_dict(torch.load(args.ce_checkpoint, map_location='cpu', weights_only=False))
    model.to(device).eval()

    train_logits = get_logits(model, train_set.tokens, device)
    val_logits = get_logits(model, val_set.tokens, device)
    test_logits = get_logits(model, test_set.tokens, device)

    mavs, shapes, scales = fit_openmax(train_logits, train_set.labels, tailsize=args.tailsize)
    print(f"Weibull shapes: {shapes.round(3)}")
    print(f"Weibull scales: {scales.round(3)}")

    val_probs = openmax_recalibrate(val_logits, mavs, shapes, scales)
    val_acc, _ = evaluate_closed_world(val_probs, val_set.labels)
    print(f"Validation closed-world accuracy under OpenMax: {val_acc:.4f}")

    test_probs = openmax_recalibrate(test_logits, mavs, shapes, scales)
    test_acc, _ = evaluate_closed_world(test_probs, test_set.labels)
    print(f"Test closed-world accuracy under OpenMax: {test_acc:.4f}")

    held_df = pd.read_csv(args.held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    held_tokens = held_df[tok_cols].values.astype('int64')
    held_logits = get_logits(model, held_tokens, device)
    held_probs = openmax_recalibrate(held_logits, mavs, shapes, scales)
    families = held_df['family'].values

    ow_df = evaluate_openworld(held_probs, families)
    print("\nOpen-world rejection under OpenMax:")
    print(ow_df.to_string(index=False))

    pd.DataFrame([{'Model': 'OpenMax (Weibull-tail on CE logits)',
                    'Closed_World_Acc': round(test_acc, 4)}]
                 ).to_csv(out_dir / 'openmax_closed_world.csv', index=False)
    ow_df.to_csv(out_dir / 'openmax_openworld.csv', index=False)
    print(f"\nSaved -> {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='final_dna_v2.csv')
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--ce_checkpoint', required=True)
    parser.add_argument('--out_dir', default='outputs/openmax_baseline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tailsize', type=int, default=20,
                        help='Number of largest per-class distances used for Weibull fitting')
    args = parser.parse_args()
    run(args)
