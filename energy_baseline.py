"""
energy_baseline.py — NEW

Energy-based out-of-distribution score (Liu et al., NeurIPS 2020,
"Energy-based Out-of-distribution Detection"), applied to the existing
CrossEntropy classifier's logits. No retraining -- loads the same
checkpoint ablation.py already saved.

This is the simplified mechanism behind the "MEM" approach Fan et al.
(2026, Appl. Sci., open-set ransomware detection) build on: they add
meta-learning on top, this script implements just the energy score
itself as a direct, honestly-scoped baseline.

Energy score:  E(x) = -log( sum_c exp(logit_c) )   (T=1, standard form)
Lower E(x) -> more in-distribution / confident.
Higher E(x) -> more out-of-distribution / novel.

Threshold calibrated on the validation split only, same protocol used
everywhere else in this repo (sweep candidate thresholds, pick the one
that maximises validation accuracy under "reject if E(x) > threshold").

Usage:
    python energy_baseline.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --ce_checkpoint outputs/ablation/crossentropy_model.pt \
        --out_dir outputs/energy_baseline/seed_0 \
        --seed 0 --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from torch.utils.data import DataLoader, TensorDataset

from ablation import BehaviourClassifier
from dataset import make_splits


@torch.no_grad()
def get_logits(model, tokens, device, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size)
    out = []
    for (batch,) in loader:
        out.append(model(batch.to(device)).cpu())
    return torch.cat(out).numpy()


def energy_score(logits, temperature=1.0):
    return -temperature * logsumexp(logits / temperature, axis=1)


def calibrate_threshold(val_energy, val_labels, val_logits, sweep_size=200):
    """Sweep thresholds spanning the observed energy range; pick the one
    maximising validation accuracy under 'reject if energy > threshold'."""
    preds = val_logits.argmax(axis=1)
    lo, hi = val_energy.min(), val_energy.max()
    sweep = np.linspace(lo, hi, sweep_size)
    best_thr, best_acc = sweep[0], -1.0
    for thr in sweep:
        kept = val_energy <= thr
        correct = int(((preds == val_labels) & kept).sum())
        acc = correct / len(val_labels)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


def evaluate_openworld(energies, preds_unused, families, threshold):
    rejected = energies > threshold
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

    val_logits = get_logits(model, val_set.tokens, device)
    test_logits = get_logits(model, test_set.tokens, device)

    val_energy = energy_score(val_logits)
    threshold, val_acc = calibrate_threshold(val_energy, val_set.labels, val_logits)
    print(f"Calibrated energy threshold (validation): {threshold:.4f}  "
          f"(val acc under this rule: {val_acc:.4f})")

    test_energy = energy_score(test_logits)
    test_preds = test_logits.argmax(axis=1)
    kept = test_energy <= threshold
    test_acc = int(((test_preds == test_set.labels) & kept).sum()) / len(test_set.labels)
    print(f"Closed-world test accuracy under energy rule: {test_acc:.4f}")

    held_df = pd.read_csv(args.held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    held_tokens = held_df[tok_cols].values.astype('int64')
    held_logits = get_logits(model, held_tokens, device)
    held_energy = energy_score(held_logits)
    families = held_df['family'].values

    ow_df = evaluate_openworld(held_energy, None, families, threshold)
    print("\nOpen-world rejection under energy rule:")
    print(ow_df.to_string(index=False))

    pd.DataFrame([{'Model': 'Energy-based OOD (Liu et al. 2020, on CE logits)',
                    'Threshold': threshold, 'Closed_World_Acc': round(test_acc, 4)}]
                 ).to_csv(out_dir / 'energy_closed_world.csv', index=False)
    ow_df.to_csv(out_dir / 'energy_openworld.csv', index=False)
    print(f"\nSaved -> {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='final_dna_v2.csv')
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--ce_checkpoint', required=True)
    parser.add_argument('--out_dir', default='outputs/energy_baseline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)
