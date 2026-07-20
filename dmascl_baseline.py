"""
dmascl_baseline.py — NEW

A simplified, honestly-labeled re-implementation of DMASCL's core idea
(Yang et al. 2025): a SINGLE model trained with a WEIGHTED SUM of
cross-entropy and supervised contrastive loss (not two separate models,
as in ablation.py's CrossEntropy vs SupCon comparison), plus Gaussian
noise augmentation as a lightweight proxy for their "contrastive tasks
constructed using Gaussian noise."

WHAT THIS IS: the closest fair proxy achievable using our own encoder,
dataset, and compute budget.
WHAT THIS IS NOT: a reproduction of DMASCL's actual hybrid BERT +
log-normalisation encoder, whose code/checkpoints are not public. Report
this distinction explicitly in the paper — do not call this "DMASCL",
call it "a DMASCL-style joint CE+SupCon baseline."

Architecture: same Transformer backbone as BehaviourEncoder (4 layers,
8 heads, d_model=128), but with TWO heads off the pooled representation:
  - classification head  (128 -> 5)   trained with cross-entropy
  - projection head       (128 -> 256, L2-normalised) trained with SupCon
Total loss = alpha * CE + (1 - alpha) * SupCon   (alpha=0.5 default,
matching DMASCL's "weighted sum" description; no exact weighting was
published so this is a reasonable default, sweep alpha if time allows).

Gaussian noise augmentation: at each training step, token embeddings are
perturbed with small Gaussian noise before pooling — a lightweight proxy
for DMASCL's noise-based contrastive task construction.

Usage:
    python dmascl_baseline.py --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --out_dir outputs/dmascl_baseline --device cuda --epochs 50
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

from dataset import make_splits, StratifiedBatchSampler, IDX_TO_FAMILY
from model import SinusoidalPositionalEncoding, AttentionPooling

try:
    from pytorch_metric_learning.losses import SupConLoss
except ImportError:
    raise ImportError("pip install pytorch-metric-learning")


class DMASCLStyleModel(nn.Module):
    """Shared backbone, two heads: classification (CE) + projection (SupCon)."""

    def __init__(self, num_classes=5, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.embedding = nn.Embedding(100, 128, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(128)
        enc_layer = nn.TransformerEncoderLayer(128, 8, 512, 0.1, 'gelu',
                                                batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, 4, enable_nested_tensor=False)
        self.pool = AttentionPooling(128)

        self.classifier = nn.Linear(128, num_classes)      # CE head
        self.projection = nn.Linear(128, 256)               # SupCon head

    def forward(self, tokens, add_noise=False):
        pad = (tokens == 0)
        x = self.embedding(tokens)
        if add_noise and self.training:
            x = x + torch.randn_like(x) * self.noise_std
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=pad)
        pooled = self.pool(x, pad)                          # (B, 128)

        logits = self.classifier(pooled)                    # CE head output
        proj = F.normalize(self.projection(pooled), dim=1)  # SupCon head output
        return logits, proj


def build_scheduler(optimizer, total_steps, warmup_frac=0.10):
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_closed_world(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    for tokens, labels in loader:
        logits, _ = model(tokens.to(device), add_noise=False)
        preds.append(logits.argmax(dim=1).cpu())
        trues.append(labels)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average=None, labels=list(range(5)), zero_division=0)
    f1_dict = {IDX_TO_FAMILY[i]: round(float(f1[i]), 4) for i in range(5)}
    return acc, f1_dict


@torch.no_grad()
def evaluate_openworld_softmax(model, held_out_csv, device, threshold=0.99, batch_size=16):
    """DMASCL itself has no open-world mechanism -- this applies the SAME
    MSP-threshold baseline used elsewhere, purely to see whether the joint
    CE+SupCon training changes softmax calibration versus CE-only."""
    df = pd.read_csv(held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    tokens = torch.tensor(df[tok_cols].values.astype('int64'), dtype=torch.long)
    families = df['family'].values

    model.eval()
    loader = DataLoader(TensorDataset(tokens), batch_size=batch_size)
    all_probs = []
    for (batch,) in loader:
        logits, _ = model(batch.to(device), add_noise=False)
        all_probs.append(F.softmax(logits, dim=1).cpu())
    all_probs = torch.cat(all_probs)
    best_prob, _ = all_probs.max(dim=1)
    best_prob = best_prob.numpy()

    rows = []
    for fam in sorted(set(families)):
        mask = (families == fam)
        n = int(mask.sum())
        rejected = int((best_prob[mask] < threshold).sum())
        rows.append({'Family': fam, 'Total': n, 'Rejected_UNKNOWN': rejected,
                     'Rejection_Rate_%': round(100 * rejected / n, 1) if n else 0.0})
    overall_rej = int((best_prob < threshold).sum())
    rows.append({'Family': 'OVERALL', 'Total': len(best_prob),
                 'Rejected_UNKNOWN': overall_rej,
                 'Rejection_Rate_%': round(100 * overall_rej / len(best_prob), 1)})
    return pd.DataFrame(rows)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = make_splits(args.csv, seed=args.seed)
    print(f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    model = DMASCLStyleModel(noise_std=args.noise_std).to(device)
    ce_loss_fn = nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    sampler = StratifiedBatchSampler(train_set.labels.tolist(), batch_size=args.batch_size)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    scheduler = build_scheduler(optimizer, args.epochs * len(loader))

    best_acc, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, proj = model(tokens, add_noise=True)
            loss_ce = ce_loss_fn(logits, labels)
            loss_sup = supcon_loss_fn(proj, labels)
            loss = args.alpha * loss_ce + (1 - args.alpha) * loss_sup
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        acc, _ = evaluate_closed_world(model, val_set, device)
        print(f"  Epoch {epoch:02d}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate_closed_world(model, test_set, device)
    print(f"\nBEST (val)={best_acc:.4f}  TEST acc={test_acc:.4f}  f1={test_f1}")

    result_row = {'Model': 'DMASCL-style (CE+SupCon joint, Ours-proxy)',
                  'Accuracy': round(test_acc, 4)}
    result_row.update({f'F1_{k}': v for k, v in test_f1.items()})
    pd.DataFrame([result_row]).to_csv(out_dir / 'dmascl_baseline_results.csv', index=False)
    print(f"Saved -> {out_dir / 'dmascl_baseline_results.csv'}")

    if Path(args.held_out_csv).exists():
        ow_df = evaluate_openworld_softmax(model, args.held_out_csv, device)
        print("\nOpen-world (MSP threshold, informational only -- DMASCL has no native open-world mechanism):")
        print(ow_df.to_string(index=False))
        ow_df.to_csv(out_dir / 'dmascl_baseline_openworld.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--out_dir', default='outputs/dmascl_baseline')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.5,
                         help='Weight on CE loss; (1-alpha) on SupCon loss')
    parser.add_argument('--noise_std', type=float, default=0.05,
                         help='Gaussian noise std added to token embeddings during training')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
