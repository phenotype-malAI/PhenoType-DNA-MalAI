"""
PHENO TYPE — train.py
Full training loop: Supervised Contrastive Loss, stratified batching,
warmup + cosine LR schedule, early stopping, checkpoint saving.

Usage:
    python train.py --csv final_dna_v2.csv --out_dir outputs --device cuda

Outputs saved to --out_dir:
    behaviour_encoder.pt        — best model checkpoint
    family_centroids.pt         — per-family mean fingerprint (L2 normalised)
    training_log.csv            — loss + per-family F1 per epoch
    test_report.json            — held-out test set evaluation
    threshold_calibration.png   — similarity score distributions for threshold tuning
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import (
    make_splits, StratifiedBatchSampler,
    CLASS_WEIGHTS, IDX_TO_FAMILY, FAMILY_TO_IDX
)
from model import BehaviourEncoder

try:
    from pytorch_metric_learning.losses import SupConLoss
except ImportError:
    raise ImportError("Run:  pip install pytorch-metric-learning")


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.10):
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers — use small batch size to avoid OOM on GPU
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_centroids(model, dataset, device, batch_size: int = 16):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_fp, all_lbl = [], []
    for tokens, labels in loader:
        fp = model(tokens.to(device)).cpu()   # move to CPU immediately to free GPU RAM
        all_fp.append(fp)
        all_lbl.append(labels)
    all_fp  = torch.cat(all_fp)
    all_lbl = torch.cat(all_lbl)
    centroids = {}
    for fam_idx in range(5):
        mask = (all_lbl == fam_idx)
        if mask.sum() == 0:
            centroids[fam_idx] = torch.zeros(all_fp.shape[1])
        else:
            c = all_fp[mask].mean(dim=0)
            centroids[fam_idx] = F.normalize(c, dim=0)
    return centroids


@torch.no_grad()
def evaluate(model, dataset, centroids, device, batch_size: int = 16):
    """Returns (accuracy, per_family_f1_dict, preds, trues)."""
    model.eval()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cent_matrix = torch.stack([centroids[i] for i in range(5)]).to(device)
    preds, trues = [], []
    for tokens, labels in loader:
        fp   = model(tokens.to(device))
        sims = fp @ cent_matrix.T
        pred = sims.argmax(dim=1).cpu()
        preds.append(pred)
        trues.append(labels)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    acc   = (preds == trues).mean()
    f1    = f1_score(trues, preds, average=None, labels=list(range(5)), zero_division=0)
    f1_dict = {IDX_TO_FAMILY[i]: float(f1[i]) for i in range(5)}
    return acc, f1_dict, preds, trues


# ─────────────────────────────────────────────────────────────────────────────
# Threshold calibration plot
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def plot_threshold_calibration(model, dataset, centroids, device,
                                batch_size: int = 16, save_path: str = None,
                                current_threshold: float = 0.85):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cent_matrix = torch.stack([centroids[i] for i in range(5)]).to(device)
    correct_scores, wrong_scores = [], []
    for tokens, labels in loader:
        fp               = model(tokens.to(device))
        sims             = fp @ cent_matrix.T
        best_score, pred = sims.max(dim=1)
        for score, p, l in zip(best_score.cpu(), pred.cpu(), labels):
            if p.item() == l.item():
                correct_scores.append(score.item())
            else:
                wrong_scores.append(score.item())

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.hist(correct_scores, bins=50, alpha=0.7, color='#2a9d8f',
            label=f'Correct ({len(correct_scores)} samples)')
    ax.hist(wrong_scores,   bins=50, alpha=0.7, color='#e63946',
            label=f'Wrong ({len(wrong_scores)} samples)')
    ax.axvline(current_threshold, color='white', linestyle='--', linewidth=1.5,
               label=f'Current threshold ({current_threshold})')
    mid = None
    if correct_scores and wrong_scores:
        mid = (np.mean(correct_scores) + np.mean(wrong_scores)) / 2
        ax.axvline(mid, color='#e9c46a', linestyle=':', linewidth=1.5,
                   label=f'Suggested ({mid:.3f})')
    ax.set_xlabel('Cosine Similarity to Nearest Centroid', color='#aaaaaa', fontsize=11)
    ax.set_ylabel('Sample Count', color='#aaaaaa', fontsize=11)
    ax.set_title('Threshold Calibration — Correct vs Wrong Attribution Scores',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.legend(facecolor='#1c1c2e', edgecolor='#333333', labelcolor='white', fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Threshold calibration plot saved → {save_path}")
    plt.close(fig)
    if mid is not None:
        print(f"\n── Threshold Calibration Summary ──────────────────────")
        print(f"  Correct mean : {np.mean(correct_scores):.4f}   "
              f"Wrong mean : {np.mean(wrong_scores):.4f}")
        print(f"  Suggested threshold : {mid:.4f}")
        print(f"  → Update THRESHOLD in attribute.py to {mid:.2f}")
        print(f"───────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    )
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset …")
    train_set, val_set, test_set = make_splits(args.csv, seed=args.seed)
    print(f"  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    batch_sampler = StratifiedBatchSampler(train_set.labels.tolist(),
                                           batch_size=args.batch_size)
    train_loader  = DataLoader(train_set, batch_sampler=batch_sampler,
                               num_workers=0, pin_memory=(device.type == 'cuda'))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BehaviourEncoder().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    loss_fn     = SupConLoss(temperature=args.temperature)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler   = build_scheduler(optimizer, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    log_rows         = []
    best_val_f1      = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            fp   = model(tokens)          # (B, 256) L2-normalised
            loss = loss_fn(fp, labels)    # SupConLoss — scalar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()              # optimizer BEFORE scheduler
            scheduler.step()

            epoch_losses.append(loss.item())

        # ── Validation (small batch to avoid OOM) ─────────────────────────────
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        centroids = compute_centroids(model, train_set, device,
                                      batch_size=args.infer_batch_size)
        val_acc, val_f1, _, _ = evaluate(model, val_set, centroids, device,
                                          batch_size=args.infer_batch_size)

        mean_f1    = np.mean(list(val_f1.values()))
        train_loss = np.mean(epoch_losses)
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_acc={val_acc:.3f}  "
            f"mean_F1={mean_f1:.3f}  "
            f"[{elapsed:.0f}s]  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in val_f1.items()))

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_acc': val_acc}
        row.update({f'f1_{k}': v for k, v in val_f1.items()})
        log_rows.append(row)

        if mean_f1 > best_val_f1:
            best_val_f1      = mean_f1
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / 'behaviour_encoder.pt')
            torch.save(centroids,          out_dir / 'family_centroids.pt')
            print(f"  ✓ Saved best checkpoint (mean F1={best_val_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── Save training log ─────────────────────────────────────────────────────
    pd.DataFrame(log_rows).to_csv(out_dir / 'training_log.csv', index=False)
    print(f"\nTraining log saved → {out_dir / 'training_log.csv'}")

    # ── Load best checkpoint ───────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(out_dir / 'behaviour_encoder.pt',
                   map_location=device, weights_only=False)
    )
    centroids = torch.load(
        out_dir / 'family_centroids.pt',
        map_location=device, weights_only=False
    )

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\n─── TEST SET EVALUATION ───")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    test_acc, test_f1, preds, trues = evaluate(
        model, test_set, centroids, device, batch_size=args.infer_batch_size
    )
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(
        trues, preds,
        target_names=[IDX_TO_FAMILY[i] for i in range(5)],
        zero_division=0
    ))
    report = {
        'test_accuracy':         float(test_acc),
        'per_family_f1':         test_f1,
        'classification_report': classification_report(
            trues, preds,
            target_names=[IDX_TO_FAMILY[i] for i in range(5)],
            zero_division=0, output_dict=True
        )
    }
    with open(out_dir / 'test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # ── Threshold calibration ──────────────────────────────────────────────────
    print("\n─── THRESHOLD CALIBRATION ───")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    plot_threshold_calibration(
        model, test_set, centroids, device,
        batch_size=args.infer_batch_size,
        save_path=str(out_dir / 'threshold_calibration.png'),
        current_threshold=args.threshold,
    )

    print(f"All outputs saved to: {out_dir.resolve()}")
    return model, centroids


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PHENO TYPE encoder')
    parser.add_argument('--csv',              default='final_dna_v2.csv')
    parser.add_argument('--out_dir',          default='outputs')
    parser.add_argument('--device',           default='cuda',  help='cuda or cpu')
    parser.add_argument('--epochs',           type=int,   default=100)
    parser.add_argument('--batch_size',       type=int,   default=64,
                        help='Training batch size')
    parser.add_argument('--infer_batch_size', type=int,   default=16,
                        help='Batch size for centroid/eval passes — keep small to avoid OOM')
    parser.add_argument('--lr',               type=float, default=3e-4)
    parser.add_argument('--temperature',      type=float, default=0.07)
    parser.add_argument('--patience',         type=int,   default=10)
    parser.add_argument('--threshold',        type=float, default=0.85,
                        help='Threshold annotated on calibration plot')
    parser.add_argument('--seed',             type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)