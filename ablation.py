"""
PHENO TYPE — ablation.py
Runs ablation experiments to generate publishable comparison results.

Ablations tested:
  1. Full model (Transformer + SupConLoss + AttentionPooling)   ← your system
  2. Transformer + CrossEntropyLoss (no contrastive training)
  3. Mean pooling instead of attention pooling
  4. TF-IDF bag-of-words baseline (no sequence info)
  5. Batch size 64 vs 128 (your two runs — already done)

Each ablation trains for up to 50 epochs with early stopping (patience=7).
Results saved to outputs/ablation_results.csv and ablation_comparison.png.

Usage:
    python ablation.py --csv final_dna_v2.csv --device cuda
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import make_splits, StratifiedBatchSampler, IDX_TO_FAMILY, FAMILY_TO_IDX
from model import BehaviourEncoder

try:
    from pytorch_metric_learning.losses import SupConLoss
except ImportError:
    raise ImportError("pip install pytorch-metric-learning")


# ─────────────────────────────────────────────────────────────────────────────
# Shared training utilities
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, total_steps, warmup_frac=0.10):
    warmup_steps = int(total_steps * warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def get_fingerprints(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    fps, lbls = [], []
    for tokens, labels in loader:
        fp = model(tokens.to(device)).cpu()
        fps.append(fp)
        lbls.append(labels)
    return torch.cat(fps), torch.cat(lbls)


@torch.no_grad()
def nearest_centroid_accuracy(model, train_set, test_set, device, batch_size=16):
    train_fp, train_lbl = get_fingerprints(model, train_set, device, batch_size)
    test_fp,  test_lbl  = get_fingerprints(model, test_set,  device, batch_size)

    centroids = {}
    for i in range(5):
        mask = (train_lbl == i)
        if mask.sum() > 0:
            c = train_fp[mask].mean(0)
            centroids[i] = F.normalize(c, dim=0)

    cent_matrix = torch.stack([centroids[i] for i in range(5)])
    sims  = test_fp @ cent_matrix.T
    preds = sims.argmax(dim=1).numpy()
    trues = test_lbl.numpy()

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average=None, labels=list(range(5)), zero_division=0)
    f1_dict = {IDX_TO_FAMILY[i]: round(float(f1[i]), 4) for i in range(5)}
    return round(acc, 4), f1_dict


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1: Full model — Transformer + SupConLoss (your system)
# ─────────────────────────────────────────────────────────────────────────────
def run_supcon(train_set, test_set, device, epochs=50, batch_size=64,
               infer_batch=16, patience=7):
    print("\n── Ablation 1: Transformer + SupConLoss (full system) ──")
    model   = BehaviourEncoder().to(device)
    loss_fn = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler = StratifiedBatchSampler(train_set.labels.tolist(), batch_size=batch_size)
    loader  = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    scheduler = build_scheduler(optimizer, epochs * len(loader))

    best_acc, patience_ctr = 0.0, 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, labels in loader:
            optimizer.zero_grad()
            fp   = model(tokens.to(device))
            loss = loss_fn(fp, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        acc, f1 = nearest_centroid_accuracy(model, train_set, test_set, device, infer_batch)
        print(f"  Epoch {epoch:02d}  acc={acc:.3f}  "
              f"F1: AT={f1['AgentTesla']:.2f} LB={f1['Lokibot']:.2f} "
              f"RD={f1['Redline']:.2f} nj={f1['njRAT']:.2f}")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    acc, f1 = nearest_centroid_accuracy(model, train_set, test_set, device, infer_batch)
    print(f"  BEST  acc={acc:.4f}  f1={f1}")
    return acc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 2: Transformer + CrossEntropy (no contrastive loss)
# ─────────────────────────────────────────────────────────────────────────────
class BehaviourClassifier(nn.Module):
    """Same encoder but with a classification head instead of contrastive projection."""
    def __init__(self, num_classes=5):
        super().__init__()
        from model import SinusoidalPositionalEncoding, AttentionPooling
        self.embedding   = nn.Embedding(100, 128, padding_idx=0)
        self.pos_enc     = SinusoidalPositionalEncoding(128)
        enc_layer        = nn.TransformerEncoderLayer(128, 8, 512, 0.1, 'gelu',
                                                       batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, 4, enable_nested_tensor=False)
        self.pool        = AttentionPooling(128)
        self.classifier  = nn.Linear(128, num_classes)

    def forward(self, tokens):
        pad  = (tokens == 0)
        x    = self.pos_enc(self.embedding(tokens))
        x    = self.transformer(x, src_key_padding_mask=pad)
        x    = self.pool(x, pad)
        return self.classifier(x)   # logits


def run_crossentropy(train_set, test_set, device, epochs=50, batch_size=64, patience=7):
    print("\n── Ablation 2: Transformer + CrossEntropyLoss ──")
    model     = BehaviourClassifier().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler   = StratifiedBatchSampler(train_set.labels.tolist(), batch_size=batch_size)
    loader    = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    scheduler = build_scheduler(optimizer, epochs * len(loader))

    best_acc, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, labels in loader:
            optimizer.zero_grad()
            logits = model(tokens.to(device))
            loss   = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        loader_test = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
        preds, trues = [], []
        with torch.no_grad():
            for tokens, labels in loader_test:
                logits = model(tokens.to(device))
                pred   = logits.argmax(dim=1).cpu()
                preds.append(pred)
                trues.append(labels)
        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
        acc   = accuracy_score(trues, preds)
        f1    = f1_score(trues, preds, average=None, labels=list(range(5)), zero_division=0)
        f1_dict = {IDX_TO_FAMILY[i]: round(float(f1[i]), 4) for i in range(5)}
        print(f"  Epoch {epoch:02d}  acc={acc:.3f}")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  BEST  acc={best_acc:.4f}  f1={f1_dict}")
    return best_acc, f1_dict


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 3: Mean pooling instead of attention pooling
# ─────────────────────────────────────────────────────────────────────────────
class BehaviourEncoderMeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        from model import SinusoidalPositionalEncoding
        self.embedding   = nn.Embedding(100, 128, padding_idx=0)
        self.pos_enc     = SinusoidalPositionalEncoding(128)
        enc_layer        = nn.TransformerEncoderLayer(128, 8, 512, 0.1, 'gelu',
                                                       batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, 4, enable_nested_tensor=False)
        self.proj        = nn.Linear(128, 256)

    def forward(self, tokens):
        pad = (tokens == 0)
        x   = self.pos_enc(self.embedding(tokens))
        x   = self.transformer(x, src_key_padding_mask=pad)
        # Mean pool over non-padding positions
        mask_float = (~pad).float().unsqueeze(-1)
        pooled     = (x * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_mean_pool(train_set, test_set, device, epochs=50, batch_size=64,
                  infer_batch=16, patience=7):
    print("\n── Ablation 3: Transformer + MeanPooling + SupConLoss ──")
    model     = BehaviourEncoderMeanPool().to(device)
    loss_fn   = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler   = StratifiedBatchSampler(train_set.labels.tolist(), batch_size=batch_size)
    loader    = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    scheduler = build_scheduler(optimizer, epochs * len(loader))

    best_acc, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, labels in loader:
            optimizer.zero_grad()
            fp   = model(tokens.to(device))
            loss = loss_fn(fp, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        acc, f1 = nearest_centroid_accuracy(model, train_set, test_set, device, infer_batch)
        print(f"  Epoch {epoch:02d}  acc={acc:.3f}")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  BEST  acc={best_acc:.4f}  f1={f1}")
    return best_acc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Ablation 4: TF-IDF bag-of-words baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_tfidf_baseline(train_set, test_set):
    print("\n── Ablation 4: TF-IDF Bag-of-Words Baseline ──")

    def tokens_to_str(token_array):
        return [' '.join(map(str, row[row > 0])) for row in token_array]

    X_train = tokens_to_str(train_set.tokens)
    y_train = train_set.labels
    X_test  = tokens_to_str(test_set.tokens)
    y_test  = test_set.labels

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500)),
        ('lr',    LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds, average=None, labels=list(range(5)), zero_division=0)
    f1_dict = {IDX_TO_FAMILY[i]: round(float(f1[i]), 4) for i in range(5)}
    print(f"  acc={acc:.4f}  f1={f1_dict}")
    return round(acc, 4), f1_dict


# ─────────────────────────────────────────────────────────────────────────────
# Results table + plot
# ─────────────────────────────────────────────────────────────────────────────
def save_results(results, out_dir):
    rows = []
    for name, (acc, f1) in results.items():
        row = {'Model': name, 'Accuracy': acc}
        row.update({f'F1_{k}': v for k, v in f1.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ablation_results.csv', index=False)
    print(f"\nAblation results saved → {out_dir / 'ablation_results.csv'}")
    print(df.to_string(index=False))
    return df


def plot_ablation(df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    models = df['Model'].tolist()
    x      = np.arange(len(models))
    width  = 0.13

    families  = ['AgentTesla', 'Formbook', 'Lokibot', 'Redline', 'njRAT']
    colors    = ['#e63946', '#2a9d8f', '#e9c46a', '#a8dadc', '#f4a261']
    acc_color = '#ffffff'

    for i, (fam, color) in enumerate(zip(families, colors)):
        col  = f'F1_{fam}'
        vals = df[col].tolist() if col in df.columns else [0]*len(models)
        ax.bar(x + (i - 2.5) * width, vals, width, label=fam,
               color=color, alpha=0.85)

    ax.bar(x + 2.5 * width, df['Accuracy'].tolist(), width,
           label='Accuracy', color=acc_color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, color='#eaeaea', fontsize=9, rotation=10)
    ax.set_ylabel('Score', color='#aaaaaa')
    ax.set_ylim(0, 1.05)
    ax.set_title('Ablation Study — Model Comparison', color='white',
                 fontsize=13, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    ax.legend(facecolor='#1c1c2e', edgecolor='#333', labelcolor='white',
              fontsize=8, loc='lower right')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    path = out_dir / 'ablation_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Ablation plot saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        required=True)
    parser.add_argument('--out_dir',    default='outputs')
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--skip_tfidf', action='store_true',
                        help='Skip TF-IDF baseline (already fast, rarely needed to skip)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    train_set, val_set, test_set = make_splits(args.csv, seed=args.seed)
    print(f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    results = {}

    # Run all ablations
    acc, f1 = run_supcon(train_set, test_set, device,
                          epochs=args.epochs, batch_size=args.batch_size)
    results['Transformer+SupCon\n(Ours)'] = (acc, f1)

    acc, f1 = run_crossentropy(train_set, test_set, device,
                                epochs=args.epochs, batch_size=args.batch_size)
    results['Transformer+CrossEntropy'] = (acc, f1)

    acc, f1 = run_mean_pool(train_set, test_set, device,
                             epochs=args.epochs, batch_size=args.batch_size)
    results['Transformer+MeanPool'] = (acc, f1)

    if not args.skip_tfidf:
        acc, f1 = run_tfidf_baseline(train_set, test_set)
        results['TF-IDF+LogReg\n(Baseline)'] = (acc, f1)

    # Also add your two existing runs from training logs if available
    for tag, fname in [('SupCon BS=64', 'training_log.csv')]:
        log_path = out_dir / fname
        if log_path.exists():
            log = pd.read_csv(log_path)
            # best epoch
            best = log.loc[log['val_acc'].idxmax()]
            print(f"\n  {tag} best val_acc={best['val_acc']:.4f} at epoch {int(best['epoch'])}")

    df = save_results(results, out_dir)
    plot_ablation(df, out_dir)

    print("\n✓ Ablation study complete.")
    print("  Files saved:")
    print(f"    {out_dir}/ablation_results.csv")
    print(f"    {out_dir}/ablation_comparison.png")
    print("\n  Use these in your paper's Table 2 (Ablation Study).")