"""
adversarial_eval.py — NEW

Tests whether PHENOTYPE's contrastive geometry is more robust to simple
evasion-style perturbations of API call sequences than a cross-entropy
classifier -- directly answering the reviewer's adversarial robustness
question. No retraining needed: this loads your ALREADY-TRAINED SupCon
model (single_run) and an ALREADY-TRAINED CrossEntropy model (from
ablation.py) and re-evaluates both under three perturbation types.

Three perturbations, each at multiple severity levels (fraction of the
sequence affected):
  1. INSERTION -- inserts random benign/no-op API call tokens at random
     positions (simulates API padding/junk-call evasion, a real technique)
  2. DELETION -- removes a fraction of HIGH_SIGNAL tokens (simulates
     evasion via suppressing observable behaviour, or incomplete
     instrumentation)
  3. REORDERING -- shuffles a fraction of the sequence's token order
     (tests reliance on positional/sequential structure)

Metrics reported per perturbation level, for BOTH models:
  - Closed-world accuracy degradation (test set)
  - Open-world rejection rate change (held-out novel set) -- does the
    perturbation push novel samples INTO acceptance, or known samples
    OUT of correct attribution?

Usage:
    python adversarial_eval.py \
        --csv final_dna_v2.csv \
        --held_out_csv data/held_out_families.csv \
        --supcon_encoder outputs/single_run/behaviour_encoder.pt \
        --supcon_centroids outputs/single_run/family_centroids.pt \
        --ce_checkpoint outputs/ablation/crossentropy_model.pt \
        --out_dir outputs/adversarial \
        --device cuda

NOTE: --ce_checkpoint requires ablation.py's run_crossentropy() to save
its best model state_dict to disk. If you haven't saved it separately,
add one line to ablation.py after `model.load_state_dict(best_state)`:
    torch.save(model.state_dict(), out_dir / 'crossentropy_model.pt')
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import BehaviourEncoder
from dataset import make_splits, IDX_TO_FAMILY

# A handful of common benign/low-signal Windows API calls, used as
# "junk" insertions -- calls that appear in almost all processes and
# carry little discriminative signal (approximate token IDs assigned
# at runtime from the vocabulary; see load_vocab()).
BENIGN_CALL_NAMES = [
    'GetTickCount', 'Sleep', 'GetSystemTimeAsFileTime',
    'GetCurrentProcessId', 'GetCurrentThreadId', 'IsDebuggerPresent',
]


def load_vocab(vocab_path='data/final_dna_v2_vocab.json'):
    import json
    with open(vocab_path) as f:
        vocab = json.load(f)
    return vocab


def get_benign_token_ids(vocab):
    """Map benign call names to their vocabulary token IDs, if present.
    Falls back to token ID 1 (the 'unknown call' token) for any name
    not in the vocabulary -- still a valid, harmless perturbation."""
    name_to_id = vocab.get('name_to_id', vocab) if isinstance(vocab, dict) else {}
    ids = []
    for name in BENIGN_CALL_NAMES:
        ids.append(name_to_id.get(name, 1))
    return ids if ids else [1]


def perturb_insert(tokens, severity, benign_ids, rng):
    """Insert benign tokens at random positions, shifting the sequence
    and truncating back to 1200. severity = fraction of length to insert."""
    tokens = tokens.copy()
    seq_len = int((tokens != 0).sum())
    n_insert = max(1, int(seq_len * severity))
    positions = rng.choice(min(seq_len, 1200), size=min(n_insert, seq_len), replace=False)
    insertions = rng.choice(benign_ids, size=len(positions))
    new_seq = tokens.tolist()
    for pos, val in sorted(zip(positions, insertions), reverse=True):
        new_seq.insert(pos, val)
    new_seq = new_seq[:1200]
    if len(new_seq) < 1200:
        new_seq += [0] * (1200 - len(new_seq))
    return np.array(new_seq, dtype=np.int64)


def perturb_delete(tokens, severity, rng):
    """Remove a fraction of non-padding tokens, zero-pad to refill."""
    tokens = tokens.copy()
    nonzero_idx = np.where(tokens != 0)[0]
    if len(nonzero_idx) == 0:
        return tokens
    n_delete = max(1, int(len(nonzero_idx) * severity))
    delete_idx = rng.choice(nonzero_idx, size=min(n_delete, len(nonzero_idx)), replace=False)
    mask = np.ones(len(tokens), dtype=bool)
    mask[delete_idx] = False
    kept = tokens[mask]
    kept = kept[kept != 0]
    new_seq = np.zeros(1200, dtype=np.int64)
    new_seq[:min(len(kept), 1200)] = kept[:1200]
    return new_seq


def perturb_reorder(tokens, severity, rng):
    """Shuffle a contiguous-ish fraction of the active (non-padding)
    tokens in place."""
    tokens = tokens.copy()
    nonzero_idx = np.where(tokens != 0)[0]
    if len(nonzero_idx) < 2:
        return tokens
    n_shuffle = max(2, int(len(nonzero_idx) * severity))
    shuffle_idx = rng.choice(nonzero_idx, size=min(n_shuffle, len(nonzero_idx)), replace=False)
    shuffled_vals = tokens[shuffle_idx].copy()
    rng.shuffle(shuffled_vals)
    tokens[shuffle_idx] = shuffled_vals
    return tokens


PERTURBATIONS = {
    'insertion': perturb_insert,
    'deletion': perturb_delete,
    'reordering': perturb_reorder,
}


def apply_perturbation(token_matrix, kind, severity, benign_ids, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(token_matrix)
    for i in range(len(token_matrix)):
        if kind == 'insertion':
            out[i] = perturb_insert(token_matrix[i], severity, benign_ids, rng)
        elif kind == 'deletion':
            out[i] = perturb_delete(token_matrix[i], severity, rng)
        elif kind == 'reordering':
            out[i] = perturb_reorder(token_matrix[i], severity, rng)
    return out


@torch.no_grad()
def supcon_closed_world_acc(model, centroids, tokens, labels, device, threshold=0.986, batch_size=16):
    """Pure argmax accuracy, no threshold -- matches how Table II/III report
    closed-world accuracy elsewhere in the paper. Threshold is reserved for
    the separate rejection-rate metric only."""
    model.eval()
    cent_matrix = torch.stack([centroids[i] for i in range(5)]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens), torch.from_numpy(labels)),
                         batch_size=batch_size)
    correct, total = 0, 0
    for batch_tok, batch_lbl in loader:
        fp = model(batch_tok.to(device))
        sims = fp @ cent_matrix.T
        pred = sims.argmax(dim=1).cpu()
        correct += int((pred == batch_lbl).sum())
        total += len(batch_lbl)
    return correct / total


@torch.no_grad()
def supcon_openworld_rejection(model, centroids, tokens, device, threshold=0.986, batch_size=16):
    model.eval()
    cent_matrix = torch.stack([centroids[i] for i in range(5)]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size)
    rejected, total = 0, 0
    for (batch_tok,) in loader:
        fp = model(batch_tok.to(device))
        sims = fp @ cent_matrix.T
        best_score, _ = sims.max(dim=1)
        rejected += int((best_score < threshold).sum())
        total += len(batch_tok)
    return rejected / total


@torch.no_grad()
def ce_closed_world_acc(model, tokens, labels, device, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens), torch.from_numpy(labels)),
                         batch_size=batch_size)
    correct, total = 0, 0
    for batch_tok, batch_lbl in loader:
        logits = model(batch_tok.to(device))
        pred = logits.argmax(dim=1).cpu()
        correct += int((pred == batch_lbl).sum())
        total += len(batch_lbl)
    return correct / total


@torch.no_grad()
def ce_openworld_rejection(model, tokens, device, threshold=0.99, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(tokens)), batch_size=batch_size)
    rejected, total = 0, 0
    for (batch_tok,) in loader:
        logits = model(batch_tok.to(device))
        probs = F.softmax(logits, dim=1)
        best_prob, _ = probs.max(dim=1)
        rejected += int((best_prob.cpu() < threshold).sum())
        total += len(batch_tok)
    return rejected / total


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load test set (known families) and held-out set (novel families) --
    _, _, test_set = make_splits(args.csv, seed=args.seed)
    test_tokens = test_set.tokens
    test_labels = test_set.labels

    held_df = pd.read_csv(args.held_out_csv)
    tok_cols = [f'tok_{i}' for i in range(1200)]
    held_tokens = held_df[tok_cols].values.astype('int64')

    # -- Benign call IDs for insertion attack --
    try:
        vocab = load_vocab(args.vocab)
        benign_ids = get_benign_token_ids(vocab)
    except FileNotFoundError:
        print("WARNING: vocab file not found, using fallback token ID for insertions")
        benign_ids = [1]

    # -- Load SupCon model --
    supcon_model = BehaviourEncoder()
    supcon_model.load_state_dict(torch.load(args.supcon_encoder, map_location='cpu', weights_only=False))
    supcon_model.to(device).eval()
    supcon_centroids = torch.load(args.supcon_centroids, map_location='cpu', weights_only=False)

    # -- Load CrossEntropy model --
    from ablation import BehaviourClassifier
    ce_model = BehaviourClassifier()
    ce_model.load_state_dict(torch.load(args.ce_checkpoint, map_location='cpu', weights_only=False))
    ce_model.to(device).eval()

    severities = [0.0, 0.05, 0.10, 0.20, 0.30]
    results = []

    for kind in PERTURBATIONS:
        for sev in severities:
            if sev == 0.0:
                pert_test_tok = test_tokens
                pert_held_tok = held_tokens
            else:
                pert_test_tok = apply_perturbation(test_tokens, kind, sev, benign_ids, seed=args.seed)
                pert_held_tok = apply_perturbation(held_tokens, kind, sev, benign_ids, seed=args.seed)

            sc_acc = supcon_closed_world_acc(supcon_model, supcon_centroids,
                                              pert_test_tok, test_labels, device)
            sc_rej = supcon_openworld_rejection(supcon_model, supcon_centroids,
                                                 pert_held_tok, device)
            ce_acc = ce_closed_world_acc(ce_model, pert_test_tok, test_labels, device)
            ce_rej = ce_openworld_rejection(ce_model, pert_held_tok, device)

            results.append({
                'perturbation': kind, 'severity': sev,
                'SupCon_closed_acc': round(sc_acc, 4),
                'SupCon_openworld_rejection': round(sc_rej, 4),
                'CrossEntropy_closed_acc': round(ce_acc, 4),
                'CrossEntropy_openworld_rejection': round(ce_rej, 4),
            })
            print(f"{kind:12s} sev={sev:.2f}  "
                  f"SupCon acc={sc_acc:.3f} rej={sc_rej:.3f}  |  "
                  f"CE acc={ce_acc:.3f} rej={ce_rej:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'adversarial_results.csv', index=False)
    print(f"\nSaved -> {out_dir / 'adversarial_results.csv'}")

    # -- Plot: accuracy degradation curves per perturbation type --
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax, kind in zip(axes, PERTURBATIONS):
        sub = df[df['perturbation'] == kind]
        ax.set_facecolor('#161b22')
        ax.plot(sub['severity'], sub['SupCon_closed_acc'], 'o-', color='#2a9d8f', label='SupCon acc')
        ax.plot(sub['severity'], sub['CrossEntropy_closed_acc'], 's-', color='#e63946', label='CrossEntropy acc')
        ax.set_title(kind.capitalize(), color='white', fontweight='bold')
        ax.set_xlabel('Perturbation severity', color='#aaaaaa')
        ax.set_ylabel('Closed-world accuracy', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.legend(facecolor='#1c1c2e', edgecolor='#333', labelcolor='white', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'adversarial_accuracy_degradation.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved -> {out_dir / 'adversarial_accuracy_degradation.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--held_out_csv', default='data/held_out_families.csv')
    parser.add_argument('--vocab', default='data/final_dna_v2_vocab.json')
    parser.add_argument('--supcon_encoder', required=True)
    parser.add_argument('--supcon_centroids', required=True)
    parser.add_argument('--ce_checkpoint', required=True)
    parser.add_argument('--out_dir', default='outputs/adversarial')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)
