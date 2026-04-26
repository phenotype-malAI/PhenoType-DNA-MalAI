"""
PHENO TYPE — explain.py
SHAP-based explanation: top-15 API calls driving each attribution decision.
Uses gradient x input attribution — fast, no extra dependencies.

Usage:
    python explain.py --csv_row final_dna_v2.csv --row_idx 0
    python explain.py --csv_row final_dna_v2.csv --row_idx 0 --method shap
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import BehaviourEncoder
from dataset import IDX_TO_FAMILY, FAMILY_TO_IDX
from attribute import load_model_and_centroids, attribute, THRESHOLD

# Load vocab
_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'final_dna_v2_vocab.json')
if os.path.exists(_VOCAB_PATH):
    import json as _json
    with open(_VOCAB_PATH) as _f:
        _VOCAB = _json.load(_f)
    ID_TO_API = {v: k for k, v in _VOCAB.items()}
else:
    ID_TO_API = {i: f'token_{i}' for i in range(100)}


# ─────────────────────────────────────────────────────────────────────────────
# Gradient x Input attribution (fast)
# ─────────────────────────────────────────────────────────────────────────────
def gradient_x_input(tokens, model, centroids, device,
                      target_family=None) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    inp = tokens.unsqueeze(0).to(device)

    # Forward with gradient tracking on the embedding
    emb = model.embedding(inp).detach().requires_grad_(True)
    pe  = model.pos_enc(emb)
    pad = (inp == 0)
    x   = model.transformer(pe, src_key_padding_mask=pad)
    pooled = model.pool(x, pad)
    fp  = F.normalize(model.proj(pooled), dim=1)

    if target_family is None:
        family, _, _, _ = attribute(tokens, model, centroids, device)
        target_family = family if family != 'UNKNOWN' else 'Redline'

    fam_idx  = FAMILY_TO_IDX.get(target_family, 0)
    centroid = centroids[fam_idx].to(device)
    sim = torch.dot(fp.squeeze(0), centroid)
    sim.backward()

    token_ids = inp.squeeze(0).cpu().numpy()

    if emb.grad is not None:
        scores_pos = (emb.grad * emb).abs().mean(dim=-1).squeeze(0).detach().cpu().numpy()
    else:
        # Fallback: attention weights
        with torch.no_grad():
            _, attn = model(inp, return_attn_weights=True)
        scores_pos = attn.squeeze(0).detach().cpu().numpy()

    # Aggregate by API call type
    api_scores = {}
    for tok_id, score in zip(token_ids, scores_pos):
        if tok_id == 0:
            continue
        name = ID_TO_API.get(int(tok_id), f'token_{tok_id}')
        api_scores[name] = api_scores.get(name, 0.0) + float(score)

    top15 = sorted(api_scores.items(), key=lambda x: -abs(x[1]))[:15]
    names  = [a[0] for a in top15]
    values = np.array([a[1] for a in top15])
    mx = np.abs(values).max()
    if mx > 0:
        values = values / mx
    return values, names


# ─────────────────────────────────────────────────────────────────────────────
# KernelSHAP attribution (rigorous, slower)
# ─────────────────────────────────────────────────────────────────────────────
def kernel_shap(tokens, model, centroids, device,
                target_family=None, nsamples=200) -> Tuple[np.ndarray, List[str]]:
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    model.eval()
    token_arr = tokens.cpu().numpy()

    if target_family is None:
        family, _, _, _ = attribute(tokens, model, centroids, device)
        target_family = family if family != 'UNKNOWN' else 'Redline'

    fam_idx  = FAMILY_TO_IDX.get(target_family, 0)
    centroid = centroids[fam_idx].to(device)

    vocab_size = 100
    base_freq  = np.bincount(token_arr, minlength=vocab_size).astype(float)
    total      = base_freq.sum()
    base_freq_norm = base_freq / (total + 1e-9)

    def predict_fn(freq_matrix):
        sims = []
        for freq in freq_matrix:
            seq = np.zeros(1200, dtype='int64')
            pos = 0
            for tok_id in range(vocab_size):
                count = int(round(freq[tok_id] * 1200))
                end   = min(pos + count, 1200)
                seq[pos:end] = tok_id
                pos = end
                if pos >= 1200:
                    break
            seq_t = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                fp  = model(seq_t).squeeze(0)
                sim = torch.dot(fp, centroid).item()
            sims.append(sim)
        return np.array(sims)

    background = np.zeros((1, vocab_size))
    explainer  = shap.KernelExplainer(predict_fn, background)
    shap_vals  = explainer.shap_values(
        base_freq_norm.reshape(1, -1), nsamples=nsamples
    ).flatten()

    names, values = [], []
    for tok_id in range(vocab_size):
        if abs(shap_vals[tok_id]) > 1e-6:
            names.append(ID_TO_API.get(tok_id, f'token_{tok_id}'))
            values.append(shap_vals[tok_id])

    top15  = sorted(zip(names, values), key=lambda x: -abs(x[1]))[:15]
    names  = [p[0] for p in top15]
    values = np.array([p[1] for p in top15])
    return values, names


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────
def explain_sample(tokens, model, centroids, device,
                   target_family=None, method='gradient', **kwargs):
    if method == 'shap':
        return kernel_shap(tokens, model, centroids, device, target_family, **kwargs)
    return gradient_x_input(tokens, model, centroids, device, target_family)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_bar(values, api_names, title='SHAP Feature Attribution',
                  save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    colors = ['#e94560' if v > 0 else '#0f3460' for v in values]
    y_pos  = range(len(api_names))
    ax.barh(list(y_pos), values, color=colors, edgecolor='none', height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(api_names, fontsize=9, color='#eaeaea',
                       fontfamily='monospace')
    ax.set_xlabel('Normalised Attribution Score', color='#aaaaaa', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=12)
    ax.axvline(0, color='#555555', linewidth=0.8)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    pos_patch = mpatches.Patch(color='#e94560', label='Increases attribution')
    neg_patch = mpatches.Patch(color='#0f3460', label='Decreases attribution')
    ax.legend(handles=[pos_patch, neg_patch], facecolor='#1a1a2e',
              edgecolor='#333333', labelcolor='#eaeaea', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"SHAP chart saved → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_row',   required=True)
    parser.add_argument('--row_idx',   type=int, default=0)
    parser.add_argument('--encoder',   default='outputs/behaviour_encoder.pt')
    parser.add_argument('--centroids', default='outputs/family_centroids.pt')
    parser.add_argument('--method',    default='gradient', choices=['gradient','shap'])
    parser.add_argument('--out',       default='outputs/shap_explanation.png')
    parser.add_argument('--device',    default='cpu')
    args = parser.parse_args()

    import pandas as pd
    df  = pd.read_csv(args.csv_row, nrows=args.row_idx + 1)
    row = df.iloc[args.row_idx]
    tok_cols = [f'tok_{i}' for i in range(1200)]
    tok = torch.tensor(row[tok_cols].values.astype('int64'), dtype=torch.long)

    model, centroids, device = load_model_and_centroids(
        args.encoder, args.centroids, args.device
    )

    family, score, _, _ = attribute(tok, model, centroids, device)
    print(f"Attribution: {family}  (score={score:.4f})")

    values, names = explain_sample(tok, model, centroids, device,
                                   target_family=family, method=args.method)
    plot_shap_bar(values, names,
                  title=f'Top API calls driving {family} attribution',
                  save_path=args.out)