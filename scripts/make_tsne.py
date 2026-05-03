#!/usr/bin/env python3
"""
PHENO TYPE — t-SNE Cluster Visualiser
=======================================
Produces a publication-quality t-SNE figure for the research paper.

Two modes (automatically detected):
  MODE A — Model Embeddings (preferred)
      Requires: outputs/batch_size64/behaviour_encoder.pt
                outputs/batch_size64/family_centroids.pt
                ../../final_dna_v2.csv   (or final_dna_v2.csv two levels up)
      Generates true 256-dim fingerprint embeddings via the trained Transformer,
      then reduces to 2D with t-SNE.  Centroids are plotted as large markers.

  MODE B — Token-Frequency Fallback (no weights needed)
      Uses raw token occurrence counts (99-dim bag-of-API-calls feature)
      from the training CSV.  Less refined than Mode A but still reveals
      cluster structure because HIGH_SIGNAL API distributions are family-specific.

Output: figs/fig_tsne.pdf

Run from phenotype-main/ directory:
    python make_tsne.py [--mode a|b] [--samples N]

Options:
    --mode   a | b   Force a specific mode (default: auto-detect)
    --samples N      Max samples per family for faster runs (default: all)
    --perplexity P   t-SNE perplexity (default: 40)

Requires: torch, scikit-learn, matplotlib, pandas, numpy
"""

import argparse
import pathlib
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE     = pathlib.Path(__file__).parent.parent
OUT_DIR  = BASE / "outputs" / "batch_size64"
FIGS     = BASE / "figs"
FIGS.mkdir(exist_ok=True)

ENCODER_PT   = OUT_DIR / "behaviour_encoder.pt"
CENTROIDS_PT = OUT_DIR / "family_centroids.pt"

# Training CSV may be in parent or root directory
_CSV_CANDIDATES = [
    BASE / "final_dna_v2.csv",
    BASE.parent / "final_dna_v2.csv",
    BASE.parent.parent / "final_dna_v2.csv",
]
TRAINING_CSV = next((p for p in _CSV_CANDIDATES if p.exists()), None)

# ─── Plot settings ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"     : "serif",
    "font.serif"      : ["Times New Roman", "Georgia", "DejaVu Serif"],
    "font.size"       : 8,
    "axes.labelsize"  : 9,
    "axes.titlesize"  : 9,
    "legend.fontsize" : 8,
    "figure.dpi"      : 150,
    "savefig.dpi"     : 300,
    "savefig.bbox"    : "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype"    : 42,
    "ps.fonttype"     : 42,
})

FAMILY_ORDER = ["AgentTesla", "Formbook", "Lokibot", "Redline", "njRAT"]
FAMILY_CLR   = {
    "AgentTesla" : "#D62728",
    "Formbook"   : "#1F77B4",
    "Lokibot"    : "#2CA02C",
    "Redline"    : "#9467BD",
    "njRAT"      : "#E78C19",
}
FAMILY_MKR   = {
    "AgentTesla" : "o",
    "Formbook"   : "s",
    "Lokibot"    : "^",
    "Redline"    : "D",
    "njRAT"      : "P",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODE A — Model Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def embed_with_model(max_samples: int = None) -> tuple:
    """
    Load trained encoder, run inference on training CSV, return
    (embeddings[N, 256], labels[N], centroid_embeddings[5, 256], centroid_labels[5]).
    """
    import torch
    sys.path.insert(0, str(BASE))
    from model import BehaviourEncoder

    print("  [MODE A] Loading encoder from", ENCODER_PT)
    encoder = BehaviourEncoder()
    encoder.load_state_dict(torch.load(ENCODER_PT, map_location="cpu"))
    encoder.eval()

    print("  [MODE A] Loading centroids from", CENTROIDS_PT)
    centroids_dict = torch.load(CENTROIDS_PT, map_location="cpu")
    # centroids_dict may be {family: tensor} or a single tensor — handle both
    if isinstance(centroids_dict, dict):
        cent_labels = list(centroids_dict.keys())
        cent_vecs   = np.stack([centroids_dict[f].numpy() for f in cent_labels])
    else:
        cent_labels = FAMILY_ORDER
        cent_vecs   = centroids_dict.numpy()

    print("  [MODE A] Loading training CSV:", TRAINING_CSV)
    df = pd.read_csv(TRAINING_CSV)

    tok_cols = [c for c in df.columns if c.startswith("tok_")]

    all_emb, all_lbl = [], []
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam]
        if max_samples and len(sub) > max_samples:
            sub = sub.sample(max_samples, random_state=42)
        tokens_np = sub[tok_cols].values.astype(np.int64)

        batch_size = 64
        embs = []
        for start in range(0, len(tokens_np), batch_size):
            batch = torch.from_numpy(tokens_np[start:start+batch_size])
            with torch.no_grad():
                embs.append(encoder(batch).numpy())
        embs = np.concatenate(embs, axis=0)
        all_emb.append(embs)
        all_lbl.extend([fam] * len(embs))
        print(f"      {fam}: {len(embs)} embeddings")

    return np.concatenate(all_emb), np.array(all_lbl), cent_vecs, cent_labels


# ═══════════════════════════════════════════════════════════════════════════════
# MODE B — Token-Frequency Fallback
# ═══════════════════════════════════════════════════════════════════════════════

def embed_with_token_freq(max_samples: int = None) -> tuple:
    """
    Build 99-dim bag-of-API-calls features (token IDs 1..99 counts per sample).
    Excludes PAD token (0).  L2-normalise before t-SNE.
    """
    print("  [MODE B] Loading training CSV:", TRAINING_CSV)
    df = pd.read_csv(TRAINING_CSV)

    tok_cols = [c for c in df.columns if c.startswith("tok_")]
    VOCAB_SIZE = 100   # 0..99

    all_feat, all_lbl = [], []
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam]
        if max_samples and len(sub) > max_samples:
            sub = sub.sample(max_samples, random_state=42)
        toks = sub[tok_cols].values  # shape (N, 1200)

        # Count each token ID per row, skip PAD (0)
        feat = np.zeros((len(toks), VOCAB_SIZE - 1), dtype=np.float32)
        for i, row in enumerate(toks):
            for tok_id in row:
                if 1 <= tok_id <= VOCAB_SIZE - 1:
                    feat[i, tok_id - 1] += 1

        # L2 normalise each row
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        feat  = feat / norms

        all_feat.append(feat)
        all_lbl.extend([fam] * len(feat))
        print(f"      {fam}: {len(feat)} samples, feature dim={feat.shape[1]}")

    features = np.concatenate(all_feat, axis=0)

    # Compute per-family centroids in feature space
    labels = np.array(all_lbl)
    cent_vecs, cent_labels = [], []
    for fam in FAMILY_ORDER:
        mask = (labels == fam)
        centroid = features[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        cent_vecs.append(centroid)
        cent_labels.append(fam)

    return features, labels, np.stack(cent_vecs), cent_labels


# ═══════════════════════════════════════════════════════════════════════════════
# t-SNE + Plot
# ═══════════════════════════════════════════════════════════════════════════════

def run_tsne(embeddings: np.ndarray, perplexity: int = 40,
             n_iter: int = 1200) -> np.ndarray:
    print(f"  Running t-SNE  (perplexity={perplexity}, n_iter={n_iter}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_jobs=-1,
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(coords: np.ndarray, labels: np.ndarray,
              cent_coords_2d: np.ndarray, cent_labels: list,
              mode_label: str):

    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    # Scatter per family
    for fam in FAMILY_ORDER:
        mask = (labels == fam)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=FAMILY_CLR[fam],
            marker=FAMILY_MKR[fam],
            s=12, alpha=0.55, linewidths=0,
            label=fam, zorder=3,
        )

    # Centroids
    for i, fam in enumerate(cent_labels):
        ax.scatter(
            cent_coords_2d[i, 0], cent_coords_2d[i, 1],
            c=FAMILY_CLR[fam],
            marker="*",
            s=240, alpha=1.0,
            edgecolors="white", linewidths=0.8,
            zorder=6,
        )
        ax.text(cent_coords_2d[i, 0] + 0.4, cent_coords_2d[i, 1],
                fam, fontsize=6.8, fontweight="bold",
                color=FAMILY_CLR[fam], zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    mode_str = ("Model Embeddings (256-dim)" if "A" in mode_label
                else "Token-Frequency Features (99-dim)")
    ax.set_title(f"t-SNE Cluster Map — {mode_str}", fontweight="bold")

    # Legend (scatter handles only — centroids indicated by ★)
    handles = [
        plt.Line2D([0], [0], marker=FAMILY_MKR[f], color="w",
                   markerfacecolor=FAMILY_CLR[f], markersize=6, label=f)
        for f in FAMILY_ORDER
    ]
    cent_handle = plt.Line2D([0], [0], marker="*", color="w",
                             markerfacecolor="0.4", markersize=9,
                             label="Centroid (★)")
    ax.legend(handles=handles + [cent_handle],
              loc="upper right", framealpha=0.85,
              fontsize=7, handlelength=0.8)

    ax.grid(True, linewidth=0.35, color="0.90")
    ax.set_aspect("auto")

    out_path = FIGS / "fig_tsne.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PHENO TYPE t-SNE figure")
    parser.add_argument("--mode",       choices=["a", "b"], default=None,
                        help="Force mode A (model embeddings) or B (token freq)")
    parser.add_argument("--samples",    type=int, default=None,
                        help="Max samples per family (default: all)")
    parser.add_argument("--perplexity", type=int, default=40,
                        help="t-SNE perplexity (default: 40)")
    parser.add_argument("--n_iter",     type=int, default=1200,
                        help="t-SNE iterations (default: 1200)")
    args = parser.parse_args()

    print("=" * 60)
    print("  PHENO TYPE — t-SNE Cluster Visualiser")
    print("=" * 60)

    if TRAINING_CSV is None:
        print("\n  [ERROR] Cannot find final_dna_v2.csv.")
        print("  Place it in the same directory as this script or one/two levels up.")
        sys.exit(1)

    # Auto-detect mode
    if args.mode == "a" or (args.mode is None and ENCODER_PT.exists() and CENTROIDS_PT.exists()):
        if not ENCODER_PT.exists():
            print(f"\n  [ERROR] --mode a requested but {ENCODER_PT} not found.")
            sys.exit(1)
        mode = "A"
        embeddings, labels, cent_vecs, cent_labels = embed_with_model(args.samples)
    else:
        if args.mode == "a":
            print(f"\n  [WARN] Model weights not found, falling back to Mode B.")
        mode = "B"
        embeddings, labels, cent_vecs, cent_labels = embed_with_token_freq(args.samples)

    print(f"\n  Total samples: {len(embeddings)}")
    print(f"  Feature dim:   {embeddings.shape[1]}")

    # Run t-SNE on combined (samples + centroids) for consistent coordinate space
    combined     = np.concatenate([embeddings, cent_vecs], axis=0)
    combined_2d  = run_tsne(combined, args.perplexity, args.n_iter)

    coords_2d    = combined_2d[:len(embeddings)]
    cent_2d      = combined_2d[len(embeddings):]

    plot_tsne(coords_2d, labels, cent_2d, cent_labels, mode_label=f"Mode {mode}")

    print("\n  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
