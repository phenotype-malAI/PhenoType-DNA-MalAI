#!/usr/bin/env python3
"""
PHENO TYPE — Paper Figure Generator
=====================================
Produces IEEE-ready PDF figures for Overleaf.

Run from phenotype-main/ directory:
    python make_paper_figs.py

Outputs (all in figs/ subdirectory):
    fig1_architecture.pdf   — Neural architecture flow diagram
    fig2_training.pdf       — Training loss + per-family F1 curves
    fig3_results.pdf        — Closed-world per-family P/R/F1 bars
    fig4_ablation.pdf       — Ablation study comparison
    fig5_openworld.pdf      — Open-world rejection rates
    fig6_cosine.pdf         — Cosine score distributions (held-out)
    fig7_dataset.pdf        — Dataset composition

Requires: matplotlib, numpy, pandas  (all in requirements.txt)
"""

import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent.parent
OUT  = BASE / "outputs" / "batch_size64"
FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

# ─── IEEE-quality plot settings ───────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Times New Roman", "Georgia", "DejaVu Serif"],
    "font.size"          : 8,
    "axes.labelsize"     : 9,
    "axes.titlesize"     : 9,
    "xtick.labelsize"    : 8,
    "ytick.labelsize"    : 8,
    "legend.fontsize"    : 7.5,
    "legend.framealpha"  : 0.85,
    "legend.edgecolor"   : "0.75",
    "figure.dpi"         : 150,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.03,
    "lines.linewidth"    : 1.5,
    "axes.linewidth"     : 0.8,
    "xtick.major.width"  : 0.8,
    "ytick.major.width"  : 0.8,
    "axes.grid"          : True,
    "grid.linewidth"     : 0.4,
    "grid.color"         : "0.88",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "pdf.fonttype"       : 42,   # TrueType -> Overleaf compatible
    "ps.fonttype"        : 42,
})

# ─── Consistent colour palette ────────────────────────────────────────────────
TRAIN_CLR = {
    "AgentTesla" : "#D62728",
    "Formbook"   : "#1F77B4",
    "Lokibot"    : "#2CA02C",
    "Redline"    : "#9467BD",
    "njRAT"      : "#E78C19",
}
FAMILY_ORDER  = ["AgentTesla", "Formbook", "Lokibot", "Redline", "njRAT"]
HELD_ORDER    = ["Amadey", "Dacic", "Qakbot", "Remcos", "Smokeloader"]
HELD_CLR      = {
    "Amadey"     : "#17BECF",
    "Dacic"      : "#BCBD22",
    "Qakbot"     : "#AEC7E8",
    "Remcos"     : "#FFBB78",
    "Smokeloader": "#98DF8A",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_training_log():
    return pd.read_csv(OUT / "training_log.csv")

def load_test_report():
    with open(OUT / "test_report.json") as f:
        return json.load(f)

def load_held_out_summary():
    return pd.read_csv(OUT / "held_out_summary.csv")

def load_held_out_results():
    with open(OUT / "held_out_results.json") as f:
        return json.load(f)


def save(fig, name, **kw):
    path = FIGS / name
    fig.savefig(path, **kw)
    print(f"  Saved -> {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig_architecture():
    """
    Horizontal pipeline.  Main labels live INSIDE each box (bold).
    Detail annotations sit in a strip BELOW the dashed separator line
    so they cannot bleed across box boundaries.
    Transformer box is 2x wider to signal its complexity.
    """
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BOX_H = 0.38          # box height (normalised)
    CY    = 0.70          # box centre-y
    SEP_Y = 0.325         # dashed separator y
    ANN_Y = 0.16          # annotation text centre-y

    # (cx, width, label_line1, label_line2, annotation, facecolour)
    # Normal boxes: w=0.085 | Transformer: w=0.175 (2x wider)
    # Annotations are kept SHORT so adjacent ones never touch in the strip below
    BOXES = [
        (0.055, 0.085, "Input",        "API Calls",   "1,200 tokens",                      "#DDEBF7"),
        (0.155, 0.085, "Token",        "Embedding",   "emb. d=128",                        "#D9E1F2"),
        (0.255, 0.085, "Sinusoidal",   "Pos. Enc.",   "sinusoidal PE",                     "#D9E1F2"),
        (0.395, 0.175, "Transformer",  "Encoder  x4", "8h | ff=512 | GELU\nPre-LN | drop=0.1", "#E2EFDA"),
        (0.540, 0.085, "Attention",    "Pooling",     "attn. query",                       "#E2EFDA"),
        (0.642, 0.085, "Linear",       "L2 Norm",     "128->256",                          "#FFF2CC"),
        (0.744, 0.085, "Cosine",       "Similarity",  "5 centroids",                       "#FCE4D6"),
    ]

    # Draw boxes + two-line labels
    for (cx, w, l1, l2, ann, fc) in BOXES:
        bx = cx - w / 2
        by = CY - BOX_H / 2
        ax.add_patch(FancyBboxPatch(
            (bx, by), w, BOX_H,
            boxstyle="round,pad=0.01",
            lw=0.8, edgecolor="0.40", facecolor=fc, zorder=3,
        ))
        ax.text(cx, CY + 0.055, l1, ha="center", va="center",
                fontsize=8.0, fontweight="bold", zorder=4)
        ax.text(cx, CY - 0.065, l2, ha="center", va="center",
                fontsize=8.0, fontweight="bold", zorder=4)

    # Horizontal arrows between boxes
    for i in range(len(BOXES) - 1):
        cx_a, wa = BOXES[i][0],   BOXES[i][1]
        cx_b, wb = BOXES[i+1][0], BOXES[i+1][1]
        ax.annotate(
            "",
            xy    =(cx_b - wb/2 - 0.005, CY),
            xytext=(cx_a + wa/2 + 0.005, CY),
            arrowprops=dict(arrowstyle="-|>", color="0.30",
                            lw=1.0, mutation_scale=10),
            zorder=5,
        )

    # Dashed separator line
    X0 = BOXES[0][0]  - BOXES[0][1] / 2
    X1 = BOXES[-1][0] + BOXES[-1][1] / 2
    ax.plot([X0, X1], [SEP_Y, SEP_Y], color="0.76", lw=0.7, ls="--")

    # Drop-lines + annotation text (safely below separator)
    for (cx, w, l1, l2, ann, fc) in BOXES:
        ax.plot([cx, cx], [CY - BOX_H/2, SEP_Y + 0.01],
                color="0.76", lw=0.5, ls=":", zorder=2)
        ax.text(cx, ANN_Y, ann, ha="center", va="center",
                fontsize=5.5, color="0.35", style="italic",
                zorder=4, linespacing=1.45)

    # Output branch boxes
    CX_LAST = BOXES[-1][0]
    W_LAST  = BOXES[-1][1]
    CX_OUT  = 0.910
    W_OUT   = 0.108

    for (cy_b, t1, t2, fc, rad) in [
        (0.76, "Family",  "score >= 0.986", "#C6EFCE", -0.30),
        (0.46, "UNKNOWN", "score < 0.986",  "#FFCCCC",  0.30),
    ]:
        ax.add_patch(FancyBboxPatch(
            (CX_OUT - W_OUT/2, cy_b - 0.135), W_OUT, 0.27,
            boxstyle="round,pad=0.01",
            lw=0.8, edgecolor="0.40", facecolor=fc, zorder=3,
        ))
        ax.text(CX_OUT, cy_b + 0.055, t1, ha="center", va="center",
                fontsize=7.8, fontweight="bold", zorder=4)
        ax.text(CX_OUT, cy_b - 0.065, t2, ha="center", va="center",
                fontsize=5.8, color="0.35", style="italic", zorder=4)
        ax.annotate(
            "",
            xy    =(CX_OUT - W_OUT/2 - 0.003, cy_b),
            xytext=(CX_LAST + W_LAST/2 + 0.004, CY),
            arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.0,
                            mutation_scale=10,
                            connectionstyle=f"arc3,rad={rad}"),
            zorder=5,
        )

    # Label sits in the gap between the two output boxes
    ax.text(0.843, 0.608, "theta=0.986",
            ha="right", va="center", fontsize=6.2,
            color="0.45", style="italic")

    ax.set_title(
        "PHENO TYPE  --  Behavioural DNA Encoding Architecture",
        fontsize=9.5, fontweight="bold", pad=4,
    )
    save(fig, "fig1_architecture.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Training Curves
# ═══════════════════════════════════════════════════════════════════════════════

def fig_training():
    df = load_training_log()
    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    fig.subplots_adjust(wspace=0.32)

    # ── Left: Training loss ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color="#1F77B4", lw=1.6, zorder=3)
    ax.fill_between(epochs, df["train_loss"], alpha=0.12, color="#1F77B4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SupCon Loss")
    ax.set_title("(a) Training Loss", fontweight="bold")
    ax.set_xlim(1, epochs[-1])
    best_ep = df.loc[df["val_acc"].idxmax(), "epoch"]
    ax.axvline(best_ep, color="0.4", lw=0.9, ls="--", zorder=4)
    ax.text(best_ep + 0.5, df["train_loss"].max() * 0.97,
            f"best val\nepoch {best_ep}", fontsize=6.5, color="0.4", va="top")

    # ── Right: Per-family F1 ─────────────────────────────────────────────────
    ax = axes[1]
    f1_cols = [f"f1_{f}" for f in FAMILY_ORDER]
    for fam, col in zip(FAMILY_ORDER, f1_cols):
        ax.plot(epochs, df[col], color=TRAIN_CLR[fam], lw=1.4,
                label=fam, zorder=3)
    ax.axvline(best_ep, color="0.4", lw=0.9, ls="--", zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("(b) Per-Family Validation F1", fontweight="bold")
    ax.set_xlim(1, epochs[-1])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=1, handlelength=1.5)

    save(fig, "fig2_training.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Closed-World Results (grouped bars: P / R / F1)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_results():
    report = load_test_report()["classification_report"]

    families = FAMILY_ORDER
    precision = [report[f]["precision"] for f in families]
    recall    = [report[f]["recall"]    for f in families]
    f1        = [report[f]["f1-score"]  for f in families]

    x   = np.arange(len(families))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    bars_p = ax.bar(x - w, precision, w, label="Precision",
                    color="#4472C4", alpha=0.88, zorder=3)
    bars_r = ax.bar(x,     recall,    w, label="Recall",
                    color="#ED7D31", alpha=0.88, zorder=3)
    bars_f = ax.bar(x + w, f1,        w, label="F1",
                    color="#70AD47", alpha=0.88, zorder=3)

    # Value labels on F1 bars only
    for bar, val in zip(bars_f, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.2,
                fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title(
        "Closed-World Evaluation per Family\n"
        "(accuracy = 71.72%  |  threshold = 0.986  |  290 test samples)",
        fontweight="bold", fontsize=8.5, linespacing=1.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.axhline(0.729, color="0.35", lw=0.9, ls=":", zorder=2)
    ax.text(len(families) - 0.52, 0.744, "Macro F1 = 0.729",
            fontsize=6.5, color="0.35", ha="right")
    # Legend in lower right — away from any data
    ax.legend(loc="lower right", ncol=1, handlelength=1.2,
              framealpha=0.90, fontsize=8)

    save(fig, "fig3_results.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Ablation Study
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation():
    """
    Panel (a): Horizontal accuracy bars — model names on the y-axis,
               no multi-line x-axis crowding.
    Panel (b): Line + dot chart — per-family F1 for each model,
               family on x-axis, one line per model.  Compact and readable.
    """
    # Data from README (batch_size64 run)
    MODEL_NAMES = ["SupCon (Ours)", "CrossEntropy", "MeanPool", "TF-IDF Baseline"]
    ACCURACY    = [75.17, 76.21, 77.24, 72.41]
    FAM_LABELS  = ["AgentTesla", "Lokibot", "Redline", "njRAT"]
    FAM_DATA    = {
        "SupCon (Ours)"  : [0.607, 0.597, 0.980, 0.938],
        "CrossEntropy"   : [0.542, 0.663, 0.973, 0.889],
        "MeanPool"       : [0.532, 0.626, 0.973, 0.896],
        "TF-IDF Baseline": [0.637, 0.447, 0.980, 0.923],
    }
    COLORS  = ["#2171B5", "#6BAED6", "#AECDE3", "#A1A1A1"]
    MARKERS = ["o", "s", "^", "D"]
    LSTYLES = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    fig.subplots_adjust(wspace=0.38)

    # ── Panel (a): Horizontal accuracy bars ──────────────────────────────────
    ax = axes[0]
    y  = np.arange(len(MODEL_NAMES))

    bars = ax.barh(y, ACCURACY, 0.52, color=COLORS, alpha=0.88,
                   edgecolor="0.35", lw=0.6, zorder=3)

    for bar, val, clr in zip(bars, ACCURACY, COLORS):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=8.0, fontweight="bold",
                color="0.2")

    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_NAMES, fontsize=8.5)
    ax.set_xlabel("Validation Accuracy (%)", fontsize=8.5)
    ax.set_title("(a) Overall Accuracy", fontweight="bold")
    ax.set_xlim(69, 83)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color("#2171B5")
    ax.invert_yaxis()   # SupCon at top

    ax.text(69.3, 3.48,
            "Raw accuracy\ndoes not capture\nopen-world ability.",
            fontsize=6.0, color="0.45", style="italic", va="top",
            linespacing=1.35)

    # ── Panel (b): Per-family F1 line chart ──────────────────────────────────
    ax = axes[1]
    xpos = np.arange(len(FAM_LABELS))

    for name, clr, mkr, ls in zip(MODEL_NAMES, COLORS, MARKERS, LSTYLES):
        vals = FAM_DATA[name]
        ax.plot(xpos, vals, color=clr, marker=mkr, markersize=6,
                ls=ls, lw=1.8, label=name, zorder=4)
        ax.fill_between(xpos, vals, alpha=0.05, color=clr)

    ax.set_xticks(xpos)
    ax.set_xticklabels(FAM_LABELS, rotation=12, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 Score", fontsize=8.5)
    ax.set_ylim(0.35, 1.08)
    ax.set_title("(b) Per-Family F1 Score", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.2, ncol=1,
              handlelength=1.8, framealpha=0.90)
    ax.axhline(0.70, color="0.80", lw=0.6, ls=":", zorder=1)
    # Reference line label
    ax.text(3.05, 0.72, "0.70", fontsize=6.0, color="0.60", va="bottom")

    save(fig, "fig4_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Open-World Novelty Detection
# ═══════════════════════════════════════════════════════════════════════════════

def fig_openworld():
    df = load_held_out_summary()
    df.columns = df.columns.str.strip()

    families  = df["Held-Out Family"].tolist()
    unknown   = df["Flagged UNKNOWN"].tolist()
    false_att = df["Incorrectly Attributed"].tolist()
    unk_rate  = df["UNKNOWN Rate (%)"].tolist()
    total     = df["Total Samples"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.subplots_adjust(wspace=0.38)

    # ── Left: Stacked horizontal bars ────────────────────────────────────────
    ax = axes[0]
    y  = np.arange(len(families))
    h  = 0.45

    ax.barh(y, unknown, h, color="#70AD47", alpha=0.88, label="Correctly -> UNKNOWN", zorder=3)
    ax.barh(y, false_att, h, left=unknown, color="#FF7070", alpha=0.88,
            label="Incorrectly Attributed", zorder=3)

    # Rejection rate labels
    for i, (unk, fa, tot) in enumerate(zip(unknown, false_att, total)):
        pct = unk / tot * 100
        ax.text(tot + 0.8, i, f"{pct:.0f}%", va="center", fontsize=7.5,
                fontweight="bold", color="#375623")

    ax.set_yticks(y)
    ax.set_yticklabels(families)
    ax.set_xlabel("Number of Samples (out of 50)")
    ax.set_title("(a) Rejection vs False Attribution", fontweight="bold")
    ax.set_xlim(0, 60)
    ax.axvline(50, color="0.5", lw=0.7, ls=":", zorder=2)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.2)

    # ── Right: Rejection rate bar chart with 80.8% line ──────────────────────
    ax = axes[1]
    clrs = [HELD_CLR[f] for f in families]
    bars = ax.bar(range(len(families)), unk_rate, color=clrs,
                  edgecolor="0.3", lw=0.6, alpha=0.9, zorder=3, width=0.6)

    for bar, val in zip(bars, unk_rate):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.axhline(80.8, color="#C55A11", lw=1.1, ls="--", zorder=4)
    ax.text(len(families) - 0.5, 82.5, "Overall\n80.8%",
            fontsize=7, color="#C55A11", ha="right", fontweight="bold")

    ax.set_ylabel("UNKNOWN Rejection Rate (%)")
    ax.set_title("(b) Per-Family Rejection Rate", fontweight="bold")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylim(0, 110)

    save(fig, "fig5_openworld.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Cosine Score Distribution (held-out families)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cosine():
    """
    Two panels:
    (a) Box plots of best cosine score per sample, grouped by held-out family.
        Shows where novel families land relative to the 0.986 threshold.
    (b) Heatmap of average cosine similarity: held-out family × training centroid.
    """
    records = load_held_out_results()

    # Organise per-family best scores and all-scores matrix
    family_scores = {f: [] for f in HELD_ORDER}
    # avg_sim[held_out][train] = mean cosine
    avg_sim = {h: {t: [] for t in FAMILY_ORDER} for h in HELD_ORDER}

    for r in records:
        hf = r["true_family"]
        family_scores[hf].append(r["best_score"])
        for tf, sc in r["all_scores"].items():
            avg_sim[hf][tf].append(sc)

    # Build matrix (rows=held-out, cols=training)
    sim_matrix = np.array([
        [np.mean(avg_sim[h][t]) for t in FAMILY_ORDER]
        for h in HELD_ORDER
    ])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.subplots_adjust(wspace=0.42)

    # ── Left: Box plots ──────────────────────────────────────────────────────
    ax = axes[0]
    data    = [family_scores[f] for f in HELD_ORDER]
    colors  = [HELD_CLR[f] for f in HELD_ORDER]

    bp = ax.boxplot(data,
                    patch_artist=True,
                    medianprops=dict(color="0.1", lw=1.5),
                    whiskerprops=dict(lw=0.9, color="0.35"),
                    capprops=dict(lw=0.9, color="0.35"),
                    flierprops=dict(marker="o", markersize=2.5,
                                   alpha=0.5, color="0.5"),
                    boxprops=dict(lw=0.8),
                    zorder=3)

    for patch, clr in zip(bp["boxes"], colors):
        patch.set_facecolor(clr)
        patch.set_alpha(0.75)

    ax.axhline(0.986, color="#C00000", lw=1.1, ls="--", zorder=4)
    ax.text(5.55, 0.9875, "θ = 0.986", fontsize=7, color="#C00000",
            va="bottom", ha="right")

    ax.set_xticks(range(1, len(HELD_ORDER)+1))
    ax.set_xticklabels(HELD_ORDER, rotation=15, ha="right", fontsize=7.5)
    ax.set_ylabel("Best Cosine Similarity Score")
    ax.set_title("(a) Score Distribution per Held-Out Family", fontweight="bold")
    ax.set_ylim(0.45, 1.02)

    # Rejection rate as text below boxes
    for i, fam in enumerate(HELD_ORDER):
        scores = family_scores[fam]
        n_unk  = sum(1 for s in scores if s < 0.986)
        ax.text(i + 1, 0.465, f"{n_unk/len(scores)*100:.0f}%\nrej.",
                ha="center", va="bottom", fontsize=6.2, color="0.4")

    # ── Right: Similarity heatmap ─────────────────────────────────────────────
    ax = axes[1]
    im = ax.imshow(sim_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0.60, vmax=1.00)

    # Annotate each cell
    for i in range(len(HELD_ORDER)):
        for j in range(len(FAMILY_ORDER)):
            val = sim_matrix[i, j]
            clr = "white" if val > 0.90 else "0.2"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6.5, color=clr, fontweight="bold")

    ax.set_xticks(range(len(FAMILY_ORDER)))
    ax.set_xticklabels(FAMILY_ORDER, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(len(HELD_ORDER)))
    ax.set_yticklabels(HELD_ORDER, fontsize=7.5)
    ax.set_title("(b) Avg Cosine Similarity vs Training Centroids", fontweight="bold")
    ax.set_xlabel("Training Family Centroid")
    ax.set_ylabel("Held-Out Family")

    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Avg Cosine Sim.", fontsize=7)

    save(fig, "fig6_cosine.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Dataset Composition
# ═══════════════════════════════════════════════════════════════════════════════

def fig_dataset():
    families = FAMILY_ORDER
    counts   = [500, 272, 436, 500, 224]
    types    = [
        "Credential Stealer / RAT",
        "Form Grabber / Keylogger",
        "Password Stealer",
        "Infostealer",
        "Remote Access Trojan",
    ]
    colors = [TRAIN_CLR[f] for f in families]

    fig, ax = plt.subplots(figsize=(4.5, 2.4))

    y    = np.arange(len(families))
    bars = ax.barh(y, counts, color=colors, alpha=0.88,
                   edgecolor="0.3", lw=0.6, zorder=3, height=0.55)

    for bar, cnt, typ in zip(bars, counts, types):
        ax.text(bar.get_width() + 6, bar.get_y() + bar.get_height()/2,
                f"{cnt}", va="center", fontsize=8, fontweight="bold")
        ax.text(8, bar.get_y() + bar.get_height()/2,
                typ, va="center", fontsize=6.8, color="white",
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(families, fontsize=8.5)
    ax.set_xlabel("Number of Samples")
    ax.set_title("Training Dataset Composition  (Total: 1,932)", fontweight="bold")
    ax.set_xlim(0, 580)
    ax.axvline(1932/5, color="0.55", lw=0.8, ls=":", zorder=2)
    # Annotation inside the plot, at the top of the equal-share line
    ax.text(1932/5 + 6, 4.42, "equal\nshare", fontsize=6.2, color="0.5",
            va="top", linespacing=1.2)

    save(fig, "fig7_dataset.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHENO TYPE — Paper Figure Generator")
    print("=" * 60)
    print(f"\n  Output directory: {FIGS}\n")

    steps = [
        ("Fig 1: Architecture diagram",           fig_architecture),
        ("Fig 2: Training curves",                fig_training),
        ("Fig 3: Closed-world results",           fig_results),
        ("Fig 4: Ablation study",                 fig_ablation),
        ("Fig 5: Open-world novelty detection",   fig_openworld),
        ("Fig 6: Cosine score distributions",     fig_cosine),
        ("Fig 7: Dataset composition",            fig_dataset),
    ]

    for name, fn in steps:
        print(f"\n  {name}")
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n" + "=" * 60)
    print("  Done.  Upload the figs/ directory to your Overleaf project.")
    print("  Reference in LaTeX: \\includegraphics{figs/figN_*.pdf}")
    print("=" * 60)
