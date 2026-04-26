"""
PHENO TYPE — dashboard.py
Streamlit dashboard: upload CAPE report.json or paste token sequence,
get predicted family, confidence score, SHAP chart, t-SNE placement.

Run:
    streamlit run dashboard.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title='PHENO TYPE — Malware Attribution',
    page_icon='🧬',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 1.2rem; margin: 0.4rem 0;
    }
    .family-badge { font-size: 1.8rem; font-weight: bold; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

FAMILY_COLORS = {
    'AgentTesla': '#e63946',
    'Formbook':   '#2a9d8f',
    'Lokibot':    '#e9c46a',
    'Redline':    '#a8dadc',
    'njRAT':      '#f4a261',
    'UNKNOWN':    '#888888',
}

@st.cache_resource
def load_resources(encoder_path, centroids_path, device_str='cpu'):
    try:
        from attribute import load_model_and_centroids
        return load_model_and_centroids(encoder_path, centroids_path, device_str)
    except Exception as e:
        return None, None, None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🧬 PHENO TYPE')
    st.caption('Behavioural DNA Framework for Malware Attribution')
    st.divider()
    encoder_path   = st.text_input('Encoder checkpoint', value='outputs/behaviour_encoder.pt')
    centroids_path = st.text_input('Centroids file',     value='outputs/family_centroids.pt')
    threshold      = st.slider('Attribution threshold', 0.90, 1.00, 0.97, 0.01)
    device_str     = st.selectbox('Device', ['cpu', 'cuda'], index=0)
    vocab_path     = st.text_input('Vocabulary JSON', value='data/final_dna_v2_vocab.json')
    embeddings_csv = st.text_input('t-SNE embeddings CSV',
                                   value='outputs/test_embeddings.csv')
    st.divider()
    st.caption('Chandigarh University · 2026')

model, centroids, device = load_resources(encoder_path, centroids_path, device_str)

# Load vocab
id_to_api = {}
if Path(vocab_path).exists():
    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_api = {v: k for k, v in vocab.items()}

st.markdown("## 🧬 PHENO TYPE — Malware Attribution Engine")
st.caption("Upload a CAPE Sandbox `report.json` **or** paste a pre-extracted 1200-token sequence.")

if model is None:
    st.warning("⚠️ Model checkpoint not found. "
               "Train the model first or check the file paths in the sidebar.")
    st.stop()

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_upload, tab_tokens = st.tabs(["📄 Upload CAPE report.json", "🔢 Paste token sequence"])
tokens = None

with tab_upload:
    uploaded = st.file_uploader("CAPE report.json", type=['json'])
    if uploaded:
        with st.spinner("Parsing CAPE report …"):
            report = json.load(uploaded)
            from run_extraction import extract_tokens
            if Path(vocab_path).exists():
                tok_list, raw_len, active = extract_tokens(report, vocab)
                tokens = torch.tensor(tok_list, dtype=torch.long)
                st.success(f"✅ Extracted {active} active tokens from {raw_len:,} raw API calls")
            else:
                st.error("Vocabulary file not found — check vocab path in sidebar.")

with tab_tokens:
    raw_input = st.text_area("Paste 1200 space-separated integer tokens", height=100,
                              placeholder="3 22 3 2 3 22 3 …")
    if st.button("Run Attribution", type="primary", disabled=not raw_input):
        try:
            vals = list(map(int, raw_input.strip().split()))
            if len(vals) != 1200:
                st.error(f"Expected 1200 tokens, got {len(vals)}")
            else:
                tokens = torch.tensor(vals, dtype=torch.long)
        except ValueError:
            st.error("Invalid token values — integers only.")

# ── Attribution ────────────────────────────────────────────────────────────────
if tokens is not None:
    from attribute import attribute
    from explain import explain_sample, plot_shap_bar

    with st.spinner("Computing fingerprint and attribution …"):
        family, best_score, all_scores, _ = attribute(
            tokens, model, centroids, device, threshold=threshold
        )

    fam_color = FAMILY_COLORS.get(family, '#888888')
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div style='color:#888;font-size:0.8rem'>PREDICTED FAMILY</div>"
            f"<div class='family-badge' style='color:{fam_color}'>{family}</div>"
            f"</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div style='color:#888;font-size:0.8rem'>CONFIDENCE SCORE</div>"
            f"<div style='font-size:2rem;font-weight:bold'>{best_score:.4f}</div>"
            f"</div>", unsafe_allow_html=True)
    with col3:
        verdict = "✅ ATTRIBUTED" if best_score >= threshold else "⚠️ UNKNOWN"
        st.markdown(
            f"<div class='metric-card'>"
            f"<div style='color:#888;font-size:0.8rem'>VERDICT</div>"
            f"<div style='font-size:1.4rem;font-weight:bold'>{verdict}</div>"
            f"</div>", unsafe_allow_html=True)

    st.divider()
    col_left, col_right = st.columns(2)

    # Similarity bar chart
    with col_left:
        st.subheader("Family Similarity Scores")
        sorted_scores = sorted(all_scores.items(), key=lambda x: -x[1])
        fig_s, ax = plt.subplots(figsize=(6, 4))
        fig_s.patch.set_facecolor('#161b22')
        ax.set_facecolor('#0d1117')
        names_s  = [s[0] for s in sorted_scores]
        vals_s   = [s[1] for s in sorted_scores]
        colors_s = [FAMILY_COLORS.get(n, '#888') for n in names_s]
        ax.barh(names_s, vals_s, color=colors_s, height=0.55)
        ax.axvline(threshold, color='white', linestyle='--', linewidth=1,
                   alpha=0.6, label=f'Threshold {threshold}')
        ax.set_xlabel('Cosine Similarity', color='#aaa')
        ax.set_xlim(0.95, 1.0)
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')
        ax.legend(facecolor='#1c1c2e', edgecolor='#333', labelcolor='white', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

    # SHAP explanation
    with col_right:
        st.subheader("Top API Call Attribution")
        with st.spinner("Computing attribution …"):
            shap_vals, api_names = explain_sample(
                tokens, model, centroids, device,
                target_family=family if family != 'UNKNOWN' else None,
                method='gradient',
            )
        fig_shap = plot_shap_bar(
            shap_vals, api_names,
            title=f"Top 15 API calls — {family}",
        )
        st.pyplot(fig_shap)

    # t-SNE placement
    if Path(embeddings_csv).exists():
        st.divider()
        st.subheader("t-SNE Cluster Placement")
        emb_df = pd.read_csv(embeddings_csv, usecols=['tsne_x', 'tsne_y', 'family'])

        fig_t, ax = plt.subplots(figsize=(10, 7))
        fig_t.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        for fam_name, color in FAMILY_COLORS.items():
            if fam_name == 'UNKNOWN':
                continue
            sub = emb_df[emb_df['family'] == fam_name]
            ax.scatter(sub['tsne_x'], sub['tsne_y'], s=12, c=color,
                       alpha=0.5, linewidths=0, label=fam_name)

        if family != 'UNKNOWN':
            sub = emb_df[emb_df['family'] == family]
            if len(sub) > 0:
                rng = np.random.default_rng(seed=42)
                cx  = sub['tsne_x'].mean() + rng.normal(0, sub['tsne_x'].std() * 0.15)
                cy  = sub['tsne_y'].mean() + rng.normal(0, sub['tsne_y'].std() * 0.15)
                ax.scatter([cx], [cy], s=220, c=FAMILY_COLORS.get(family, 'white'),
                           marker='*', edgecolors='white', linewidths=1.5,
                           zorder=10, label='Query ★')

        patches = [mpatches.Patch(color=c, label=f)
                   for f, c in FAMILY_COLORS.items() if f != 'UNKNOWN']
        ax.legend(handles=patches, facecolor='#1c1c2e', edgecolor='#333',
                  labelcolor='white', fontsize=9)
        ax.set_title('Behavioural Fingerprint Space (t-SNE)', color='white', fontsize=12)
        ax.tick_params(colors='#444')
        for sp in ax.spines.values():
            sp.set_edgecolor('#222')
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)
    else:
        st.info("💡 Run `python visualise.py --csv final_dna_v2.csv` "
                "to generate t-SNE embeddings for the map view.")

    with st.expander("Raw similarity scores"):
        st.dataframe(
            pd.DataFrame(list(all_scores.items()), columns=['Family', 'Cosine Similarity'])
            .sort_values('Cosine Similarity', ascending=False).reset_index(drop=True),
            use_container_width=True
        )