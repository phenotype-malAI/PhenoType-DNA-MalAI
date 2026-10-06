#!/usr/bin/env python3
"""
PHENOTYPE reviewer-requested analyses (ICISS-2026). INFERENCE ONLY, no retraining.

  Task 0  file check + consistency check against the saved test_scores.csv
  Task A  per-family one-vs-rest ROC (score = cosine to that family's centroid)
  Task B  empirical accuracy vs max centroid cosine (equal-count bins)
  Task C  latency / throughput (stages A, B and optionally C)

Usage (repo = the unzipped PhenoType_v3 folder):
  python reviewer_analyses.py --repo . --out outputs/reviewer_analyses
  python reviewer_analyses.py --repo . --tasks A B          # skip latency
  python reviewer_analyses.py --repo . --tasks C --raw_reports /path/to/WinMET_json_dir
"""
import argparse, glob, importlib.util, json, os, platform, subprocess, sys, time, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

THETA = 0.986
FAMS = ["AgentTesla", "Formbook", "Lokibot", "Redline", "njRAT"]
REP_SEED = 42            # "representative run" lives in outputs/single_run
V3 = {"closed_acc_mean": 71.5, "closed_acc_std": 1.5, "rej_mean": 84.6, "rej_std": 4.4,
      "auroc_known_vs_novel_seed42": 0.85, "acc_seed42": 74.8,
      "corr_vs_incorr_auroc_seed42": 0.67, "corr_vs_incorr_auroc_mean": 0.49,
      "corr_vs_incorr_auroc_range": (0.38, 0.61)}

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 120})


# ----------------------------------------------------------------- helpers
def ckpt_dir(repo, seed):
    return repo / "outputs" / ("single_run" if seed == REP_SEED else f"multi_seed/seed_{seed}")


def load_model(repo, seed, device):
    from model import BehaviourEncoder
    d = ckpt_dir(repo, seed)
    m = BehaviourEncoder()
    m.load_state_dict(torch.load(d / "behaviour_encoder.pt", map_location="cpu", weights_only=False))
    m.to(device).eval()
    cents = torch.load(d / "family_centroids.pt", map_location="cpu", weights_only=False)
    C = torch.stack([cents[i] for i in range(5)]).to(device)   # (5,256), rows L2-normalised
    return m, C


@torch.inference_mode()
def embed(model, tokens_np, device, bs=16):
    out = []
    for i in range(0, len(tokens_np), bs):
        t = torch.as_tensor(tokens_np[i:i + bs], dtype=torch.long, device=device)
        out.append(model(t).float().cpu())
    return torch.cat(out)


def env_info(device_names):
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    cpu = sh("lscpu | grep -m1 'Model name' | sed 's/.*: *//'") or platform.processor()
    mem = ""
    try:
        mem = f"{int(open('/proc/meminfo').readline().split()[1]) / 1024 / 1024:.1f} GB"
    except Exception:
        pass
    info = {"cpu_model": cpu, "cpu_logical_cores": os.cpu_count(), "ram": mem,
            "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "devices_timed": device_names}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def savefig(fig, base):
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ----------------------------------------------------------------- Task 0 + scores
def regenerate_scores(repo, seeds, out, device, log):
    """Inference with saved checkpoints on the exact test split of each seed."""
    from dataset import make_splits, IDX_TO_FAMILY
    csv = repo / "final_dna_v2.csv"
    meta = pd.read_csv(csv, usecols=["sha256", "family"])
    res, checks = {}, []
    for s in seeds:
        d = ckpt_dir(repo, s)
        model, C = load_model(repo, s, device)
        _, _, test = make_splits(str(csv), seed=s)
        fp = embed(model, test.tokens, device)
        cos = (fp @ C.cpu().T).numpy()
        y = test.labels.copy()
        pred = cos.argmax(1)
        df = pd.DataFrame(cos, columns=[f"cos_{f}" for f in FAMS])
        df.insert(0, "sha256", test.sha256)
        df.insert(1, "true_family", [IDX_TO_FAMILY[i] for i in y])
        df.insert(2, "pred_family", [IDX_TO_FAMILY[i] for i in pred])
        df["max_cos"] = cos.max(1)
        df["correct"] = (pred == y).astype(int)
        (out / "per_sample").mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_sample" / f"known_test_scores_seed{s}.csv", index=False, float_format="%.6f")
        # ---- consistency with the scores saved at training time
        ts = pd.read_csv(d / "test_scores.csv")
        same_lab = bool((ts["true_family"].values == df["true_family"].values).all())
        max_abs = float(np.abs(ts["best_score"].values - df["max_cos"].values).max())
        pred_match = float((ts["predicted_family"].values == df["pred_family"].values).mean())
        acc = float(df["correct"].mean())
        rep = json.load(open(d / "test_report.json"))["test_accuracy"]
        checks.append(dict(seed=s, n=len(df), labels_identical_order=same_lab,
                           max_abs_diff_best_score=max_abs, pred_agreement=pred_match,
                           acc_regenerated=acc, acc_saved_report=rep))
        log(f"  seed {s}: n={len(df)} labels_match={same_lab} max|dscore|={max_abs:.2e} "
            f"pred_agree={pred_match:.4f} acc={acc:.4f} (saved {rep:.4f})")
        res[s] = dict(cos=cos, y=y, pred=pred, df=df, model=model, C=C)
    pd.DataFrame(checks).to_csv(out / "consistency_check.csv", index=False)
    return res


def held_out_sanity(repo, res, device, out, log):
    """Seed-42 known-vs-novel AUROC on the 250 held-out samples, vs saved open_world_auroc.json."""
    ho = pd.read_csv(repo / "data" / "held_out_families.csv")
    tok = ho[[f"tok_{i}" for i in range(1200)]].values.astype("int64") if "tok_0" in ho.columns else None
    if tok is None:   # held_out_families.csv has no token columns -> tokens live elsewhere
        log("  held-out CSV has no tok_* columns; trying held_out_full_scores.csv instead")
        hs = pd.read_csv(repo / "outputs" / "single_run" / "held_out_full_scores.csv")
        novel = hs["best_score"].values
    else:
        r = res[REP_SEED]
        fp = embed(r["model"], tok, device)
        novel = (fp @ r["C"].cpu().T).numpy().max(1)
    known = res[REP_SEED]["cos"].max(1)
    auroc = roc_auc_score(np.r_[np.ones(len(known)), np.zeros(len(novel))], np.r_[known, novel])
    saved = json.load(open(repo / "outputs" / "single_run" / "open_world_auroc.json"))
    out_d = dict(auroc_regenerated=float(auroc), auroc_saved=saved["auroc"],
                 tpr_at_theta_regenerated=float((known >= THETA).mean()), tpr_saved=saved["tpr_at_deployed"],
                 fpr_at_theta_regenerated=float((novel >= THETA).mean()), fpr_saved=saved["fpr_at_deployed"],
                 n_known=len(known), n_novel=len(novel))
    json.dump(out_d, open(out / "heldout_sanity_seed42.json", "w"), indent=2)
    log(f"  known-vs-novel AUROC seed 42: regenerated {auroc:.4f} vs saved {saved['auroc']:.4f}; "
        f"novel rejected {(1 - out_d['fpr_at_theta_regenerated']) * 100:.1f}%")
    return out_d


# ----------------------------------------------------------------- Task A
def task_A(res, seeds, out, log):
    rows, per_seed_auc = [], {f: {} for f in FAMS}
    for s in seeds:
        for k, f in enumerate(FAMS):
            yk = (res[s]["y"] == k).astype(int)
            per_seed_auc[f][s] = roc_auc_score(yk, res[s]["cos"][:, k])
    r42 = res[REP_SEED]
    cols = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    for k, f in enumerate(FAMS):
        yk = (r42["y"] == k).astype(int)
        sc = r42["cos"][:, k]
        fpr, tpr, _ = roc_curve(yk, sc)
        auc42 = per_seed_auc[f][REP_SEED]
        vals = np.array([per_seed_auc[f][s] for s in seeds if s != REP_SEED])
        m, sd = vals.mean(), vals.std(ddof=1)
        ax.plot(fpr, tpr, color=cols[k], lw=1.4,
                label=f"{f}  AUC {auc42:.3f}  (seeds 0-4: {m:.3f}$\\pm${sd:.3f})")
        ax.plot((sc[yk == 0] >= THETA).mean(), (sc[yk == 1] >= THETA).mean(), marker="o", ms=5,
                mfc="white", mec=cols[k], mew=1.4, ls="none")
        rows.append(dict(family=f, n_pos_seed42=int(yk.sum()), n_neg_seed42=int((1 - yk).sum()),
                         auc_seed42=auc42, auc_mean_seeds0_4=m, auc_std_seeds0_4=sd,
                         **{f"auc_seed{s}": per_seed_auc[f][s] for s in seeds if s != REP_SEED},
                         tpr_at_theta_seed42=float((sc[yk == 1] >= THETA).mean()),
                         fpr_at_theta_seed42=float((sc[yk == 0] >= THETA).mean())))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.plot([], [], marker="o", mfc="white", mec="k", ls="none", label=f"$\\theta$ = {THETA}")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Per-family one-vs-rest ROC (known test set, seed 42)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, framealpha=0.9)
    savefig(fig, out / "fig_roc_per_family")
    df = pd.DataFrame(rows)
    df.to_csv(out / "roc_auc_per_family.csv", index=False, float_format="%.4f")
    log("\n" + df[["family", "auc_seed42", "auc_mean_seeds0_4", "auc_std_seeds0_4"]].to_string(index=False,
        float_format=lambda v: f"{v:.3f}"))
    return df


# ----------------------------------------------------------------- Task B
def quantile_bins(x, correct, nb):
    """Equal-count bins by sorted rank (stable, so ties never produce empty bins)."""
    order = np.argsort(x, kind="stable")
    parts = np.array_split(order, nb)
    rows = []
    for i, idx in enumerate(parts):
        k, n = int(correct[idx].sum()), len(idx)
        lo, hi = wilson(k, n)
        rows.append(dict(bin=i + 1, x_min=float(x[idx].min()), x_max=float(x[idx].max()),
                         x_median=float(np.median(x[idx])), count=n, n_correct=k,
                         accuracy=k / n, wilson95_lo=lo, wilson95_hi=hi))
    return pd.DataFrame(rows)


def task_B(res, seeds, out, nb, log):
    r = res[REP_SEED]
    x42, c42 = r["df"]["max_cos"].values, r["df"]["correct"].values
    pool_x = np.concatenate([res[s]["df"]["max_cos"].values for s in seeds if s != REP_SEED])
    pool_c = np.concatenate([res[s]["df"]["correct"].values for s in seeds if s != REP_SEED])
    b42 = quantile_bins(x42, c42, nb);  b42.insert(0, "scope", "seed42")
    bpool = quantile_bins(pool_x, pool_c, nb); bpool.insert(0, "scope", "pooled_seeds0-4")
    pd.concat([b42, bpool]).to_csv(out / "acc_vs_cosine_bins.csv", index=False, float_format="%.6f")

    # correct-vs-incorrect AUROC of max-cosine
    au = {s: roc_auc_score(res[s]["df"]["correct"], res[s]["df"]["max_cos"]) for s in seeds}
    a05 = np.array([au[s] for s in seeds if s != REP_SEED])
    a_pool = roc_auc_score(pool_c, pool_x)
    summ = dict(auroc_correct_vs_incorrect_seed42=au[REP_SEED],
                auroc_per_seed={str(s): au[s] for s in seeds},
                auroc_mean_seeds0_4=float(a05.mean()), auroc_std_seeds0_4=float(a05.std(ddof=1)),
                auroc_min_seeds0_4=float(a05.min()), auroc_max_seeds0_4=float(a05.max()),
                auroc_pooled_seeds0_4=float(a_pool), n_bins=nb,
                acc_seed42=float(c42.mean()), frac_known_above_theta_seed42=float((x42 >= THETA).mean()),
                acc_above_theta_seed42=float(c42[x42 >= THETA].mean()) if (x42 >= THETA).any() else None,
                acc_below_theta_seed42=float(c42[x42 < THETA].mean()) if (x42 < THETA).any() else None,
                v3_reference=V3)
    json.dump(summ, open(out / "acc_vs_cosine_summary.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, b, ttl in [(axes[0], b42, f"(a) seed 42, n={len(x42)}"),
                       (axes[1], bpool, f"(b) pooled seeds 0-4, n={len(pool_x)}")]:
        pos = np.arange(len(b))
        yerr = np.vstack([b["accuracy"] - b["wilson95_lo"], b["wilson95_hi"] - b["accuracy"]])
        ax.bar(pos, b["accuracy"], width=0.8, color="#8fb4d9", edgecolor="#3b6ea5", lw=0.6)
        ax.errorbar(pos, b["accuracy"], yerr=yerr, fmt="none", ecolor="k", elinewidth=0.7, capsize=2)
        for p, n, a in zip(pos, b["count"], b["accuracy"]):
            ax.text(p, 0.015, f"n={n}", ha="center", va="bottom", fontsize=5.5, rotation=90)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{m:.4f}" for m in b["x_median"]], rotation=60, ha="right", fontsize=6)
        # theta between the bins that straddle it (bins are ordered by cosine)
        below = int((b["x_max"] < THETA).sum())          # bins entirely below theta
        if below < len(b) and b["x_min"].iloc[below] <= THETA:   # theta falls INSIDE bin `below`
            lo, hi = b["x_min"].iloc[below], b["x_max"].iloc[below]
            xpos = below - 0.4 + 0.8 * (THETA - lo) / (hi - lo)
        else:                                            # theta falls in the gap between two bins
            xpos = below - 0.5
        ax.axvline(xpos, color="crimson", ls="--", lw=1.1)
        ax.text(xpos + 0.05, 1.04, r"$\theta$=0.986", color="crimson", fontsize=6.5, va="bottom")
        ax.axhline(b["n_correct"].sum() / b["count"].sum(), color="gray", ls=":", lw=0.8)
        ax.set_ylim(0, 1.12); ax.set_title(ttl)
        ax.set_xlabel("max centroid cosine (bin median; equal-count bins)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("empirical accuracy of argmax attribution")
    fig.tight_layout()
    savefig(fig, out / "fig_acc_vs_cosine")
    log(f"  correct-vs-incorrect AUROC: seed42={au[REP_SEED]:.3f}; seeds0-4 mean={a05.mean():.3f} "
        f"(range {a05.min():.2f}-{a05.max():.2f}); V3 reports 0.67 / 0.49 (0.38-0.61)")
    log("\n" + b42[["bin", "x_min", "x_max", "count", "accuracy"]].to_string(index=False,
        float_format=lambda v: f"{v:.4f}"))
    return summ


# ----------------------------------------------------------------- Task C
def _pcts(ts_ms):
    a = np.asarray(ts_ms)
    return float(np.median(a)), float(np.percentile(a, 95)), float(np.percentile(a, 99)), float(a.mean())


def time_fn(fn, sync, warmup, reps):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def task_C(repo, res, out, devices, batch_sizes, reps, warmup, raw_dir, log, cpu_warmup=None, cpu_batch_sizes=None):
    from model import BehaviourEncoder
    d = ckpt_dir(repo, REP_SEED)
    state = torch.load(d / "behaviour_encoder.pt", map_location="cpu", weights_only=False)
    cents = torch.load(d / "family_centroids.pt", map_location="cpu", weights_only=False)
    from dataset import make_splits
    _, _, test = make_splits(str(repo / "final_dna_v2.csv"), seed=REP_SEED)
    pool = torch.as_tensor(test.tokens, dtype=torch.long)
    rows = []
    nparams = None
    for dev in devices:
        device = torch.device(dev)
        model = BehaviourEncoder(); model.load_state_dict(state); model.to(device).eval()
        nparams = sum(p.numel() for p in model.parameters())
        C = torch.stack([cents[i] for i in range(5)]).to(device)
        sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
        for bs in (batch_sizes if dev == "cuda" else (cpu_batch_sizes or batch_sizes)):
            idx = torch.arange(bs) % len(pool)
            x = pool[idx].to(device)              # input already on device: H2D copy is excluded

            def stageA():
                model(x)

            def stageB():
                fp = model(x)
                sims = fp @ C.T
                mx, am = sims.max(dim=1)
                acc = mx >= THETA
                return am.cpu(), mx.cpu(), acc.cpu()   # decision returned to host (D2H included)

            with torch.inference_mode():
                for stage, fn in (("A", stageA), ("B", stageB)):
                    r = reps[bs] if dev == "cuda" else reps[("cpu", bs)]
                    w = (warmup if dev == "cuda" else (cpu_warmup if cpu_warmup is not None else warmup)) if bs == 1 else (warmup if dev == "cuda" else max(3, warmup // 10))
                    log(f"  ... timing {dev} stage {stage} bs {bs} (warmup {w}, reps {r})")
                    try:
                        ts = time_fn(fn, sync, w, r)
                    except RuntimeError as e:
                        log(f"  [{dev} stage {stage} bs {bs}] FAILED: {str(e)[:120]}")
                        continue
                    med, p95, p99, mean = _pcts(ts)
                    rows.append(dict(device=dev, stage=stage, batch=bs, warmup=w, reps=r, median_ms=med,
                                     p95_ms=p95, p99_ms=p99, mean_ms=mean, throughput_samples_s=bs / (med / 1e3)))
                    log(f"  {dev:4s} stage {stage} bs {bs:3d}: median {med:9.2f} ms  p95 {p95:9.2f}  "
                        f"-> {bs / (med / 1e3):9.1f} samples/s  (reps {r})")
                    pd.DataFrame(rows).to_csv(out / "latency_partial.csv", index=False, float_format="%.4f")
        # ---- Stage C: raw CAPE JSON -> tokens -> B   (only with raw reports)
        if raw_dir:
            rows += stage_C(repo, model, C, device, dev, test, raw_dir, warmup, log)
    if not raw_dir:
        log("  Stage C (end-to-end from raw CAPE JSON): NOT MEASURED, no raw reports supplied.")
    df = pd.DataFrame(rows)
    colab = ("COLAB_GPU" in os.environ) or ("COLAB_RELEASE_TAG" in os.environ)
    pre = "Colab " if colab else ""
    lab = {"cpu": f"{pre}CPU ({os.cpu_count()} vCPU, {torch.get_num_threads()} threads)"}
    if torch.cuda.is_available():
        lab["cuda"] = f"{pre}{torch.cuda.get_device_name(0)} GPU"
    if len(df):
        df["device_label"] = df["device"].map(lab)
    df.to_csv(out / "latency.csv", index=False, float_format="%.4f")
    # ready-to-paste tables
    if len(df):
        t = df[df.batch.isin([1, 64])].copy()
        md = ["| Device | Stage | Batch | Median (ms) | P95 (ms) | Throughput (samples/s) |",
              "|---|---|---:|---:|---:|---:|"]
        tex = [r"\begin{tabular}{llrrrr}", r"\hline", r"Device & Stage & Batch & Median (ms) & P95 (ms) & Throughput (samples/s) \\", r"\hline"]
        for _, r in t.iterrows():
            md.append(f"| {r.device_label} | {r.stage} | {int(r.batch)} | {r.median_ms:.2f} | {r.p95_ms:.2f} | {r.throughput_samples_s:.1f} |")
            tex.append(f"{r.device_label} & {r.stage} & {int(r.batch)} & {r.median_ms:.2f} & {r.p95_ms:.2f} & {r.throughput_samples_s:.1f} \\\\")
        tex += [r"\hline", r"\end{tabular}"]
        (out / "latency_table.md").write_text("\n".join(md) + "\n")
        (out / "latency_table.tex").write_text("\n".join(tex) + "\n")
    return df, nparams


def stage_C(repo, model, C, device, dev, test, raw_dir, warmup, log):
    spec = importlib.util.spec_from_file_location("run_extraction", repo / "scripts" / "run_extraction.py")
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
    vocab = json.load(open(repo / "data" / "final_dna_v2_vocab.json"))
    files = {Path(p).stem: Path(p) for p in glob.glob(str(Path(raw_dir) / "**" / "*.json"), recursive=True)}
    shas = [s for s in test.sha256 if s in files]
    log(f"  Stage C: {len(shas)}/{len(test.sha256)} test reports found under {raw_dir}")
    if not shas:
        return []
    # fidelity check: re-tokenised raw report must equal the stored CSV tokens
    pos = {s: i for i, s in enumerate(test.sha256)}
    eq = np.mean([ex.tokenize(ex.extract_sequence(files[s]), vocab) == test.tokens[pos[s]].tolist() for s in shas[:50]])
    log(f"  Stage C tokenisation reproduces stored tokens for {eq * 100:.1f}% of 50 checked samples")
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)

    def one(s):
        seq = ex.extract_sequence(files[s])                       # JSON read+parse, HIGH_SIGNAL filter, time sort
        tok = torch.tensor(ex.tokenize(seq, vocab), dtype=torch.long).unsqueeze(0).to(device)
        fp = model(tok)
        mx, am = (fp @ C.T).max(dim=1)
        return am.cpu(), mx.cpu(), (mx >= THETA).cpu()

    with torch.inference_mode():
        for s in shas[:10]:
            one(s)
        ts = []
        for s in shas:
            sync(); t0 = time.perf_counter(); one(s); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    med, p95, p99, mean = _pcts(ts)
    log(f"  {dev:4s} stage C bs 1: median {med:.2f} ms p95 {p95:.2f} (n={len(ts)} distinct reports)")
    return [dict(device=dev, stage="C", batch=1, warmup=10, reps=len(ts), median_ms=med, p95_ms=p95, p99_ms=p99,
                 mean_ms=mean, throughput_samples_s=1 / (med / 1e3))]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="unzipped PhenoType_v3 folder")
    ap.add_argument("--out", default="outputs/reviewer_analyses")
    ap.add_argument("--tasks", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--raw_reports", default=None, help="folder with raw CAPE/WinMET <sha256>.json (enables Stage C)")
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 64])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--reps_bs1", type=int, default=1000)
    ap.add_argument("--reps_large", type=int, default=100, help="reps for batch>1 on GPU")
    ap.add_argument("--cpu_reps_large", type=int, default=20, help="reps for batch>1 on CPU")
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--cpu_reps_bs1", type=int, default=None, help="reps for batch 1 on CPU (default: --reps_bs1)")
    ap.add_argument("--cpu_warmup", type=int, default=None, help="warmup for batch 1 on CPU (default: --warmup)")
    ap.add_argument("--cpu_batch_sizes", nargs="+", type=int, default=None, help="batch sizes to time on CPU (default: --batch_sizes)")
    args = ap.parse_args()

    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    logf = open(out / "run_log.txt", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    seeds = sorted(set(args.seeds) | {REP_SEED})
    need_scores = any(t in args.tasks for t in "AB")
    dev_infer = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 70 + "\nTask 0: file check\n" + "=" * 70)
    inv = []
    for s in seeds:
        d = ckpt_dir(repo, s)
        for f in ("behaviour_encoder.pt", "family_centroids.pt", "test_scores.csv", "test_report.json"):
            inv.append(dict(seed=s, file=str((d / f).relative_to(repo)), exists=(d / f).exists()))
    for f in ("final_dna_v2.csv", "data/final_dna_v2_vocab.json", "data/held_out_families.csv",
              "model.py", "dataset.py", "attribute.py", "scripts/run_extraction.py",
              "outputs/single_run/held_out_full_scores.csv"):
        inv.append(dict(seed="-", file=f, exists=(repo / f).exists()))
    pd.DataFrame(inv).to_csv(out / "file_inventory.csv", index=False)
    log(pd.DataFrame(inv).to_string(index=False))

    res = None
    if need_scores:
        log("\nRegenerating per-sample cosines by INFERENCE from saved checkpoints "
            f"(device={dev_infer}, no training)")
        res = regenerate_scores(repo, seeds, out, dev_infer, log)
        log("\nHeld-out sanity check (seed 42)")
        try:
            held_out_sanity(repo, res, dev_infer, out, log)
        except Exception as e:
            log(f"  held-out sanity check skipped: {e}")
    if "A" in args.tasks:
        log("\n" + "=" * 70 + "\nTask A: per-family ROC\n" + "=" * 70)
        task_A(res, seeds, out, log)
    if "B" in args.tasks:
        log("\n" + "=" * 70 + "\nTask B: accuracy vs max centroid cosine\n" + "=" * 70)
        task_B(res, seeds, out, args.bins, log)
    if "C" in args.tasks:
        log("\n" + "=" * 70 + "\nTask C: latency / throughput\n" + "=" * 70)
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and not args.cpu_only else [])
        reps = {bs: (args.reps_bs1 if bs == 1 else args.reps_large) for bs in args.batch_sizes}
        reps.update({("cpu", bs): ((args.cpu_reps_bs1 or args.reps_bs1) if bs == 1 else args.cpu_reps_large)
                     for bs in set(args.batch_sizes) | set(args.cpu_batch_sizes or [])})
        df, npar = task_C(repo, res, out, devices, args.batch_sizes, reps, args.warmup, args.raw_reports, log,
                          cpu_warmup=args.cpu_warmup, cpu_batch_sizes=args.cpu_batch_sizes)
        env = env_info(devices); env["parameters"] = npar
        env["fp32_model_MB"] = round(npar * 4 / 1e6, 2) if npar else None
        json.dump(env, open(out / "environment.json", "w"), indent=2)
        log("\nEnvironment: " + json.dumps(env, indent=2))
    shutil.copy(__file__, out / Path(__file__).name)
    log("\nDone. Outputs in " + str(out))


if __name__ == "__main__":
    main()
