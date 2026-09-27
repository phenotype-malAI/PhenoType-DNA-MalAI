"""
significance_test.py — NEW

Paired comparison, across the 5 existing seeds, between SupCon's
open-world rejection rate and every baseline that has been run
per-seed (CADE, OpenMax, Energy, One-Class SVM, and the MSP-thresholded
CrossEntropy baseline once multi_seed_ablation.py has produced it).
Answers "is the 84.6% vs 70.0%-style gap real, or within noise?"

Two tests are reported, since n=5 is small enough that neither alone
is fully convincing:
  - Paired t-test (scipy.stats.ttest_rel): assumes the per-seed
    differences are roughly normal; standard, easy to defend, but
    shakier with only 5 pairs.
  - Wilcoxon signed-rank test (scipy.stats.wilcoxon): distribution-free,
    more appropriate for n=5, but has very limited power at this
    sample size, so a non-significant result here is uninformative --
    report both rather than picking whichever looks better.

Also reports a percentile bootstrap 95% CI on SupCon's MEAN rejection
rate, resampling across the 5 seed-level point estimates (with
replacement, 10,000 resamples). NOTE: this is a bootstrap over
per-seed summary statistics, not over the underlying 250 held-out
samples themselves, because the baseline scripts (cade_baseline.py,
openmax_baseline.py, etc.) currently save only per-family aggregate
counts, not per-sample accept/reject flags. A finer per-sample
bootstrap is possible but would require adding a per-sample dump to
each of those scripts first -- flagged here rather than silently
implied.

Usage (run after cade_baseline.py, openmax_baseline.py,
energy_baseline.py, oneclass_svm_baseline.py, and
multi_seed_ablation.py have all been run across seeds 0-4):
    python significance_test.py \
        --multi_seed_csv outputs/multi_seed/multi_seed_results.csv \
        --out_dir outputs/significance
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [0, 1, 2, 3, 4]

# (label, path template with {seed} placeholder, csv filename, column, row filter)
BASELINE_SOURCES = [
    ('CADE (MAD-normalised distance)',
     'outputs/cade_baseline/seed_{seed}/cade_openworld.csv'),
    ('OpenMax (Weibull-tail on logits)',
     'outputs/openmax_baseline/seed_{seed}/openmax_openworld.csv'),
    ('Energy-based OOD (Liu et al. 2020)',
     'outputs/energy_baseline/seed_{seed}/energy_openworld.csv'),
    ('One-Class SVM (per-family)',
     'outputs/oneclass_svm_baseline/seed_{seed}/oneclass_svm_openworld.csv'),
    ('MSP-thresholded CrossEntropy',
     'outputs/ablation_multi_seed/seed_{seed}/ablation_openworld.csv'),
]


def load_overall_rejection(path_template):
    """Reads the 'OVERALL' row's Rejection_Rate_% from each seed's CSV.
    Returns None (and prints a warning) if any seed's file is missing,
    rather than silently dropping seeds -- a paired test needs the SAME
    5 seeds on both sides."""
    values = []
    for seed in SEEDS:
        path = Path(path_template.format(seed=seed))
        if not path.exists():
            print(f"  [skip] missing: {path}")
            return None
        df = pd.read_csv(path)
        row = df[df['Family'] == 'OVERALL']
        if row.empty:
            print(f"  [skip] no OVERALL row in: {path}")
            return None
        values.append(float(row['Rejection_Rate_%'].iloc[0]))
    return np.array(values)


def bootstrap_ci(values, n_resamples=10000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_resamples)
    ])
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    supcon_df = pd.read_csv(args.multi_seed_csv).sort_values('seed')
    supcon_values = supcon_df['open_world_rejection_pct'].values
    print(f"SupCon per-seed rejection rates: {supcon_values}")
    print(f"SupCon mean +/- std: {supcon_values.mean():.2f} +/- {supcon_values.std():.2f}")

    ci_lo, ci_hi = bootstrap_ci(supcon_values)
    print(f"SupCon 95% bootstrap CI (over the 5 seed means): [{ci_lo:.2f}, {ci_hi:.2f}]")

    results = []
    for label, path_template in BASELINE_SOURCES:
        print(f"\n--- {label} ---")
        baseline_values = load_overall_rejection(path_template)
        if baseline_values is None:
            print(f"  Skipping — not all 5 seeds available yet for this baseline.")
            continue

        print(f"  Per-seed rejection rates: {baseline_values}")
        print(f"  Mean +/- std: {baseline_values.mean():.2f} +/- {baseline_values.std():.2f}")

        diffs = supcon_values - baseline_values
        t_stat, t_p = stats.ttest_rel(supcon_values, baseline_values)
        try:
            w_stat, w_p = stats.wilcoxon(supcon_values, baseline_values)
        except ValueError as e:
            # Wilcoxon fails if all differences are zero or n too small for exact test
            w_stat, w_p = float('nan'), float('nan')
            print(f"  [Wilcoxon could not be computed: {e}]")

        base_ci_lo, base_ci_hi = bootstrap_ci(baseline_values)

        print(f"  Mean difference (SupCon - baseline): {diffs.mean():.2f} pts")
        print(f"  Paired t-test:       t={t_stat:.3f}, p={t_p:.4f}")
        print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.4f}" if not np.isnan(w_p)
              else "  Wilcoxon signed-rank: not computable at n=5")

        results.append({
            'Baseline': label,
            'SupCon_mean': round(supcon_values.mean(), 2),
            'Baseline_mean': round(baseline_values.mean(), 2),
            'Mean_diff_pts': round(diffs.mean(), 2),
            'Baseline_95pct_CI_low': round(base_ci_lo, 2),
            'Baseline_95pct_CI_high': round(base_ci_hi, 2),
            'paired_t_stat': round(t_stat, 4),
            'paired_t_p': round(t_p, 4),
            'wilcoxon_stat': w_stat if not np.isnan(w_p) else None,
            'wilcoxon_p': round(w_p, 4) if not np.isnan(w_p) else None,
        })

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(out_dir / 'significance_results.csv', index=False)
        print(f"\n\nSaved -> {out_dir / 'significance_results.csv'}")
        print(summary.to_string(index=False))
    else:
        print("\nNo baselines had all 5 seeds available -- nothing to compare yet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_seed_csv', default='outputs/multi_seed/multi_seed_results.csv')
    parser.add_argument('--out_dir', default='outputs/significance')
    args = parser.parse_args()
    run(args)
