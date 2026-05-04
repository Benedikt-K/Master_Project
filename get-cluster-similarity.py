#!/usr/bin/env python3
"""
Cluster Composition Comparison Script
======================================
Compares how CRISPR-evOr arrays are grouped across different clustering
configurations, using the prediction CSVs already produced by
compare_crispr_orientations.py.
 
Usage:
    python compare_clusterings.py [--results-dir DIR] [--out-dir DIR]
 
Defaults:
    --results-dir  ./crispr_comparison_results/
    --out-dir      ./crispr_comparison_results/cluster_analysis/
"""
 
import argparse
from pathlib import Path
from itertools import combinations
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import adjusted_rand_score
from scipy.stats import chi2_contingency
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
 
CLUSTERING_NAMES = [
    "out-cluster-size-50",
    "out-cluster-size-150",
    "out-cluster-size-300",
    "out-SP-cluster-smaller200",
    "out-sp_style_clusters",
]
 
SHORT_NAMES = {
    "out-cluster-size-50":       "size-50",
    "out-cluster-size-150":      "size-150",
    "out-cluster-size-300":      "size-300",
    "out-SP-cluster-smaller200": "SP-<200",
    "out-sp_style_clusters":     "SP-style",
}
 
COLORS = {
    "out-cluster-size-50":       "#4C72B0",
    "out-cluster-size-150":      "#55A868",
    "out-cluster-size-300":      "#C44E52",
    "out-SP-cluster-smaller200": "#DD8452",
    "out-sp_style_clusters":     "#8172B2",
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────
 
def load_predictions(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the pre-built evor prediction CSVs."""
    dfs = {}
    for name in CLUSTERING_NAMES:
        path = results_dir / f"evor_{name}_predictions.csv"
        if not path.exists():
            print(f"  [WARN] Not found: {path}")
            continue
        df = pd.read_csv(path, dtype=str)
        # exclude singletons for cluster-composition analysis
        df_clustered = df[df["group_name"] != "singleton"].copy()
        dfs[name] = df_clustered
        n_arrays   = len(df_clustered)
        n_clusters = df_clustered["group_name"].nunique()
        n_sing     = (df["group_name"] == "singleton").sum()
        # report how many were flip-corrected (SP-style only)
        if "was_flipped" in df_clustered.columns:
            n_flip = (df_clustered["was_flipped"] == "True").sum()
            print(f"  [{SHORT_NAMES[name]}]  "
                  f"{n_arrays} arrays in {n_clusters} clusters  "
                  f"(+{n_sing} singletons excluded, {n_flip} flip-corrected)")
        else:
            print(f"  [{SHORT_NAMES[name]}]  "
                  f"{n_arrays} arrays in {n_clusters} clusters  "
                  f"(+{n_sing} singletons excluded)")
    return dfs
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. CLUSTER SIZE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_cluster_size_distributions(dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
 
    for ax, (name, df) in zip(axes, dfs.items()):
        sizes = df.groupby("group_name").size()
        short = SHORT_NAMES[name]
        color = COLORS[name]
 
        ax.hist(sizes, bins=50, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(short, fontsize=13, fontweight="bold")
        ax.set_xlabel("Arrays per cluster", fontsize=10)
        ax.set_ylabel("Number of clusters", fontsize=10)
 
        stats_txt = (f"n clusters: {len(sizes)}\n"
                     f"median size: {sizes.median():.0f}\n"
                     f"max size: {sizes.max()}\n"
                     f"mean size: {sizes.mean():.1f}")
        ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
 
    fig.suptitle("Cluster size distributions across clustering strategies",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = out_dir / "cluster_size_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. ADJUSTED RAND INDEX
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_ari_matrix(dfs: dict) -> pd.DataFrame:
    """
    Compute pairwise ARI scores.
    Only arrays present in BOTH clusterings are used (inner join).
    Singletons are already excluded from dfs.
    """
    names = list(dfs.keys())
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
 
    for n in names:
        matrix.loc[n, n] = 1.0
 
    for a, b in combinations(names, 2):
        merged = dfs[a][["array_name", "group_name"]].rename(
            columns={"group_name": "grp_a"}
        ).merge(
            dfs[b][["array_name", "group_name"]].rename(
                columns={"group_name": "grp_b"}),
            on="array_name", how="inner"
        )
        if len(merged) < 2:
            continue
        ari = adjusted_rand_score(merged["grp_a"], merged["grp_b"])
        matrix.loc[a, b] = round(ari, 4)
        matrix.loc[b, a] = round(ari, 4)
 
    return matrix
 
 
def plot_ari_heatmap(ari_matrix: pd.DataFrame, out_dir: Path):
    short_labels = [SHORT_NAMES[n] for n in ari_matrix.index]
    data = ari_matrix.values.astype(float)
 
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Adjusted Rand Index")
 
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)
 
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = data[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
 
    ax.set_title("Pairwise Adjusted Rand Index\n(1 = identical clustering, 0 = random)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "ari_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. SHARED / SPLIT / MERGED CLUSTER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
 
def cluster_overlap_stats(df_a: pd.DataFrame, df_b: pd.DataFrame,
                           name_a: str, name_b: str) -> dict:
    """
    For each cluster in A, check what happens to its members in B:
      - 'intact'  : all members are in the same B-cluster
      - 'split'   : members are spread across multiple B-clusters
    And vice versa (merged = multiple A-clusters collapse into one B-cluster).
    """
    merged = df_a[["array_name", "group_name"]].rename(
        columns={"group_name": "grp_a"}
    ).merge(
        df_b[["array_name", "group_name"]].rename(
            columns={"group_name": "grp_b"}),
        on="array_name", how="inner"
    )
 
    # From A's perspective: how many distinct B-clusters does each A-cluster map to?
    a_to_b = merged.groupby("grp_a")["grp_b"].nunique()
    n_intact_a = (a_to_b == 1).sum()
    n_split_a  = (a_to_b > 1).sum()
 
    # From B's perspective: how many distinct A-clusters does each B-cluster map to?
    b_to_a = merged.groupby("grp_b")["grp_a"].nunique()
    n_intact_b  = (b_to_a == 1).sum()
    n_merged_b  = (b_to_a > 1).sum()
 
    return {
        "pair": f"{SHORT_NAMES[name_a]} vs {SHORT_NAMES[name_b]}",
        "n_shared_arrays": len(merged),
        "n_clusters_a": merged["grp_a"].nunique(),
        "n_clusters_b": merged["grp_b"].nunique(),
        f"intact_in_{SHORT_NAMES[name_b]}": int(n_intact_a),
        f"split_in_{SHORT_NAMES[name_b]}":  int(n_split_a),
        f"intact_in_{SHORT_NAMES[name_a]}": int(n_intact_b),
        f"merged_in_{SHORT_NAMES[name_a]}": int(n_merged_b),
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. SP-CLUSTER FORWARD BIAS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
 
def analyse_sp_forward_bias(dfs: dict, out_dir: Path):
    """
    Investigate whether SP-cluster's Forward bias is universal or confined
    to specific clusters / array subsets.
    """
    sp_name = "out-SP-cluster-smaller200"
    ref_name = "out-cluster-size-150"   # most stable reference
 
    if sp_name not in dfs or ref_name not in dfs:
        print("  [SKIP] SP or reference clustering not loaded.")
        return
 
    sp_df  = dfs[sp_name][["array_name", "group_name", "evor_direction"]]
    ref_df = dfs[ref_name][["array_name", "group_name", "evor_direction"]].rename(
        columns={"group_name": "ref_group", "evor_direction": "ref_direction"})
 
    merged = sp_df.merge(ref_df, on="array_name", how="inner")
 
    # ── 4a. Global direction counts ─────────────────────────────────────────
    print("\n  SP-cluster direction distribution:")
    sp_counts = merged["evor_direction"].value_counts()
    print(sp_counts.to_string())
 
    print("\n  Reference (size-150) direction distribution:")
    ref_counts = merged["ref_direction"].value_counts()
    print(ref_counts.to_string())
 
    # ── 4b. Per SP-cluster: Forward fraction ────────────────────────────────
    per_cluster = merged.groupby("group_name")["evor_direction"].apply(
        lambda x: (x == "Forward").mean()
    ).reset_index()
    per_cluster.columns = ["sp_cluster", "fwd_fraction"]
 
    all_fwd  = (per_cluster["fwd_fraction"] == 1.0).sum()
    all_rev  = (per_cluster["fwd_fraction"] == 0.0).sum()
    mixed    = ((per_cluster["fwd_fraction"] > 0) &
                (per_cluster["fwd_fraction"] < 1)).sum()
 
    print(f"\n  SP-cluster forward-fraction per cluster:")
    print(f"    100% Forward : {all_fwd} clusters")
    print(f"    100% Reverse : {all_rev} clusters")
    print(f"    Mixed        : {mixed} clusters")
 
    # ── 4c. Crosstab: SP orientation vs reference orientation ───────────────
    ct = pd.crosstab(merged["evor_direction"], merged["ref_direction"])
    print("\n  SP direction vs reference direction (crosstab):")
    print(ct.to_string())
 
    # Chi-squared test to quantify the association
    if ct.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(ct)
        print(f"\n  Chi² = {chi2:.2f}, p = {p:.2e}")
 
    # ── 4d. Plot: histogram of per-SP-cluster Forward fraction ───────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_cluster["fwd_fraction"], bins=20,
            color=COLORS[sp_name], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Fraction of arrays called 'Forward' within SP-cluster", fontsize=11)
    ax.set_ylabel("Number of SP-clusters", fontsize=11)
    ax.set_title("SP-cluster: Forward fraction per cluster\n"
                 "(a spike at 1.0 = all clusters unanimously call Forward)",
                 fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    path = out_dir / "sp_cluster_forward_bias.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
 
    # ── 4e. Do arrays that the reference calls 'Reverse' cluster together? ───
    rev_in_ref = merged[merged["ref_direction"] == "Reverse"].copy()
    print(f"\n  Arrays the reference calls 'Reverse': {len(rev_in_ref)}")
    print(f"  How SP-cluster labels them:")
    print(rev_in_ref["evor_direction"].value_counts().to_string())
    print(f"  How many distinct SP-clusters do they fall into: "
          f"{rev_in_ref['group_name'].nunique()}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. SP-STYLE FLIP CORRECTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
 
def analyse_sp_style_correction(dfs: dict, out_dir: Path):
    """
    For the SP-style clustering (which had orientation normalised before
    clustering), show:
      a) How many arrays had their direction corrected (was_flipped=True)
      b) Pre- vs post-correction direction distribution
      c) Agreement with the reference (size-150) before and after correction
      d) A side-by-side bar chart of pre/post distributions
    """
    sp_style_name = "out-sp_style_clusters"
    ref_name      = "out-cluster-size-150"
 
    if sp_style_name not in dfs:
        print(f"  [SKIP] {sp_style_name} not loaded.")
        return
    if ref_name not in dfs:
        print(f"  [SKIP] Reference {ref_name} not loaded.")
        return
 
    sp_df  = dfs[sp_style_name].copy()
    ref_df = dfs[ref_name][["array_name", "evor_direction"]].rename(
        columns={"evor_direction": "ref_direction"})
 
    # ── 5a. Flip correction summary ──────────────────────────────────────────
    if "was_flipped" not in sp_df.columns or "evor_direction_tool" not in sp_df.columns:
        print("  [SKIP] SP-style CSV missing was_flipped / evor_direction_tool columns.")
        print("         Re-run compare-results.py with --sp-metadata to generate them.")
        return
 
    n_total   = len(sp_df)
    n_flipped = (sp_df["was_flipped"] == "True").sum()
    print(f"\n  SP-style arrays: {n_total} total, {n_flipped} had direction corrected "
          f"({100*n_flipped/n_total:.1f}%)")
 
    # ── 5b. Pre/post direction distributions ────────────────────────────────
    pre_counts  = sp_df["evor_direction_tool"].value_counts()
    post_counts = sp_df["evor_direction"].value_counts()
 
    print("\n  Direction distribution BEFORE correction (raw tool output):")
    print(pre_counts.to_string())
    print("\n  Direction distribution AFTER correction (genomic strand):")
    print(post_counts.to_string())
 
    # ── 5c. Agreement with reference before and after ────────────────────────
    merged = sp_df.merge(ref_df, on="array_name", how="inner")
    comparable = merged[
        (~merged["ref_direction"].isin(["Unknown", "Singleton"])) &
        (~merged["evor_direction"].isin(["Unknown", "Singleton"]))
    ]
    comparable_pre = merged[
        (~merged["ref_direction"].isin(["Unknown", "Singleton"])) &
        (~merged["evor_direction_tool"].isin(["Unknown", "Singleton"]))
    ]
 
    if len(comparable_pre) > 0:
        agree_pre  = (comparable_pre["evor_direction_tool"] ==
                      comparable_pre["ref_direction"]).sum()
        pct_pre    = 100 * agree_pre / len(comparable_pre)
        print(f"\n  Agreement with {ref_name} BEFORE correction: "
              f"{agree_pre}/{len(comparable_pre)} ({pct_pre:.1f}%)")
 
    if len(comparable) > 0:
        agree_post = (comparable["evor_direction"] == comparable["ref_direction"]).sum()
        pct_post   = 100 * agree_post / len(comparable)
        print(f"  Agreement with {ref_name} AFTER  correction: "
              f"{agree_post}/{len(comparable)} ({pct_post:.1f}%)")
 
    # ── 5d. Crosstab after correction vs reference ───────────────────────────
    ct = pd.crosstab(merged["evor_direction"], merged["ref_direction"],
                     rownames=["SP-style (corrected)"],
                     colnames=[f"Reference ({ref_name})"])
    print("\n  Crosstab (post-correction SP-style vs reference):")
    print(ct.to_string())
 
    # ── 5e. Side-by-side bar chart ───────────────────────────────────────────
    cats = ["Forward", "Reverse", "Unknown"]
    pre_vals  = [pre_counts.get(c, 0)  for c in cats]
    post_vals = [post_counts.get(c, 0) for c in cats]
 
    x = range(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_pre  = ax.bar([i - width/2 for i in x], pre_vals,  width,
                       label="Before correction (tool output)",
                       color=COLORS.get(sp_style_name, "#8172B2"),
                       alpha=0.6, edgecolor="white")
    bars_post = ax.bar([i + width/2 for i in x], post_vals, width,
                       label="After correction (genomic strand)",
                       color=COLORS.get(sp_style_name, "#8172B2"),
                       alpha=1.0, edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Number of arrays", fontsize=11)
    ax.set_title("SP-style clustering: orientation before vs after flip correction",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    for bar in list(bars_pre) + list(bars_post):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 10,
                    str(int(h)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "sp_style_flip_correction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
 
def print_and_save_summary(ari_matrix: pd.DataFrame,
                           overlap_rows: list[dict],
                           out_dir: Path):
    print("\n=== ARI Matrix ===")
    short_matrix = ari_matrix.rename(
        index=SHORT_NAMES, columns=SHORT_NAMES)
    print(short_matrix.to_string())
    ari_matrix.rename(index=SHORT_NAMES, columns=SHORT_NAMES).to_csv(
        out_dir / "ari_matrix.csv")
 
    print("\n=== Cluster overlap summary ===")
    overlap_df = pd.DataFrame(overlap_rows)
    print(overlap_df.to_string(index=False))
    overlap_df.to_csv(out_dir / "cluster_overlap_summary.csv", index=False)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Compare CRISPR-evOr cluster compositions across "
                    "clustering strategies."
    )
    parser.add_argument(
        "--results-dir", default="./crispr_comparison_results",
        help="Directory containing the evor_*_predictions.csv files "
             "(output of compare_crispr_orientations.py)"
    )
    parser.add_argument(
        "--out-dir", default="./crispr_comparison_results/cluster_analysis",
        help="Directory to write analysis outputs"
    )
    args = parser.parse_args()
 
    results_dir = Path(args.results_dir).resolve()
    out_dir     = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Results directory : {results_dir}")
    print(f"Output directory  : {out_dir}\n")
 
    # ── Load ──────────────────────────────────────────────────────────────────
    print("=== Loading prediction CSVs ===")
    dfs = load_predictions(results_dir)
    if len(dfs) < 2:
        print("Need at least 2 clusterings to compare. Exiting.")
        return
 
    # ── 1. Size distributions ─────────────────────────────────────────────────
    print("\n=== Cluster size distributions ===")
    plot_cluster_size_distributions(dfs, out_dir)
 
    # ── 2. ARI ───────────────────────────────────────────────────────────────
    print("\n=== Adjusted Rand Index ===")
    ari_matrix = compute_ari_matrix(dfs)
    plot_ari_heatmap(ari_matrix, out_dir)
 
    # ── 3. Overlap stats ──────────────────────────────────────────────────────
    print("\n=== Cluster overlap (split / merge analysis) ===")
    overlap_rows = []
    for name_a, name_b in combinations(dfs.keys(), 2):
        stats = cluster_overlap_stats(dfs[name_a], dfs[name_b], name_a, name_b)
        overlap_rows.append(stats)
        print(f"\n  {stats['pair']}")
        for k, v in stats.items():
            if k != "pair":
                print(f"    {k}: {v}")
 
    # ── 4. SP Forward bias ───────────────────────────────────────────────────
    print("\n=== SP-cluster Forward bias analysis ===")
    analyse_sp_forward_bias(dfs, out_dir)
 
    # ── 5. SP-style flip correction analysis ─────────────────────────────────
    print("\n=== SP-style clustering: flip correction analysis ===")
    analyse_sp_style_correction(dfs, out_dir)
 
    # ── 6. Summary ───────────────────────────────────────────────────────────
    print_and_save_summary(ari_matrix, overlap_rows, out_dir)
 
    print(f"\n✓ All outputs written to: {out_dir}")
 
 
if __name__ == "__main__":
    main()