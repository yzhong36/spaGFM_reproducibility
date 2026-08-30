# Panel: figure3_supp_g_prep
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig2_c_spearman_tertiles_ridgefit.py
# Original filename: fig2_c_spearman_tertiles_ridgefit.py
# Last modified: 2026-08-18 10:29:10  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad + 317M inference set
# Output: data/work/fig2_c/spearman_long_2arm.csv
# Match evidence: Ridge(alpha=1.0) head that produces the per-perturbation Spearman rho table the figure script reads.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""
STEP 1 -- the Ridge fit. TWO ARMS: spaGFM (317M features) and GenePT*.

THIS IS A NEW RIDGE FIT, NOT A REPRODUCTION of the published figure. The rho
DEFINITION is carried over from ridge_spearman_3methods.py verbatim; the fit is new.

Recipe, unchanged from the published script:
  - true LFC read from the stored `stage2_best_mean_knn_target_lfc` layer
  - 10 perturbation-level CV splits; perturbations with >= MIN_CELLS (20) cells
  - per split: StandardScaler fit on the TRAIN fold -> Ridge(alpha=1.0) -> predict held-out
  - restricted to cells carrying a spaGFM embedding, so both arms see identical cells
  - per perturbation: concat held-out cells across all 10 splits, mean over cells ->
    one 500-d true and one 500-d predicted vector -> Spearman across the 500 genes
  - one rho per perturbation

THE ONE DELIBERATE DEPARTURE: spaGFM features come from the 317M inference set
(spatial_perturb_06_29_26/split_10_317M) instead of the published 2026-04-23 set.

BUILT-IN ISOLATION CHECK. The GenePT* arm uses `GPT_3_5_gene_embeddings` -- byte-for-byte
the same feature bank the published "MLP" arm used, under the same recipe. So IF the 317M
and 04-23 inference sets mark the same cells, GenePT* must reproduce the published MLP rho
exactly. That comparison is computed here and reported. It is the evidence for whether the
spaGFM delta is attributable to the embedding swap alone or is confounded by a cell-set
change -- rather than an assumption either way.

Outputs (all new, this directory only):
  spearman_long_2arm.csv       method, perturbation, spearman_r, n_test_cells
  ridge_vs_published_comparison.csv
  run_ridge_2arm.log
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/fig2_c"
H5AD = ("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw"
        "/perturb_fish_spatial.h5ad")
CACHE = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived/cache_spagfm317M_restricted.npz"
PUBLISHED = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived/published_spearman_long_ridge.csv"

# constants verbatim from ridge_spearman_3methods.py
NUM_SPLITS = 10
MIN_CELLS = 20
PERTURB_KEY = "perturbation"
Y_KEY = "stage2_best_mean_knn_target_lfc"
GENE_KEY = "GPT_3_5_gene_embeddings"
RIDGE_ALPHA = 1.0

lines: list[str] = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


def filter_min_cells(sub, min_cells=MIN_CELLS):
    counts = sub.obs[PERTURB_KEY].value_counts()
    keep = counts[counts >= min_cells].index
    return sub[sub.obs[PERTURB_KEY].isin(keep)].copy()


def run_ridge_for_feature(adata, feat_key):
    """{perturbation: (spearman_r, n_test_cells)} for one feature bank."""
    gt_accum = defaultdict(list)
    pred_accum = defaultdict(list)

    for split_idx in range(NUM_SPLITS):
        split_col = f"train_test_split_{split_idx}"

        tr_all = filter_min_cells(adata[adata.obs[split_col] == "train"])
        tr_has = tr_all.obs["has_spaGFM_emb"].values
        Xtr = tr_all.obsm[feat_key][tr_has]
        Ytr = tr_all.obsm[Y_KEY][tr_has]
        if Xtr.shape[0] < 10:
            continue

        scaler = StandardScaler().fit(Xtr)
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(Xtr), Ytr)

        te_all = filter_min_cells(adata[adata.obs[split_col] == "test"])
        for perturb in te_all.obs[PERTURB_KEY].unique():
            te_mask = (te_all.obs[PERTURB_KEY] == perturb).values
            te_has = te_all.obs["has_spaGFM_emb"].values[te_mask]
            if te_has.sum() < 3:
                continue
            Xte = te_all.obsm[feat_key][te_mask][te_has]
            Yte = te_all.obsm[Y_KEY][te_mask][te_has]
            gt_accum[perturb].append(Yte)
            pred_accum[perturb].append(ridge.predict(scaler.transform(Xte)))

    out = {}
    for p in sorted(gt_accum.keys()):
        gt = np.concatenate(gt_accum[p], 0).mean(0)
        pred = np.concatenate(pred_accum[p], 0).mean(0)
        rho = float(stats.spearmanr(gt, pred)[0])
        out[p] = (rho, int(np.concatenate(gt_accum[p], 0).shape[0]))
    return out


def main() -> None:
    say("=" * 78)
    say("STEP 1 -- RIDGE FIT, 2 ARMS (spaGFM 317M, GenePT*)")
    say("NOT a reproduction of the published figure: new fit, published rho definition")
    say("=" * 78)

    say("")
    say("Loading h5ad …")
    adata = sc.read_h5ad(H5AD)
    adata.obsm[Y_KEY] = np.asarray(adata.layers[Y_KEY], dtype=np.float32)
    say(f"  adata: {adata.shape}")

    say("Restoring 317M spaGFM features from the step-0b cache …")
    z = np.load(CACHE, allow_pickle=False)
    has_emb = z["has_emb"]
    spa_r = z["spagfm"]
    cache_ids = z["cell_ids"].astype(str)
    dim = spa_r.shape[1]
    spa = np.zeros((adata.n_obs, dim), dtype=np.float32)
    spa[has_emb] = spa_r
    # the cache was written in adata.obs_names order; verify rather than trust
    assert (np.asarray(adata.obs_names)[has_emb] == cache_ids).all(), \
        "cache cell order does not match adata.obs_names"
    adata.obsm["spaGFM_emb"] = spa
    adata.obs["has_spaGFM_emb"] = has_emb
    say(f"  317M dim={dim}  embedded cells={has_emb.sum():,} / {adata.n_obs:,}")
    say("  cache cell order verified against adata.obs_names")

    FEATURE_BANKS = {"spaGFM": "spaGFM_emb", "GenePT*": GENE_KEY}

    rows = []
    for name, key in FEATURE_BANKS.items():
        say("")
        say(f"Running Ridge for {name} ({key}) …")
        res = run_ridge_for_feature(adata, key)
        for p, (rho, n) in res.items():
            rows.append({"method": name, "perturbation": p,
                         "spearman_r": rho, "n_test_cells": n})
        rr = np.array([v[0] for v in res.values()])
        say(f"  {name}: {len(res)} perturbations, mean rho={rr.mean():.3f}, "
            f"median={np.median(rr):.3f}, min={rr.min():.3f}, max={rr.max():.3f}")

    long = pd.DataFrame(rows)
    long.to_csv(f"{HERE}/spearman_long_2arm.csv", index=False)
    say("")
    say(f"Saved spearman_long_2arm.csv ({len(long)} rows)")

    # ---- comparison against the published values ---------------------------------
    say("")
    say("=" * 78)
    say("COMPARISON AGAINST THE PUBLISHED VALUES (per perturbation)")
    say("=" * 78)
    pub = pd.read_csv(PUBLISHED)
    pub_sp = pub[pub.method == "spaGFM"][["perturbation", "spearman_r"]] \
        .rename(columns={"spearman_r": "published_spaGFM_0423"})
    pub_mlp = pub[pub.method == "MLP"][["perturbation", "spearman_r"]] \
        .rename(columns={"spearman_r": "published_MLP"})
    new_sp = long[long.method == "spaGFM"][["perturbation", "spearman_r"]] \
        .rename(columns={"spearman_r": "new_spaGFM_317M"})
    new_gp = long[long.method == "GenePT*"][["perturbation", "spearman_r"]] \
        .rename(columns={"spearman_r": "new_GenePT_star"})

    cmp = (new_sp.merge(pub_sp, on="perturbation", how="outer")
                 .merge(new_gp, on="perturbation", how="outer")
                 .merge(pub_mlp, on="perturbation", how="outer")
                 .sort_values("perturbation").reset_index(drop=True))
    cmp["delta_spaGFM_317M_minus_0423"] = (cmp.new_spaGFM_317M
                                           - cmp.published_spaGFM_0423)
    cmp["delta_GenePT_star_minus_publishedMLP"] = (cmp.new_GenePT_star
                                                   - cmp.published_MLP)
    cmp.to_csv(f"{HERE}/ridge_vs_published_comparison.csv", index=False)
    say(f"  saved ridge_vs_published_comparison.csv ({len(cmp)} perturbations)")

    d_sp = cmp["delta_spaGFM_317M_minus_0423"]
    d_gp = cmp["delta_GenePT_star_minus_publishedMLP"]
    say("")
    say("  spaGFM: 317M vs published 04-23 features")
    say(f"    mean new={cmp.new_spaGFM_317M.mean():.4f}  "
        f"mean published={cmp.published_spaGFM_0423.mean():.4f}")
    say(f"    delta: mean={d_sp.mean():+.4f}  median={d_sp.median():+.4f}  "
        f"min={d_sp.min():+.4f}  max={d_sp.max():+.4f}  "
        f"n_improved={(d_sp>0).sum()}/{d_sp.notna().sum()}")

    say("")
    say("  ISOLATION CHECK -- GenePT* vs published MLP (IDENTICAL feature bank)")
    say(f"    mean new={cmp.new_GenePT_star.mean():.4f}  "
        f"mean published={cmp.published_MLP.mean():.4f}")
    say(f"    delta: mean={d_gp.mean():+.6f}  max|delta|={d_gp.abs().max():.2e}")
    identical = bool(d_gp.abs().max() < 1e-9)
    say(f"    GenePT* reproduces published MLP exactly: {identical}")
    if identical:
        say("    => the two inference sets mark the SAME cells, so the spaGFM delta")
        say("       above is attributable to the embedding values alone.")
    else:
        say("    => the arms do NOT land on identical cells; the spaGFM delta is")
        say("       therefore CONFOUNDED (features + cell set) and must not be")
        say("       attributed to the embedding swap alone.")

    # ---- the only between-arm statistic --------------------------------------
    say("")
    say("=" * 78)
    say("STATISTICS -- spaGFM vs GenePT* (the only comparison reported)")
    say("=" * 78)
    m = new_sp.merge(new_gp, on="perturbation", how="inner").sort_values("perturbation")
    a = m.new_spaGFM_317M.to_numpy(float)
    b = m.new_GenePT_star.to_numpy(float)
    n = len(m)
    assert m.perturbation.nunique() == n, "a perturbation appears more than once"
    w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided")
    say(f"  test      : Wilcoxon signed-rank, two-sided, PAIRED on perturbation")
    say(f"  pairing   : same perturbation, same held-out cells, both arms")
    say(f"  n         : {n} perturbations")
    say(f"  repeated units: NONE -- each perturbation appears exactly once "
        f"({m.perturbation.nunique()} distinct / {n} rows)")
    say(f"  => the descriptive-only caveat that applies to the 70-pair (split,")
    say(f"     perturbation) setting does NOT apply here.")
    say(f"  mean spaGFM={a.mean():.4f}  mean GenePT*={b.mean():.4f}  "
        f"mean diff={a.mean()-b.mean():+.4f}")
    say(f"  W={w_stat:.1f}  p={w_p:.4e}")
    pd.DataFrame([{"comparison": "spaGFM vs GenePT*", "test":
                   "wilcoxon signed-rank (two-sided, paired on perturbation)",
                   "n": n, "repeated_units": False,
                   "mean_spaGFM": float(a.mean()), "mean_GenePT_star": float(b.mean()),
                   "mean_diff": float(a.mean() - b.mean()),
                   "statistic": float(w_stat), "p_value": float(w_p)}]) \
        .to_csv(f"{HERE}/stats_spagfm_vs_genept.csv", index=False)
    say("  saved stats_spagfm_vs_genept.csv")

    with open(f"{HERE}/run_ridge_2arm.log", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    say("")
    say("STEP 1 COMPLETE.")


if __name__ == "__main__":
    main()
