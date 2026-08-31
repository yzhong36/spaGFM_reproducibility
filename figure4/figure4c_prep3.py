# Panel: figure3c_prep3
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/panelF_ownscgpt_20260827/code/panelF_own_scgpt.py
# Original filename: panelF_own_scgpt.py
# Last modified: 2026-08-27 14:48:30  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: fig1_d/tcell_proximity_auc.csv; fig1_d/residual_r2_results_v2.csv; figd_r2_scgpt_20260827/data/per_gene_r2.csv; figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad
# Output: panelF_ownscgpt_20260827/data/metrics.json + residual_r2_results_v2_with_own_scgpt.csv
# Match evidence: Upstream step 3: computes own-cell scGPT AUC (0.7213 +/- 0.0077) and merges the own-scGPT R^2 column. Writes the two files figure3c.py reads.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""
panelF_auc_r2_restyled regenerated with the scGPT arm changed from mean-scGPT of neighbours
to the OWN-CELL scGPT embedding. Everything else -- palette, figsize, width ratios, bar and
box parameters, y-limits, ticks, labels, significance bracket -- is copied verbatim from
fig1_d_tcell_proximity_auc_step2_figure.py.

Inputs: the bundle's own cached CSVs, plus one new column (own-cell scGPT) computed with the
original's own helpers. The recomputed mean-scGPT and spaGFM R^2 reproduce the published CSV
to max|diff| = 0.0, which is what licenses swapping a single arm.

Two outputs:
  _boxonly   the literal request: ONLY the R^2 box changes; the AUC bar is left as published
             (so its 'scGPT' bar is still mean-scGPT).
  _both      both panels use own-cell scGPT, so the label 'scGPT' means one thing in the
             figure. Needs a wider AUC y-limit because own-cell AUC falls below 0.75.
"""
import json
import numpy as np, pandas as pd, scanpy as sc, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUN = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817")
WORK = BUN / "data/work/fig1_d"
RAW = BUN / "data/raw"
Z = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/panelF_ownscgpt_20260827")
MINE = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figd_r2_scgpt_20260827")

# ---- own-cell scGPT AUC, using the original's auc_cv recipe verbatim
adata = sc.read_h5ad(RAW / "perturb_fish_spatial.h5ad")
inf = torch.load(RAW / "inference_set_wl_9_317M.pt", map_location="cpu", weights_only=False)
asg = np.zeros(adata.n_obs, bool)
for g in inf:
    m = np.asarray(g.y == g.perturbation)
    idx = adata.obs_names.get_indexer(np.asarray(g.cell_id)[m]); asg[idx[idx >= 0]] = True
del inf
cnt = adata.obs["perturbation"].value_counts()
valid = cnt[(cnt >= 20) & (cnt.index != "Control")].index
pa = adata[(adata.obs["perturbation"].isin(valid) & asg).values].copy()
y = pa.obs["stage2_best_mean_knn_has_tcell_neighbor"].astype(float).fillna(0).values.astype(int)

def auc_cv(X, y, n_pca=None, C=1.0):                      # verbatim from step1
    if n_pca and n_pca < X.shape[1]:
        X = PCA(n_pca, random_state=42).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    return cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring="roc_auc")
own = auc_cv(pa.obsm["X_scGPT"].astype(np.float32), y, n_pca=50)
print(f"own-cell scGPT AUC {own.mean():.4f} +/- {own.std():.4f}")

auc = pd.read_csv(WORK / "tcell_proximity_auc.csv").set_index("method")
r2 = pd.read_csv(WORK / "residual_r2_results_v2.csv")
mine = pd.read_csv(MINE / "data/per_gene_r2.csv")
assert list(r2["gene"]) == list(mine["gene"])
assert np.allclose(r2["r2_mean_scgpt"], mine["mean_scGPT"]) and \
       np.allclose(r2["r2_spagfm"], mine["spaGFM"]), "recomputation does not match the bundle"
r2 = r2.assign(r2_own_scgpt=mine["own_scGPT"].values)
r2.to_csv(Z / "data/residual_r2_results_v2_with_own_scgpt.csv", index=False)

SCGPT_F, SCGPT_E = "#F4E6C4", "#CBA84E"
SPAGFM_F, SPAGFM_E = "#EAA9A8", "#C15C58"
GRAY_F, GRAY_E = "#BEBEBE", "#5F5F5F"

def render(tag, auc_scgpt_m, auc_scgpt_s, r2_scgpt, ylim1, yticks1):
    a_sp_m, a_sp_s = auc.loc["spagfm", ["auc_mean", "auc_std"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 4.0),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.42))
    means, stds = [auc_scgpt_m, a_sp_m], [auc_scgpt_s, a_sp_s]
    for i, (m, s, f, e) in enumerate(zip(means, stds, [SCGPT_F, SPAGFM_F], [SCGPT_E, SPAGFM_E])):
        ax1.bar(i, m, yerr=s, color=f, edgecolor=e, linewidth=1.3, width=0.62,
                capsize=5, error_kw=dict(linewidth=1.3, ecolor="black"))
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["scGPT", "spaGFM"], fontsize=12)
    ax1.set_ylim(*ylim1); ax1.set_yticks(yticks1)
    ax1.set_ylabel("AUC", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)
    ysig = max(m + s for m, s in zip(means, stds)) + 0.015 * (ylim1[1] - ylim1[0]) / 0.25
    h = 0.006 * (ylim1[1] - ylim1[0]) / 0.25
    ax1.plot([0, 0, 1, 1], [ysig, ysig + h, ysig + h, ysig], color="black", linewidth=1.1)
    ax1.text(0.5, ysig + 1.5 * h, "***", ha="center", va="bottom", fontsize=13)

    data = [r2["r2_simple_spa"].values, r2_scgpt, r2["r2_spagfm"].values]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", linewidth=2),
                     whiskerprops=dict(color="black", linewidth=1.1),
                     capprops=dict(color="black", linewidth=1.1),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="#9A9A9A",
                                     markeredgecolor="#9A9A9A", alpha=0.6))
    for patch, f, e in zip(bp["boxes"], [GRAY_F, SCGPT_F, SPAGFM_F], [GRAY_E, SCGPT_E, SPAGFM_E]):
        patch.set_facecolor(f); patch.set_edgecolor(e); patch.set_linewidth(1.3)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Pure spatial", "scGPT", "spaGFM"], fontsize=12, rotation=30, ha="right")
    ax2.set_ylim(-0.07, 0.52); ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax2.set_ylabel(r"$R^2$", fontsize=13)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(Z / f"figures/panelF_auc_r2_{tag}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {tag}: AUC scGPT {auc_scgpt_m:.3f}+/-{auc_scgpt_s:.3f} | "
          f"R2 scGPT mean {np.mean(r2_scgpt):.3f}")

# literal request: ONLY the box changes
render("boxonly", *auc.loc["mean_scgpt", ["auc_mean", "auc_std"]],
       r2["r2_own_scgpt"].values, (0.75, 1.0), [.75, .80, .85, .90, .95, 1.00])
# consistent: both panels own-cell (needs a lower y-limit; own-cell AUC < 0.75)
render("both", own.mean(), own.std(), r2["r2_own_scgpt"].values,
       (0.60, 1.0), [.6, .7, .8, .9, 1.0])

json.dump(dict(auc_own_scgpt_mean=float(own.mean()), auc_own_scgpt_std=float(own.std()),
               auc_mean_scgpt_published=float(auc.loc["mean_scgpt", "auc_mean"]),
               auc_spagfm_published=float(auc.loc["spagfm", "auc_mean"]),
               r2_means={k: float(r2[k].mean()) for k in
                         ["r2_simple_spa", "r2_mean_scgpt", "r2_own_scgpt", "r2_spagfm"]}),
          open(Z / "data/metrics.json", "w"), indent=1)
print("done")
