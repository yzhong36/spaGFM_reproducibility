# Panel: figure3_supp_b_scgpt
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/umap_scgpt_20260826/code/scgpt_umap_fig1c_style.py
# Original filename: scgpt_umap_fig1c_style.py
# Last modified: 2026-08-27 14:07:18  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: umap_scgpt_20260826/data/umap_coords.npz (key 'own_scgpt')
# Output: umap_scgpt_20260826/figures/scgpt_umap_fig1c_style{,_bare}.{png,svg}
# Match evidence: scGPT half of the panel. Explicitly redrawn through fig1_c's own sc.pl.umap call so the dot size matches the spaGFM panel by construction (both n=4,886 -> scanpy default 120000/n_obs = 24.56); same orange/purple palette.
# Other candidates considered: umap_scgpt_20260826/code/scgpt_umap_png.py -- earlier single-PNG version with hand-set dot size, superseded.
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""
scGPT UMAP redrawn through sc.pl.umap with fig1_c's exact call, so the dot size is
identical to the spaGFM panel (scanpy's default size = 120000/n_obs; both panels have
n=4,886, so the sizes match by construction rather than by a hand-tuned `s`).

Coordinates come from the cached own-cell scGPT UMAP; only the plotting is redone.
"""
import numpy as np, anndata as ad, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/umap_scgpt_20260826"
z = np.load(f"{W}/data/umap_coords.npz")
V, lab = z["own_scgpt"], z["lab"]
key = "stage2_best_mean_knn_neighbor_group"
palette = {"with_tcell_neighbor": "#ff7f0e", "without_tcell_neighbor": "#9467bd"}

A = ad.AnnData(X=np.zeros((len(V), 1), dtype=np.float32),
               obs=pd.DataFrame({key: pd.Categorical(
                   np.where(lab == 1, "with_tcell_neighbor", "without_tcell_neighbor"),
                   categories=["with_tcell_neighbor", "without_tcell_neighbor"])},
                   index=[f"c{i}" for i in range(len(V))]))
A.obsm["X_umap"] = np.asarray(V, dtype=np.float32)
print(f"n_obs={A.n_obs}  scanpy default point size = {120000/A.n_obs:.2f}")

# labelled version (fig1_c figsize (6,5))
fig, ax = plt.subplots(figsize=(6, 5))
sc.pl.umap(A, color=key, palette=palette, na_color="lightgray", ax=ax, show=False)
ax.set_title("scGPT embedding — stage2_best_mean_knn_neighbor_group")
fig.savefig(f"{W}/figures/scgpt_umap_fig1c_style.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{W}/figures/scgpt_umap_fig1c_style.svg", format="svg", bbox_inches="tight")
plt.close(fig)

# bare version (fig1_c figsize (5,5), no title/legend/axes)
fig2, ax2 = plt.subplots(figsize=(5, 5))
sc.pl.umap(A, color=key, palette=palette, na_color="lightgray", ax=ax2, show=False,
           legend_loc=None)
ax2.set_title(""); ax2.set_xlabel(""); ax2.set_ylabel("")
ax2.set_xticks([]); ax2.set_yticks([])
for s in ax2.spines.values(): s.set_visible(False)
fig2.savefig(f"{W}/figures/scgpt_umap_fig1c_style_bare.png", dpi=200, bbox_inches="tight")
fig2.savefig(f"{W}/figures/scgpt_umap_fig1c_style_bare.svg", format="svg", bbox_inches="tight")
plt.close(fig2)
print("wrote scgpt_umap_fig1c_style{,_bare}.{png,svg}")
