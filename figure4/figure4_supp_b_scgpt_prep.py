# Panel: figure3_supp_b_scgpt_prep
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/umap_scgpt_20260826/code/umap_scgpt.py
# Original filename: umap_scgpt.py
# Last modified: 2026-08-27 11:41:37  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad; data/raw/inference_set_wl_9_317M.pt
# Output: umap_scgpt_20260826/data/umap_coords.npz + metrics.json; figures/umap_perturbed_scgpt_vs_spagfm.{png,pdf}
# Match evidence: Computes the scGPT and spaGFM UMAP coordinates cached in umap_coords.npz. Also emits its own side-by-side 2-panel figure, which is an ALTERNATIVE rendering of supp_b (different styling, kNN-purity subtitles).
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""
UMAP of the PERTURBED cells in scGPT space, coloured by T-cell-neighbour status.

Cell set and pipeline are identical to fig1_c so the panels are directly comparable:
perturbations with >=20 cells, Control excluded, PCA -> neighbours -> UMAP, random_state=0.

Three spaces, because fig1_d's bar chart compares two DIFFERENT scGPT quantities:
  own_scgpt   obsm['X_scGPT'] of the cell itself                      (512-d)
  mean_scgpt  mean X_scGPT of NON-T-CELL neighbours within r=300      (512-d)
              ^ this is the feature the 'scGPT' bar in fig1_d actually uses
  spagfm      obsm spaGFM wl_9 317M                                   (1280-d)
kNN label purity vs chance is reported for each, which is the UMAP-space analogue of the
AUC bar and shows where the gap comes from.
"""
import json
import numpy as np, scanpy as sc, anndata as ad, torch
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RAW = ("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/"
       "figure_bundle_20260817/data/raw")
OUT = ("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/"
       "umap_scgpt_20260826")
WITH, WITHOUT, INK, MUTED, SURFACE, BG = "#ff7f0e", "#9467bd", "#0b0b0b", "#898781", "#fcfcfb", "#e6e5df"
GKEY, SEED, R = "stage2_best_mean_knn_neighbor_group", 0, 300

adata = sc.read_h5ad(f"{RAW}/perturb_fish_spatial.h5ad")
inf = torch.load(f"{RAW}/inference_set_wl_9_317M.pt", map_location="cpu", weights_only=False)
adata.obsm["spaGFM_emb"] = np.zeros((adata.n_obs, 1280), dtype=np.float32)
assigned = np.zeros(adata.n_obs, dtype=bool)
for gph in inf:
    m = np.asarray(gph.y == gph.perturbation)
    idx = adata.obs_names.get_indexer(np.asarray(gph.cell_id)[m])
    v = idx >= 0
    adata.obsm["spaGFM_emb"][idx[v]] = gph.node_emb.detach().numpy().astype(np.float32)[v]
    assigned[idx[v]] = True
del inf

cnt = adata.obs["perturbation"].value_counts()
valid = cnt[(cnt >= 20) & (cnt.index != "Control")].index
sel = (adata.obs["perturbation"].isin(valid) & assigned).values
pa = adata[sel].copy()
lab = (pa.obs[GKEY].astype(str) == "with_tcell_neighbor").values.astype(int)
print(f"perturbed cells {pa.n_obs:,}  with {int(lab.sum()):,}  without {int((1-lab).sum()):,}")

# mean-scGPT of NON-T-cell neighbours within R -- the fig1_d 'scGPT' feature
tree = cKDTree(adata.obsm["spatial"])
sg_all = adata.obsm["X_scGPT"].astype(np.float32)
ct_all = adata.obs["cell_type"].values
mean_sg = np.zeros((pa.n_obs, sg_all.shape[1]), dtype=np.float32)
for i, (cx, cy) in enumerate(pa.obsm["spatial"]):
    nb = tree.query_ball_point([cx, cy], R)
    keep = [j for j in nb if ct_all[j] != "T_cell"]
    if keep: mean_sg[i] = sg_all[keep].mean(0)
print("mean-scGPT feature built")

SPACES = [("own_scgpt",  pa.obsm["X_scGPT"].astype(np.float32), "scGPT (own cell)"),
          ("mean_scgpt", mean_sg,                               f"mean-scGPT of non-T neighbours (r={R})"),
          ("spagfm",     pa.obsm["spaGFM_emb"],                 "spaGFM wl_9 (317M)")]

def purity(X, y, k=30):
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return float(np.mean(y[idx[:, 1:]] == y[:, None]))
ch = float(lab.mean() ** 2 + (1 - lab.mean()) ** 2)

res, coords = {}, {}
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), facecolor=SURFACE)
rng = np.random.default_rng(SEED)
for ax, (key, X, ttl) in zip(axes, SPACES):
    S = ad.AnnData(X=np.ascontiguousarray(X), obs=pa.obs.copy())
    sc.pp.pca(S, random_state=SEED); sc.pp.neighbors(S, random_state=SEED)
    sc.tl.umap(S, random_state=SEED)
    V = S.obsm["X_umap"]; coords[key] = V
    res[key] = dict(purity_umap=purity(V, lab), purity_pca=purity(S.obsm["X_pca"], lab),
                    chance=ch, dim=int(X.shape[1]))
    sh = rng.permutation(len(V))
    ax.scatter(V[sh, 0], V[sh, 1], s=4, c=np.where(lab == 1, WITH, WITHOUT)[sh],
               lw=0, alpha=.8, rasterized=True)
    ax.set_title(f"{ttl}\nkNN purity {res[key]['purity_umap']:.3f} (UMAP) · "
                 f"{res[key]['purity_pca']:.3f} (PCA) · chance {ch:.3f}",
                 fontsize=9.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_xlabel("UMAP1", fontsize=9, color=MUTED); axes[0].set_ylabel("UMAP2", fontsize=9, color=MUTED)
fig.text(.008, .965, "Perturbed cells — UMAP in scGPT vs spaGFM space, coloured by "
                     "T-cell-neighbour status", fontsize=13.5, color=INK, va="top")
fig.text(.008, .922, f"{pa.n_obs:,} perturbed cells, {len(valid)} perturbations (>=20 cells, "
                     f"Control excluded) — the same cell set and pipeline as fig1_c. The middle "
                     f"panel is the feature the 'scGPT' bar in fig1_d actually uses.",
         fontsize=9, color=MUTED, va="top")
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mfc=WITH, mec="none",
                           label=f"with T-cell neighbour (n={int(lab.sum()):,})"),
                    Line2D([], [], marker="o", ls="", ms=8, mfc=WITHOUT, mec="none",
                           label=f"without (n={int((1-lab).sum()):,})")],
           loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(.5, 0))
fig.tight_layout(rect=[0, .05, 1, .885])
for e in ("pdf", "png"):
    fig.savefig(f"{OUT}/figures/umap_perturbed_scgpt_vs_spagfm.{e}",
                dpi=230 if e == "png" else None, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
np.savez_compressed(f"{OUT}/data/umap_coords.npz", lab=lab, **coords)
json.dump(dict(n_cells=int(pa.n_obs), n_with=int(lab.sum()), n_without=int((1-lab).sum()),
               n_perturbations=int(len(valid)), radius=R, spaces=res),
          open(f"{OUT}/data/metrics.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print("done")
