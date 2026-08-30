# Panel: figure3_supp_b
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_c_umap_tcell_neighbor.py
# Original filename: fig1_c_umap_tcell_neighbor.py
# Last modified: 2026-08-18 15:09:25  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad; data/raw/inference_set_wl_9_317M.pt
# Output: data/work/fig1_c/wl9_umap_stage2_best_mean_knn_recolored_orange_purple{,_bare}.{png,svg}
# Match evidence: spaGFM half of the panel. Palette #ff7f0e / #9467bd (orange/purple = with/without T-cell neighbour), n=4,886 perturbed cells, Control excluded. The '_bare' variant has no title/legend/spines, consistent with the arrow-style axes drawn in PowerPoint.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
"""Reproduce the wl_9 spaGFM UMAP colored by stage2_best_mean_knn_neighbor_group,
mirroring 04_check_spaGFM_emb.ipynb exactly, then recolor:
  with_tcell_neighbor    -> #ff7f0e
  without_tcell_neighbor -> #9467bd
Identical pipeline/seeds to redo_wl9_umap_svg.py, only the palette differs.
Saves SVG (+ PNG for verification)."""
# 2026-08-18: embedding switched from inference_set_wl_9_0423.pt (the 2026-04-23 set)
# to inference_set_wl_9_317M.pt (mtime 2026-06-29), so this panel uses the SAME
# embedding as fig1_d and fig1_g. The UMAP is recomputed from it (PCA -> neighbors ->
# umap, scanpy defaults, random_state=0), so the coordinates change; the palette and
# every other setting are unchanged. Previous version backed up in
# ../backup_fig1_c_0423emb_20260818/.

import scanpy as sc
import torch
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

file_path = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw'
OUT = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/fig1_c'

# ── cell 2 ─────────────────────────────────────────────────────────────────
adata = sc.read_h5ad(file_path + '/perturb_fish_spatial.h5ad')

# ── cell 3 (wl_9 only) ─────────────────────────────────────────────────────
wl = 9
inf = torch.load('/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/inference_set_wl_9_317M.pt',
                 map_location='cpu', weights_only=False)

# ── cell 4: assign embeddings ──────────────────────────────────────────────
d_spaGFM = 1280
adata.obsm[f'spaGFM_emb_wl_{wl}'] = np.zeros((adata.n_obs, d_spaGFM), dtype=np.float32)

def assign_embeddings(adata, inference_list, obsm_key):
    for item in inference_list:
        perturb_emb = item.node_emb.numpy()
        idx = adata.obs_names.get_indexer(item.cell_id[item.y == item.perturbation])
        if np.any(idx < 0):
            raise ValueError('Missing cell IDs while assigning ' + obsm_key)
        adata.obsm[obsm_key][idx] = perturb_emb

assign_embeddings(adata, inf, f'spaGFM_emb_wl_{wl}')

# ── cell 5: valid perturbations ────────────────────────────────────────────
min_cell_count = 20
counts = adata.obs["perturbation"].value_counts()
valid_perts = counts[counts >= min_cell_count].index
all_data_perturb = adata[(~adata.obs['perturbation'].isna())
                         & (adata.obs['perturbation'] != 'Control')
                         & (adata.obs['perturbation'].isin(valid_perts))].copy()
all_perturbs = all_data_perturb.obs['perturbation'].values.unique()

# ── cell 6: build wl_9 adata ───────────────────────────────────────────────
spaGFM_wl_9_adata = ad.AnnData(X=adata.obsm[f'spaGFM_emb_wl_{wl}'], obs=adata.obs.copy())
spaGFM_wl_9_adata = spaGFM_wl_9_adata[
    spaGFM_wl_9_adata.obs['perturbation'].isin(all_perturbs)].copy()

# ── cell 7: pca -> neighbors -> umap (deterministic, default seeds) ─────────
# 2026-08-18: seeds written out explicitly rather than relying on scanpy defaults.
# These ARE scanpy's defaults (random_state=0), so output is unchanged -- verified by
# re-running and comparing SHA-256. Stated explicitly so the panel does not silently
# change if a future scanpy alters its defaults.
SEED = 0
sc.pp.pca(spaGFM_wl_9_adata, random_state=SEED)
sc.pp.neighbors(spaGFM_wl_9_adata, random_state=SEED)
sc.tl.umap(spaGFM_wl_9_adata, random_state=SEED)

# ── recolored plot ─────────────────────────────────────────────────────────
palette = {
    'with_tcell_neighbor':    '#ff7f0e',
    'without_tcell_neighbor': '#9467bd',
}
key = 'stage2_best_mean_knn_neighbor_group'
spaGFM_wl_9_adata.obs[key] = spaGFM_wl_9_adata.obs[key].astype('category')

TAG = 'orange_purple'

fig, ax = plt.subplots(figsize=(6, 5))
sc.pl.umap(spaGFM_wl_9_adata, color=key, palette=palette,
           na_color='lightgray', ax=ax, show=False)
ax.set_title('stage2_best_mean_knn_neighbor_group')
fig.savefig(f'{OUT}/wl9_umap_stage2_best_mean_knn_recolored_{TAG}.svg',
            format='svg', bbox_inches='tight')
fig.savefig(f'{OUT}/wl9_umap_stage2_best_mean_knn_recolored_{TAG}.png',
            dpi=200, bbox_inches='tight')
print('SAVED', f'{OUT}/wl9_umap_stage2_best_mean_knn_recolored_{TAG}.svg')

# also save the bare cropped scatter (no title/legend/axes) to match the shown crop
fig2, ax2 = plt.subplots(figsize=(5, 5))
sc.pl.umap(spaGFM_wl_9_adata, color=key, palette=palette,
           na_color='lightgray', ax=ax2, show=False, legend_loc=None)
ax2.set_title(''); ax2.set_xlabel(''); ax2.set_ylabel('')
ax2.set_xticks([]); ax2.set_yticks([])
for s in ax2.spines.values(): s.set_visible(False)
fig2.savefig(f'{OUT}/wl9_umap_stage2_best_mean_knn_recolored_{TAG}_bare.svg',
             format='svg', bbox_inches='tight')
fig2.savefig(f'{OUT}/wl9_umap_stage2_best_mean_knn_recolored_{TAG}_bare.png',
             dpi=200, bbox_inches='tight')
print('SAVED bare version too')
print('Done.')
