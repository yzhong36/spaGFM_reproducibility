# Panel: figure3_supp_a
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_b_fig2_a_spatial_maps.py
# Original filename: fig1_b_fig2_a_spatial_maps.py
# Last modified: 2026-08-18 10:29:09  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/all_qc_passing_cells_stage2.h5ad
# Output: figure_bundle_20260817/data/work/spatial/spatial_plot.svg (collected as fig2_a_celltype_spatial.svg)
# Match evidence: Bundle figures_index.csv maps this script to fig2_a_celltype_spatial.svg, the cell-type-coloured whole-tissue map. Same script also emits perturbation_spatial.svg; the cell-type output is the supp_a panel.
# Other candidates considered: spatial_per_perturbation_20260826/code/draw_box_pair.py -- same tissue but IRAK1-subset with insets (figure3b).
# ---- copied verbatim below; NOT modified ----
import scanpy as sc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

H5AD = ('/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/'
        ''
        'all_qc_passing_cells_stage2.h5ad')
OUTDIR = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/spatial'

adata = sc.read(H5AD)

# ============================================================
# Figure 1: Tumor-T cell neighborhood
# ============================================================
adata.obs["plot_celltype"] = adata.obs["cell_type"].astype(str)

tumor_mask = adata.obs["cell_type"] == "tumor"
tcell_mask = adata.obs["cell_type"] == "T_cell"

with_mask = (tumor_mask &
    (adata.obs["stage2_best_mean_knn_neighbor_group"] == "with_tcell_neighbor"))
without_mask = (tumor_mask &
    (adata.obs["stage2_best_mean_knn_neighbor_group"] == "without_tcell_neighbor"))

adata.obs.loc[with_mask, "plot_celltype"] = "Tumor_with_T_neighbor"
adata.obs.loc[without_mask, "plot_celltype"] = "Tumor_without_T_neighbor"

adata.obs["plot_celltype"] = adata.obs["plot_celltype"].replace("nan", np.nan).astype("category")

palette = {
    "T_cell": "#2ca02c",
    "Tumor_with_T_neighbor": "#d62728",
    "Tumor_without_T_neighbor": "#1f77b4",
}

fig, ax = plt.subplots(figsize=(12, 10))
sc.pl.spatial(adata, color="plot_celltype", spot_size=80, palette=palette,
              na_color="lightgray", legend_loc=None, ax=ax, show=False)
sc.pl.spatial(adata[tcell_mask], color="plot_celltype", spot_size=200,
              palette=palette, legend_loc=None, ax=ax, show=False)
sc.pl.spatial(adata[with_mask], color="plot_celltype", spot_size=200,
              palette=palette, legend_loc=None, ax=ax, show=False)
sc.pl.spatial(adata[without_mask], color="plot_celltype", spot_size=200,
              palette=palette, legend_loc=None, ax=ax, show=False)

ax.invert_yaxis()

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='T_cell',
           markerfacecolor='#2ca02c', markersize=14),
    Line2D([0], [0], marker='o', color='w', label='Tumor_with_T_neighbor',
           markerfacecolor='#d62728', markersize=14),
    Line2D([0], [0], marker='o', color='w', label='Tumor_without_T_neighbor',
           markerfacecolor='#1f77b4', markersize=14),
    Line2D([0], [0], marker='o', color='w', label='NA',
           markerfacecolor='lightgray', markersize=14),
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
          frameon=False, fontsize=14)
ax.set_title("Tumor-T cell neighborhood", fontsize=20)
ax.set_xticks([]); ax.set_yticks([])
plt.savefig(f"{OUTDIR}/spatial_plot.svg", format="svg", bbox_inches="tight")
plt.close()
print("wrote spatial_plot.svg")

# ============================================================
# Figure 2: Perturbation spatial distribution
# ============================================================
adata.obs["perturbation_plot"] = (
    adata.obs["perturbation"].astype(str).replace("nan", np.nan).astype("category"))

cats = adata.obs["perturbation_plot"].cat.categories
colors = sns.husl_palette(len(cats), s=0.95, l=0.55)
palette = dict(zip(cats, colors))

valid_mask = adata.obs["perturbation_plot"].notna()
na_mask = adata.obs["perturbation_plot"].isna()

fig, ax = plt.subplots(figsize=(12, 10))
sc.pl.spatial(adata[na_mask], color="perturbation_plot", spot_size=80,
              palette={"NA": "lightgray"}, na_color="lightgray",
              legend_loc=None, ax=ax, show=False)
sc.pl.spatial(adata[valid_mask], color="perturbation_plot", spot_size=200,
              palette=palette, legend_loc="right margin", ax=ax, show=False)
ax.invert_yaxis()
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Perturbation spatial distribution", fontsize=20)
plt.savefig(f"{OUTDIR}/perturbation_spatial.svg", format="svg", bbox_inches="tight")
plt.close()
print("wrote perturbation_spatial.svg")
