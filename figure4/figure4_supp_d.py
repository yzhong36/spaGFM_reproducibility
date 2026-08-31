# Panel: figure3_supp_d
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig2_f_attention_arrows.py
# Original filename: fig2_f_attention_arrows.py
# Last modified: 2026-08-18 10:29:10  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad; data/work/attention/per_cell_table.csv; data/work/attention/vectorfield/cache/{pert}_attrib.npz
# Output: data/work/attention/attention_arrows_spatial.{pdf,png}
# Match evidence: per_cell_table.csv row 847 = RELA / RawRow_108886 with 77 T cells and 108 neighbours -> the left title's '77 T / 108 reachable'; row 4269 = NFKBIA / RawRow_44857 with 1 T cell -> the right title's '1 T'. Script colours arrows RED to a T cell and BLUE to a non-T cell, as described.
# Other candidates considered: tcell_attention/make_arrow_fig.py -- pre-bundle original, same cells. make_arrow_fig_region.py -- a regional variant, not the two-cell panel.
# ---- copied verbatim below; NOT modified ----
"""Spatial attention-arrow figure (2 subplots), reusing cached intermediates only.

Subplot 1: a with-T-group perturbation cell with the MOST reachable T cells.
Subplot 2: a without-T-group perturbation cell with the FEWEST reachable T cells (>=1 so
           red arrows are visible).
For each center cell, draw an attention arrow from the cell to every walk-reachable neighbor
(cached _attrib.npz support, self-excluded), colored RED if the neighbor is a T cell, BLUE if
non-T; arrow width/opacity scale with the neighbor's attention attribution.
Dots: purple = the perturbation (center) cell; green = T-cell neighbors; light gray = non-T.

Attention = information-flow direction (where the masked cell reads context from), not physical
spread; probe attention is unscaled (M = walk tokens, not genes).
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE  = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/attention"
CACHE = os.path.join(BASE, "cache")
ADATA = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad"
PURPLE, GREEN, GRAY = "#7b3fa0", "#2ca02c", "#d0d0d0"
RED, BLUE = "#d62728", "#1f77b4"

tab = pd.read_csv(os.path.join(BASE, "per_cell_table.csv"))
print("loading adata (cell_type + spatial) ...", flush=True)
ad = sc.read_h5ad(ADATA)
cell_type = ad.obs["cell_type"].astype(object).to_numpy()
spatial = np.asarray(ad.obsm["spatial"])

# ---- select the two center cells ----
with_pool = tab[(tab.group == "with_T") & (~tab.boundary_T)]
with_cell = with_pool.sort_values(["n_tcell_nodes", "n_acc_nodes"], ascending=[False, False]).iloc[0]
wo_pool = tab[(tab.group == "without_T") & (tab.n_tcell_nodes >= 1)]
wo_cell = wo_pool.sort_values(["n_tcell_nodes", "n_acc_nodes"], ascending=[True, False]).iloc[0]
print(f"with-T center : {with_cell.perturbation}/{with_cell.cell_id} "
      f"T_reachable={int(with_cell.n_tcell_nodes)} total_nbr={int(with_cell.n_acc_nodes)}")
print(f"without-T ctr : {wo_cell.perturbation}/{wo_cell.cell_id} "
      f"T_reachable={int(wo_cell.n_tcell_nodes)} total_nbr={int(wo_cell.n_acc_nodes)}")


def neighbors(cell):
    z = np.load(os.path.join(CACHE, f"{cell.perturbation}_attrib.npz"))
    p = int(cell.pos_in_block); a, b = int(z["indptr"][p]), int(z["indptr"][p + 1])
    nidx = z["neigh_adata_idx"][a:b]; w = z["attrib_mean"][a:b].astype(float)
    assert (b - a) == int(cell.n_acc_nodes)
    self_xy = spatial[int(cell.adata_idx)]
    nxy = spatial[nidx]
    is_t = (cell_type[nidx] == "T_cell")
    return self_xy, nxy, w, is_t


def draw(ax, cell, title):
    self_xy, nxy, w, is_t = neighbors(cell)
    wn = w / w.max() if w.max() > 0 else w
    # arrows: center -> each neighbor, color by T/non-T, width/alpha by attention
    for j in range(len(nxy)):
        col = RED if is_t[j] else BLUE
        lw = 0.4 + 4.5 * wn[j]; al = 0.30 + 0.65 * wn[j]
        ax.annotate("", xy=(nxy[j, 0], nxy[j, 1]), xytext=(self_xy[0], self_xy[1]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, alpha=al,
                                    shrinkA=2.5, shrinkB=3, mutation_scale=8 + 8 * wn[j]),
                    zorder=2)
    # dots: non-T (gray), T (green), center (purple)
    ax.scatter(nxy[~is_t, 0], nxy[~is_t, 1], s=42, c=GRAY, edgecolors="0.5",
               linewidths=0.4, zorder=3, label="non-T cell")
    ax.scatter(nxy[is_t, 0], nxy[is_t, 1], s=52, c=GREEN, edgecolors="0.25",
               linewidths=0.5, zorder=4, label="T cell")
    ax.scatter([self_xy[0]], [self_xy[1]], s=170, c=PURPLE, edgecolors="k",
               linewidths=0.8, marker="o", zorder=5, label="perturbation cell")
    # frame to the local neighborhood
    allx = np.r_[nxy[:, 0], self_xy[0]]; ally = np.r_[nxy[:, 1], self_xy[1]]
    padx = 0.06 * (allx.max() - allx.min() + 1); pady = 0.06 * (ally.max() - ally.min() + 1)
    ax.set_xlim(allx.min() - padx, allx.max() + padx)
    ax.set_ylim(ally.min() - pady, ally.max() + pady)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
draw(axes[0], with_cell,
     f"WITH T-cell neighbors\n{with_cell.perturbation} cell {with_cell.cell_id}  "
     f"({int(with_cell.n_tcell_nodes)} T / {int(with_cell.n_acc_nodes)} reachable)")
draw(axes[1], wo_cell,
     f"WITHOUT T-cell neighbors\n{wo_cell.perturbation} cell {wo_cell.cell_id}  "
     f"({int(wo_cell.n_tcell_nodes)} T / {int(wo_cell.n_acc_nodes)} reachable)")

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE, markeredgecolor="k",
                 markersize=12, label="perturbation cell (center)"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markeredgecolor="0.25",
                 markersize=10, label="T cell"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY, markeredgecolor="0.5",
                 markersize=10, label="non-T cell"),
          Line2D([0], [0], color=RED, lw=3, label="attention arrow -> T cell"),
          Line2D([0], [0], color=BLUE, lw=3, label="attention arrow -> non-T cell")]
fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=10, frameon=False)
fig.suptitle("Attention vector field around perturbation cells (arrow width/opacity ~ attention; "
             "walk-reachable neighbors)", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(BASE, f"attention_arrows_spatial.{ext}"), dpi=200, bbox_inches="tight")
print("wrote attention_arrows_spatial.png/pdf", flush=True)
