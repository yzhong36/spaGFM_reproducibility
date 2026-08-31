# Panel: figure3b
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/spatial_per_perturbation_20260826/code/draw_box_pair.py
# Original filename: draw_box_pair.py
# Last modified: 2026-08-26 22:33:24  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/all_qc_passing_cells_stage2.h5ad; spatial_per_perturbation_20260826/data/box_search.json
# Output: spatial_per_perturbation_20260826/figures/IRAK1_box_pair*.{pdf,png,svg}
# Match evidence: argparse default --gene IRAK1; draws full-tissue scatter + TWO zoom insets via Rectangle + dashed leader lines; palette TCELL=#2ca02c (green), WITH=#ff7f0e (orange), WITHOUT=#9467bd (purple), other = light grey -- the exact four categories and colours in the panel. Box A = perturbed-WITH hugging T cells (orange/green-rich), Box B = perturbed-WITHOUT with no T cell in frame (purple-rich), matching top/bottom insets.
# Other candidates considered: figure_bundle_20260817/code/fig1_b_fig2_a_spatial_maps.py -- same tissue and 4 categories but NO insets and no per-perturbation subsetting; it is the supp_a/whole-tissue panel instead.
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""Overview + two zoom insets for the perturbation with the cleanest illustrative box pair.
Box A: perturbed-WITH cells hugging T cells, controls all control_with, no perturbed-without.
Box B: perturbed-WITHOUT cells with NO T cell in frame, controls all control_without."""
import argparse, json
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.legend_handler import HandlerTuple

# white lightning-bolt glyph drawn inside every perturbed cell
_BOLT = MPath([(0.28, 1.00), (-0.38, 0.06), (-0.04, 0.06), (-0.28, -1.00),
               (0.38, -0.04), (0.04, -0.04), (0.28, 1.00)],
              [MPath.MOVETO] + [MPath.LINETO] * 5 + [MPath.CLOSEPOLY])

SRC = ("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/"
       "figure_bundle_20260817/data/raw/all_qc_passing_cells_stage2.h5ad")
ROOT = ("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/"
        "spatial_per_perturbation_20260826")
TCELL, WITH, WITHOUT = "#2ca02c", "#ff7f0e", "#9467bd"
BG, INK, MUTED, SURFACE = "#e8e7e1", "#0b0b0b", "#898781", "#fcfcfb"

def dec(x): return x.decode() if isinstance(x, bytes) else x
ap = argparse.ArgumentParser(); ap.add_argument("--gene", default="IRAK1")
ap.add_argument("--s-overview", type=float, default=4.0,
                help="ONE marker size for every labelled cell in the overview")
ap.add_argument("--s-zoom", type=float, default=30.0,
                help="ONE marker size for every labelled cell in the zooms")
ap.add_argument("--s-pert-zoom", type=float, default=170.0,
                help="perturbed-cell size in the zooms (may exceed --s-zoom)")
ap.add_argument("--s-pert-overview", type=float, default=24.0,
                help="perturbed-cell size in the overview; must be big enough to host the "
                     "bolt glyph, so it necessarily exceeds --s-overview")
ap.add_argument("--alpha", type=float, default=0.85,
                help="ONE alpha for every labelled cell")
ap.add_argument("--suffix", default="")
a = ap.parse_args()
rec = [r for r in json.load(open(f"{ROOT}/data/box_search.json")) if r["gene"] == a.gene][0]
A, B, L = rec["boxA"], rec["boxB"], rec["L"]

with h5py.File(SRC, "r") as f:
    def cat(c):
        n = f["obs"][c]; cs = [dec(v) for v in n["categories"][:]]
        return np.array([cs[i] if i >= 0 else None for i in n["codes"][:]], dtype=object)
    ct, pert, grp = cat("cell_type"), cat("perturbation"), cat("stage2_best_mean_knn_neighbor_group")
    x, y = f["obs"]["spatial_x"][:], f["obs"]["spatial_y"][:]
w = grp == "with_tcell_neighbor"; wo = grp == "without_tcell_neighbor"
isC, isT, isP = pert == "Control", ct == "T_cell", pert == a.gene
LAY = [("all other cells", np.ones(len(x), bool), BG, 1.0, 0),
       ("T cell",          isT,        TCELL,   1.0, 2),
       ("control, without", isC & wo,  WITHOUT, 1.0, 3),
       ("control, with",    isC & w,   WITH,    1.0, 3),
       (f"{a.gene}, without", isP & wo, WITHOUT, 1.0, 5),
       (f"{a.gene}, with",    isP & w,  WITH,    1.0, 5)]

fig = plt.figure(figsize=(14.5, 6.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], left=.03, right=.99, top=.86,
                      bottom=.13, wspace=.10, hspace=.16)
axo = fig.add_subplot(gs[:, 0])
# Every labelled category shares ONE size and ONE alpha; only the grey "all other cells"
# base differs. The perturbation is separated by a dark ring alone -- an edge, not size or
# opacity -- because at identical size and alpha it is otherwise indistinguishable from the
# same-coloured control cells.
for _, m, c, _, z in LAY:
    s = .5 if c == BG else (a.s_pert_overview if z == 5 else a.s_overview)
    al = 1.0 if c == BG else a.alpha
    axo.scatter(x[m], y[m], s=s, c=c, lw=0, alpha=al, rasterized=True, zorder=z)
    if z == 5:   # white lightning bolt inside each perturbed cell
        axo.scatter(x[m], y[m], s=s * .40, marker=_BOLT, c="white", lw=0,
                    rasterized=True, zorder=z + 1)
for bx, lab in ((A, "A"), (B, "B")):
    axo.add_patch(Rectangle((bx["cx"] - L/2, bx["cy"] - L/2), L, L, fill=False,
                            ec=INK, lw=1.6, zorder=10))
axo.set_aspect("equal"); axo.invert_yaxis(); axo.set_xticks([]); axo.set_yticks([])
for sp in axo.spines.values(): sp.set_visible(False)
axo.set_title(f"{a.gene} — whole section", fontsize=11, color=INK, loc="left")

for i, (bx, lab, note) in enumerate(
        ((A, "A", f"{A['n_pert']} {a.gene} WITH · {A['n_ctrl_same']} control_with · "
                  f"0 control_without · {A['n_tcell']} T cells"),
         (B, "B", f"{B['n_pert']} {a.gene} WITHOUT · {B['n_ctrl_same']} control_without · "
                  f"0 control_with · 0 T cells (nearest {B['dist_nearest_T']:.0f} away)"))):
    ax = fig.add_subplot(gs[i, 1])
    sel = ((np.abs(x - bx["cx"]) <= L/2) & (np.abs(y - bx["cy"]) <= L/2))
    for _, m, c, _, z in LAY:
        mm = m & sel
        s = 14 if c == BG else (a.s_pert_zoom if z == 5 else a.s_zoom)
        ax.scatter(x[mm], y[mm], s=s, c=c, lw=0,
                   alpha=(1.0 if c == BG else a.alpha), zorder=z)
        if z == 5:
            ax.scatter(x[mm], y[mm], s=s * .40, marker=_BOLT, c="white", lw=0, zorder=z + 1)
    ax.set_xlim(bx["cx"] - L/2, bx["cx"] + L/2); ax.set_ylim(bx["cy"] - L/2, bx["cy"] + L/2)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(1.6)
    ax.set_title(note, fontsize=8.8, color=INK, loc="left", pad=5)

fig.text(.03, .965, f"{a.gene} — perturbed cells with vs without a T-cell neighbour",
         fontsize=14, color=INK, ha="left", va="top")
pert_keys = [((Line2D([], [], marker="o", ls="", ms=12, mfc=c, mec="none"),
               Line2D([], [], marker=_BOLT, ls="", ms=7, mfc="white", mec="none")), n)
             for n, _, c, _, z in LAY[::-1] if z >= 5]
fig.legend(handles=[k for k, _ in pert_keys]
                 + [Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec="none", label=n)
                    for n, _, c, _, z in LAY if z in (2, 3)]
                 + [Line2D([], [], marker="o", ls="", ms=5, mfc=BG, mec="none",
                           label="all other cells")],
           labels=[n for _, n in pert_keys]
                  + [n for n, _, _, _, z in LAY if z in (2, 3)] + ["all other cells"],
           handler_map={tuple: HandlerTuple(ndivide=None)},
           loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, .005))
_equal = (a.s_overview == a.s_pert_overview) and (a.s_zoom == a.s_pert_zoom)
_cap = (f"Boxes are {L:.0f}x{L:.0f} units. "
        + (f"Every labelled cell is drawn at one size and one opacity ({a.alpha:g}); perturbed "
           f"cells are marked by a white bolt only."
           if _equal else
           f"T cells and control cells share one size and one opacity ({a.alpha:g}); perturbed "
           f"cells carry a white bolt and are drawn larger so the glyph is legible.")
        + " Neighbour status from stage2_best_mean_knn_neighbor_group.")
fig.text(.5, .055, _cap,
         ha="center", va="bottom", fontsize=7.5, color=MUTED)
for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/{a.gene}_box_pair{a.suffix}.{ext}", dpi=300 if ext == "png" else None,
                facecolor=SURFACE, bbox_inches="tight")
print("wrote", f"{ROOT}/figures/{a.gene}_box_pair{a.suffix}.pdf")
