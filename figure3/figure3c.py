# Panel: figure3c
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/panelF_ownscgpt_20260827/code/panelF_stacked.py
# Original filename: panelF_stacked.py
# Last modified: 2026-08-28 09:59:27  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/work/fig1_d/tcell_proximity_auc.csv; panelF_ownscgpt_20260827/data/residual_r2_results_v2_with_own_scgpt.csv; panelF_ownscgpt_20260827/data/metrics.json
# Output: panelF_ownscgpt_20260827/figures/panelF_stacked_297x293pt.{pdf,svg,png} (+_exactpx.png)
# Match evidence: The ONLY script in the tree that stacks AUC above R^2 sharing one column (all others are side-by-side). AUC bars = 0.7213 (pale orange, own-cell scGPT) and 0.9071 (pink, spaGFM) with error bars and a *** bracket; AUC ylim (0.60,1.00) matches the described 0.6-1.0 axis. R^2 boxes = Pure spatial (grey) / scGPT (pale) / spaGFM (pink) with rotated x tick labels.
# Other candidates considered: figure_bundle_20260817/code/fig1_d_tcell_proximity_auc_step2_figure.py -- same content but SIDE-BY-SIDE, and its scGPT bar is mean-scGPT (0.8395), not the ~0.72 shown. panelF_own_scgpt.py -- correct numbers but still side-by-side; it is the prep, copied as figure3c_prep3.py.
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""
Panel F on a fixed 297 x 293 canvas, AUC above R^2 (stacked, not side by side).

Both arms are OWN-CELL scGPT (not mean-scGPT); spaGFM and 'Pure spatial' are the bundle's
cached values, unchanged.

CANVAS. 297x293 POINTS -> figsize = (297/72, 293/72) in. bbox_inches='tight' is deliberately
NOT used: it recrops and would silently change the output size. Margins are set explicitly so
the axes fill the canvas without crowding.
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

BUN = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/fig1_d")
Z = Path("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/panelF_ownscgpt_20260827")
ap = argparse.ArgumentParser()
ap.add_argument("--w", type=float, default=297); ap.add_argument("--h", type=float, default=293)
ap.add_argument("--unit", choices=["pt", "px"], default="pt")
ap.add_argument("--dpi", type=float, default=600)
ap.add_argument("--tag", default="stacked_297x293pt")
a = ap.parse_args()
per_inch = 72.0 if a.unit == "pt" else a.dpi
FIGW, FIGH = a.w / per_inch, a.h / per_inch
print(f"canvas {a.w}x{a.h} {a.unit} -> figsize {FIGW:.4f} x {FIGH:.4f} in")

auc = pd.read_csv(BUN / "tcell_proximity_auc.csv").set_index("method")
r2 = pd.read_csv(Z / "data/residual_r2_results_v2_with_own_scgpt.csv")
import json
OWN = json.load(open(Z / "data/metrics.json"))
a_sc_m, a_sc_s = OWN["auc_own_scgpt_mean"], OWN["auc_own_scgpt_std"]
a_sp_m, a_sp_s = auc.loc["spagfm", ["auc_mean", "auc_std"]]

SCGPT_F, SCGPT_E = "#F4E6C4", "#CBA84E"
SPAGFM_F, SPAGFM_E = "#EAA9A8", "#C15C58"
GRAY_F, GRAY_E = "#BEBEBE", "#5F5F5F"
FS_TICK, FS_LAB = 8.0, 9.0

fig = plt.figure(figsize=(FIGW, FIGH), facecolor="white")
# explicit margins: fill the canvas, leave just enough for labels
# hspace kept just large enough for the top panel's x tick labels; the panels then
# occupy nearly the full height instead of leaving a dead band in the middle.
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.12],
                      left=0.175, right=0.985, top=0.965, bottom=0.155, hspace=0.30)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- top: AUC bars
means, stds = [a_sc_m, a_sp_m], [a_sc_s, a_sp_s]
for i, (m, s, f, e) in enumerate(zip(means, stds, [SCGPT_F, SPAGFM_F], [SCGPT_E, SPAGFM_E])):
    ax1.bar(i, m, yerr=s, color=f, edgecolor=e, linewidth=1.1, width=0.62,
            capsize=3.5, error_kw=dict(linewidth=1.1, ecolor="black"))
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["scGPT", "spaGFM"], fontsize=FS_LAB)
ax1.set_ylim(0.60, 1.0); ax1.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax1.set_ylabel("AUC", fontsize=FS_LAB, labelpad=2)
ax1.tick_params(axis="both", labelsize=FS_TICK, length=2.5, pad=1.5)
ax1.spines[["top", "right"]].set_visible(False)
ysig = max(m + s for m, s in zip(means, stds)) + 0.024
ax1.plot([0, 0, 1, 1], [ysig, ysig + .010, ysig + .010, ysig], color="black", linewidth=1.0)
ax1.text(0.5, ysig + .014, "***", ha="center", va="bottom", fontsize=10)
ax1.set_xlim(-0.62, 1.62)

# ---- bottom: R2 boxes
data = [r2["r2_simple_spa"].values, r2["r2_own_scgpt"].values, r2["r2_spagfm"].values]
bp = ax2.boxplot(data, patch_artist=True, widths=0.6,
                 medianprops=dict(color="black", linewidth=1.6),
                 whiskerprops=dict(color="black", linewidth=1.0),
                 capprops=dict(color="black", linewidth=1.0),
                 flierprops=dict(marker="o", markersize=2.2, markerfacecolor="#9A9A9A",
                                 markeredgecolor="#9A9A9A", alpha=0.6))
for patch, f, e in zip(bp["boxes"], [GRAY_F, SCGPT_F, SPAGFM_F], [GRAY_E, SCGPT_E, SPAGFM_E]):
    patch.set_facecolor(f); patch.set_edgecolor(e); patch.set_linewidth(1.1)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(["Pure spatial", "scGPT", "spaGFM"], fontsize=FS_LAB,
                    rotation=28, ha="right", rotation_mode="anchor")
ax2.set_ylim(-0.07, 0.52); ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_ylabel(r"$R^2$", fontsize=FS_LAB + 1, labelpad=2)
ax2.tick_params(axis="both", labelsize=FS_TICK, length=2.5, pad=1.5)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlim(0.4, 3.6)

for ext in ("pdf", "svg"):
    fig.savefig(Z / f"figures/panelF_{a.tag}.{ext}", facecolor="white")   # NO bbox_inches
fig.savefig(Z / f"figures/panelF_{a.tag}.png", dpi=a.dpi, facecolor="white")
# a PNG at 72 dpi is exactly W x H PIXELS, covering the "297x293 px" reading of the spec
fig.savefig(Z / f"figures/panelF_{a.tag}_exactpx.png", dpi=72, facecolor="white")
plt.close(fig)
print(f"wrote panelF_{a.tag}.{{pdf,svg,png}}")
