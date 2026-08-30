# Panel: figure3e
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_f_lfc_confidence_triangles.py
# Original filename: fig1_f_lfc_confidence_triangles.py
# Last modified: 2026-08-27 14:29:58  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/derived/procB_triangle_full_data.npz; data/raw/perturb_fish_spatial.h5ad
# Output: figure_bundle_20260817/data/work/fig1_f/{gt_lfc_5rows_procB_joint, spagfm_conf_5rows_procB_joint}.{pdf,png,svg}
# Match evidence: PERTS = ['MAP3K7','MYD88','CHUK','RELA','NFKB1'] and GENES = ['LGALS9','BTLA','FFAR2','VEGFA','TWIST1','PDGFB','TLR9','ITGB2'] match the 5x8 grid exactly, in order. Emits the two stacked diagonal-split triangle heatmaps (Ground Truth LFC, -2..2; spaGFM confidence, -1..1) with BH significance dots.
# Other candidates considered: main_figure_perturbation*/panel_e_triangle_heatmap/panelE_5rows.py and draw_selected_triangle*.py -- the pre-bundle lineage; the 4-row ancestors lack the NFKB1 row. backup_fig1_f_4row_20260818/ is the 4-row backup.
# ---- copied verbatim below; NOT modified ----
"""
BUNDLE fig1_f — REPLACED 2026-08-18 with the 5-row version.

Supersedes the previous 4-row draw_selected_triangle lineage (MAP3K7, MYD88, CHUK, RELA),
which is preserved in ../backup_fig1_f_4row_20260818/. This is panelE_5rows.py adapted to
the bundle's path conventions; predictions come from the bundle's own
data/derived/procB_triangle_full_data.npz, i.e. the same procB_joint fit that backs the
spaGFM arm of fig1_e.

Panel E, 5-row version = the `_reselected` grid plus one extra perturbation row.

  rows  = MAP3K7, MYD88, CHUK, RELA, NFKB1   (NFKB1 is the added row)
  cols  = LGALS9, BTLA, FFAR2, VEGFA, TWIST1, PDGFB, TLR9, ITGB2

NFKB1 was picked as the highest-accuracy perturbation not already in the grid:
14/16 direction calls correct on these 8 genes (tied with MAP2K7, but NFKB1 has the
larger mean confidence margin, 0.219 vs 0.183, and flips direction with/without a
T-cell neighbour in 7/8 genes).  Grid accuracy goes 62/64 -> 76/80.

Predictions come from the procB_joint (317M, pipeline B) npz -- the same fit that backs
figure_bundle_20260817's fig1_f triangle and the spaGFM arm of its fig1_e violin.
GT LFC + within-grid BH significance dots are recomputed from adata (GT is fit-independent
and is bit-identical between the old and new npz).  Note the BH correction is done WITHIN the grid, so the extra
row slightly changes q-values for the original 4 rows.
Writes the `_5rows` panel-E files.
"""
import warnings
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
warnings.filterwarnings('ignore')

BASE     = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817'
# 2026-08-18: repointed from the pre-317M npz to the procB_joint / 317M predictions,
# so this panel uses the SAME fit as figure_bundle_20260817 (its fig1_f triangle and the
# spaGFM arm of its fig1_e violin both trace to this file). Byte-identical (sha256
# 60a0b70b0ca85994...) to figure_bundle_20260817/data/derived/procB_triangle_full_data.npz
# and to perturbation_317M_outputs/benchmark_run_new/nb05_triangle_full_data.npz.
# Previous value was f'{BASE}/data/nb05_triangle_full_data.npz' (pre-317M fit,
# sp_proba_w sha256 9e3374e340acabea...); backup in backup_pre317M_20260818/.
NPZ      = f'{BASE}/data/derived/procB_triangle_full_data.npz'
ADATA    = f'{BASE}/data/raw/perturb_fish_spatial.h5ad'
OUT_DIR  = f'{BASE}/data/work/fig1_f'
TARGET_KEY = 'stage2_best_mean_knn_target_lfc'
IMMUNE_COL = 'stage2_best_mean_knn_has_tcell_neighbor'
PERTURB_KEY= 'perturbation'
Q_THRESH   = 0.1

PERTS = ['MAP3K7', 'MYD88', 'CHUK', 'RELA', 'NFKB1']
GENES = ['LGALS9', 'BTLA', 'FFAR2', 'VEGFA', 'TWIST1', 'PDGFB', 'TLR9', 'ITGB2']
n_p, n_g = len(PERTS), len(GENES)

# predictions from npz
d = np.load(NPZ, allow_pickle=True)
perts_all = [str(p) for p in d['perts']]; genes_all = [str(g) for g in d['genes']]
pidx = [perts_all.index(p) for p in PERTS]; gidx = [genes_all.index(g) for g in GENES]
conf_w  = np.asarray(d['sp_proba_w'],  float)[np.ix_(pidx, gidx)]
conf_wo = np.asarray(d['sp_proba_wo'], float)[np.ix_(pidx, gidx)]

# GT LFC + within-grid BH significance from adata
print('Loading adata ...', flush=True)
adata = ad.read_h5ad(ADATA)
pert_lab  = adata.obs[PERTURB_KEY].astype(str).values
immune    = np.asarray(adata.obs[IMMUNE_COL].fillna(False).astype(bool))
ctrl_mask = pert_lab == 'Control'
Y_lfc     = np.asarray(adata.layers[TARGET_KEY], dtype=np.float32)
X_expr    = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
v2i       = {str(v): i for i, v in enumerate(adata.var_names)}
gcols     = [v2i[g] for g in GENES]
ctrl_w_e  = X_expr[ctrl_mask & immune, :]; ctrl_wo_e = X_expr[ctrl_mask & ~immune, :]

lfc_w = np.full((n_p, n_g), np.nan, np.float32); lfc_wo = np.full((n_p, n_g), np.nan, np.float32)
pv_w  = np.full((n_p, n_g), np.nan);             pv_wo  = np.full((n_p, n_g), np.nan)
for pi, p in enumerate(PERTS):
    pm = pert_lab == p
    Yw, Ywo = Y_lfc[pm & immune], Y_lfc[pm & ~immune]
    ew, ewo = X_expr[pm & immune], X_expr[pm & ~immune]
    for gl, gc in enumerate(gcols):
        vw  = Yw[:, gc][~np.isnan(Yw[:, gc])];  vwo = Ywo[:, gc][~np.isnan(Ywo[:, gc])]
        if vw.size:  lfc_w[pi, gl]  = np.mean(vw)
        if vwo.size: lfc_wo[pi, gl] = np.mean(vwo)
        if ew[:, gc].size  >= 3: pv_w[pi, gl]  = stats.mannwhitneyu(ew[:, gc],  ctrl_w_e[:, gc],  alternative='two-sided').pvalue
        if ewo[:, gc].size >= 3: pv_wo[pi, gl] = stats.mannwhitneyu(ewo[:, gc], ctrl_wo_e[:, gc], alternative='two-sided').pvalue
allpv = np.concatenate([pv_w.ravel(), pv_wo.ravel()]); ok = ~np.isnan(allpv)
q = np.full_like(allpv, np.nan)
if ok.any():
    _, qv, _, _ = multipletests(allpv[ok], method='fdr_bh'); q[ok] = qv
qw = q[:n_p*n_g].reshape(n_p, n_g); qwo = q[n_p*n_g:].reshape(n_p, n_g)

# drawing
def draw_split(ax, col, row, c_ul, c_lr, cell):
    x0, y0 = col*cell, row*cell; x1, y1 = x0+cell, y0+cell
    ax.add_patch(Polygon([[x0,y0],[x0,y1],[x1,y1]], closed=True, facecolor=c_ul, edgecolor='none', zorder=2))
    ax.add_patch(Polygon([[x0,y0],[x1,y0],[x1,y1]], closed=True, facecolor=c_lr, edgecolor='none', zorder=2))
def borders(ax, cell):
    for c in range(n_g):
        for r in range(n_p):
            ax.plot([c*cell,(c+1)*cell],[r*cell,(r+1)*cell], color='white', lw=0.8, zorder=3)
    for c in range(n_g+1): ax.axvline(c*cell, color='white', lw=0.8, zorder=3)
    for r in range(n_p+1): ax.axhline(r*cell, color='white', lw=0.8, zorder=3)
def setup(ax, cell, pw, ph):
    ax.set_xlim(0, pw); ax.set_ylim(0, ph); ax.set_aspect('equal')
    ax.set_xticks([(i+0.5)*cell for i in range(n_g)]); ax.set_xticklabels(GENES, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([(i+0.5)*cell for i in range(n_p)]); ax.set_yticklabels(PERTS[::-1], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

cell = 0.85; pw, ph = n_g*cell, n_p*cell
lm, bm, rm, tm = 1.4, 1.0, 1.5, 0.5; fw, fh = lm+pw+rm, bm+ph+tm

# LEFT: GT LFC
# 2026-08-27: colour scale FIXED at +/-2 at Xiaojie's request (was data-driven,
# round(max|LFC|*1.05,1) = 3.2 for this grid). Values beyond +/-2 saturate; the
# np.clip below already handles that. Data unchanged, only the mapping.
vmax = 2.0
_data_max = max(float(np.nanmax(np.abs(lfc_w))), float(np.nanmax(np.abs(lfc_wo))))
print(f'Colour scale fixed at +/-{vmax}; data max |LFC| = {_data_max:.2f} '
      f'({int(np.nansum(np.abs(lfc_w) > vmax) + np.nansum(np.abs(lfc_wo) > vmax))} of '
      f'{2*n_p*n_g} triangles saturate)', flush=True)
cmap_l = plt.cm.RdBu_r; norm_l = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
fig1 = plt.figure(figsize=(fw, fh)); fig1.patch.set_facecolor('white')
ax1 = fig1.add_axes([lm/fw, bm/fh, pw/fw, ph/fh]); setup(ax1, cell, pw, ph)
for ri in range(n_p):
    y = n_p-1-ri
    for gl in range(n_g):
        vw, vwo = lfc_w[ri, gl], lfc_wo[ri, gl]
        c_ul = cmap_l(norm_l(float(np.clip(vw,  -vmax, vmax)))) if not np.isnan(vw)  else '#C0C0C0'
        c_lr = cmap_l(norm_l(float(np.clip(vwo, -vmax, vmax)))) if not np.isnan(vwo) else '#C0C0C0'
        draw_split(ax1, gl, y, c_ul, c_lr, cell)
        if not np.isnan(qw[ri,gl])  and qw[ri,gl]  < Q_THRESH: ax1.plot(gl*cell+cell*0.28, y*cell+cell*0.72, 'k.', ms=5, zorder=5)
        if not np.isnan(qwo[ri,gl]) and qwo[ri,gl] < Q_THRESH: ax1.plot(gl*cell+cell*0.72, y*cell+cell*0.28, 'k.', ms=5, zorder=5)
borders(ax1, cell)
fig1.suptitle('Ground-Truth LFC\n▲ upper-left: with T-cell  |  ▼ lower-right: without T-cell\n• = significant (Wilcoxon BH q < 0.1)',
              fontsize=9, fontweight='bold', x=(lm+pw/2)/fw, y=1.04)
cax1 = fig1.add_axes([(lm+pw+0.15)/fw, bm/fh, 0.14/fw, ph/fh])
sm1 = ScalarMappable(cmap=cmap_l, norm=norm_l); sm1.set_array([])
cb1 = fig1.colorbar(sm1, cax=cax1); cb1.set_label(f'LFC (clipped ±{vmax:g})', fontsize=8); cb1.ax.tick_params(labelsize=7)
for ext in ('png','svg','pdf'):
    fig1.savefig(f'{OUT_DIR}/gt_lfc_5rows_procB_joint.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print('Saved gt_lfc_5rows_procB_joint.*', flush=True)

# RIGHT: confidence
cmap_r = plt.cm.RdBu_r; norm_r = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
def cconf(p): return '#FFFFFF' if np.isnan(p) else cmap_r(norm_r(float(p)))
fig2 = plt.figure(figsize=(fw, fh)); fig2.patch.set_facecolor('white')
ax2 = fig2.add_axes([lm/fw, bm/fh, pw/fw, ph/fh]); setup(ax2, cell, pw, ph)
for ri in range(n_p):
    y = n_p-1-ri
    for gl in range(n_g):
        draw_split(ax2, gl, y, cconf(conf_w[ri, gl]), cconf(conf_wo[ri, gl]), cell)
borders(ax2, cell)
fig2.suptitle('spaGFM Prediction Confidence\n▲ upper-left: with T-cell  |  ▼ lower-right: without T-cell\nRed = predicts up  |  Blue = predicts down  |  Dark = high confidence',
              fontsize=9, fontweight='bold', x=(lm+pw/2)/fw, y=1.04)
cax2 = fig2.add_axes([(lm+pw+0.15)/fw, bm/fh, 0.14/fw, ph/fh])
sm2 = ScalarMappable(cmap=cmap_r, norm=norm_r); sm2.set_array([])
cb2 = fig2.colorbar(sm2, cax=cax2); cb2.set_ticks([0.0,0.25,0.5,0.75,1.0])
cb2.set_ticklabels(['0.0\n(down)','0.25','0.5','0.75','1.0\n(up)'])
cb2.set_label('predicted P(up)  —  darker = more confident', fontsize=8); cb2.ax.tick_params(labelsize=7)
for ext in ('png','svg','pdf'):
    fig2.savefig(f'{OUT_DIR}/spagfm_conf_5rows_procB_joint.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print('Saved spagfm_conf_5rows_procB_joint.*', flush=True)

# verify counts (overall + per-row, so the added row can be audited)
cw  = (conf_w  > 0.5) == (lfc_w  > 0); cwo = (conf_wo > 0.5) == (lfc_wo > 0)
opp = (lfc_w > 0) != (lfc_wo > 0)
print(f'\nVerify: correctness={int(np.nansum(cw)+np.nansum(cwo))}/{2*n_p*n_g}, '
      f'flip cells={int(np.nansum(opp))}/{n_p*n_g}, '
      f'sig triangles={int(np.nansum(qw<Q_THRESH)+np.nansum(qwo<Q_THRESH))}/{2*n_p*n_g}', flush=True)
print('\nPer-row direction accuracy (out of 16):')
for pi, p in enumerate(PERTS):
    print(f'  {p:8s} {int(np.nansum(cw[pi])+np.nansum(cwo[pi])):2d}/16', flush=True)
print('DONE.', flush=True)
