# Panel: figure2i
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/TLS_figure_317M/redo_figure_B.py
# Original filename: redo_figure_B.py
# Last modified: 2026-07-19 10:01:46  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: pick/GSM5924031_ffpe_c_3.h5ad; spaGFM_predict overwritten with new_model_results/GSM5924031_ffpe_c_3.h5ad obs['317M_binary_predict']. Score = mean log1p-CP10K over the 21-gene tls_markers list defined in the script.
# Output: TLS_figure_317M/figure_B.{png,pdf,svg}  (two subplots: A = prediction classes, B = ground-truth label)
# Match evidence: all four violins in the slide match figure_B numerically - Predicted TLS median 0.93 / box 0.73-1.09, Predicted non-TLS median 0.46 / box 0.38-0.56 (subplot A, n=17 and n=4738); Annotated TLS median 0.68 / box 0.56-0.85, Annotated non-TLS median 0.46 / box 0.38-0.55 (subplot B, n=257 and n=4498). Same blue (#1f77b4) / light-grey (#d3d3d3) fills, the same violin + inner box + thick median stick style, the same *** brackets, and the same 0.0-1.4 "TLS Signature Score" y-axis.
# Slide edits made outside this script: the two subplots were composited into one axis with a shared y-axis; titles, p-values, "mean=" annotations, n counts and the yellow note box were dropped; x labels shortened to "Predicted TLS / Predicted non-TLS / Annotated TLS / Annotated non-TLS".
# Other candidates considered: TLS_figure/redo_figure_B.py (2026-06-11) is the pre-317M ancestor with the old spaGFM head; TLS_figure/redo_figure_B_violin_only.py drops all the text and brackets, so it cannot be the source of the *** brackets; tls_validation_figures{,_multi}.py draw the same two violin pairs inside a larger sheet.
# DATA DRIFT (verified 2026-08-30): the column this script reads, new_model_results/
#   GSM5924031_ffpe_c_3.h5ad obs['317M_binary_predict'], was re-pointed to a different model
#   head when that file was rewritten 2026-07-29 13:49. It now yields 111 TLS spots; the archived
#   figure was drawn from 17. The original values survive untouched in obs['317M_8_binary_predict']
#   (n=17, mean 0.903 / 0.487, KW p=8.38e-10 -- an exact match to figure_B on all four statistics,
#   and no other head matches). Swapping that one column name reproduces the archived figure
#   bit-for-bit; see ../figure2_figures/README.md section 2 and _logs/i_verify.log.
# ---- copied verbatim below; NOT modified ----
"""
Redo figure_B in the ORIGINAL full style: violin + inner box/whiskers + median
sticks, plus all text (titles with stats, axis labels, n counts, mean= annotations,
significance brackets, note box). Single sample GSM5924031.
Outputs figure_B.pdf and figure_B.svg (and figure_B.png for preview).
"""
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import scipy.sparse as sp
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/KC/test/pick/GSM5924031_ffpe_c_3.h5ad"

adata = sc.read_h5ad(DATA_PATH)
# --- spaGFM prediction now sourced from the NEW 317M model results ---
_new_adata = sc.read_h5ad(DATA_PATH.replace('/pick/', '/pick/new_model_results/'))
assert list(_new_adata.obs_names) == list(adata.obs_names), 'cell order mismatch (new vs old)'
adata.obs['spaGFM_predict'] = _new_adata.obs['317M_binary_predict'].values
print('spaGFM_predict replaced with 317M_binary_predict from new_model_results')
print(f"Loaded: {adata.shape[0]} spots × {adata.shape[1]} genes")

# Log-normalised copy for marker scoring
adata_pp = adata.copy()
if sp.issparse(adata_pp.X):
    adata_pp.X = adata_pp.X.toarray()
adata_pp.var_names = adata_pp.var['gene_name'].values
adata_pp.var_names_make_unique()
sc.pp.normalize_total(adata_pp, target_sum=1e4)
sc.pp.log1p(adata_pp)

class_map   = {0: 'TLS', 2: 'Non-TLS'}
class_order = ['TLS', 'Non-TLS']
palette_cls = {'TLS': '#1f77b4', 'Non-TLS': '#d3d3d3'}  # TLS keeps Mature-TLS color

adata.obs['best_label'] = adata.obs['spaGFM_predict'].map(class_map)

tls_markers = ['MS4A1','CD79A','CD79B','MZB1','IGHG1','CR2','BCL6','AICDA',
               'MKI67','CD3D','CD3E','CD8A','CXCL13','CXCR5','CCL19','CCL21',
               'CCR7','SELL','PDCD1','LAMP3','PECAM1']
marker_idx = [adata_pp.var_names.get_loc(g) for g in tls_markers]
adata.obs['TLS_score'] = adata_pp.X[:, marker_idx].mean(axis=1)

score_by_class = {cls: adata.obs.loc[adata.obs['best_label'] == cls, 'TLS_score'].values
                  for cls in class_order}
kw_stat, kw_p = stats.kruskal(*[score_by_class[c] for c in class_order])

# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A
ax = axes[0]
parts = ax.violinplot([score_by_class[c] for c in class_order],
                       positions=[0, 1], showmedians=True, showextrema=False)
for pc, cls in zip(parts['bodies'], class_order):
    pc.set_facecolor(palette_cls[cls]); pc.set_alpha(0.7)
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
ax.boxplot([score_by_class[c] for c in class_order], positions=[0, 1],
           widths=0.12, patch_artist=True, showfliers=False,
           medianprops=dict(color='black', linewidth=2),
           boxprops=dict(facecolor='white', alpha=0.8),
           whiskerprops=dict(color='gray'), capprops=dict(color='gray'))
ax.set_xticks([0, 1])
ax.set_xticklabels([f'{c}\n(n={len(score_by_class[c])})' for c in class_order], fontsize=10)
ax.set_ylabel('TLS Signature Score', fontsize=11)
ax.set_title(f'A. TLS Score by spaGFM Prediction Class\n(Kruskal-Wallis p={kw_p:.2e})',
             fontsize=11, fontweight='bold')
ymax = max([v.max() for v in score_by_class.values()])
for x1, x2, y_off, txt in [(0, 1, 0.02, '***')]:
    ax.plot([x1, x1, x2, x2], [ymax+y_off, ymax+y_off+0.02, ymax+y_off+0.02, ymax+y_off],
            lw=1, c='k')
    ax.text((x1+x2)/2, ymax+y_off+0.025, txt, ha='center', va='bottom', fontsize=9)
ax.set_ylim(-0.05, ymax + 0.22)
for i, cls in enumerate(class_order):
    ax.text(i, -0.03, f'mean={score_by_class[cls].mean():.3f}',
            ha='center', va='top', fontsize=8, color='#333333')

# Panel B
ax2 = axes[1]
gt_tls    = adata.obs.loc[adata.obs['label'] == 0, 'TLS_score'].values
gt_nontls = adata.obs.loc[adata.obs['label'] == 2, 'TLS_score'].values
parts2 = ax2.violinplot([gt_tls, gt_nontls], positions=[0, 1],
                        showmedians=True, showextrema=False)
for pc, col in zip(parts2['bodies'], ['#1f77b4', '#d3d3d3']):
    pc.set_facecolor(col); pc.set_alpha(0.7)
parts2['cmedians'].set_color('black'); parts2['cmedians'].set_linewidth(2)
ax2.boxplot([gt_tls, gt_nontls], positions=[0, 1], widths=0.12, patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(facecolor='white', alpha=0.8),
            whiskerprops=dict(color='gray'), capprops=dict(color='gray'))
ax2.set_xticks([0, 1])
ax2.set_xticklabels([f'True TLS\n(n={len(gt_tls)})', f'True Non-TLS\n(n={len(gt_nontls)})'],
                    fontsize=10)
ax2.set_ylabel('TLS Signature Score', fontsize=11)
mwu_stat, mwu_p = stats.mannwhitneyu(gt_tls, gt_nontls, alternative='two-sided')
ax2.set_title(f'B. TLS Score by Ground Truth Label\n(Mann-Whitney U p={mwu_p:.2e})',
              fontsize=11, fontweight='bold')
ymax2 = max(gt_tls.max(), gt_nontls.max())
ax2.plot([0, 0, 1, 1], [ymax2+0.02, ymax2+0.04, ymax2+0.04, ymax2+0.02], lw=1, c='k')
ax2.text(0.5, ymax2+0.045, '***', ha='center', va='bottom', fontsize=9)
ax2.set_ylim(-0.05, ymax2 + 0.12)
for i, (v, lbl) in enumerate([(gt_tls, 'TLS'), (gt_nontls, 'Non-TLS')]):
    ax2.text(i, -0.03, f'mean={v.mean():.3f}', ha='center', va='top', fontsize=8, color='#333333')
ax2.text(0.02, 0.97,
         'Note: True TLS spans a wide score range\nbecause it includes spots at all TLS\nmaturation stages (Early→Mature)',
         transform=ax2.transAxes, fontsize=8, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.suptitle('TLS Marker Gene Signature Score Validation', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figure_B.pdf", bbox_inches='tight')
plt.savefig("figure_B.svg", bbox_inches='tight')
plt.savefig("figure_B.png", dpi=150, bbox_inches='tight')
plt.close()
print("figure_B.pdf, figure_B.svg, figure_B.png saved (original full style).")
