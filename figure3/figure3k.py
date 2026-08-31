# Panel: figure2k  (sub-images 1 and 2 of 3: "21 TLS signatures" and "CCL19")
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/TLS_marker_panels_317M_HEV21_20260811/make_spatial_score.py
# Original filename: make_spatial_score.py   (imports panel_common.py, copied alongside)
# Last modified: 2026-08-11 21:45:51  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: pick/<sample>.h5ad x5 (counts, var['gene_name'], uns['spatial'], obsm['spatial']); gene list imported from panel_common.ORDERED_GENES so the score covers exactly the 21 genes drawn in figure2j
# Output: TLS_marker_panels_317M_HEV21_20260811/spatial/GSM5924038_ffpe_c_36/03_tls_marker_score_HEV21.png  and  .../CCL19_spatial.png
# Match evidence: the slide's first two sub-images are these two files for GSM5924038_ffpe_c_36 - same wedge-shaped tissue crop, same YlOrRd spot colouring on H&E at alpha 0.4, and the same hot cluster in the lower-right quadrant. The colourbar label in the source reads "TLS Signature Score (mean log-norm, 21 markers)", i.e. the slide's "21 TLS signatures".
# Slide edits made outside this script: the three per-image titles and colourbars were replaced by one shared "Low / Medium / High - Expression" YlOrRd legend.
# Note: panel k is a THREE-image composite drawn by TWO scripts. The third sub-image (MS4A1) comes from figure2k_ms4a1.py.
# Other candidates considered: TLS_figure_317M/redo_sample_panels_all_corrected.py panel 03 uses the OLD 18-gene list (figures/<sample>/03_tls_marker_score.png) and emits no CCL19 map; this script is its 21-gene successor and imports the gene list rather than hard-coding it.
# ---- copied verbatim below; NOT modified ----
"""
Spatial H&E overlays for the 21-marker HEV panel, all 5 KC samples.

Per sample, writes into spatial/<sample>/:
  03_tls_marker_score_HEV21.png   TLS signature score over the NEW 21-gene panel
  CCL19_spatial.png               CCL19 alone, same drawing recipe

Drawing recipe (EllipseCollection spots at the true Visium footprint, H&E at
alpha 0.4, YlOrRd, vmax = 99th percentile, dpi 180) is carried over unchanged
from TLS_figure_317M/redo_sample_panels_all_corrected.py panel 03, so these are
drop-in comparable with the earlier signature maps.

Source data is the same as the marker heatmaps in this folder:
  pick/*.h5ad  (counts, var/gene_name, uns['spatial'], obsm['spatial'])
The gene list is imported from panel_common.ORDERED_GENES, so the score always
covers exactly the genes drawn in fig1-fig4 -- edit MODULES there, not here.

Note the score is a plain mean of log1p CP10K over the panel, unweighted. Genes
that sit near the detection floor in this data (CHST4, MADCAM1) therefore pull
the mean toward zero rather than contributing signal; the per-gene CCL19 map is
the honest read of one strong marker.
"""
import os
_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)

import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
import scipy.sparse as sp

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

from panel_common import FILES, ORDERED_GENES

OUT = 'spatial'
SINGLE_GENES = ['CCL19']


def _base_fig(hires, title, he_alpha=1.0):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hires, origin='upper', alpha=he_alpha)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    return fig, ax


def _spots(ax, x, y, diam, alpha, cmap, norm, array):
    ec = EllipseCollection(widths=diam, heights=diam, angles=0, units='xy',
                           offsets=np.column_stack([x, y]),
                           offset_transform=ax.transData,
                           edgecolors='none', linewidths=0, alpha=alpha,
                           cmap=cmap, norm=norm)
    ec.set_array(np.asarray(array))
    ax.add_collection(ec, autolim=False)
    return ec


def draw(hires, x_s, y_s, diam, values, title, cbar_label, path):
    """One continuous-valued overlay. vmax is the 99th percentile so a handful
    of very hot spots cannot flatten the rest of the section."""
    fig, ax = _base_fig(hires, title, he_alpha=0.4)
    vmax = float(np.percentile(values, 99))
    if not np.isfinite(vmax) or vmax <= 0:      # gene absent / all-zero section
        vmax = float(np.max(values)) if np.max(values) > 0 else 1.0
    ec = _spots(ax, x_s, y_s, diam, 0.9, 'YlOrRd', Normalize(vmin=0, vmax=vmax), values)
    plt.colorbar(ec, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"    {path}  (vmax={vmax:.3f})", flush=True)


print(f"{len(ORDERED_GENES)}-gene panel: {', '.join(ORDERED_GENES)}\n", flush=True)

for f in FILES:
    sample = os.path.basename(f).replace('.h5ad', '')
    outdir = os.path.join(OUT, sample)
    os.makedirs(outdir, exist_ok=True)
    print(f"=== {sample} ===", flush=True)

    adata = sc.read_h5ad(f)

    # log-normalised copy for scoring -- CP10K then log1p, as in panel 03
    pp = adata.copy()
    if sp.issparse(pp.X):
        pp.X = pp.X.toarray()
    pp.var_names = pp.var['gene_name'].values
    pp.var_names_make_unique()
    sc.pp.normalize_total(pp, target_sum=1e4)
    sc.pp.log1p(pp)

    missing = [g for g in ORDERED_GENES if g not in pp.var_names]
    if missing:
        raise SystemExit(f"STOP: gene(s) absent in {sample}: {missing}. No substitution made.")
    idx = [pp.var_names.get_loc(g) for g in ORDERED_GENES]
    score = pp.X[:, idx].mean(axis=1)

    sk = list(adata.uns['spatial'].keys())[0]
    hires = adata.uns['spatial'][sk]['images']['hires']
    scalef = adata.uns['spatial'][sk]['scalefactors']['tissue_hires_scalef']
    diam = adata.uns['spatial'][sk]['scalefactors']['spot_diameter_fullres'] * scalef
    x_s = adata.obsm['spatial'][:, 0] * scalef
    y_s = adata.obsm['spatial'][:, 1] * scalef

    draw(hires, x_s, y_s, diam, score,
         f'{sample}\nTLS Marker Signature Score',
         f'TLS Signature Score\n(mean log-norm, {len(ORDERED_GENES)} markers)',
         os.path.join(outdir, '03_tls_marker_score_HEV21.png'))

    for g in SINGLE_GENES:
        v = pp.X[:, pp.var_names.get_loc(g)]
        draw(hires, x_s, y_s, diam, v,
             f'{sample}\n{g} expression',
             f'{g}\n(log-norm CP10K)',
             os.path.join(outdir, f'{g}_spatial.png'))

print("\nDone.", flush=True)
