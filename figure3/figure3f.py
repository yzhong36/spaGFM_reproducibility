# Panel: figure2f
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/TLS_prediction_outline_20260812/draw_prediction_outline.py
# Original filename: draw_prediction_outline.py   (SAME script as figure2e - see the caveat below)
# Last modified: 2026-08-12 15:28:25  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: same h5ad as figure2e, but the GROUND-TRUTH column obs['label'] / obs['TLS_2_cat'] (0 = TLS, 2 = Non-TLS, n_TLS = 255) instead of obs['317M_predict']
# Output: NO on-disk output exists for this panel (see "Match evidence")
# Match evidence: the slide's panel f is drawn with the identical outline recipe as panel e (same crop, colour, line width, region wash), and its two orange regions match the GROUND-TRUTH TLS spots exactly - compare TLS_figure_317M_annotated/figures/GSM5924038_ffpe_c_36/01_ground_truth.png ("TLS (n=255)"), whose elongated upper-right blob and large triangular lower-right blob are the same shapes. This script is the only code under graph_foundation_model/ that draws traced region outlines, and its CLI (`python draw_prediction_outline.py <obs_column> <output_stem>`) accepts any binary 0/2 column, so `label` produces panel f.
# CAVEAT - NOT REPRODUCED: no panel_f_* file exists on disk and no run with the `label` column appears in TLS_prediction_outline_20260812/logs/ (only 317M_predict and spaGFM_binary_predict were run, jobs 6814299 / 6814314). PANEL_LETTER='e' and PANEL_TITLE='Prediction' are module constants with no CLI override, so whoever made panel f either edited those two lines or relabelled the figure afterwards. Copied here verbatim and unmodified; to regenerate panel f, set PANEL_LETTER='f', PANEL_TITLE='Annotation' and run with the ground-truth column.
# Other candidates considered: redo_annotation_panels.py (01_ground_truth.png) uses the right data but flat fills with a legend, no outlines; tls_validation_figures_multi.py likewise. Neither matches the slide's rendering.
# ---- copied verbatim below; NOT modified ----
"""
Panel "e — Prediction": H&E with the predicted TLS spots drawn as orange dots and
the contiguous predicted regions traced with a thick orange outline.

Data (self-contained: H&E + coords + predictions in one file):
    /fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/KC/test/pick/new_model_results/
        GSM5924038_ffpe_c_36_corrected.h5ad

Lineage: the spot-fill version of this panel is
    TLS_figure_317M_annotated/redo_annotation_panels.py            (02/03 panels)
    TLS_figure_317M/redo_sample_panels_GSM5924038.py               (02 panel)
Neither of those draws region outlines, so the outline pass here is new.  The
drawing recipe for the spots themselves (EllipseCollection at the true Visium
footprint, hires scalefactor) is kept identical to those scripts.

NOTE on the prediction column: as of 2026-08-12 the *_corrected.h5ad file no
longer carries `317M_binary_predict`; `317M_predict` is itself binary here
(0 = TLS, 2 = Non-TLS, n_TLS = 271).  PRED_KEY below is asserted to be binary so
a future 3-class file fails loudly instead of silently mis-colouring.

Reads with h5py (not scanpy) so the X matrix is never loaded.
"""
import os
import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from scipy import ndimage as ndi
from skimage import measure

# ── configuration ────────────────────────────────────────────────────────────
H5AD = ("/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/KC/test/pick/"
        "new_model_results/GSM5924038_ffpe_c_36_corrected.h5ad")
PRED_KEY = '317M_predict'      # 0 = TLS, 2 = Non-TLS
TLS_CODE = 0

OUTDIR = os.path.dirname(os.path.abspath(__file__))
STEM   = 'panel_e_prediction_GSM5924038_ffpe_c_36'

COL_TLS   = '#b15929'          # same burnt orange as the earlier TLS panels
ALPHA_DOT = 0.75               # predicted-TLS spot fill
ALPHA_REG = 0.10               # wash inside an outlined region
LW_OUTLINE = 2.6

MIN_SPOTS_PER_REGION = 2       # lone spots stay as bare dots, no outline
CROP_MARGIN_PX       = 15      # hires px of padding around the tissue bbox

PANEL_LETTER = 'e'
PANEL_TITLE  = 'Prediction'


def read_obs_int(f, key):
    """obs column -> int array, handling both plain datasets and categoricals."""
    node = f['obs'][key]
    if isinstance(node, h5py.Group):                  # categorical
        cats = node['categories'][()]
        return np.asarray(cats)[node['codes'][()]].astype(int)
    return node[()].astype(int)


def load():
    with h5py.File(H5AD, 'r') as f:
        sk = list(f['uns/spatial'].keys())[0]
        sp = f[f'uns/spatial/{sk}']
        hires  = sp['images/hires'][()]
        scalef = float(sp['scalefactors/tissue_hires_scalef'][()])
        diam   = float(sp['scalefactors/spot_diameter_fullres'][()]) * scalef

        coords = f['obsm/spatial'][()]
        x = coords[:, 0] * scalef
        y = coords[:, 1] * scalef

        pred     = read_obs_int(f, PRED_KEY)
        in_tiss  = read_obs_int(f, 'in_tissue') if 'in_tissue' in f['obs'] else None
    codes = set(np.unique(pred).tolist())
    assert codes <= {0, 2}, f'{PRED_KEY} is not binary here: found codes {sorted(codes)}'
    return hires, scalef, diam, x, y, pred, in_tiss, sk


def region_mask(shape, x, y, radius, close_px):
    """Binary mask of the spot footprints, morphologically closed so that
    neighbouring Visium spots merge into one blob."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    r = int(np.ceil(radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= radius ** 2
    for cx, cy in zip(x, y):
        ci, cj = int(round(cy)), int(round(cx))
        i0, i1 = max(0, ci - r), min(h, ci + r + 1)
        j0, j1 = max(0, cj - r), min(w, cj + r + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        m[i0:i1, j0:j1] |= disk[i0 - (ci - r): i1 - (ci - r),
                                j0 - (cj - r): j1 - (cj - r)]
    rr = int(np.ceil(close_px))
    yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    se = (xx ** 2 + yy ** 2) <= close_px ** 2
    m = ndi.binary_closing(m, structure=se)
    # a fiducial or an unassayed spot inside a region would otherwise punch a
    # hole that find_contours traces as a second, inner outline
    return ndi.binary_fill_holes(m)


def tissue_bbox(hires, x, y, in_tiss):
    """(x0, y0, x1, y1) of the tissue section in hires pixels.

    Otsu on the greyscale image separates the pale slide background from the
    H&E-stained section; the largest connected component is the section.  Falls
    back to the spot bounding box if that segmentation looks implausible.
    """
    sel = in_tiss.astype(bool) if in_tiss is not None else np.ones(len(x), bool)
    fallback = (x[sel].min(), y[sel].min(), x[sel].max(), y[sel].max())
    try:
        from skimage.filters import threshold_otsu
        grey = hires.astype(float).mean(axis=2)
        m = grey < threshold_otsu(grey)                     # tissue is darker
        m = ndi.binary_closing(m, structure=np.ones((9, 9)))
        m = ndi.binary_opening(m, structure=np.ones((9, 9)))
        lab, n = ndi.label(m)
        if n == 0:
            return fallback
        sizes = ndi.sum(m, lab, range(1, n + 1))
        big = (lab == (int(np.argmax(sizes)) + 1))
        if big.sum() < 0.05 * grey.size:                    # segmentation failed
            return fallback
        ys, xs = np.where(big)
        return xs.min(), ys.min(), xs.max(), ys.max()
    except Exception as exc:                                # noqa: BLE001
        print(f'  tissue segmentation failed ({exc}); cropping to the spot bbox')
        return fallback


def spots(ax, x, y, diam, alpha, colour, edge='none', lw=0):
    if len(x) == 0:
        return
    ec = EllipseCollection(widths=diam, heights=diam, angles=0, units='xy',
                           offsets=np.column_stack([x, y]),
                           offset_transform=ax.transData,
                           facecolors=colour, edgecolors=edge, linewidths=lw,
                           alpha=alpha)
    ax.add_collection(ec, autolim=False)


def main(pred_key=None, stem=None):
    global PRED_KEY, STEM
    if pred_key:
        PRED_KEY = pred_key
    if stem:
        STEM = stem
    hires, scalef, diam, x, y, pred, in_tiss, sample = load()
    H, W = hires.shape[:2]
    tls = pred == TLS_CODE
    print(f'{sample}: {len(pred)} spots, predicted TLS = {int(tls.sum())} '
          f'({PRED_KEY}), hires {W}x{H}, spot diam {diam:.1f} px')

    # Visium pitch is 100 um centre-to-centre vs a 55 um spot -> close with a bit
    # more than the gap between two neighbouring footprints.
    pitch    = diam * 100.0 / 55.0
    close_px = 0.60 * (pitch - diam) + 2.0

    mask = region_mask((H, W), x[tls], y[tls], diam / 2.0, close_px)
    lab, n = ndi.label(mask)
    print(f'  {n} connected predicted regions before the size filter')

    # keep only regions holding >= MIN_SPOTS_PER_REGION spots
    ids_at_spot = lab[np.clip(np.round(y[tls]).astype(int), 0, H - 1),
                      np.clip(np.round(x[tls]).astype(int), 0, W - 1)]
    keep = {i for i in np.unique(ids_at_spot) if i > 0
            and int((ids_at_spot == i).sum()) >= MIN_SPOTS_PER_REGION}
    n_in = int(np.isin(ids_at_spot, list(keep)).sum()) if keep else 0
    print(f'  {len(keep)} regions with >= {MIN_SPOTS_PER_REGION} spots '
          f'({n_in} spots inside them)')

    kept = np.isin(lab, list(keep)) if keep else np.zeros_like(mask)
    # light smoothing so the outline follows the region instead of every spot rim
    smooth = ndi.gaussian_filter(kept.astype(float), sigma=diam * 0.35)
    smooth = ndi.binary_fill_holes(smooth >= 0.5).astype(float)
    smooth = ndi.gaussian_filter(smooth, sigma=diam * 0.20)

    # ── draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(hires, origin='upper', interpolation='bilinear')

    for c in measure.find_contours(smooth, 0.5):
        ax.fill(c[:, 1], c[:, 0], facecolor=COL_TLS, alpha=ALPHA_REG,
                edgecolor='none', zorder=2)
        ax.plot(c[:, 1], c[:, 0], color=COL_TLS, lw=LW_OUTLINE,
                solid_joinstyle='round', zorder=4)

    spots(ax, x[tls], y[tls], diam, ALPHA_DOT, COL_TLS)

    # crop to the tissue itself, not to the spot array (the array extends well
    # past the section on this slide)
    x0, y0, x1, y1 = tissue_bbox(hires, x, y, in_tiss)
    x0, x1 = x0 - CROP_MARGIN_PX, x1 + CROP_MARGIN_PX
    y0, y1 = y0 - CROP_MARGIN_PX, y1 + CROP_MARGIN_PX
    ax.set_xlim(max(0, x0), min(W, x1))
    ax.set_ylim(min(H, y1), max(0, y0))          # inverted: image coords
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(PANEL_TITLE, fontsize=15, pad=10)
    fig.text(0.015, 0.975, PANEL_LETTER, fontsize=20, fontweight='bold',
             ha='left', va='top')

    for ext, dpi in (('png', 400), ('pdf', 400), ('svg', 400)):
        p = os.path.join(OUTDIR, f'{STEM}.{ext}')
        fig.savefig(p, dpi=dpi, bbox_inches='tight',
                    facecolor='white', transparent=False)
        print(f'  saved {p}')
    plt.close(fig)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(pred_key=sys.argv[1],
             stem=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
