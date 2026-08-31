# Panel: figure2j
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/TLS_marker_panels_317M_HEV21_20260811/make_figures.py
# Original filename: make_figures.py   (imports panel_common.py, copied alongside as panel_common.py)
# Last modified: 2026-08-11 21:27:25  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: pick/new_model_results/<sample>_corrected.h5ad x5, obs['label'] (ground truth) and obs['317M_binary_predict'] (drawn as the "spaGFM" row); 21-gene HEV panel defined in panel_common.MODULES
# Output: TLS_marker_panels_317M_HEV21_20260811/fig1_lfc/fig1_lfc_pooled.{png,pdf}  (the fig1 family; the script also writes fig2/fig3/fig4 and 5 per-sample panels each)
# Match evidence: PIXEL-IDENTICAL. figure2/fig1j.png and fig1_lfc/fig1_lfc_pooled.png are byte-for-byte equal in RGB (both 4405x928; numpy array_equal == True). No ambiguity.
# Note: run it with run_panels.slurm in the source folder (env /fs/ess/PAS1475/Xiaojie/spatialQC/pyenv_spatialqc0/bin/python). It regenerates all 24 panels; fig1_lfc_pooled is the one used here.
# Other candidates considered: make_fig1_notext.py (2026-08-20) in the same folder redraws fig1 WITHOUT the per-cell numbers and stars -> fig1_lfc_notext/; the slide panel has both, so it is not the source. TLS_marker_panels_317M_corrected_20260730/make_figures.py is the 18-gene ancestor (no HEV/addressin block, so no CHST4/CHST2/MADCAM1 columns); TLS_marker_panels_20260728/ is the older spaGFM-head version.
# ---- copied verbatim below; NOT modified ----
"""
Build the four TLS marker-panel figure families, pooled + one per sample
(6 figures each, 24 total).

2026-08-11 rebuild of TLS_marker_panels_317M_corrected_20260730 with the
21-marker panel requested by the user (HEV/addressin block added; AICDA, MKI67,
PECAM1 dropped) -- see the MODULES comment in panel_common.py. Everything else
(prediction source, CP10K normalisation, statistics, layout) is unchanged.

Prediction source, inherited from the 20260730 rebuild:
Only the prediction source changed (see panel_common.load_samples):
    was  pick/<s>.h5ad             spaGFM_binary_predict / spaGFM_predict
    now  new_model_results/<s>_corrected.h5ad   317M_binary_predict / 317M_predict
Ground truth, the gene panel, CP10K normalisation, colour scheme, statistics
and layout are untouched, so these panels are drop-in comparable with the old
ones. The row labels still read "spaGFM" (the model family) as before.

  fig1  log2 fold change, 2 rows, with FDR significance stars
          GT TLS / GT Non-TLS
          spaGFM TLS / spaGFM Non-TLS          (binary head)
  fig2  mean expression of 4 groups:  GT-TLS, GT-Non-TLS,
                                      spaGFM-TLS, spaGFM-Non-TLS
  fig3  mean expression of 5 groups:  GT-TLS, GT-Non-TLS, spaGFM-Mature-TLS,
                                      spaGFM-Early-TLS, spaGFM-Non-TLS
  fig4  log2 fold change, 3 rows, with FDR significance stars
          GT TLS / GT Non-TLS
          spaGFM Mature TLS / spaGFM Non-TLS   (3-class head)
          spaGFM Early  TLS / spaGFM Non-TLS   (3-class head)

Conventions (see README.md for the reasoning):
  * expression is CP10K = counts / library_size * 1e4, computed per spot on the
    full 17527-gene matrix, group means taken on the LINEAR CP10K scale
  * fig1/4 colour = log2((mean_TLS + 1) / (mean_NonTLS + 1))
  * fig2/3 colour = z-score of the group means WITHIN each gene column, so
    every gene is legible; the number printed in the cell is the raw mean CP10K
  * stars are BH-FDR over the 21 genes within each row, from a two-sided
    Mann-Whitney U test on the per-spot CP10K values
  * colour scales are shared across all 6 panels of a family, so the pooled and
    per-sample figures are directly comparable
"""
import os
_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from panel_common import (ORDERED_GENES, MODULES, N_GENES,
                          CODE_TLS, CODE_EARLY, CODE_NON,
                          load_samples, pool, bh_fdr, stars, mwu_pvals,
                          fmt_expr, draw_panel)

FIG1 = 'fig1_lfc'
FIG2 = 'fig2_expr_4group'
FIG3 = 'fig3_expr_5group'
FIG4 = 'fig4_lfc_3row'
TAB = 'tables'

GENE2MOD = {g: m for m, gs in MODULES for g in gs}


def short(name):
    """GSM5924031_ffpe_c_3 -> ffpe_c_3"""
    return name.split('_', 1)[1] if name.startswith('GSM') else name


def group_means(cp, masks):
    """(n_groups, N_GENES) mean CP10K; NaN row where the group has no spots."""
    out = np.full((len(masks), cp.shape[1]), np.nan)
    for i, m in enumerate(masks):
        if m.sum() > 0:
            out[i] = cp[m].mean(axis=0)
    return out


def column_z(G):
    """z-score down each gene column, ignoring NaN rows."""
    mu = np.nanmean(G, axis=0)
    sd = np.nanstd(G, axis=0, ddof=0)
    sd = np.where(sd > 0, sd, np.nan)
    Z = (G - mu) / sd
    return np.where(np.isnan(Z) & ~np.isnan(G), 0.0, Z)


# ═══════════════════════════════════════════════════════════════════════════
print("Loading samples ...", flush=True)
samples = load_samples()
datasets = [pool(samples)] + samples
print(f"{len(datasets)} datasets (1 pooled + {len(samples)} samples)\n", flush=True)


# ── log2FC families (fig1, fig4) ────────────────────────────────────────────
def build_lfc(fam, row_specs, outdir, title_stub):
    """
    row_specs: list of (name, contrast_text, mask_a_fn, mask_b_fn)
    Row value = log2((mean CP10K of a + 1) / (mean CP10K of b + 1)).
    A row whose numerator or denominator group is empty is drawn as n/a.
    """
    print(f"\nComputing {fam} ({len(row_specs)} rows, log2FC + FDR stars) ...", flush=True)
    nrow = len(row_specs)
    packs = []
    for d in datasets:
        cp = d['cp10k']
        M = np.full((nrow, cp.shape[1]), np.nan)
        P = np.full((nrow, cp.shape[1]), np.nan)
        na = np.zeros((nrow, cp.shape[1]), dtype=bool)
        labels, counts = [], []
        for i, (name, contrast, fa, fb) in enumerate(row_specs):
            ma, mb = fa(d), fb(d)
            counts.append((int(ma.sum()), int(mb.sum())))
            if ma.sum() == 0 or mb.sum() == 0:
                na[i] = True
            else:
                a, b = cp[ma].mean(axis=0), cp[mb].mean(axis=0)
                M[i] = np.log2((a + 1.0) / (b + 1.0))
                P[i] = mwu_pvals(cp, ma, mb)
            labels.append(f'{name}\n{contrast}\nn={int(ma.sum()):,} / {int(mb.sum()):,}')
        Q = np.vstack([bh_fdr(P[i]) for i in range(nrow)])
        packs.append(dict(d=d, M=M, P=P, Q=Q, labels=labels, na=na, counts=counts))

    finite = np.concatenate([p['M'] for p in packs])
    vmax = float(np.nanmax(np.abs(finite)))
    print(f"  shared symmetric scale: +/-{vmax:.3f}", flush=True)

    for p in packs:
        d = p['d']
        is_pooled = d['name'].startswith('pooled')
        tag = 'pooled' if is_pooled else short(d['name'])
        text = [[f'{v:.2f}' if np.isfinite(v) else '' for v in row] for row in p['M']]
        star = [[stars(q) for q in row] for row in p['Q']]
        n = d['cp10k'].shape[0]
        title = (f'{title_stub} - all 5 samples pooled' if is_pooled
                 else f'{title_stub} - {short(d["name"])}')
        draw_panel(p['M'], text, p['labels'], title,
                   'log$_2$ fold change (TLS / Non-TLS)',
                   os.path.join(outdir, f'{fam}_{tag}'),
                   vmax=vmax, star_M=star, na_M=p['na'],
                   subtitle=(f'n = {n:,} spots   |   colour = log2FC of mean CP10K   |   '
                             f'* q<0.05  ** q<0.01  *** q<0.001 '
                             f'(Mann-Whitney U, BH-FDR over {N_GENES} genes)'))
        rec = []
        for i, (name, contrast, _, _) in enumerate(row_specs):
            na_n, nb_n = p['counts'][i]
            for j, g in enumerate(ORDERED_GENES):
                rec.append(dict(dataset=tag, row=name, contrast=contrast,
                                n_numerator=na_n, n_denominator=nb_n,
                                module=GENE2MOD[g], gene=g,
                                log2FC=p['M'][i, j], p_value=p['P'][i, j],
                                q_value=p['Q'][i, j], star=stars(p['Q'][i, j])))
        pd.DataFrame(rec).to_csv(os.path.join(TAB, f'{fam}_{tag}.csv'), index=False)


build_lfc(
    'fig1_lfc',
    [('Ground truth', '(TLS / Non-TLS)',
      lambda d: d['gt'] == CODE_TLS, lambda d: d['gt'] == CODE_NON),
     ('spaGFM', '(TLS / Non-TLS)',
      lambda d: d['pred2'] == CODE_TLS, lambda d: d['pred2'] == CODE_NON)],
    FIG1, 'TLS marker enrichment')


# ── fig2 / fig3: group mean expression ──────────────────────────────────────
def build_expr(fam, row_specs, outdir, title_stub):
    print(f"\nComputing {fam} ({len(row_specs)} groups) ...", flush=True)
    packs = []
    for d in datasets:
        cp = d['cp10k']
        masks = [spec[1](d) for spec in row_specs]
        G = group_means(cp, masks)
        Z = column_z(G)
        na = np.zeros(G.shape, dtype=bool)
        for i, m in enumerate(masks):
            if m.sum() == 0:
                na[i] = True
        labels = [f'{spec[0]}\n(n={int(m.sum()):,})' for spec, m in zip(row_specs, masks)]
        packs.append(dict(d=d, G=G, Z=Z, na=na, labels=labels, masks=masks))

    vmax = float(np.nanmax(np.abs(np.concatenate([p['Z'] for p in packs]))))
    print(f"  shared symmetric scale: +/-{vmax:.3f}", flush=True)

    for p in packs:
        d = p['d']
        is_pooled = d['name'].startswith('pooled')
        tag = 'pooled' if is_pooled else short(d['name'])
        text = [[fmt_expr(v) for v in row] for row in p['G']]
        n = d['cp10k'].shape[0]
        title = (f'{title_stub} - all 5 samples pooled' if is_pooled
                 else f'{title_stub} - {short(d["name"])}')
        draw_panel(p['Z'], text, p['labels'], title,
                   'z-score of group means (within gene)',
                   os.path.join(outdir, f'{fam}_{tag}'),
                   vmax=vmax, na_M=p['na'],
                   subtitle=(f'n = {n:,} spots   |   cell value = mean CP10K   |   '
                             'colour = z-score of the group means within each gene column'))
        rec = []
        for i, spec in enumerate(row_specs):
            for j, g in enumerate(ORDERED_GENES):
                rec.append(dict(dataset=tag, group=spec[0], n_spots=int(p['masks'][i].sum()),
                                module=GENE2MOD[g], gene=g,
                                mean_cp10k=p['G'][i, j], z_within_gene=p['Z'][i, j]))
        pd.DataFrame(rec).to_csv(os.path.join(TAB, f'{fam}_{tag}.csv'), index=False)


build_expr(
    'fig2_expr4',
    [('Ground truth TLS',    lambda d: d['gt'] == CODE_TLS),
     ('Ground truth Non-TLS', lambda d: d['gt'] == CODE_NON),
     ('spaGFM TLS',          lambda d: d['pred2'] == CODE_TLS),
     ('spaGFM Non-TLS',      lambda d: d['pred2'] == CODE_NON)],
    FIG2, 'TLS marker expression by group')

build_expr(
    'fig3_expr5',
    [('Ground truth TLS',      lambda d: d['gt'] == CODE_TLS),
     ('Ground truth Non-TLS',   lambda d: d['gt'] == CODE_NON),
     ('spaGFM Mature TLS',     lambda d: d['pred3'] == CODE_TLS),
     ('spaGFM Early TLS',      lambda d: d['pred3'] == CODE_EARLY),
     ('spaGFM Non-TLS',        lambda d: d['pred3'] == CODE_NON)],
    FIG3, 'TLS marker expression by maturity')

# fig4: same log2FC treatment as fig1, but the model rows come from the 3-class
# head, each contrasted against the SAME 3-class Non-TLS denominator.
build_lfc(
    'fig4_lfc3',
    [('Ground truth', '(TLS / Non-TLS)',
      lambda d: d['gt'] == CODE_TLS, lambda d: d['gt'] == CODE_NON),
     ('spaGFM Mature TLS', '(vs spaGFM Non-TLS)',
      lambda d: d['pred3'] == CODE_TLS, lambda d: d['pred3'] == CODE_NON),
     ('spaGFM Early TLS', '(vs spaGFM Non-TLS)',
      lambda d: d['pred3'] == CODE_EARLY, lambda d: d['pred3'] == CODE_NON)],
    FIG4, 'TLS marker enrichment by maturity')

print("\nDONE", flush=True)
