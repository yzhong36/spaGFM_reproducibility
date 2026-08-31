# Panel: figure3f
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_g_nfkb_network_step3_figures.py
# Original filename: fig1_g_nfkb_network_step3_figures.py
# Last modified: 2026-08-18 10:29:10  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/work/panelg/pergene/upfrac_500d.npz; the step2 context CSV; data/raw/spatial_perturb_04_23_26/perturb_fish_spatial.h5ad
# Output: figure_bundle_20260817/data/work/panelg/network_figures/network_context_2panel.{pdf,png,svg}
# Match evidence: SOURCES = ['IRAK3','CYLD','UBE2N','STAT1'] (cls neg/neg/pos/pos -> the red/blue box colouring) and TARGETS = ['IL6','NFKB1','NFKB2','SOD2','CXCL1','CXCL2'] match both node columns exactly. The 'network_context_2panel' output is the two side-by-side with/without-T-cell bipartite networks.
# Other candidates considered: main_figure_perturbation*/panel_g_network/run_network_figures.py and panel_g_network_procB/draw_panelG_procB.py -- the pre-bundle pipeline-A lineage; backup_fig1_g_pipelineA_20260818/ holds the superseded pipeline-A copy.
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python
"""
Publication-quality regulatory-network diagrams.

Visualises the predicted directional regulatory output of four Tier A
(zero-shot) MAPK/NF-kB pathway regulators on six representative NF-kB
immune-response target genes.

Model output = per-cell up-regulation fraction (up-frac in [0,1]) from the
Two-Branch Spatial FM directional classifier (run_pergene.py:288).
We render it as a signed directional score  s = 2*upfrac - 1  in [-1, +1]:
    sign(s)  -> direction   (red = predicted UP, blue = predicted DOWN)
    |s|      -> confidence   (edge width + opacity)

Two figures are produced:
  1. network_main_bipartite  -- integrated regulator -> target network
  2. network_panels_fan      -- per-regulator small-multiple fans
plus a supplementary value table (CSV).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

warnings.filterwarnings('ignore')

OUTDIR = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/panelg'
FIGDIR = os.path.join(OUTDIR, 'network_figures')
os.makedirs(FIGDIR, exist_ok=True)

# ── load predicted up-fractions (500-d, all panel + Tier A genes) ─────────────
npz = np.load(os.path.join(OUTDIR, 'pergene', 'upfrac_500d.npz'))

import scanpy as sc
adata = sc.read_h5ad('/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/'
                     'spatial_perturb_04_23_26/perturb_fish_spatial.h5ad', backed='r')
var_symbols = list(adata.var['gene_symbol'])

# ── four Tier A regulators (source nodes) ─────────────────────────────────────
# Selected for class-coherence: negative regulators whose perturbation is predicted
# to UP-regulate NF-κB targets (red node → red edges) and positive nodes predicted
# to DOWN-regulate them (blue node → blue edges).
#   IRAK3 5/6 · CYLD 4/6 coherent (neg) ; UBE2N 5/6 · STAT1 4/6 coherent (pos)
SOURCES = ['IRAK3', 'CYLD', 'UBE2N', 'STAT1']
SOURCE_META = {
    'IRAK3':   dict(cls='neg', name='IRAK3',   alias='IRAK-M',  role='decoy kinase',
                    layer='TLR / IL-1R'),
    'CYLD':    dict(cls='neg', name='CYLD',    alias='CYLD',    role='K63 deubiquitinase',
                    layer='Ubiquitin / DUB'),
    'UBE2N':   dict(cls='pos', name='UBE2N',   alias='Ubc13',   role='K63 Ub-conjugating E2',
                    layer='Ubiquitin / E2'),
    'STAT1':   dict(cls='pos', name='STAT1',   alias='STAT1',   role='ISG transcription factor',
                    layer='IFN / ISG'),
    'TNFAIP3': dict(cls='neg', name='TNFAIP3', alias='A20',     role='deubiquitinase',
                    layer='Ubiquitin / A20'),
    'TNIP1':   dict(cls='neg', name='TNIP1',   alias='ABIN-1',  role='A20-complex adaptor',
                    layer='Ubiquitin / A20'),
    'OTULIN':  dict(cls='neg', name='OTULIN',  alias='OTULIN',  role='linear-Ub deubiquitinase',
                    layer='Ubiquitin / LUBAC'),
    'TNF':     dict(cls='neg', name='TNF',     alias='TNF-α',   role='autocrine cytokine feedback',
                    layer='Autocrine'),
    'IKBKG':   dict(cls='pos', name='IKBKG',   alias='NEMO',    role='IKK regulatory subunit',
                    layer='IKK core'),
    'REL':     dict(cls='pos', name='REL',     alias='c-Rel',   role='NF-κB transcription factor',
                    layer='NF-κB TF'),
    'MAPK14':  dict(cls='unl', name='MAPK14',  alias='p38α',    role='stress MAP kinase',
                    layer='MAPK'),
}

# ── six immune-response targets selected by DIRECTIONAL PREDICTION ACCURACY ────
# For each candidate NF-κB core gene, accuracy = fraction of the 33 panel
# perturbations (which have ground truth) where the model's predicted up/down
# direction matches the observed up/down direction (kNN reference; control
# reference used only to break the 0.788 6th-place tie). The Tier A regulators
# themselves have no ground truth, so accuracy is a transferable per-target metric.
# Ranking: CXCL1 .818 · CXCL2 .818 · SOD2 .818 · NFKB1 .818 · IL6 .818 · NFKB2 .788
TARGETS = ['IL6', 'NFKB1', 'NFKB2', 'SOD2', 'CXCL1', 'CXCL2']   # grouped for readability
TARGET_META = {
    'IL6':    dict(full='IL-6',   group='Cytokine',  role='pleiotropic cytokine',   acc=0.818),
    'NFKB1':  dict(full='NF-κB1', group='NF-κB',     role='p50 subunit',            acc=0.818),
    'NFKB2':  dict(full='NF-κB2', group='NF-κB',     role='p52 subunit',            acc=0.788),
    'SOD2':   dict(full='SOD2',   group='Effector',  role='MnSOD · antioxidant',    acc=0.818),
    'CXCL1':  dict(full='CXCL1',  group='Chemokine', role='GROα chemokine',         acc=0.818),
    'CXCL2':  dict(full='CXCL2',  group='Chemokine', role='MIP-2α chemokine',       acc=0.818),
    'NFKBIA': dict(full='IκBα',   group='NF-κB',     role='NF-κB feedback inhibitor',acc=0.788),
    'CXCL8':  dict(full='CXCL8',  group='Chemokine', role='IL-8 chemokine',         acc=0.788),
    'NOS2':   dict(full='iNOS',   group='Effector',  role='inducible NO synthase',  acc=0.758),
    'IL1B':   dict(full='IL-1β',  group='Cytokine',  role='pro-inflammatory cytokine',acc=0.727),
    'ICAM1':  dict(full='ICAM-1', group='Adhesion',  role='leukocyte adhesion',     acc=0.697),
}

# full Tier A pool + full NF-κB readout panel (for the all-vs-all figures)
ALL_SOURCES = ['IRAK3', 'CYLD', 'TNFAIP3', 'TNIP1', 'OTULIN', 'TNF',      # neg-reg
               'UBE2N', 'STAT1', 'IKBKG', 'REL',                          # pos-node
               'MAPK14']                                                  # unlabeled
ALL_TARGETS = ['IL6', 'IL1B',                                            # cytokine
               'CXCL8', 'CXCL1', 'CXCL2',                                # chemokine
               'ICAM1',                                                  # adhesion
               'NOS2', 'SOD2',                                           # effector
               'NFKB1', 'NFKB2', 'NFKBIA']                               # NF-κB
CORE_IDX = {g: var_symbols.index(g) for g in ALL_TARGETS}

# ── predicted matrix:  upfrac[source][target]  and signed score s = 2u-1 ──────
# computed over the FULL pool; curated figures index subsets of it
upfrac = {s: {t: float(npz[s][CORE_IDX[t]]) for t in ALL_TARGETS} for s in ALL_SOURCES}
signed = {s: {t: 2.0 * upfrac[s][t] - 1.0 for t in ALL_TARGETS} for s in ALL_SOURCES}

# ── palette ───────────────────────────────────────────────────────────────────
C_UP   = '#D6453D'   # red  — predicted up-regulation
C_DOWN = '#2D6FB3'   # blue — predicted down-regulation
NEUTRAL = '#9AA0A6'

SRC_EDGE = {'neg': '#B23A2E', 'pos': '#1F5C8B', 'unl': '#6B7177'}
SRC_FILL = {'neg': '#FBE3DF', 'pos': '#DCEAF6', 'unl': '#ECEDEE'}

GROUP_FILL = {
    'Cytokine':  '#F6C9C2',
    'Chemokine': '#F7D9A8',
    'Effector':  '#D9C7E8',
    'Adhesion':  '#BDE3C6',
    'NF-κB':     '#BCD6EE',
}
GROUP_EDGE = '#6B7177'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.0,
    'figure.dpi': 300,
    'pdf.fonttype': 42,   # editable text in Illustrator
    'ps.fonttype': 42,
})


def edge_style(s):
    """signed score -> (color, linewidth, alpha)"""
    mag = abs(s)
    col = C_UP if s > 0 else C_DOWN
    lw  = 1.1 + 5.0 * mag          # 1.1 – 6.1 pt
    al  = 0.30 + 0.62 * mag        # 0.30 – 0.92
    return col, lw, al


def source_box(ax, x, y, key, w=0.150, h=0.105):
    m = SOURCE_META[key]
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.008,rounding_size=0.018',
        linewidth=2.0, edgecolor=SRC_EDGE[m['cls']],
        facecolor=SRC_FILL[m['cls']], zorder=5,
        mutation_aspect=0.62))
    ax.text(x, y + 0.022, m['name'], ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=SRC_EDGE[m['cls']], zorder=6)
    ax.text(x, y - 0.012, f"({m['alias']})", ha='center', va='center',
            fontsize=7.2, color='#444', style='italic', zorder=6)
    ax.text(x, y - 0.036, m['role'], ha='center', va='center',
            fontsize=6.0, color='#666', zorder=6)


def target_box(ax, x, y, key, w=0.150, h=0.092):
    m = TARGET_META[key]
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.008,rounding_size=0.018',
        linewidth=1.3, edgecolor=GROUP_EDGE,
        facecolor=GROUP_FILL[m['group']], zorder=5,
        mutation_aspect=0.60))
    ax.text(x, y + 0.018, m['full'], ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='#1A1A1A', zorder=6)
    ax.text(x, y - 0.018, m['role'], ha='center', va='center',
            fontsize=5.8, color='#555', style='italic', zorder=6)
    # directional-accuracy badge (right edge of node)
    ax.text(x + w/2 + 0.012, y, f"acc\n{m['acc']:.2f}", ha='left', va='center',
            fontsize=5.8, color='#2C3E50', fontweight='bold', zorder=6,
            linespacing=0.9,
            bbox=dict(boxstyle='round,pad=0.18', fc='#EBF1F5', ec='#B7C4CE', lw=0.6))


# ══════════════════════════════════════════════════════════════════════════════
# reusable bipartite-network renderer
# ══════════════════════════════════════════════════════════════════════════════
def draw_bipartite(ax, upf, sources=None, targets=None,
                   show_col_headers=True, src_label_fs=10.5, tgt_label_fs=10.5,
                   tgt_subtitle='top-6 by directional accuracy'):
    """Render a regulator -> target network on `ax` from upfrac dict
    upf[source][target] in [0,1]."""
    sources = sources or SOURCES
    targets = targets or TARGETS
    sgn = {s: {t: 2.0 * upf[s][t] - 1.0 for t in targets} for s in sources}
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    xs, xt = 0.165, 0.835
    ys = np.linspace(0.80, 0.16, len(sources))
    yt = np.linspace(0.86, 0.10, len(targets))
    SPOS = {g: (xs, ys[i]) for i, g in enumerate(sources)}
    TPOS = {g: (xt, yt[i]) for i, g in enumerate(targets)}

    if show_col_headers:
        ax.text(xs, 0.93, 'Tier A regulators', ha='center', va='center',
                fontsize=src_label_fs, fontweight='bold', color='#222')
        ax.text(xs, 0.905, 'zero-shot perturbations', ha='center', va='center',
                fontsize=7.0, color='#777', style='italic')
        ax.text(xt, 0.95, 'NF-κB targets', ha='center', va='center',
                fontsize=tgt_label_fs, fontweight='bold', color='#222')
        ax.text(xt, 0.925, tgt_subtitle, ha='center', va='center',
                fontsize=7.0, color='#777', style='italic')

    order = sorted([(s, t) for s in sources for t in targets],
                   key=lambda st: abs(sgn[st[0]][st[1]]))
    hw = 0.150 / 2
    for s, t in order:
        x0, y0 = SPOS[s]; x1, y1 = TPOS[t]
        col, lw, al = edge_style(sgn[s][t])
        ti = targets.index(t)
        rad = 0.10 * ((ti - (len(targets) - 1) / 2) / max(1, (len(targets) - 1) / 2))
        ax.add_patch(FancyArrowPatch(
            (x0 + hw, y0), (x1 - hw, y1),
            connectionstyle=f'arc3,rad={rad:.3f}',
            arrowstyle='-|>', mutation_scale=11,
            linewidth=lw, color=col, alpha=al, zorder=3,
            shrinkA=1.0, shrinkB=1.0, capstyle='round'))
    for g in sources:
        source_box(ax, *SPOS[g], g)
    for g in targets:
        target_box(ax, *TPOS[g], g)


def edge_node_legend_handles():
    edge = [
        Line2D([0], [0], color=C_UP,   lw=4.0, solid_capstyle='round'),
        Line2D([0], [0], color=C_DOWN, lw=4.0, solid_capstyle='round'),
        Line2D([0], [0], color=NEUTRAL, lw=1.3, solid_capstyle='round'),
        Line2D([0], [0], color=NEUTRAL, lw=5.5, solid_capstyle='round'),
    ]
    edge_lab = ['Predicted up-regulation', 'Predicted down-regulation',
                'Low confidence', 'High confidence']
    node = [
        mpatches.Patch(fc=SRC_FILL['neg'], ec=SRC_EDGE['neg'], lw=1.8, label='Regulator · negative regulator'),
        mpatches.Patch(fc=SRC_FILL['pos'], ec=SRC_EDGE['pos'], lw=1.8, label='Regulator · positive node'),
        mpatches.Patch(fc=GROUP_FILL['Cytokine'],  ec=GROUP_EDGE, label='Target · cytokine'),
        mpatches.Patch(fc=GROUP_FILL['Chemokine'], ec=GROUP_EDGE, label='Target · chemokine'),
        mpatches.Patch(fc=GROUP_FILL['Effector'],  ec=GROUP_EDGE, label='Target · effector enzyme'),
        mpatches.Patch(fc=GROUP_FILL['NF-κB'],     ec=GROUP_EDGE, label='Target · NF-κB subunit'),
    ]
    return edge, edge_lab, node


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — integrated bipartite regulatory network (pooled host pool)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11.0, 8.4))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0.02, 0.06, 0.96, 0.84])
draw_bipartite(ax, upfrac)

# title
fig.text(0.5, 0.975, 'Predicted NF-κB regulatory network under Tier A gene perturbation',
         ha='center', va='top', fontsize=13.5, fontweight='bold', color='#111')
fig.text(0.5, 0.945,
         'Two-Branch Spatial FM · zero-shot direction · targets = 6 most accurately '
         'predicted NF-κB genes (per-target acc on 33 panel perturbations)',
         ha='center', va='top', fontsize=8.4, color='#666', style='italic')

# legend (bottom)
edge_handles = [
    Line2D([0], [0], color=C_UP,   lw=4.0, solid_capstyle='round'),
    Line2D([0], [0], color=C_DOWN, lw=4.0, solid_capstyle='round'),
    Line2D([0], [0], color=NEUTRAL, lw=1.3, solid_capstyle='round'),
    Line2D([0], [0], color=NEUTRAL, lw=5.5, solid_capstyle='round'),
]
edge_labels = ['Predicted up-regulation', 'Predicted down-regulation',
               'Low confidence', 'High confidence']
leg1 = ax.legend(edge_handles, edge_labels, loc='upper left',
                 bbox_to_anchor=(0.0, 0.045), ncol=2, frameon=True,
                 framealpha=0.95, edgecolor='#D5D5D5', fontsize=8.2,
                 title='Edge   (width & opacity ∝ |2·up-frac − 1|)',
                 title_fontsize=8.4, handlelength=2.2, columnspacing=1.6,
                 borderpad=0.7, labelspacing=0.6)
leg1._legend_box.align = 'left'
ax.add_artist(leg1)

node_handles = [
    mpatches.Patch(fc=SRC_FILL['neg'], ec=SRC_EDGE['neg'], lw=1.8, label='Regulator · negative regulator'),
    mpatches.Patch(fc=SRC_FILL['pos'], ec=SRC_EDGE['pos'], lw=1.8, label='Regulator · positive node'),
    mpatches.Patch(fc=GROUP_FILL['Cytokine'],  ec=GROUP_EDGE, label='Target · cytokine'),
    mpatches.Patch(fc=GROUP_FILL['Chemokine'], ec=GROUP_EDGE, label='Target · chemokine'),
    mpatches.Patch(fc=GROUP_FILL['Effector'],  ec=GROUP_EDGE, label='Target · effector enzyme'),
    mpatches.Patch(fc=GROUP_FILL['NF-κB'],     ec=GROUP_EDGE, label='Target · NF-κB subunit'),
]
leg2 = ax.legend(handles=node_handles, loc='upper right',
                 bbox_to_anchor=(1.0, 0.05), ncol=2, frameon=True,
                 framealpha=0.95, edgecolor='#D5D5D5', fontsize=8.0,
                 title='Node', title_fontsize=8.4, handlelength=1.4,
                 columnspacing=1.2, borderpad=0.7, labelspacing=0.5)
leg2._legend_box.align = 'left'

plt.savefig(os.path.join(FIGDIR, 'network_main_bipartite.png'),
            dpi=350, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIGDIR, 'network_main_bipartite.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: network_main_bipartite.png / .pdf')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b — two-context bipartite network (with vs without T-cell neighbor)
# ══════════════════════════════════════════════════════════════════════════════
CTX_CSV = os.path.join(FIGDIR, 'context_predicted_values.csv')
if os.path.exists(CTX_CSV):
    cdf = pd.read_csv(CTX_CSV)
    upf_ctx = {}
    for ctx in ['with', 'without']:
        sub = cdf[cdf.context == ctx]
        upf_ctx[ctx] = {s: {t: float(sub[(sub.regulator == s) & (sub.target == t)]['upfrac'].iloc[0])
                            for t in TARGETS} for s in SOURCES}

    fig1b, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 7.6))
    fig1b.patch.set_facecolor('white')
    draw_bipartite(axL, upf_ctx['with'])
    draw_bipartite(axR, upf_ctx['without'])

    panel_titles = [
        (axL, 'With T-cell neighbor',    '#B23A2E'),
        (axR, 'Without T-cell neighbor', '#1F5C8B'),
    ]
    for ax, t, c in panel_titles:
        ax.text(0.5, 1.005, t, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=12.5, fontweight='bold', color=c)
    axL.text(-0.02, 1.04, 'A', transform=axL.transAxes, fontsize=16,
             fontweight='bold', color='#222', va='top')
    axR.text(-0.02, 1.04, 'B', transform=axR.transAxes, fontsize=16,
             fontweight='bold', color='#222', va='top')

    fig1b.text(0.5, 0.99,
               'Spatial-context dependence of predicted NF-κB regulatory output',
               ha='center', va='top', fontsize=14, fontweight='bold', color='#111')
    fig1b.text(0.5, 0.955,
               'Two-Branch Spatial FM · zero-shot directional prediction · '
               'host pool split by T-cell-neighbor status',
               ha='center', va='top', fontsize=8.8, color='#666', style='italic')

    e_h, e_l, n_h = edge_node_legend_handles()
    leg = fig1b.legend(e_h + n_h, e_l + [p.get_label() for p in n_h],
                       loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02),
                       fontsize=8.0, frameon=True, framealpha=0.95, edgecolor='#D5D5D5',
                       handlelength=1.7, columnspacing=1.3,
                       title='Edge width & opacity ∝ |2·up-frac − 1|   ·   red = up, blue = down',
                       title_fontsize=8.6)
    fig1b.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.11, wspace=0.04)
    plt.savefig(os.path.join(FIGDIR, 'network_context_2panel.png'),
                dpi=350, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIGDIR, 'network_context_2panel.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIGDIR, 'network_context_2panel.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: network_context_2panel.png / .pdf / .svg')
else:
    print(f'[skip] {CTX_CSV} not found — run run_context_predictions.py first')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — per-regulator fan small-multiples (2×2)
# ══════════════════════════════════════════════════════════════════════════════
def draw_fan(ax, src):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    m = SOURCE_META[src]
    hx, hy = 0.135, 0.50
    tx = 0.80
    tys = np.linspace(0.90, 0.10, len(TARGETS))

    # edges
    for i, t in enumerate(TARGETS):
        col, lw, al = edge_style(signed[src][t])
        ar = FancyArrowPatch(
            (hx + 0.085, hy), (tx - 0.095, tys[i]),
            connectionstyle='arc3,rad=0.04',
            arrowstyle='-|>', mutation_scale=10,
            linewidth=lw, color=col, alpha=al, zorder=3,
            shrinkA=1.0, shrinkB=1.0, capstyle='round')
        ax.add_patch(ar)
        # numeric up-frac label near target
        u = upfrac[src][t]
        lcol = C_UP if u > 0.5 else C_DOWN
        lx = tx - 0.095 - 0.085
        ly = (hy + tys[i]) / 2 + (tys[i] - hy) * 0.18
        ax.text(lx, ly, f'{u:.2f}', ha='center', va='center', fontsize=6.6,
                color=lcol, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.82))

    # hub node
    ax.add_patch(FancyBboxPatch(
        (hx - 0.092, hy - 0.075), 0.184, 0.150,
        boxstyle='round,pad=0.008,rounding_size=0.02',
        linewidth=2.2, edgecolor=SRC_EDGE[m['cls']],
        facecolor=SRC_FILL[m['cls']], zorder=5, mutation_aspect=0.85))
    ax.text(hx, hy + 0.032, m['name'], ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=SRC_EDGE[m['cls']], zorder=6)
    ax.text(hx, hy + 0.002, f"({m['alias']})", ha='center', va='center',
            fontsize=7.5, color='#444', style='italic', zorder=6)
    badge = 'negative regulator' if m['cls'] == 'neg' else 'positive node'
    ax.text(hx, hy - 0.042, badge, ha='center', va='center', fontsize=6.0,
            color='white', fontweight='bold', zorder=7,
            bbox=dict(boxstyle='round,pad=0.22', fc=SRC_EDGE[m['cls']], ec='none'))

    # target nodes
    for i, t in enumerate(TARGETS):
        tm = TARGET_META[t]
        ax.add_patch(FancyBboxPatch(
            (tx - 0.095, tys[i] - 0.052), 0.215, 0.104,
            boxstyle='round,pad=0.006,rounding_size=0.016',
            linewidth=1.2, edgecolor=GROUP_EDGE,
            facecolor=GROUP_FILL[tm['group']], zorder=5, mutation_aspect=0.55))
        ax.text(tx + 0.012, tys[i] + 0.013, tm['full'], ha='center', va='center',
                fontsize=8.8, fontweight='bold', color='#1A1A1A', zorder=6)
        ax.text(tx + 0.012, tys[i] - 0.020, tm['group'].lower(), ha='center', va='center',
                fontsize=5.6, color='#555', style='italic', zorder=6)

    ax.text(0.5, 0.985, f"{m['layer']}", ha='center', va='top',
            fontsize=7.6, color='#777', style='italic', transform=ax.transAxes)


fig2, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
fig2.patch.set_facecolor('white')
labels = ['A', 'B', 'C', 'D']
for k, (src, ax) in enumerate(zip(SOURCES, axes.ravel())):
    draw_fan(ax, src)
    ax.text(0.02, 0.99, labels[k], transform=ax.transAxes,
            fontsize=15, fontweight='bold', color='#222', va='top')

fig2.text(0.5, 0.985,
          'Predicted directional output of individual Tier A regulators on NF-κB targets',
          ha='center', va='top', fontsize=13, fontweight='bold', color='#111')
fig2.text(0.5, 0.958,
          'Edge labels = predicted up-regulation fraction (>0.50 ⇒ up, red; <0.50 ⇒ down, blue) · '
          'width & opacity ∝ confidence',
          ha='center', va='top', fontsize=8.4, color='#666', style='italic')

fan_handles = [
    Line2D([0], [0], color=C_UP,   lw=4, solid_capstyle='round', label='Predicted up-regulation'),
    Line2D([0], [0], color=C_DOWN, lw=4, solid_capstyle='round', label='Predicted down-regulation'),
    mpatches.Patch(fc=SRC_FILL['neg'], ec=SRC_EDGE['neg'], lw=1.8, label='neg-regulator'),
    mpatches.Patch(fc=SRC_FILL['pos'], ec=SRC_EDGE['pos'], lw=1.8, label='pos-node'),
    mpatches.Patch(fc=GROUP_FILL['Cytokine'],  ec=GROUP_EDGE, label='cytokine'),
    mpatches.Patch(fc=GROUP_FILL['Chemokine'], ec=GROUP_EDGE, label='chemokine'),
    mpatches.Patch(fc=GROUP_FILL['Effector'],  ec=GROUP_EDGE, label='effector'),
    mpatches.Patch(fc=GROUP_FILL['NF-κB'],     ec=GROUP_EDGE, label='NF-κB subunit'),
]
fig2.legend(handles=fan_handles, loc='lower center', ncol=9,
            bbox_to_anchor=(0.5, -0.005), fontsize=7.8, frameon=True,
            framealpha=0.95, edgecolor='#D5D5D5', handlelength=1.5, columnspacing=1.1)

fig2.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.05, hspace=0.10, wspace=0.04)
plt.savefig(os.path.join(FIGDIR, 'network_panels_fan.png'),
            dpi=350, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIGDIR, 'network_panels_fan.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: network_panels_fan.png / .pdf')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 3/4 — ALL Tier A perturbations × ALL NF-κB targets  (full panel)
#   rendered as a function of an up-fraction dict, so the same code serves the
#   pooled host pool and each spatial context (with / without immune neighbour).
# ══════════════════════════════════════════════════════════════════════════════
def compact_box(ax, x, y, label, w, h, ec, fc, fs=8.0, tc='#1A1A1A'):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.004,rounding_size=0.010',
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=5, mutation_aspect=0.42))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=6)


def all_network(upf, subtitle, fname):
    sgn = {s: {t: 2 * upf[s][t] - 1 for t in ALL_TARGETS} for s in ALL_SOURCES}
    figA = plt.figure(figsize=(10.5, 12.2)); figA.patch.set_facecolor('white')
    axA = figA.add_axes([0.02, 0.05, 0.96, 0.88])
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis('off')
    xsA, xtA = 0.17, 0.83
    ysA = np.linspace(0.94, 0.045, len(ALL_SOURCES))
    ytA = np.linspace(0.94, 0.045, len(ALL_TARGETS))
    SP = {g: (xsA, ysA[i]) for i, g in enumerate(ALL_SOURCES)}
    TP = {g: (xtA, ytA[i]) for i, g in enumerate(ALL_TARGETS)}
    wbox, hbox = 0.150, 0.052
    for s, t in sorted([(s, t) for s in ALL_SOURCES for t in ALL_TARGETS],
                       key=lambda st: abs(sgn[st[0]][st[1]])):
        x0, y0 = SP[s]; x1, y1 = TP[t]
        v = sgn[s][t]; mag = abs(v)
        col = C_UP if v > 0 else C_DOWN
        ti = ALL_TARGETS.index(t)
        rad = 0.06 * ((ti - (len(ALL_TARGETS) - 1) / 2) / ((len(ALL_TARGETS) - 1) / 2))
        axA.add_patch(FancyArrowPatch(
            (x0 + wbox/2, y0), (x1 - wbox/2, y1),
            connectionstyle=f'arc3,rad={rad:.3f}', arrowstyle='-|>',
            mutation_scale=8, linewidth=0.6 + 4.2 * mag, color=col,
            alpha=0.10 + 0.72 * mag, zorder=3, shrinkA=0.5, shrinkB=0.5, capstyle='round'))
    for g in ALL_SOURCES:
        m = SOURCE_META[g]
        compact_box(axA, *SP[g], g, wbox, hbox, SRC_EDGE[m['cls']], SRC_FILL[m['cls']],
                    fs=8.2, tc=SRC_EDGE[m['cls']])
    for g in ALL_TARGETS:
        m = TARGET_META[g]
        compact_box(axA, *TP[g], m['full'], wbox, hbox, GROUP_EDGE, GROUP_FILL[m['group']], fs=8.0)
    axA.text(xsA, 0.985, 'Tier A perturbations (n=11)', ha='center', va='center',
             fontsize=10.5, fontweight='bold', color='#222')
    axA.text(xtA, 0.985, 'NF-κB target genes (n=11)', ha='center', va='center',
             fontsize=10.5, fontweight='bold', color='#222')
    figA.text(0.5, 0.985, 'Predicted NF-κB regulatory network — full Tier A × target panel',
              ha='center', va='top', fontsize=13.5, fontweight='bold', color='#111')
    figA.text(0.5, 0.962, subtitle, ha='center', va='top', fontsize=8.6,
              color='#444', style='italic')
    e_h, e_l, n_h = edge_node_legend_handles()
    n_h = n_h + [mpatches.Patch(fc=SRC_FILL['unl'], ec=SRC_EDGE['unl'], lw=1.6,
                                label='Regulator · unlabeled'),
                 mpatches.Patch(fc=GROUP_FILL['Adhesion'], ec=GROUP_EDGE, label='Target · adhesion')]
    figA.legend(e_h + n_h, e_l + [p.get_label() for p in n_h],
                loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.005),
                fontsize=7.4, frameon=True, framealpha=0.95, edgecolor='#D5D5D5',
                handlelength=1.6, columnspacing=1.2)
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(FIGDIR, f'{fname}.{ext}'),
                    dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {fname}.png / .pdf')


def all_heatmap(upf, title, fname):
    M = np.array([[2 * upf[s][t] - 1 for t in ALL_TARGETS] for s in ALL_SOURCES])
    U = np.array([[upf[s][t] for t in ALL_TARGETS] for s in ALL_SOURCES])
    figH, axH = plt.subplots(figsize=(11.0, 9.6))
    im = axH.imshow(M, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axH.set_xticks(range(len(ALL_TARGETS)))
    axH.set_xticklabels([TARGET_META[t]['full'] for t in ALL_TARGETS], rotation=45, ha='right', fontsize=9)
    axH.set_yticks(range(len(ALL_SOURCES)))
    axH.set_yticklabels([SOURCE_META[s]['name'] for s in ALL_SOURCES], fontsize=9)
    for i in range(len(ALL_SOURCES)):
        for j in range(len(ALL_TARGETS)):
            axH.text(j, i, f'{U[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                     color='white' if abs(M[i, j]) > 0.55 else '#333')
    for tl, s in zip(axH.get_yticklabels(), ALL_SOURCES):
        tl.set_color(SRC_EDGE[SOURCE_META[s]['cls']]); tl.set_fontweight('bold')
    for tl, t in zip(axH.get_xticklabels(), ALL_TARGETS):
        tl.set_color(GROUP_EDGE)
    n_neg = sum(SOURCE_META[s]['cls'] == 'neg' for s in ALL_SOURCES)
    n_pos = sum(SOURCE_META[s]['cls'] == 'pos' for s in ALL_SOURCES)
    for yb in [n_neg - 0.5, n_neg + n_pos - 0.5]:
        axH.axhline(yb, color='#444', lw=1.2)
    axH.text(-1.4, (n_neg - 1) / 2, 'negative\nregulators', rotation=90, va='center', ha='center',
             fontsize=7.5, color=SRC_EDGE['neg'], fontweight='bold')
    axH.text(-1.4, n_neg + (n_pos - 1) / 2, 'positive\nnodes', rotation=90, va='center', ha='center',
             fontsize=7.5, color=SRC_EDGE['pos'], fontweight='bold')
    cb = figH.colorbar(im, ax=axH, fraction=0.040, pad=0.02)
    cb.set_label('signed score  (2·up-frac − 1):   blue = down  ·  red = up', fontsize=8.5)
    axH.set_title(title, fontsize=12, fontweight='bold', pad=12)
    figH.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(FIGDIR, f'{fname}.{ext}'),
                    dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {fname}.png / .pdf')


# pooled host pool
all_network(upfrac,
            'Two-Branch Spatial FM · zero-shot direction · red = up, blue = down · '
            'width & opacity ∝ confidence (weak edges fade)',
            'network_all_bipartite')
all_heatmap(upfrac,
            'Predicted direction — all Tier A perturbations × all NF-κB targets (pooled)\n'
            'cell value = predicted up-fraction (>0.50 ⇒ up / red)',
            'network_all_heatmap')

# ── two spatial contexts (same model; host pool split by immune neighbour) ────
CTX_CSV = os.path.join(FIGDIR, 'context_predicted_values.csv')
if os.path.exists(CTX_CSV):
    cdf = pd.read_csv(CTX_CSV)
    for ctx, tag in [('with', 'with immune (T-cell) neighbour'),
                     ('without', 'without immune (T-cell) neighbour')]:
        sub = cdf[cdf.context == ctx]
        upf_c = {s: {t: float(sub[(sub.regulator == s) & (sub.target == t)]['upfrac'].iloc[0])
                     for t in ALL_TARGETS} for s in ALL_SOURCES}
        all_network(upf_c,
                    f'Host cells {tag} · same model (trained jointly) · red = up, blue = down · '
                    'width & opacity ∝ confidence',
                    f'network_all_bipartite_{ctx}')
        all_heatmap(upf_c,
                    f'Predicted direction — all Tier A × all NF-κB targets · {tag}\n'
                    'cell value = predicted up-fraction (>0.50 ⇒ up / red)',
                    f'network_all_heatmap_{ctx}')
else:
    print(f'[skip context figures] {CTX_CSV} not found — run run_context_predictions.py first')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — STORY figure: 2 pos + 2 neg regulators × 6 targets, without vs with
#   selection (data-driven): regulators/targets that are most class-coherent in
#   the WITHOUT-neighbour condition AND change most between contexts.
# ══════════════════════════════════════════════════════════════════════════════
STORY_SRC = ['IRAK3', 'TNF', 'IKBKG', 'UBE2N']       # 2 neg (red) + 2 pos (blue)
STORY_TGT = ['ICAM1', 'CXCL2', 'CXCL8', 'NFKB2', 'NFKBIA', 'SOD2']

if os.path.exists(CTX_CSV):
    cdf = pd.read_csv(CTX_CSV)
    story = {}
    for ctx in ['without', 'with']:
        sub = cdf[cdf.context == ctx]
        story[ctx] = {s: {t: float(sub[(sub.regulator == s) & (sub.target == t)]['upfrac'].iloc[0])
                          for t in STORY_TGT} for s in STORY_SRC}

    figS, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 7.8))
    figS.patch.set_facecolor('white')
    draw_bipartite(axL, story['without'], sources=STORY_SRC, targets=STORY_TGT,
                   tgt_subtitle='NF-κB response genes')
    draw_bipartite(axR, story['with'], sources=STORY_SRC, targets=STORY_TGT,
                   tgt_subtitle='NF-κB response genes')
    for ax, t, c in [(axL, 'WITHOUT immune (T-cell) neighbour', '#1F5C8B'),
                     (axR, 'WITH immune (T-cell) neighbour', '#B23A2E')]:
        ax.text(0.5, 1.005, t, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=12.5, fontweight='bold', color=c)
    axL.text(-0.02, 1.05, 'A', transform=axL.transAxes, fontsize=16, fontweight='bold',
             color='#222', va='top')
    axR.text(-0.02, 1.05, 'B', transform=axR.transAxes, fontsize=16, fontweight='bold',
             color='#222', va='top')

    figS.text(0.5, 0.99, 'Spatial-context switch in predicted NF-κB regulatory output',
              ha='center', va='top', fontsize=14, fontweight='bold', color='#111')
    figS.text(0.5, 0.955,
              'Baseline (A): regulators follow expected biology (neg-reg → up / red, pos-node → down / blue).  '
              'With a T-cell neighbour (B): IRAK3 & IKBKG drive the feedback brakes IκBα & SOD2 up.',
              ha='center', va='top', fontsize=8.4, color='#555', style='italic')

    e_h, e_l, n_h = edge_node_legend_handles()
    figS.legend(e_h + n_h, e_l + [p.get_label() for p in n_h],
                loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02),
                fontsize=8.0, frameon=True, framealpha=0.95, edgecolor='#D5D5D5',
                handlelength=1.7, columnspacing=1.3,
                title='Edge width & opacity ∝ |2·up-frac − 1|   ·   red = up, blue = down',
                title_fontsize=8.6)
    figS.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.12, wspace=0.04)
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(FIGDIR, f'network_story_context.{ext}'),
                    dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: network_story_context.png / .pdf')

    # story value table
    srows = []
    for s in STORY_SRC:
        for t in STORY_TGT:
            srows.append(dict(regulator=s, cls=SOURCE_META[s]['cls'], target=t,
                              upfrac_without=story['without'][s][t],
                              upfrac_with=story['with'][s][t],
                              delta=round(story['with'][s][t] - story['without'][s][t], 3)))
    pd.DataFrame(srows).to_csv(os.path.join(FIGDIR, 'network_story_values.csv'), index=False)
    print('Saved: network_story_values.csv')


# ── supplementary value table ─────────────────────────────────────────────────
rows = []
for s in SOURCES:
    for t in TARGETS:
        u = upfrac[s][t]
        rows.append(dict(regulator=s, reg_class=SOURCE_META[s]['cls'],
                         target=t, target_group=TARGET_META[t]['group'],
                         upfrac=round(u, 4), signed_score=round(2*u - 1, 4),
                         direction='UP' if u > 0.5 else 'DOWN'))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(FIGDIR, 'network_predicted_values.csv'), index=False)
print('Saved: network_predicted_values.csv')

print('\nPredicted up-fraction matrix (rows=regulator, cols=target):')
mat = pd.DataFrame({s: {t: upfrac[s][t] for t in TARGETS} for s in SOURCES}).T[TARGETS]
print(mat.round(2).to_string())
print(f'\nAll outputs -> {FIGDIR}/')
