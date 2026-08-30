# Panel: figure2g
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/TLS_figure_317M/redo_figure_D.py
# Original filename: redo_figure_D.py
# Last modified: 2026-07-19 10:10:37  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: pick/GSM5924031_ffpe_c_3.h5ad (+ the other 4 pick/*.h5ad for the mean panel); spaGFM_predict is overwritten with new_model_results/*.h5ad obs['317M_binary_predict']
# Output: TLS_figure_317M/figure_D_mean.{png,pdf,svg}  --  LEFT subplot ("A. Spatial Autocorrelation (Moran's I)")
# Match evidence: bar lengths in the slide read 0.85 / 0.62 / 0.24 / 0.21 / 0.19 / 0 / 0, matching figure_D_mean's printed values 0.852 (Ground Truth), 0.617 (spaGFM), 0.239 (finetune-scGPT), 0.216 (scGPT), 0.187 (NOVAE), n/a (NicheCompass), -0.000 (scGPT-spatial). Row order, the per-method fill colours (grey / #E8888C-pink / pale pink / peach / light blue) and the "Moran's I" axis label all match. The single-sample figure_D.{png,pdf,svg} from the same script does NOT match (0.872 / 0.429 / 0.322 / 0.285 / 0.260).
# Slide edits made outside this script: the "Scanpy" row (PCA_predict, 0.040) was deleted, and three labels were renamed - "Ground Truth" -> "Annotation", "finetune-scGPT" -> "Finetuned scGPT", "NOVAE" -> "Novae". The value annotations and the "All p < 1e-300" note were also removed. These are cosmetic post-edits; no code here produces the relabelled version.
# Note: this ONE script emits both figure2g and figure2h (left and right subplots of figure_D_mean). figure2h.py is the same file, kept under its own panel name so each panel is findable.
# Other candidates considered: TLS_figure/redo_figure_D.py (2026-06-11) is the pre-317M ancestor - same layout, older numbers; tls_validation_figures{,_multi}.py compute the same two metrics but draw them inside a large multi-panel sheet, not as this standalone 2-subplot figure.
# ---- copied verbatim below; NOT modified ----
"""
Redo figure_D: Spatial Coherence Analysis (Moran's I + neighborhood purity),
single sample GSM5924031, now including ALL benchmarked methods.

Method columns -> display labels:
    spaGFM_predict            -> spaGFM
    X_finetune_scGPT_predict  -> finetune-scGPT
    X_NicheCompass_predict    -> NicheCompass
    X_scGPT_predict           -> scGPT
    novae_latent_predict      -> NOVAE
    spatial_predict           -> scGPT-spatial
    PCA_predict               -> Scanpy
    label (==0)               -> Ground Truth  (reference, neutral gray)

Outputs figure_D.pdf (and figure_D.png for preview).
"""
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

import warnings
warnings.filterwarnings('ignore')

import glob
import scanpy as sc
import numpy as np
from scipy.spatial import cKDTree
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

colors = {
    "Scanpy": "#D5EAD9",
    "NicheCompass": "#7DC69B",
    "NOVAE": "#9BD7F3",
    "finetune-scGPT": "#FBDDDD",
    "scGPT-spatial": "#D8EEFB",
    "spaGFM": "#F2A1A7",
    "scGPT": "#FCE6CF",
}
GT_COLOR = "#BFBFBF"  # neutral gray for Ground Truth reference

# label -> obs column ('GT' is special: uses adata.obs['label'])
# all methods use their binary (TLS vs Non-TLS) prediction columns
method_cols = {
    "spaGFM":          "spaGFM_predict",                    # = 317M_binary_predict
    "finetune-scGPT":  "X_finetune_scGPT_binary_predict",
    "NicheCompass":    "X_NicheCompass_binary_predict",
    "scGPT":           "X_scGPT_binary_predict",
    "NOVAE":           "novae_latent_binary_predict",
    "scGPT-spatial":   "spatial_binary_predict",
    "Scanpy":          "PCA_binary_predict",
}

# ── spatial neighbour graph (k=6 nearest, exclude self) ──────────────────────
coords = adata.obsm['spatial'].astype(float)
_, idx = cKDTree(coords).query(coords, k=7)
nb = idx[:, 1:]


def morans_i(values, neighbor_idx):
    n = len(values); z = values - values.mean()
    W = numerator = 0.0
    for i in range(n):
        for j in neighbor_idx[i]:
            numerator += z[i] * z[j]; W += 1.0
    denom = np.sum(z ** 2)
    if denom == 0 or W == 0:
        return np.nan, np.nan
    I = (n / W) * (numerator / denom)
    E_I = -1 / (n - 1)
    S1, S2 = 2 * W, 4 * W
    k4 = n * np.sum(z ** 4) / (np.sum(z ** 2) ** 2)
    var_I = (n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*W**2) -
             k4 * ((n**2 - n)*S1 - 2*n*S2 + 6*W**2)) / \
            ((n-1)*(n-2)*(n-3)*W**2) - E_I**2
    z_score = (I - E_I) / np.sqrt(max(var_I, 1e-10))
    p = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return I, p


def neighborhood_purity(labels, neighbor_idx):
    return np.array([(labels[neighbor_idx[i]] == labels[i]).sum() / len(neighbor_idx[i])
                     for i in range(len(labels))])


# ── compute Moran's I (binary TLS indicator) + TLS-spot purity per method ─────
morans, purity = {}, {}

# Ground Truth
gt_int = adata.obs['label'].values.astype(int)
morans["Ground Truth"] = morans_i((gt_int == 0).astype(float), nb)[0]
pur = neighborhood_purity(gt_int, nb)
purity["Ground Truth"] = pur[gt_int != 2].mean()

for lbl, col in method_cols.items():
    arr = adata.obs[col].values.astype(int)
    morans[lbl] = morans_i((arr != 2).astype(float), nb)[0]
    pur = neighborhood_purity(arr, nb)
    tls_mask = arr != 2
    purity[lbl] = pur[tls_mask].mean() if tls_mask.sum() > 0 else np.nan
    print(f"  {lbl:16s} MoranI={morans[lbl]:.3f}  purity={purity[lbl]:.3f}")

def fmt(v):
    return 'n/a' if (v is None or np.isnan(v)) else f'{v:.3f}'

def safe(v):
    return 0.0 if (v is None or np.isnan(v)) else v

# ── order bars by Moran's I ascending (worst at top, best at bottom) ─────────
order = sorted(morans, key=lambda m: safe(morans[m]))
bar_colors = [GT_COLOR if m == "Ground Truth" else colors[m] for m in order]

# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
vals = [safe(morans[m]) for m in order]
bars = ax.barh(order, vals, color=bar_colors, edgecolor='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel("Moran's I", fontsize=11)
ax.set_title("A. Spatial Autocorrelation (Moran's I)\n(higher = more spatially clustered)",
             fontsize=11, fontweight='bold')
for bar, m in zip(bars, order):
    ax.text(safe(morans[m]) + 0.01, bar.get_y() + bar.get_height()/2,
            fmt(morans[m]), va='center', ha='left', fontsize=10)
ax.set_xlim(0, 1.05)
ax.text(0.98, 0.02, 'All p < 10⁻³⁰⁰', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9, style='italic', color='gray')

ax2 = axes[1]
vals2 = [safe(purity[m]) for m in order]
bars2 = ax2.barh(order, vals2, color=bar_colors, edgecolor='black', linewidth=1.2)
ax2.axvline(1/6, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='Random (1/6)')
ax2.set_xlabel('Neighborhood Purity (TLS spots)', fontsize=11)
ax2.set_title('B. Neighborhood Purity for TLS-Predicted Spots\n'
              '(fraction of neighbors also predicted TLS)',
              fontsize=11, fontweight='bold')
for bar, m in zip(bars2, order):
    ax2.text(safe(purity[m]) + 0.005, bar.get_y() + bar.get_height()/2,
             fmt(purity[m]), va='center', ha='left', fontsize=10)
ax2.set_xlim(0, 1.05); ax2.legend(fontsize=9)

plt.suptitle('Spatial Coherence Analysis — spaGFM vs Baselines',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figure_D.pdf", bbox_inches='tight')
plt.savefig("figure_D.svg", bbox_inches='tight')
plt.savefig("figure_D.png", dpi=150, bbox_inches='tight')
plt.close()
print("figure_D.pdf, figure_D.svg and figure_D.png saved.")


# ═════════════════════════════════════════════════════════════════════════════
# figure_D_mean : Moran's I + purity averaged (mean ± std) across ALL samples
#                 every method uses its binary (TLS vs Non-TLS) prediction column
# ═════════════════════════════════════════════════════════════════════════════
PICK_DIR = "/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/KC/test/pick"
FILES = sorted(glob.glob(os.path.join(PICK_DIR, "*.h5ad")))
methods_all = ["Ground Truth", "spaGFM", "finetune-scGPT", "NicheCompass",
               "scGPT", "NOVAE", "scGPT-spatial", "Scanpy"]
# per-method binary column (baselines); spaGFM comes from new_model_results 317M_binary
binary_col = {
    "finetune-scGPT":  "X_finetune_scGPT_binary_predict",
    "NicheCompass":    "X_NicheCompass_binary_predict",
    "scGPT":           "X_scGPT_binary_predict",
    "NOVAE":           "novae_latent_binary_predict",
    "scGPT-spatial":   "spatial_binary_predict",
    "Scanpy":          "PCA_binary_predict",
}
morans_ps = {m: [] for m in methods_all}
purity_ps = {m: [] for m in methods_all}

for f in FILES:
    a  = sc.read_h5ad(f)
    nw = sc.read_h5ad(f.replace('/pick/', '/pick/new_model_results/'))
    assert list(nw.obs_names) == list(a.obs_names), 'cell order mismatch (new vs old)'
    coords = a.obsm['spatial'].astype(float)
    _, idx = cKDTree(coords).query(coords, k=7)
    nb_s = idx[:, 1:]

    # Ground Truth (label 0 = TLS, 2 = Non-TLS)
    gi = a.obs['label'].values.astype(int)
    morans_ps["Ground Truth"].append(morans_i((gi == 0).astype(float), nb_s)[0])
    pur = neighborhood_purity(gi, nb_s)
    purity_ps["Ground Truth"].append(pur[gi != 2].mean())

    # spaGFM (317M binary)
    sarr = nw.obs['317M_binary_predict'].values.astype(int)
    morans_ps["spaGFM"].append(morans_i((sarr != 2).astype(float), nb_s)[0])
    pur = neighborhood_purity(sarr, nb_s); sm = sarr != 2
    purity_ps["spaGFM"].append(pur[sm].mean() if sm.sum() > 0 else np.nan)

    # baselines (their own binary columns)
    for m, col in binary_col.items():
        arr = a.obs[col].values.astype(int)
        morans_ps[m].append(morans_i((arr != 2).astype(float), nb_s)[0])
        pur = neighborhood_purity(arr, nb_s); mm = arr != 2
        purity_ps[m].append(pur[mm].mean() if mm.sum() > 0 else np.nan)
    print(f"  [mean] {os.path.basename(f)} done")

morans_mean = {m: np.nanmean(morans_ps[m]) for m in methods_all}
morans_sd   = {m: np.nanstd(morans_ps[m])  for m in methods_all}
purity_mean = {m: np.nanmean(purity_ps[m]) for m in methods_all}
purity_sd   = {m: np.nanstd(purity_ps[m])  for m in methods_all}

order_m = sorted(methods_all, key=lambda m: safe(morans_mean[m]))
bar_colors_m = [GT_COLOR if m == "Ground Truth" else colors[m] for m in order_m]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
vals  = [safe(morans_mean[m]) for m in order_m]
bars  = ax.barh(order_m, vals, color=bar_colors_m,
                edgecolor='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel("Moran's I", fontsize=11)
ax.set_title("A. Spatial Autocorrelation (Moran's I)\n(mean across samples)",
             fontsize=11, fontweight='bold')
for bar, m in zip(bars, order_m):
    ax.text(safe(morans_mean[m]) + 0.01,
            bar.get_y() + bar.get_height()/2,
            fmt(morans_mean[m]), va='center', ha='left', fontsize=10)
ax.set_xlim(0, 1.15)

ax2 = axes[1]
vals2 = [safe(purity_mean[m]) for m in order_m]
bars2 = ax2.barh(order_m, vals2, color=bar_colors_m,
                 edgecolor='black', linewidth=1.2)
ax2.axvline(1/6, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='Random (1/6)')
ax2.set_xlabel('Neighborhood Purity (TLS spots)', fontsize=11)
ax2.set_title('B. Neighborhood Purity for TLS-Predicted Spots\n(mean across samples)',
              fontsize=11, fontweight='bold')
for bar, m in zip(bars2, order_m):
    ax2.text(safe(purity_mean[m]) + 0.005,
             bar.get_y() + bar.get_height()/2,
             fmt(purity_mean[m]), va='center', ha='left', fontsize=10)
ax2.set_xlim(0, 1.15); ax2.legend(fontsize=9)

plt.suptitle(f'Spatial Coherence Analysis — mean across {len(FILES)} samples',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figure_D_mean.pdf", bbox_inches='tight')
plt.savefig("figure_D_mean.svg", bbox_inches='tight')
plt.savefig("figure_D_mean.png", dpi=150, bbox_inches='tight')
plt.close()
print("figure_D_mean.pdf, figure_D_mean.svg and figure_D_mean.png saved.")
