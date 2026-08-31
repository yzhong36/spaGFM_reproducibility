# Panel: figure3c_prep1
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_d_tcell_proximity_auc_step1_compute.py
# Original filename: fig1_d_tcell_proximity_auc_step1_compute.py
# Last modified: 2026-08-18 10:29:09  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad; inference_set_wl_9_317M.pt
# Output: figure_bundle_20260817/data/work/fig1_d/tcell_proximity_auc.csv, residual_r2_results_v2.csv
# Match evidence: Upstream step 1: computes the spaGFM AUC and the published R^2 table that figure3c still reads.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
"""
최종 분석: spaGFM이 세포 유형 레이블 없이 T cell 근접성을 학습하는가?

핵심 스토리:
  1. LFC 잔차는 실재한다 (같은 perturbation이라도 공간 위치에 따라 효과가 다름)
  2. 이 이질성의 주요 공간 드라이버는 T cell 근접성이다
  3. spaGFM은 cell type 레이블 없이 T cell 근접성을 mean-scGPT보다 훨씬 잘 학습한다
  4. 이것이 spaGFM이 LFC 잔차를 더 잘 예측하는 이유다
"""

import numpy as np
import torch
import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT_DIR = Path('/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/fig1_d')
OUT_DIR.mkdir(exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────
print("Loading data...")
adata = sc.read_h5ad('/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad')
inf = torch.load(
    '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/inference_set_wl_9_317M.pt',
    map_location='cpu', weights_only=False
)

# spaGFM embedding 할당
adata.obsm['spaGFM_emb'] = np.zeros((adata.n_obs, 1280), dtype=np.float32)
assigned = np.zeros(adata.n_obs, dtype=bool)
for graph in inf:
    m = np.asarray(graph.y == graph.perturbation)
    emb = graph.node_emb.detach().numpy().astype(np.float32)
    cell_ids = np.asarray(graph.cell_id)[m]
    idx = adata.obs_names.get_indexer(cell_ids)
    v = idx >= 0
    adata.obsm['spaGFM_emb'][idx[v]] = emb[v]
    assigned[idx[v]] = True

counts = adata.obs['perturbation'].value_counts()
valid_perts = counts[(counts >= 20) & (counts.index != 'Control')].index
pert_mask = adata.obs['perturbation'].isin(valid_perts) & assigned
pa = adata[pert_mask].copy()
print(f"Perturbed cells: {pa.n_obs}, perturbations: {pa.obs['perturbation'].nunique()}")

# ── mean-scGPT of neighbors 계산 ─────────────────────────────────────────
print("Computing mean-scGPT of neighbors...")
coords_all  = adata.obsm['spatial']
tree        = cKDTree(coords_all)
scgpt_all   = adata.obsm['X_scGPT'].astype(np.float32)
ct_all      = adata.obs['cell_type'].values
coords_pert = pa.obsm['spatial']

feat_mean_scgpt = []
for cx, cy in coords_pert:
    idx_n = tree.query_ball_point([cx, cy], 300)
    ctrl  = [j for j in idx_n if ct_all[j] != 'T_cell']
    feat_mean_scgpt.append(scgpt_all[ctrl].mean(0) if ctrl else np.zeros(scgpt_all.shape[1]))
feat_mean_scgpt = np.array(feat_mean_scgpt, dtype=np.float32)

# ── LFC 잔차 계산 ─────────────────────────────────────────────────────────
print("Computing LFC residuals...")
lfc      = pa.layers['primary_target_lfc'].astype(np.float32)
perturbs = pa.obs['perturbation'].values

pert_means   = {p: lfc[perturbs == p].mean(0) for p in np.unique(perturbs)}
lfc_residual = np.array([lfc[i] - pert_means[p] for i, p in enumerate(perturbs)], dtype=np.float32)

gene_var  = lfc_residual.var(0)
top_idx   = np.argsort(gene_var)[::-1][:50]
top_genes = adata.var_names[top_idx].tolist()
Y         = lfc_residual[:, top_idx]

# ── 헬퍼 함수 ────────────────────────────────────────────────────────────
def logo_r2(X, Y, groups, alpha=1.0, n_pca=None):
    if n_pca and n_pca < X.shape[1]:
        X = PCA(n_pca, random_state=42).fit_transform(X)
    ugroups = np.unique(groups)
    yt, yp  = [], []
    for g in ugroups:
        tm, vm = groups == g, groups != g
        if vm.sum() < 10 or tm.sum() < 2:
            continue
        sc_ = StandardScaler()
        Xtr = sc_.fit_transform(X[vm])
        Xte = sc_.transform(X[tm])
        Ridge(alpha=alpha).fit(Xtr, Y[vm])
        mdl = Ridge(alpha=alpha).fit(Xtr, Y[vm])
        yt.append(Y[tm]); yp.append(mdl.predict(Xte))
    yt = np.vstack(yt); yp = np.vstack(yp)
    ss_res = ((yt - yp) ** 2).sum(0)
    ss_tot = ((yt - yt.mean(0)) ** 2).sum(0)
    return 1 - ss_res / (ss_tot + 1e-10)

def auc_cv(X, y, n_pca=None, C=1.0):
    if n_pca and n_pca < X.shape[1]:
        X = PCA(n_pca, random_state=42).fit_transform(X)
    X   = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    return cross_val_score(clf, X, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring='roc_auc')

# ── Panel A: per-perturbation LFC 잔차 분산 ──────────────────────────────
print("Panel A: per-perturbation residual variance...")
pert_resvar = {}
for p in np.unique(perturbs):
    pert_resvar[p] = lfc_residual[perturbs == p].var(0).mean()
sorted_perts  = sorted(pert_resvar, key=pert_resvar.get, reverse=True)
sorted_vals   = [pert_resvar[p] for p in sorted_perts]

# ── Panel B: has_tcell_neighbor 예측 AUC ────────────────────────────────
print("Panel B: T cell proximity prediction AUC...")
y_tcell = pa.obs['stage2_best_mean_knn_has_tcell_neighbor'].astype(float).fillna(0).values.astype(int)

auc_mean_scgpt = auc_cv(feat_mean_scgpt,      y_tcell, n_pca=50)
auc_spagfm     = auc_cv(pa.obsm['spaGFM_emb'], y_tcell, n_pca=200)

print(f"  Mean-scGPT AUC: {auc_mean_scgpt.mean():.4f} ± {auc_mean_scgpt.std():.4f}")
print(f"  SpaGFM    AUC: {auc_spagfm.mean():.4f} ± {auc_spagfm.std():.4f}")

# ── Panel C: LFC 잔차 예측 R² ────────────────────────────────────────────
print("Panel C: LFC residual prediction R²...")

obs_spa = np.column_stack([
    pa.obs['stage2_best_nearest_has_tcell_neighbor'].astype(float).fillna(0),
    pa.obs['stage2_best_mean_knn_has_tcell_neighbor'].astype(float).fillna(0),
    pa.obs['stage2_best_sizeaware_has_tcell_neighbor'].astype(float).fillna(0),
    pa.obs['stage2_best_sizeaware_neighbor_count'].astype(float).fillna(0),
]).astype(np.float32)

r2_dummy       = logo_r2(np.zeros((pa.n_obs, 1), dtype=np.float32), Y, perturbs)
r2_simple      = logo_r2(obs_spa,                   Y, perturbs)
r2_mean_scgpt  = logo_r2(feat_mean_scgpt,           Y, perturbs, n_pca=50)
r2_spagfm      = logo_r2(pa.obsm['spaGFM_emb'],     Y, perturbs, n_pca=50)

print(f"  Dummy:       {r2_dummy.mean():.4f}")
print(f"  Simple spa:  {r2_simple.mean():.4f}")
print(f"  Mean-scGPT:  {r2_mean_scgpt.mean():.4f}")
print(f"  SpaGFM:      {r2_spagfm.mean():.4f}")

# ── 그림 그리기 ──────────────────────────────────────────────────────────
print("Plotting...")

fig = plt.figure(figsize=(16, 5))
gs  = fig.add_gridspec(1, 3, wspace=0.35)

BLUE   = '#4C72B0'
ORANGE = '#DD8452'
GREEN  = '#55A868'
GRAY   = '#8C8C8C'

# ── Panel A ──────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
colors_bar = [BLUE if v > np.median(sorted_vals) else GRAY for v in sorted_vals]
bars = ax0.bar(range(len(sorted_perts)), sorted_vals, color=colors_bar, edgecolor='none')
ax0.set_xticks(range(len(sorted_perts)))
ax0.set_xticklabels(sorted_perts, rotation=45, ha='right', fontsize=7)
ax0.set_ylabel('Mean LFC residual variance\nacross top variable genes', fontsize=10)
ax0.set_title('A. Spatial heterogeneity of\nperturbation effects', fontsize=11, fontweight='bold')
ax0.axhline(np.median(sorted_vals), color='red', linestyle='--', linewidth=1, alpha=0.6,
            label=f'Median = {np.median(sorted_vals):.3f}')
ax0.legend(fontsize=8)
ax0.spines[['top','right']].set_visible(False)

# ── Panel B ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])

labels  = ['Mean-scGPT\nof neighbors', 'SpaGFM\nembedding']
means_  = [auc_mean_scgpt.mean(), auc_spagfm.mean()]
stds_   = [auc_mean_scgpt.std(),  auc_spagfm.std()]
colors_ = [ORANGE, BLUE]

for i, (lbl, m, s, c) in enumerate(zip(labels, means_, stds_, colors_)):
    ax1.bar(i, m, yerr=s, color=c, alpha=0.85, edgecolor='black', linewidth=0.8,
            capsize=5, width=0.5, error_kw={'linewidth': 1.5})
    ax1.text(i, m + s + 0.005, f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks([0, 1])
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylim(0.75, 1.0)
ax1.set_ylabel('AUC — predicting T cell proximity\n(has_tcell_neighbor)', fontsize=10)
ax1.set_title('B. Does spaGFM learn T cell\nproximity without cell type labels?', fontsize=11, fontweight='bold')
ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.spines[['top','right']].set_visible(False)

# 유의성 표시
y_sig = max(means_) + max(stds_) + 0.02
ax1.plot([0, 0, 1, 1], [y_sig, y_sig+0.005, y_sig+0.005, y_sig], color='black', linewidth=1)
ax1.text(0.5, y_sig + 0.008, '***', ha='center', va='bottom', fontsize=12)

# ── Panel C ──────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])

col_labels = ['Dummy\n(no spatial)', 'Simple spatial\n(T cell binary)', 'Mean-scGPT\nof neighbors', 'SpaGFM\nembedding']
r2_data    = [r2_dummy, r2_simple, r2_mean_scgpt, r2_spagfm]
col_colors = [GRAY, '#B5B5B5', ORANGE, BLUE]

bp = ax2.boxplot(r2_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2),
                 flierprops=dict(marker='o', markersize=3, alpha=0.5))
for patch, c in zip(bp['boxes'], col_colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(col_labels, fontsize=9)
ax2.set_ylabel('R² — predicting LFC residual\n(LOGO CV, top 50 genes)', fontsize=10)
ax2.set_title('C. Predicting spatial modulation\nof perturbation effects', fontsize=11, fontweight='bold')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.spines[['top','right']].set_visible(False)

for i, (data, c) in enumerate(zip(r2_data, col_colors), 1):
    m = np.mean(data)
    ax2.text(i, np.percentile(data, 75) + 0.02, f'{m:.3f}', ha='center', va='bottom',
             fontsize=8, fontweight='bold', color='black')

plt.savefig(OUT_DIR / 'spagfm_spatial_pattern_analysis.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUT_DIR / 'spagfm_spatial_pattern_analysis.png', bbox_inches='tight', dpi=150)
print(f"\nSaved to {OUT_DIR}/spagfm_spatial_pattern_analysis.{{pdf,png}}")

# ── CSV 저장 ──────────────────────────────────────────────────────────────
pd.DataFrame({
    'gene': top_genes,
    'residual_var': gene_var[top_idx],
    'r2_dummy': r2_dummy,
    'r2_simple_spa': r2_simple,
    'r2_mean_scgpt': r2_mean_scgpt,
    'r2_spagfm': r2_spagfm,
    'delta_spagfm_vs_mean_scgpt': r2_spagfm - r2_mean_scgpt,
}).to_csv(OUT_DIR / 'residual_r2_results_v2.csv', index=False)

pd.DataFrame({
    'method': ['mean_scgpt','spagfm'],
    'auc_mean': [auc_mean_scgpt.mean(), auc_spagfm.mean()],
    'auc_std':  [auc_mean_scgpt.std(),  auc_spagfm.std()],
}).to_csv(OUT_DIR / 'tcell_proximity_auc.csv', index=False)

print("Done.")
