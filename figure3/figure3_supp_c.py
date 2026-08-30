# Panel: figure3_supp_c
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig2_e_attention_reach_field.py
# Original filename: fig2_e_attention_reach_field.py
# Last modified: 2026-08-18 10:29:10  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad; data/work/attention/per_cell_table.csv; data/work/attention/vectorfield/cache/{pert}_attrib.npz
# Output: data/work/attention/vectorfield/metric_reach.png (+ per_metric_stats.{csv,json}, summary_forest.*)
# Match evidence: per_metric_stats.csv row 'reach' gives cliffs_delta = -0.08367 -> the -0.084 in the title, p_bh = 1.369e-05 -> p_BH=1.4e-05, and perm_p = 0.00049975 -> perm p=0.0005. All three annotated numbers reproduce exactly. n_with_T=2534, n_without_T=1785.
# Other candidates considered: tcell_attention/vectorfield/build_field.py -- the pre-bundle original this was copied and path-patched from; same numbers, non-portable paths.
# ---- copied verbatim below; NOT modified ----
"""
Attention vector-field analysis around masked perturbation cells, by local T-cell-neighbor status.

Builds on the completed T-cell attention run; REUSES cached per-neighbor attribution
(cache/{pert}_attrib.npz: indptr, neigh_adata_idx, attrib_mean, visits) — no re-extraction.
The attention kernel, M<->patterns alignment, self-node exclusion, and traversal-weighting are
inherited unchanged from upstream (they define the cached _attrib.npz support).

Same cell set / grouping / boundary exclusions as the prior run:
  group = stage2_best_mean_knn_has_tcell_neighbor (with_T / without_T); boundary_T cells excluded.

STEP 1  influence vector  F = sum_j w_j * (coord_j - coord_self), RAW DISPLACEMENT (default).
        w_j = attrib_mean normalized to sum 1 over the cell's walk-accessible neighbors.
        magnitude |F| and reach (attention-weighted mean distance) saved SEPARATELY so direction
        and reach are tested independently. Unit-direction variants saved for sensitivity.
STEP 2  field metrics over the SAME walk-accessible neighbor set (self excluded, traversal-weighted;
        NOT kNN / radius). Meshless weighted least-squares Jacobian A of the influence-vector field
        g_j = w_j*(coord_j-coord_self) on the neighbor positions r_j (residual weights = visits_j):
            divergence  = tr(A)            (source/sink character of the influence field)
            curl        = A[1,0]-A[0,1]    (signed; |curl| = rotational strength, tested)
            anisotropy  = (|l1|-|l2|)/(|l1|+|l2|) of sym(A) eigenvalues (directional stretching)
        Asserts the neighbor set per cell == the cached _attrib.npz block (same support).
STEP 3  per metric: Mann-Whitney U (two-sided) + Cliff's delta + medians/IQR; BH across metrics.
STEP 4  (a) label-permutation null per metric; (b) confound sensitivity: stratify Cliff's delta on
        local neighbor count, and recompute MWU on a set matched on (n_acc_nodes, total mass).

INTERPRETATION: the field is from ATTENTION ATTRIBUTION -> information-flow direction (where the
masked cell reads context from), NOT physical perturbation spread. Probe attention is unscaled;
M = random-walk tokens, not genes. All claims relational.
"""
import os, json, sys
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from scipy import stats

BASE   = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/attention'
CACHE  = os.path.join(BASE, 'cache')
OUT    = os.path.join(BASE, 'vectorfield')
OCACHE = os.path.join(OUT, 'cache')
ADATA  = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad'
N_PERM = 2000
RNG    = np.random.default_rng(0)
METRICS = ['magnitude', 'reach', 'divergence', 'abs_curl', 'anisotropy']

df = pd.read_csv(os.path.join(BASE, 'per_cell_table.csv'))
with h5py.File(ADATA, 'r') as f:
    spatial = f['obsm/spatial'][:]            # [N,2]
print(f"cells: {len(df)}  spatial: {spatial.shape}", flush=True)

# ---------- Step 1+2: per-cell field metrics ----------
def jacobian(r, g, w):
    """weighted LSQ fit g ≈ A r + b ; return A (2x2) or None if underdetermined."""
    k = r.shape[0]
    if k < 3:
        return None
    X = np.hstack([r, np.ones((k, 1))])       # [k,3]
    sw = np.sqrt(w)[:, None]
    Xw, gw = X * sw, g * sw
    if np.linalg.matrix_rank(Xw) < 3:
        return None
    B, *_ = np.linalg.lstsq(Xw, gw, rcond=None)   # [3,2]; columns gx,gy
    A = B[:2, :].T                                # A[i,j] = d g_i / d r_j
    return A

rows = []
field_cache = {}   # pert -> dict of arrays for divergence-source reuse
for pert, g_df in df.groupby('perturbation', sort=False):
    npz = os.path.join(CACHE, f'{pert}_attrib.npz')
    if not os.path.exists(npz):
        continue
    z = np.load(npz)
    indptr, nidx_all, attrib_all, visits_all = z['indptr'], z['neigh_adata_idx'], z['attrib_mean'], z['visits']
    fc = dict(adata_idx=[], Fx=[], Fy=[], magnitude=[], reach=[], divergence=[], curl=[], anisotropy=[])
    for _, row in g_df.iterrows():
        p = int(row['pos_in_block'])
        a, b = int(indptr[p]), int(indptr[p + 1])
        w = attrib_all[a:b].astype(float); vis = visits_all[a:b].astype(float); nidx = nidx_all[a:b]
        self_idx = int(row['adata_idx'])
        # assertion: neighbor set matches cached support (count == upstream n_acc_nodes)
        assert (b - a) == int(row['n_acc_nodes']), \
            f"{pert} cell {p}: block {b-a} != n_acc_nodes {row['n_acc_nodes']}"
        rec = dict(perturbation=pert, cell_id=row['cell_id'], adata_idx=self_idx,
                   group=row['group'], boundary_T=bool(row['boundary_T']),
                   n_acc_nodes=int(row['n_acc_nodes']), total_mass=float(w.sum()))
        if w.size == 0 or w.sum() == 0:
            for m in ('Fx','Fy','magnitude','reach','divergence','curl','abs_curl','anisotropy',
                      'mag_unit','divergence_unit','curl_unit','anisotropy_unit'):
                rec[m] = np.nan
            rows.append(rec); continue
        r = spatial[nidx] - spatial[self_idx]      # [k,2] displacement self->neighbor
        d = np.linalg.norm(r, axis=1)
        wn = w / w.sum()
        # Step 1 (raw displacement)
        F = (wn[:, None] * r).sum(0)
        rec['Fx'], rec['Fy'] = float(F[0]), float(F[1])
        rec['magnitude'] = float(np.linalg.norm(F))
        rec['reach'] = float((wn * d).sum())
        # unit-direction net (sensitivity): mean resultant length in [0,1]
        nz = d > 0
        u = np.zeros_like(r); u[nz] = r[nz] / d[nz, None]
        rec['mag_unit'] = float(np.linalg.norm((wn[:, None] * u).sum(0)))
        # Step 2 Jacobian on raw influence field g=wn*r (primary), weights=visits
        A = jacobian(r, wn[:, None] * r, vis)
        if A is None:
            rec['divergence'] = rec['curl'] = rec['abs_curl'] = rec['anisotropy'] = np.nan
        else:
            div = float(np.trace(A)); curl = float(A[1, 0] - A[0, 1])
            S = (A + A.T) / 2; ev = np.linalg.eigvalsh(S); ae = np.abs(ev)
            aniso = float((ae.max() - ae.min()) / (ae.sum() + 1e-30))
            rec['divergence'], rec['curl'], rec['abs_curl'], rec['anisotropy'] = div, curl, abs(curl), aniso
        # unit-direction Jacobian (sensitivity)
        Au = jacobian(r, wn[:, None] * u, vis)
        if Au is None:
            rec['divergence_unit'] = rec['curl_unit'] = rec['anisotropy_unit'] = np.nan
        else:
            Su = (Au + Au.T) / 2; evu = np.abs(np.linalg.eigvalsh(Su))
            rec['divergence_unit'] = float(np.trace(Au))
            rec['curl_unit'] = float(Au[1, 0] - Au[0, 1])
            rec['anisotropy_unit'] = float((evu.max() - evu.min()) / (evu.sum() + 1e-30))
        rows.append(rec)
        for kk, vv in (('adata_idx', self_idx), ('Fx', rec['Fx']), ('Fy', rec['Fy']),
                       ('magnitude', rec['magnitude']), ('reach', rec['reach']),
                       ('divergence', rec['divergence']), ('curl', rec['curl']),
                       ('anisotropy', rec['anisotropy'])):
            fc[kk].append(vv)
    field_cache[pert] = {k: np.asarray(v) for k, v in fc.items()}
    np.savez(os.path.join(OCACHE, f'{pert}_field.npz'), **field_cache[pert])

fdf = pd.DataFrame(rows)
# same cell set as prior tested run: known group + not boundary_T
tested = fdf[(fdf['group'].isin(['with_T', 'without_T'])) & (~fdf['boundary_T'])].copy()
tested.to_csv(os.path.join(OUT, 'per_cell_metrics.csv'), index=False)
print(f"tested cells: {len(tested)}  (with_T={int((tested.group=='with_T').sum())}, "
      f"without_T={int((tested.group=='without_T').sum())})", flush=True)


# ---------- Step 3+4: per-metric tests ----------
def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    # delta = P(x>y) - P(x<y); positive => with_T larger
    gt = lt = 0
    for xb in np.array_split(x, max(1, len(x) // 2000 + 1)):
        gt += np.sum(xb[:, None] > y[None, :]); lt += np.sum(xb[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))

def iqr(v):
    q1, q3 = np.percentile(v, [25, 75]); return q1, q3

stat_rows = []
null_store = {}
for m in METRICS:
    a = tested.loc[tested.group == 'with_T', m].to_numpy(); a = a[np.isfinite(a)]
    b = tested.loc[tested.group == 'without_T', m].to_numpy(); b = b[np.isfinite(b)]
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    delta = cliffs_delta(a, b)
    rb = 1 - 2 * U / (len(a) * len(b))    # rank-biserial (== -? ) reported alongside delta
    q1a, q3a = iqr(a); q1b, q3b = iqr(b)
    # 4a permutation null on Cliff's delta
    pooled = np.concatenate([a, b]); n1 = len(a)
    nd = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = RNG.permutation(pooled)
        nd[i] = cliffs_delta(perm[:n1], perm[n1:])
    emp_p = (np.sum(np.abs(nd) >= abs(delta)) + 1) / (N_PERM + 1)
    null_store[m] = nd
    stat_rows.append(dict(metric=m,
        n_with_T=len(a), n_without_T=len(b),
        median_with_T=float(np.median(a)), iqr_with_T=[float(q1a), float(q3a)],
        median_without_T=float(np.median(b)), iqr_without_T=[float(q1b), float(q3b)],
        U=float(U), p_raw=float(p), cliffs_delta=float(delta), rank_biserial=float(rb),
        null_delta_mean=float(nd.mean()),
        null_delta_ci95=[float(np.percentile(nd, 2.5)), float(np.percentile(nd, 97.5))],
        perm_p=float(emp_p)))

sdf = pd.DataFrame(stat_rows)
# Benjamini-Hochberg across metrics (manual, no extra deps)
def bh_adjust(p):
    p = np.asarray(p, float); n = p.size; order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[idx] * n / k); adj[idx] = prev
    return np.clip(adj, 0, 1)
sdf['p_bh'] = bh_adjust(sdf['p_raw'].to_numpy())

# ---------- Step 4b: confound sensitivity (match on n_acc_nodes + total_mass) ----------
def greedy_match(t):
    """1:1 nearest matching (without replacement) of the smaller group to the larger on
    standardized [n_acc_nodes, log1p(total_mass)]."""
    cov = t[['n_acc_nodes', 'total_mass']].to_numpy().astype(float)
    cov[:, 1] = np.log1p(cov[:, 1])
    cov = (cov - cov.mean(0)) / (cov.std(0) + 1e-9)
    gi = (t['group'] == 'with_T').to_numpy()
    A_idx = np.where(gi)[0]; B_idx = np.where(~gi)[0]
    small, big = (A_idx, B_idx) if len(A_idx) <= len(B_idx) else (B_idx, A_idx)
    used = np.zeros(len(big), dtype=bool); pairs = []
    for s in small:
        dd = np.sum((cov[big] - cov[s]) ** 2, axis=1)
        dd[used] = np.inf
        j = int(np.argmin(dd))
        if np.isfinite(dd[j]):
            used[j] = True; pairs.append((s, big[j]))
    keep = np.array([p for pr in pairs for p in pr])
    return t.iloc[keep]

matched = greedy_match(tested.reset_index(drop=True))
for i, m in enumerate(METRICS):
    a = matched.loc[matched.group == 'with_T', m].to_numpy(); a = a[np.isfinite(a)]
    b = matched.loc[matched.group == 'without_T', m].to_numpy(); b = b[np.isfinite(b)]
    if len(a) > 0 and len(b) > 0:
        U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        d = cliffs_delta(a, b)
    else:
        U, p, d = np.nan, np.nan, np.nan
    sdf.loc[sdf.metric == m, 'matched_n'] = min(len(a), len(b))
    sdf.loc[sdf.metric == m, 'matched_p'] = p
    sdf.loc[sdf.metric == m, 'matched_cliffs_delta'] = d
    sdf.loc[sdf.metric == m, 'survives_matching'] = bool(
        np.isfinite(d) and abs(d) >= 0.5 * abs(sdf.loc[sdf.metric == m, 'cliffs_delta'].iloc[0])
        and np.isfinite(p) and p < 0.05)

# stratified Cliff's delta on n_acc_nodes quartiles
strat = {}
qs = np.quantile(tested['n_acc_nodes'], [0, .25, .5, .75, 1.0])
tested['nbin'] = np.clip(np.digitize(tested['n_acc_nodes'], qs[1:-1]), 0, 3)
for m in METRICS:
    ds = []
    for bb in range(4):
        sub = tested[tested.nbin == bb]
        a = sub.loc[sub.group == 'with_T', m].to_numpy(); a = a[np.isfinite(a)]
        b = sub.loc[sub.group == 'without_T', m].to_numpy(); b = b[np.isfinite(b)]
        ds.append(cliffs_delta(a, b) if len(a) and len(b) else np.nan)
    strat[m] = ds
sdf['cliffs_delta_by_nbin'] = sdf['metric'].map(lambda m: [round(x, 3) for x in strat[m]])

sdf.to_csv(os.path.join(OUT, 'per_metric_stats.csv'), index=False)
with open(os.path.join(OUT, 'per_metric_stats.json'), 'w') as f:
    json.dump(sdf.to_dict(orient='records'), f, indent=2, default=float)


# ---------- figures ----------
labels = {'magnitude': 'net influence |F| (µm)', 'reach': 'attention-weighted reach (µm)',
          'divergence': 'divergence tr(A)', 'abs_curl': '|curl|', 'anisotropy': 'anisotropy'}
colors = {'with_T': '#d1495b', 'without_T': '#3f7cac'}
for m in METRICS:
    r = sdf[sdf.metric == m].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    a = tested.loc[tested.group == 'with_T', m].to_numpy(); a = a[np.isfinite(a)]
    b = tested.loc[tested.group == 'without_T', m].to_numpy(); b = b[np.isfinite(b)]
    lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
    for i, (grp, v) in enumerate([('with_T', a), ('without_T', b)]):
        vv = v[(v >= lo) & (v <= hi)]
        if vv.size >= 2:
            bp = ax[0].violinplot(vv, positions=[i], widths=0.8, showmedians=True, showextrema=False)
            for bd in bp['bodies']:
                bd.set_facecolor(colors[grp]); bd.set_alpha(0.55)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['with-T', 'without-T'])
    ax[0].set_ylabel(labels[m])
    ax[0].set_title(f"{m}\nδ={r['cliffs_delta']:.3f}  p_BH={r['p_bh']:.1e}")
    # null overlay (permutation null of Cliff's delta)
    nd = null_store[m]
    ax[1].hist(nd, bins=40, color='grey', alpha=0.6)
    ax[1].axvline(r['cliffs_delta'], color='r', lw=2, label=f"observed δ={r['cliffs_delta']:.3f}")
    ax[1].axvline(0, color='k', lw=0.8)
    ax[1].set_xlabel("Cliff's δ"); ax[1].set_title(f"label-permutation null\nperm p={r['perm_p']:.3g}")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f'metric_{m}.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

# summary forest plot
fig, ax = plt.subplots(figsize=(7, 4))
y = np.arange(len(METRICS))
for i, m in enumerate(METRICS):
    r = sdf[sdf.metric == m].iloc[0]
    ci = r['null_delta_ci95']
    ax.plot([ci[0], ci[1]], [i, i], color='grey', lw=6, alpha=0.4)
    sig = r['p_bh'] < 0.05
    ax.plot(r['cliffs_delta'], i, 'o', color='r' if sig else 'k', ms=8)
    ax.text(r['cliffs_delta'], i + 0.18, f"δ={r['cliffs_delta']:.3f}, p_BH={r['p_bh']:.1e}"
            + (" *" if sig else ""), fontsize=8, ha='center')
ax.axvline(0, color='k', lw=1)
ax.set_yticks(y); ax.set_yticklabels(METRICS); ax.set_xlabel("Cliff's δ (with-T vs without-T)")
ax.set_title("Attention vector-field metrics by T-cell-neighbor status\n(grey = permutation-null 95% of δ; red = BH-significant)")
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_forest.png'), dpi=180, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'summary_forest.pdf'), bbox_inches='tight')

# ---------- console summary ----------
print("\n==================== VECTOR-FIELD SUMMARY ====================")
for _, r in sdf.iterrows():
    print(f"\n{r['metric']}: median with_T={r['median_with_T']:.4g} without_T={r['median_without_T']:.4g}")
    print(f"   U={r['U']:.0f} p_raw={r['p_raw']:.2e} p_BH={r['p_bh']:.2e} "
          f"Cliff_delta={r['cliffs_delta']:.3f}")
    print(f"   perm_null δ={r['null_delta_mean']:.3f} CI95={r['null_delta_ci95']} perm_p={r['perm_p']:.3g}")
    print(f"   matched: n={r.get('matched_n')} p={r.get('matched_p')} δ={r.get('matched_cliffs_delta')} "
          f"survives={r.get('survives_matching')}")
    print(f"   δ by n_acc_nodes quartile: {r['cliffs_delta_by_nbin']}")
print("\nwrote per_cell_metrics.csv, per_metric_stats.csv/json, metric_*.png, summary_forest.png/pdf, cache/*_field.npz")
