# Panel: figure3_supp_g
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig2_c_spearman_tertiles_figure.py
# Original filename: fig2_c_spearman_tertiles_figure.py
# Last modified: 2026-08-18 10:29:10  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: data/work/fig2_c/spearman_long_2arm.csv; figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad
# Output: data/work/fig2_c/fig_spearman_by_tertile_2arm_317M.{pdf,png,svg} + tertiles_2arm_317M.csv
# Match evidence: PANELS = [('spaGFM','spaGFM (ours)'), ('GenePT*','GenePT*')] and PAL = Low #4878CF (blue) / Medium #6ACC65 (green) / High #D65F5F (red) match the two stacked panels and the three tertile colours. Recomputing the Kruskal-Wallis test from spearman_long_2arm.csv gives p=0.033 (spaGFM) and p=0.422 (GenePT*) -- both titles reproduce exactly; rho range 0-0.94 matches the 0-0.9 axis.
# Other candidates considered: spearman_ridge_317M_2arm_20260817/make_figure_2arm.py -- the pre-bundle original this was copied from.
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""
STEP 2 -- the figure. Reuse of
  spearman_by_tertile_4methods/make_figure_plainLFC_celcomen.py
with ONLY the input, the panel list and the output names changed.

UNTOUCHED from the original: PAL colours, TERTILE_ORDER, rcParams, the plain
mean|LFC| tertile binning (qcut into 3 over the evaluated perturbations), violin
construction, jitter (rng seed 0, +/-0.07, s=45, alpha 0.85), per-panel KW +
Mann-Whitney annotation, axis labels, suptitle wording, tight_layout rect, and the
png/pdf/svg save loop.

CHANGED, and only this:
  - input  : spearman_long_2arm.csv (this directory) instead of the three published CSVs
  - PANELS : 2 arms (spaGFM, GenePT*) instead of 4
  - grid   : 1x2 instead of 2x2, figsize 12x4.5 instead of 12x9, so EACH PANEL keeps
             the original 6.0 x 4.5 inch geometry. Dropping two arms forces this.
  - output : fig_spearman_by_tertile_2arm_317M.{png,pdf,svg}, tertiles_2arm_317M.csv

NOTE the per-panel "KW p" and "LvM/LvH/MvH" annotations are inherited from the reused
plotting code. They are WITHIN-arm comparisons across effect-size tertiles -- they are NOT
the spaGFM-vs-GenePT* comparison, which is reported separately in stats_spagfm_vs_genept.csv
and in run_ridge_2arm.log.

NOT A REPRODUCTION of the published figure -- new Ridge fit, published rho definition.
See RUN_NOTES.md.
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/fig2_c"
H5AD=("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw"
      "/perturb_fish_spatial.h5ad")
Y_KEY="stage2_best_mean_knn_target_lfc"; PK="perturbation"

PANELS=[("spaGFM","spaGFM (ours)"),("GenePT*","GenePT*")]
PAL={"Low":"#4878CF","Medium":"#6ACC65","High":"#D65F5F"}
TERTILE_ORDER=["Low","Medium","High"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
rng=np.random.default_rng(0)

# ── per-perturbation Spearman (from THIS run's Ridge fit) ────────────────────
long=pd.read_csv(f"{OUT}/spearman_long_2arm.csv")[["method","perturbation","spearman_r"]]

# ── PLAIN mean |LFC| tertiles (overall effect size) ──────────────────────────
print("Loading adata for plain mean|LFC| …", flush=True)
ad=sc.read_h5ad(H5AD)
Y=np.asarray(ad.layers[Y_KEY],dtype=np.float32)
ridge_perts=sorted(long["perturbation"].unique())
plain={p:float(np.mean(np.abs(Y[(ad.obs[PK]==p).values].mean(0)))) for p in ridge_perts}
tdf=pd.DataFrame({"perturbation":list(plain.keys()),
                  "plain_mean_abs_LFC":list(plain.values())})
tdf["tertile"]=pd.qcut(tdf["plain_mean_abs_LFC"],3,labels=TERTILE_ORDER)
tdf.to_csv(f"{OUT}/tertiles_2arm_317M.csv",index=False)
print(f"Saved tertiles_2arm_317M.csv ({len(tdf)} perturbations)")
long=long.merge(tdf[["perturbation","tertile"]],on="perturbation",how="left")

# ── plot ─────────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,(method,label) in zip(axes.flat,PANELS):
    sub=long[long["method"]==method]
    data3=[sub.loc[sub["tertile"]==t,"spearman_r"].dropna().values for t in TERTILE_ORDER]
    if all(len(d)>0 for d in data3):
        parts=ax.violinplot(data3,positions=[0,1,2],showmedians=True,showextrema=True)
        for pc,t in zip(parts["bodies"],TERTILE_ORDER):
            pc.set_facecolor(PAL[t]); pc.set_alpha(0.60)
    for pos,(t,d) in enumerate(zip(TERTILE_ORDER,data3)):
        if len(d)==0: continue
        jit=rng.uniform(-0.07,0.07,len(d))
        ax.scatter(pos+jit,d,color=PAL[t],s=45,alpha=0.85,zorder=3,
                   edgecolors="white",linewidths=0.4)
    nonempty=[d for d in data3 if len(d)>0]
    kp=stats.kruskal(*data3)[1] if len(nonempty)==3 else float("nan")
    sigs=[]
    for ai,bi,lab in [(0,1,"LvM"),(0,2,"LvH"),(1,2,"MvH")]:
        if len(data3[ai]) and len(data3[bi]):
            mwp=stats.mannwhitneyu(data3[ai],data3[bi],alternative="two-sided")[1]
            star="***" if mwp<.001 else("**" if mwp<.01 else("*" if mwp<.05 else"ns"))
        else: star="na"
        sigs.append(f"{lab}:{star}")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(TERTILE_ORDER)
    ax.set_ylabel("Spearman ρ",fontsize=10); ax.set_xlabel("Effect-size tertile",fontsize=9)
    kp_str=f"{kp:.3f}" if np.isfinite(kp) else "na"
    ax.set_title(f"{label}  |  KW p={kp_str}\n{' | '.join(sigs)}",fontsize=10)

fig.suptitle(
    "Spearman ρ (predicted vs. true LFC) by perturbation effect size\n"
    "(Low / Medium / High = tertiles of mean |LFC|)",
    fontsize=12,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.94])
for ext in ("png","pdf","svg"):
    plt.savefig(f"{OUT}/fig_spearman_by_tertile_2arm_317M.{ext}",bbox_inches="tight")
plt.close()
print(f"Saved fig_spearman_by_tertile_2arm_317M.[png,pdf,svg] -> {OUT}")
print("\nper-tertile mean rho:")
print(long.pivot_table(index="method",columns="tertile",values="spearman_r",
      aggfunc="mean").reindex(columns=TERTILE_ORDER).round(3).to_string())
