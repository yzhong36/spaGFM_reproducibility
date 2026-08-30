# Panel: figure3c_prep2
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figd_r2_scgpt_20260827/code/r2_pergene_check.py
# Original filename: r2_pergene_check.py
# Last modified: 2026-08-27 14:40:34  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: figure_bundle_20260817/data/raw/perturb_fish_spatial.h5ad (X_scGPT)
# Output: figd_r2_scgpt_20260827/data/per_gene_r2.csv
# Match evidence: Upstream step 2: per-gene own-cell scGPT R^2 consumed by figure3c_prep3.py.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python3
"""Diagnostic: which of my two changes flipped the R^2 result?
Runs the ORIGINAL per-gene logo_r2 (n=50, pooled held-out block) for four arms, so the
published panel is reproduced and the own-cell-vs-mean-scGPT change is isolated."""
import json, numpy as np, scanpy as sc, torch, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
RAW=("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/"
     "figure_bundle_20260817/data/raw")
OUT=("/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/"
     "figd_r2_scgpt_20260827")
SEED,ALPHA,NPCA,NTOP,R=42,1.0,50,50,300
adata=sc.read_h5ad(f"{RAW}/perturb_fish_spatial.h5ad")
inf=torch.load(f"{RAW}/inference_set_wl_9_317M.pt",map_location="cpu",weights_only=False)
adata.obsm["spaGFM_emb"]=np.zeros((adata.n_obs,1280),dtype=np.float32)
asg=np.zeros(adata.n_obs,bool)
for g in inf:
    m=np.asarray(g.y==g.perturbation); idx=adata.obs_names.get_indexer(np.asarray(g.cell_id)[m]); v=idx>=0
    adata.obsm["spaGFM_emb"][idx[v]]=g.node_emb.detach().numpy().astype(np.float32)[v]; asg[idx[v]]=True
del inf
cnt=adata.obs["perturbation"].value_counts()
valid=cnt[(cnt>=20)&(cnt.index!="Control")].index
pa=adata[(adata.obs["perturbation"].isin(valid)&asg).values].copy()
perturbs=pa.obs["perturbation"].astype(str).values
lfc=pa.layers["primary_target_lfc"].astype(np.float32)
pm={p:lfc[perturbs==p].mean(0) for p in np.unique(perturbs)}
resid=np.array([lfc[i]-pm[p] for i,p in enumerate(perturbs)],dtype=np.float32)
top=np.argsort(resid.var(0))[::-1][:NTOP]; Y=resid[:,top]
tree=cKDTree(adata.obsm["spatial"]); sg=adata.obsm["X_scGPT"].astype(np.float32)
ct=adata.obs["cell_type"].values
mean_sg=np.zeros((pa.n_obs,sg.shape[1]),dtype=np.float32)
for i,(cx,cy) in enumerate(pa.obsm["spatial"]):
    k=[j for j in tree.query_ball_point([cx,cy],R) if ct[j]!="T_cell"]
    if k: mean_sg[i]=sg[k].mean(0)

def logo_r2_pergene(X,Y,groups,n_pca=NPCA):
    if n_pca and n_pca<X.shape[1]: X=PCA(n_pca,random_state=SEED).fit_transform(X)
    yt,yp=[],[]
    for g in np.unique(groups):
        te,tr=groups==g,groups!=g
        if tr.sum()<10 or te.sum()<2: continue
        s=StandardScaler().fit(X[tr])
        mdl=Ridge(alpha=ALPHA).fit(s.transform(X[tr]),Y[tr])
        yt.append(Y[te]); yp.append(mdl.predict(s.transform(X[te])))
    yt=np.vstack(yt); yp=np.vstack(yp)
    return 1-((yt-yp)**2).sum(0)/(((yt-yt.mean(0))**2).sum(0)+1e-10)

arms={"own_scGPT":pa.obsm["X_scGPT"].astype(np.float32),
      "mean_scGPT":mean_sg, "spaGFM":pa.obsm["spaGFM_emb"]}
res={k:logo_r2_pergene(v,Y,perturbs) for k,v in arms.items()}
print("PER-GENE R2 (n=50), the published definition:")
for k,v in res.items(): print(f"   {k:<11} mean {v.mean():+.4f}  median {np.median(v):+.4f}")
for a_,b_ in [("mean_scGPT","spaGFM"),("own_scGPT","spaGFM")]:
    _,p=wilcoxon(res[a_],res[b_])
    print(f"   {a_} vs {b_}: spaGFM higher in {int((res[b_]>res[a_]).sum())}/50, p={p:.3e}")
pd.DataFrame({"gene":pa.var_names[top],**res}).to_csv(f"{OUT}/data/per_gene_r2.csv",index=False)
json.dump({k:float(v.mean()) for k,v in res.items()},open(f"{OUT}/data/per_gene_means.json","w"),indent=1)
print("wrote per_gene_r2.csv")
