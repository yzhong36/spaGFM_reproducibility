# Panel: figure3f_prep2
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_g_nfkb_network_step2_context.py
# Original filename: fig1_g_nfkb_network_step2_context.py
# Last modified: 2026-08-18 12:09:04  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: data/work/panelg/pergene/upfrac_500d.npz
# Output: the with/without context prediction CSV read by step3
# Match evidence: Step 2 of the 3-step panel-g chain.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python
"""
Context-split predicted up-fractions for ALL Tier A perturbations x ALL NF-kB
targets, consistent with the pooled figure.

Reproduces the EXACT 33-panel-gene model from run_pergene.py (same architecture,
hyper-params, seed). Training uses the full panel pool with NO context split
(trained together). Only at INFERENCE is the host pool split into two contexts:
  - with    immune (T-cell) neighbour   (TCELL_COL == True)
  - without immune (T-cell) neighbour   (TCELL_COL == False)
so the two figures differ only in the host cells fed to the same trained model.

Output: zeroshot_tierA/network_figures/context_predicted_values.csv
        columns: regulator, context, target, upfrac
"""
# ---------------------------------------------------------------------------
# 2026-08-18: TRAINING PROCEDURE SWITCHED FROM PIPELINE A TO PIPELINE B, so this
# panel matches fig1_e / fig1_f. The six audited axes now follow B:
#   1+2 NaN target rows dropped before scaling/splitting (scaler on filtered rows)
#   3   validation permutation from np.random.default_rng(42)
#   4   val_n = max(1, int(n*0.1)), NO clamp to n-1
#   5   pos_weight without .clamp(min=1e-3)
#   6   torch.manual_seed(42) immediately BEFORE constructing the model
# Architecture/hyperparameters unchanged. Pipeline-A version backed up in
# ../backup_fig1_g_pipelineA_20260818/.
# UNCHANGED: this panel still trains on the full pool and scores those same cells
# (host pool = training pool, split by T-cell neighbour only at inference).
# ---------------------------------------------------------------------------

import os, pickle, time, warnings
# cap threads BEFORE importing torch/numpy — avoids login-node oversubscription
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc

warnings.filterwarnings('ignore')

BASE      = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw'
ADATA     = BASE + '/perturb_fish_spatial.h5ad'
INFER_WL9 = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/inference_set_wl_9_317M.pt'
GENEPT    = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived/GenePT_gene_embedding_ada_text.pickle'
OUTDIR    = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/panelg'
FIGDIR    = os.path.join(OUTDIR, 'network_figures')
TCELL_COL = 'stage2_best_mean_knn_has_tcell_neighbor'
Path(FIGDIR).mkdir(parents=True, exist_ok=True)

# all 11 Tier A perturbations  ×  all 11 NF-kB targets  (match run_network_figures.py)
SOURCES = ['IRAK3','CYLD','TNFAIP3','TNIP1','OTULIN','TNF','UBE2N','STAT1','IKBKG','REL','MAPK14']
TARGETS = ['IL6','IL1B','CXCL8','CXCL1','CXCL2','ICAM1','NOS2','SOD2','NFKB1','NFKB2','NFKBIA']

# ── model utilities (verbatim from run_pergene.py) ────────────────────────────
def binarize_lfc(Y): return np.where(np.asarray(Y,np.float32)>0,1.,-1.).astype(np.float32)
def labels_to_binary(Y): return (np.asarray(Y,np.float32)>0).astype(np.int64)

class NumpyStandardScaler:
    def __init__(self, eps=1e-6): self.eps=eps; self.mean_=self.scale_=None
    def fit(self,X):
        X=np.asarray(X,np.float32); self.mean_=X.mean(0,keepdims=True)
        self.scale_=X.std(0,keepdims=True); self.scale_[self.scale_<self.eps]=1.; return self
    def transform(self,X): return (np.asarray(X,np.float32)-self.mean_)/self.scale_

def build_mlp(d_in,hidden,d_out,drop=0.):
    L=[]; prev=d_in
    for h in hidden: L+=[nn.Linear(prev,h),nn.ReLU()]; (L.append(nn.Dropout(drop)) if drop>0 else None); prev=h
    L.append(nn.Linear(prev,d_out)); return nn.Sequential(*L)

def build_encoder(d_in,hidden,drop=0.):
    if not hidden: return nn.Identity(),d_in
    L=[]; prev=d_in
    for h in hidden: L+=[nn.Linear(prev,h),nn.ReLU()]; (L.append(nn.Dropout(drop)) if drop>0 else None); prev=h
    return nn.Sequential(*L),prev

class TwoBranchMLP(nn.Module):
    def __init__(self,gene_input_dim,spa_input_dim,output_dim,
                 gene_hidden_dims=(256,),spa_hidden_dims=(256,),
                 fusion_hidden_dims=(256,128),dropout=0.1):
        super().__init__()
        self.ge,ge=build_encoder(gene_input_dim,gene_hidden_dims,dropout)
        self.se,se=build_encoder(spa_input_dim, spa_hidden_dims, dropout)
        self.fh=build_mlp(ge+se,fusion_hidden_dims,output_dim,dropout)
    def forward(self,gx,sx): return self.fh(torch.cat([self.ge(gx),self.se(sx)],1))

class MMWrapper:
    def __init__(self,builder,mkw,tkw):
        self.builder=builder; self.mkw=dict(mkw); self.tkw=dict(tkw)
        self.model=self.gs=self.ss=self.dev=None
    def _prep(self,Xg,Xs,fit=False):
        Xg=np.asarray(Xg,np.float32); Xs=np.asarray(Xs,np.float32)
        if fit or self.gs is None: self.gs=NumpyStandardScaler().fit(Xg)
        if fit or self.ss is None: self.ss=NumpyStandardScaler().fit(Xs)
        return self.gs.transform(Xg), self.ss.transform(Xs)
    def fit(self,Xg,Xs,Y):
        # B (axes 1+2): drop NaN target rows BEFORE scaling and splitting
        Y=np.asarray(Y,np.float32)
        valid=~np.isnan(Y).any(1)
        Xg=np.asarray(Xg,np.float32)[valid]; Xs=np.asarray(Xs,np.float32)[valid]; Y=Y[valid]
        Xg,Xs=self._prep(Xg,Xs,fit=True)
        Y=(Y>0).astype(np.float32)
        mk=dict(self.mkw); mk['gene_input_dim']=Xg.shape[1]; mk['spa_input_dim']=Xs.shape[1]; mk['output_dim']=Y.shape[1]
        torch.manual_seed(self.tkw.get('random_state',42))   # B (axis 6)
        self.model=self.builder(**mk); self.model,self.dev=_train(self.model,(Xg,Xs),Y,**self.tkw); return self
    def predict_logits(self,Xg,Xs,bs=1024):
        Xg,Xs=self._prep(Xg,Xs)
        ds=TensorDataset(*[torch.as_tensor(p,dtype=torch.float32) for p in (Xg,Xs)])
        lo=DataLoader(ds,batch_size=min(bs,len(Xg)),shuffle=False)
        out=[]; m=self.model.to(self.dev); m.eval()
        with torch.no_grad():
            for b in lo: out.append(m(*[t.to(self.dev) for t in b]).cpu().numpy())
        return np.concatenate(out,0)

def _bce_pw(Yt):
    # B (axis 5): no .clamp(min=1e-3)
    Y=Yt.detach().cpu().numpy(); pos=Y.sum(0); neg=Y.shape[0]-pos
    return torch.tensor(np.where(pos>0,neg/np.clip(pos,1e-9,None),1.0).astype(np.float32))

def _train(model,Xp,Y,epochs=200,batch_size=256,lr=1e-3,weight_decay=1e-4,
           val_fraction=0.1,patience=20,device=None,random_state=42):
    dev=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    Xts=[torch.as_tensor(p,dtype=torch.float32) for p in Xp]; Yt=torch.as_tensor(Y,dtype=torch.float32)
    n=Yt.shape[0]
    # B (axes 3+4): numpy PCG64 permutation; val_n=max(1,int(n*0.1)) with NO clamp
    rng=np.random.default_rng(random_state)
    pm=torch.from_numpy(rng.permutation(n).astype('int64'))
    vs=max(1,int(n*val_fraction)) if n>=10 else 0
    vi=pm[:vs]; ti=pm[vs:]
    loader=DataLoader(TensorDataset(*[t[ti] for t in Xts],Yt[ti]),batch_size=min(batch_size,len(ti)),shuffle=True)
    model=model.to(dev); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    crit=nn.BCEWithLogitsLoss(pos_weight=_bce_pw(Yt[ti]).to(dev))
    best=float('inf'); wait=0; bst=None
    for _ in range(epochs):
        model.train()
        for batch in loader:
            *inp,by=batch; inp=[t.to(dev) for t in inp]; by=by.to(dev)
            opt.zero_grad(); crit(model(*inp),by).backward(); opt.step()
        if not len(vi): continue
        model.eval()
        with torch.no_grad():
            vl=crit(model(*[t[vi].to(dev) for t in Xts]),Yt[vi].to(dev)).item()
        if vl<best: best=vl; bst={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    if bst: model.load_state_dict(bst)
    return model.eval(), dev

MKW={'gene_hidden_dims':(256,),'spa_hidden_dims':(256,),'fusion_hidden_dims':(256,128),'dropout':0.1}
TKW={'epochs':200,'batch_size':256,'lr':1e-3,'weight_decay':1e-4,'patience':20,'val_fraction':0.1,'random_state':42}

def assign_embeddings(adata,inf_list,key):
    for item in inf_list:
        emb=item.node_emb.numpy()
        idx=adata.obs_names.get_indexer(item.cell_id[item.y==item.perturbation])
        adata.obsm[key][idx]=emb

def filter_min_cells(adata,grp='perturbation',mc=20):
    c=adata.obs[grp].value_counts(); keep=c[c>=mc].index
    return adata[adata.obs[grp].isin(keep)].copy()

def upfrac(model,v,spa):
    L=model.predict_logits(np.tile(v,(spa.shape[0],1)),spa)   # [N,500]
    return (L>0).astype(np.float32).mean(0)                    # [500]

# ── load data (verbatim from run_pergene.py) ──────────────────────────────────
print(f'[{datetime.now():%H:%M:%S}] Loading data...',flush=True)
adata=sc.read_h5ad(ADATA)
adata.obsm['stage2_best_mean_knn_target_lfc']=adata.layers['stage2_best_mean_knn_target_lfc']
adata.obsm['spaGFM_emb']=np.zeros((adata.n_obs,1280),np.float32)
inf_l=torch.load(INFER_WL9,map_location='cpu',weights_only=False); assign_embeddings(adata,inf_l,'spaGFM_emb')
var_symbols=list(adata.var['gene_symbol'])
TGT_IDX={t:var_symbols.index(t) for t in TARGETS}

with open(GENEPT,'rb') as f: gpt=pickle.load(f)
def gvec(g): v=gpt.get(g); return np.asarray(v,np.float32) if v else None

# training pool = all panel genes >=20 cells (trained together, NO context split)
train_adata=filter_min_cells(adata[(adata.obs['perturbation'].notna())&
                                    (adata.obs['perturbation']!='Control')].copy())
panel_genes=sorted(train_adata.obs['perturbation'].unique().tolist())
print(f'Panel genes (train, together): {len(panel_genes)}',flush=True)

# host pool = same cells; split by immune(T-cell)-neighbour ONLY at inference
host=train_adata
with_mask=(host.obs[TCELL_COL]==True).values
without_mask=(host.obs[TCELL_COL]==False).values
spa_with=host.obsm['spaGFM_emb'][with_mask]
spa_without=host.obsm['spaGFM_emb'][without_mask]
print(f'Host pool: total={host.n_obs}  with_immune={with_mask.sum()}  without_immune={without_mask.sum()}',flush=True)

# ── train ONE model on the full pool (same seed/arch as run_pergene.py final) ─
print(f'[{datetime.now():%H:%M:%S}] Training (single fit = run_pergene final model)...',flush=True)
t0=time.time()
model=MMWrapper(TwoBranchMLP,MKW,TKW)
model.fit(train_adata.obsm['GPT_3_5_gene_embeddings'],
          train_adata.obsm['spaGFM_emb'],
          train_adata.obsm['stage2_best_mean_knn_target_lfc'])
print(f'[{datetime.now():%H:%M:%S}] Trained in {time.time()-t0:.1f}s',flush=True)

# ── context-split inference for all 11 sources × 11 targets ───────────────────
rows=[]
for g in SOURCES:
    v=gvec(g)
    if v is None: print(f'  SKIP {g}: no GenePT'); continue
    uf_w =upfrac(model,v,spa_with)
    uf_wo=upfrac(model,v,spa_without)
    for t in TARGETS:
        rows.append(dict(regulator=g,context='with',   target=t,upfrac=round(float(uf_w [TGT_IDX[t]]),4)))
        rows.append(dict(regulator=g,context='without',target=t,upfrac=round(float(uf_wo[TGT_IDX[t]]),4)))
    print(f'  {g:8s} with[NFKB1]={uf_w[TGT_IDX["NFKB1"]]:.2f}  without[NFKB1]={uf_wo[TGT_IDX["NFKB1"]]:.2f}',flush=True)

df=pd.DataFrame(rows)
out=os.path.join(FIGDIR,'context_predicted_values.csv')
df.to_csv(out,index=False)
print(f'\nSaved: {out}  ({len(df)} rows)',flush=True)
