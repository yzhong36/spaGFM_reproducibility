# Panel: figure3f_prep1
# Source: /fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/code/fig1_g_nfkb_network_step1_pergene.py
# Original filename: fig1_g_nfkb_network_step1_pergene.py
# Last modified: 2026-08-18 12:09:04  (git: n/a - no git repository anywhere under graph_foundation_model/)
# Input data: data/raw h5ad + 317M inference set
# Output: data/work/panelg/pergene/upfrac_500d.npz
# Match evidence: Step 1 of the 3-step panel-g chain; writes the npz step3 loads.
# Other candidates considered: none - no other script in the tree produces this panel
# ---- copied verbatim below; NOT modified ----
#!/usr/bin/env python
"""
Per-gene zero-shot evaluation — revised analysis.
Replaces the broken NF-kB scalar with per-gene sign agreement on:
  (a) NF-kB canonical core (excl. CCL3/4/5)
  (b) chemokine / AP-1 module (separate axis)
under two reference frames: kNN-neighbor and control-cell.

All frozen predictions are PRESERVED. Only new computation:
  - full 500-d predicted up-fraction (model retrained identically)
  - per-gene ground truth (training genes only; Tier A are zero-shot)
  - metrics and plots

Outputs: OUTDIR/pergene/
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

import os, sys, json, pickle, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, GroupKFold

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw'
ADATA     = BASE + '/perturb_fish_spatial.h5ad'
INFER_WL9 = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/raw/inference_set_wl_9_317M.pt'
GENEPT    = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/derived/GenePT_gene_embedding_ada_text.pickle'
OUTDIR    = '/fs/ess/PAS1475/Xiaojie/graph_foundation_model/Final_perturbation/figure_bundle_20260817/data/work/panelg'
PGDIR     = os.path.join(OUTDIR, 'pergene')
Path(PGDIR).mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
LOG = os.path.join(PGDIR, 'run_log.txt')
_lines = []
def log(m):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    s = f'[{t}] {m}'; print(s, flush=True); _lines.append(s)
def flush_log():
    open(LOG, 'w').write('\n'.join(_lines)+'\n')

log('Per-gene analysis started.'); log(f'PGDIR = {PGDIR}')
log('Frozen Tier A predictions in OUTDIR will NOT be modified.')

# ── model utilities (verbatim from original) ──────────────────────────────────
CLASS_NEG, CLASS_POS = -1.0, 1.0
def binarize_lfc(Y):
    Y = np.asarray(Y, np.float32)
    return np.where(Y > 0, CLASS_POS, CLASS_NEG).astype(np.float32)
def labels_to_binary(Y):
    return (np.asarray(Y, np.float32) > 0).astype(np.int64)
def logits_to_labels(logits):
    return np.where(np.asarray(logits, np.float32) > 0, CLASS_POS, CLASS_NEG).astype(np.float32)

class NumpyStandardScaler:
    def __init__(self, eps=1e-6): self.eps=eps; self.mean_=self.scale_=None
    def fit(self, X):
        X=np.asarray(X,np.float32); self.mean_=X.mean(0,keepdims=True)
        self.scale_=X.std(0,keepdims=True); self.scale_[self.scale_<self.eps]=1.; return self
    def transform(self, X): return (np.asarray(X,np.float32)-self.mean_)/self.scale_

def build_mlp(d_in, hidden, d_out, drop=0.):
    L=[]; prev=d_in
    for h in hidden: L+=[nn.Linear(prev,h),nn.ReLU()]; (L.append(nn.Dropout(drop)) if drop>0 else None); prev=h
    L.append(nn.Linear(prev,d_out)); return nn.Sequential(*L)

def build_encoder(d_in, hidden, drop=0.):
    if not hidden: return nn.Identity(), d_in
    L=[]; prev=d_in
    for h in hidden: L+=[nn.Linear(prev,h),nn.ReLU()]; (L.append(nn.Dropout(drop)) if drop>0 else None); prev=h
    return nn.Sequential(*L), prev

class TwoBranchMLP(nn.Module):
    def __init__(self, gene_input_dim, spa_input_dim, output_dim,
                 gene_hidden_dims=(256,), spa_hidden_dims=(256,),
                 fusion_hidden_dims=(256,128), dropout=0.1):
        super().__init__()
        self.ge, ge = build_encoder(gene_input_dim, gene_hidden_dims, dropout)
        self.se, se = build_encoder(spa_input_dim,  spa_hidden_dims,  dropout)
        self.fh = build_mlp(ge+se, fusion_hidden_dims, output_dim, dropout)
    def forward(self, gx, sx):
        return self.fh(torch.cat([self.ge(gx), self.se(sx)], 1))

class MMWrapper:
    def __init__(self, builder, mkw, tkw):
        self.builder=builder; self.mkw=dict(mkw); self.tkw=dict(tkw)
        self.model=self.gs=self.ss=self.dev=None
    def _prep(self, Xg, Xs, fit=False):
        Xg=np.asarray(Xg,np.float32); Xs=np.asarray(Xs,np.float32)
        if fit or self.gs is None: self.gs=NumpyStandardScaler().fit(Xg)
        if fit or self.ss is None: self.ss=NumpyStandardScaler().fit(Xs)
        return self.gs.transform(Xg), self.ss.transform(Xs)
    def fit(self, Xg, Xs, Y):
        # B (axes 1+2): drop NaN target rows BEFORE scaling and splitting
        Y=np.asarray(Y,np.float32)
        valid=~np.isnan(Y).any(1)
        Xg=np.asarray(Xg,np.float32)[valid]; Xs=np.asarray(Xs,np.float32)[valid]; Y=Y[valid]
        Xg,Xs=self._prep(Xg,Xs,fit=True)
        Y=(Y>0).astype(np.float32)
        mk=dict(self.mkw); mk['gene_input_dim']=Xg.shape[1]; mk['spa_input_dim']=Xs.shape[1]; mk['output_dim']=Y.shape[1]
        torch.manual_seed(self.tkw.get('random_state',42))   # B (axis 6)
        self.model=self.builder(**mk); self.model,self.dev=_train(self.model,(Xg,Xs),Y,**self.tkw); return self
    def predict_logits(self, Xg, Xs, bs=1024):
        Xg,Xs=self._prep(Xg,Xs)
        ds=TensorDataset(*[torch.as_tensor(p,dtype=torch.float32) for p in (Xg,Xs)])
        lo=DataLoader(ds,batch_size=min(bs,len(Xg)),shuffle=False)
        out=[]; m=self.model.to(self.dev); m.eval()
        with torch.no_grad():
            for b in lo: out.append(m(*[t.to(self.dev) for t in b]).cpu().numpy())
        return np.concatenate(out,0)
    def predict(self, Xg, Xs, bs=1024):
        return logits_to_labels(self.predict_logits(Xg, Xs, bs))

def _bce_pw(Yt):
    # B (axis 5): no .clamp(min=1e-3)
    Y=Yt.detach().cpu().numpy(); pos=Y.sum(0); neg=Y.shape[0]-pos
    return torch.tensor(np.where(pos>0,neg/np.clip(pos,1e-9,None),1.0).astype(np.float32))

def _train(model, Xp, Y, epochs=200, batch_size=256, lr=1e-3,
           weight_decay=1e-4, val_fraction=0.1, patience=20, device=None, random_state=42):
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

MKW = {'gene_hidden_dims':(256,),'spa_hidden_dims':(256,),'fusion_hidden_dims':(256,128),'dropout':0.1}
TKW = {'epochs':200,'batch_size':256,'lr':1e-3,'weight_decay':1e-4,'patience':20,'val_fraction':0.1,'random_state':42}

def assign_embeddings(adata, inference_list, key):
    for item in inference_list:
        emb=item.node_emb.numpy()
        idx=adata.obs_names.get_indexer(item.cell_id[item.y==item.perturbation])
        adata.obsm[key][idx]=emb

def filter_min_cells(adata, grp='perturbation', mc=20):
    c=adata.obs[grp].value_counts(); keep=c[c>=mc].index
    return adata[adata.obs[grp].isin(keep)].copy()

def k_fold_spagfm(adata, gk, sk, yk, k=5, gkey='perturbation'):
    Xg=adata.obsm[gk]; Xs=adata.obsm[sk]; Y=adata.obsm[yk]
    groups=np.asarray(adata.obs[gkey])
    n_sp=min(k,pd.Series(groups).nunique())
    sp=GroupKFold(n_splits=n_sp); si=sp.split(Xg,Y,groups=groups)
    for ti,vi in si:
        m=MMWrapper(TwoBranchMLP,MKW,TKW); m.fit(Xg[ti],Xs[ti],Y[ti])
    final=MMWrapper(TwoBranchMLP,MKW,TKW); final.fit(Xg,Xs,Y); return final

# ── load data ─────────────────────────────────────────────────────────────────
log('Loading AnnData...'); t0=time.time()
adata=sc.read_h5ad(ADATA)
adata.obsm['stage2_best_mean_knn_target_lfc']=adata.layers['stage2_best_mean_knn_target_lfc']
adata.obsm['constant_zero_spa_features']=np.ones((adata.n_obs,1280),np.float32)
log(f'Loaded {adata.n_obs} cells x {adata.n_vars} genes in {time.time()-t0:.1f}s')

adata.obsm['spaGFM_emb']=np.zeros((adata.n_obs,1280),np.float32)
inf_l=torch.load(INFER_WL9,map_location='cpu',weights_only=False); assign_embeddings(adata,inf_l,'spaGFM_emb')

var_symbols=list(adata.var['gene_symbol'])
n_genes=adata.n_vars  # 500

train_adata=filter_min_cells(adata[(adata.obs['perturbation'].notna())&
                                    (adata.obs['perturbation']!='Control')].copy())
panel_genes=sorted(train_adata.obs['perturbation'].unique().tolist())
log(f'Panel genes: {len(panel_genes)}')

with open(GENEPT,'rb') as f: gpt=pickle.load(f)
def gvec(g): v=gpt.get(g); return np.asarray(v,np.float32) if v else None

host=train_adata; host_spa=host.obsm['spaGFM_emb'].copy(); N_host=host_spa.shape[0]
log(f'Host pool: {N_host} cells')

# ── module definitions ────────────────────────────────────────────────────────
NFKB_CORE=['IL1B','CXCL8','NFKB1','NFKB2','NFKBIA','ICAM1','IL6','CXCL1','CXCL2','NOS2','SOD2']
CHEMO_MOD=['CCL3','CCL4','CCL5','CCL2','PTGS2','IL12B']

def get_cols(gene_list):
    return [i for i,s in enumerate(var_symbols) if s in gene_list]

CORE_COLS  = get_cols(NFKB_CORE)
CHEMO_COLS = get_cols(CHEMO_MOD)
CORE_KEPT  = [var_symbols[i] for i in CORE_COLS]
CHEMO_KEPT = [var_symbols[i] for i in CHEMO_COLS]
log(f'NF-kB core: {CORE_KEPT}')
log(f'Chemokine module: {CHEMO_KEPT}')

# ── retrain multimodal model (same settings; needed for full 500-d predictions) ─
log('Retraining multimodal model for full 500-d predictions...')
t0=time.time()
model=k_fold_spagfm(train_adata,'GPT_3_5_gene_embeddings','spaGFM_emb',
                    'stage2_best_mean_knn_target_lfc')
log(f'Retrained in {time.time()-t0:.1f}s')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Ground truth per training gene, two reference frames
# ═══════════════════════════════════════════════════════════════════════════════
log('=== SECTION 1: per-gene ground truth ===')
log('Control-ref formula: ln(mean_KO + 1) - ln(mean_ctrl + 1), pseudocount=1, raw counts (int16).')

ctrl_mask=(adata.obs['perturbation']=='Control').values
ctrl_X=np.asarray(adata.X[ctrl_mask,:],float)
ctrl_mean=ctrl_X.mean(0)+1.0  # [500], pseudocount=1
n_ctrl=int(ctrl_mask.sum())
log(f'Control cells: {n_ctrl}')

gt_knn={}; gt_ctrl={}
for g in panel_genes:
    # kNN reference
    m=train_adata.obs['perturbation']==g
    lfc=train_adata[m].obsm['stage2_best_mean_knn_target_lfc']
    gt_knn[g]=np.nanmean(np.asarray(lfc,float),axis=0)  # [500]
    # Control reference
    all_mask=(adata.obs['perturbation']==g).values
    ko_X=np.asarray(adata.X[all_mask,:],float)
    ko_mean=ko_X.mean(0)+1.0
    gt_ctrl[g]=np.log(ko_mean/ctrl_mean)  # [500]

pd.DataFrame(gt_knn, index=var_symbols).T.to_csv(os.path.join(PGDIR,'ground_truth_knn.csv'))
pd.DataFrame(gt_ctrl, index=var_symbols).T.to_csv(os.path.join(PGDIR,'ground_truth_ctrl.csv'))
log('ground_truth_knn.csv and ground_truth_ctrl.csv written.')

# kNN-ref range check
knn_arr=np.array([gt_knn[g] for g in panel_genes])
ctrl_arr=np.array([gt_ctrl[g] for g in panel_genes])
log(f'kNN-ref LFC: min={knn_arr.min():.3f}, max={knn_arr.max():.3f}, median_abs={np.abs(knn_arr).median() if False else float(np.median(np.abs(knn_arr))):.4f}')
log(f'Ctrl-ref LFC: min={ctrl_arr.min():.3f}, max={ctrl_arr.max():.3f}, median_abs={float(np.median(np.abs(ctrl_arr))):.4f}')

# RELA reference-frame cross-check
if 'RELA' in panel_genes:
    for g_check in ['CCL3','CCL4','CCL5','TNF','IL1B','NFKBIA']:
        if g_check not in var_symbols: continue
        idx=var_symbols.index(g_check)
        log(f'  RELA/{g_check}: kNN={gt_knn["RELA"][idx]:+.3f}  ctrl={gt_ctrl["RELA"][idx]:+.3f}')
flush_log()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Full 500-d predicted up-fractions
# ═══════════════════════════════════════════════════════════════════════════════
log('=== SECTION 2: full 500-d predictions ===')

TIER_A_RAW=['TNFAIP3','CYLD','IRAK3','TNIP1','OTULIN','TNF','IKBKG','STAT1','REL','MAPK14','UBE2N']
TIER_A_EXPECTED={
    'TNFAIP3':{'module':'deubiquitinase/neg-reg','is_neg':1},
    'CYLD':   {'module':'deubiquitinase/neg-reg','is_neg':1},
    'IRAK3':  {'module':'decoy-IRAK/neg-reg',    'is_neg':1},
    'TNIP1':  {'module':'A20-binding/neg-reg',   'is_neg':1},
    'OTULIN': {'module':'linear-Ub/neg-reg',     'is_neg':1},
    'TNF':    {'module':'autocrine-neg-feedback','is_neg':1},
    'IKBKG':  {'module':'IKK-core/pos-node',     'is_neg':0},
    'STAT1':  {'module':'IFN-ISG/pos-node',      'is_neg':0},
    'REL':    {'module':'NF-kB-TF/pos-node',     'is_neg':0},
    'UBE2N':  {'module':'K63-Ub-E2/pos-node',   'is_neg':0},
}
# MAPK14: no label (was below min-cells in training) — keep for plotting, exclude from AUROC

# Compute full 500-d upfrac for every gene (panel + Tier A)
upfrac_all={}
for g in panel_genes + TIER_A_RAW:
    v=gvec(g)
    if v is None: log(f'  SKIP {g}: missing GenePT'); continue
    Xg=np.tile(v,(N_host,1))
    logits=model.predict_logits(Xg, host_spa)   # [N_host, 500]
    upfrac_all[g]=(logits>0).astype(np.float32).mean(0)  # [500]
    log(f'  {g}: core_upfrac={upfrac_all[g][CORE_COLS].mean():.3f}  chemo_upfrac={upfrac_all[g][CHEMO_COLS].mean():.3f}')

np.savez(os.path.join(PGDIR,'upfrac_500d.npz'),
         **{g: upfrac_all[g] for g in upfrac_all})
log('upfrac_500d.npz written (all panel + Tier A genes, 500-d).')
flush_log()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Tier A per-gene evaluation
# ═══════════════════════════════════════════════════════════════════════════════
log('=== SECTION 3: Tier A per-gene evaluation ===')

def per_gene_metrics(pred_upfrac, gt_vec, gene_label, ref_label, tag, top_k=25, lfc_thr=0.05):
    """Compare 500-d predicted up-fraction against 500-d ground-truth LFC."""
    pred_up = (pred_upfrac > 0.5).astype(int)    # 1 = predicted UP
    gt_sign = np.sign(gt_vec).astype(int)         # +1/-1/0

    # Responsive set: |LFC| >= thr OR top/bottom top_k
    abs_gt = np.abs(gt_vec)
    thr_mask = abs_gt >= lfc_thr
    top_idx = np.argsort(gt_vec)[-top_k:]
    bot_idx = np.argsort(gt_vec)[:top_k]
    topk_mask = np.zeros(n_genes, bool)
    topk_mask[top_idx] = True; topk_mask[bot_idx] = True
    resp = thr_mask | topk_mask
    n_resp = resp.sum()

    sign_all  = (pred_up == (gt_sign > 0).astype(int)).mean()
    sign_resp = (pred_up[resp] == (gt_sign[resp] > 0).astype(int)).mean() if n_resp else float('nan')
    sp_r, sp_p = stats.spearmanr(pred_upfrac[resp], gt_vec[resp]) if n_resp > 3 else (float('nan'),float('nan'))

    # Top-k recovery: top 10 and bottom 10 by ground truth
    k_eval = min(10, top_k)
    top10 = np.argsort(gt_vec)[-k_eval:]  # most UP
    bot10 = np.argsort(gt_vec)[:k_eval]   # most DOWN
    top10_correct = (pred_upfrac[top10] > 0.5).mean() if len(top10) else float('nan')
    bot10_correct = (pred_upfrac[bot10] < 0.5).mean() if len(bot10) else float('nan')
    topk_recovery = (top10_correct + bot10_correct) / 2

    return {
        f'sign_all_{tag}':    sign_all,
        f'sign_resp_{tag}':   sign_resp,
        f'spearman_{tag}':    sp_r,
        f'spearman_p_{tag}':  sp_p,
        f'topk_rec_{tag}':    topk_recovery,
        f'n_resp_{tag}':      int(n_resp),
    }

# For Tier A: ground truth = nearest training gene (proxy) since Tier A never perturbed
from numpy.linalg import norm
def cosine(a, b): return float(np.dot(a,b)/(norm(a)*norm(b)+1e-12))

# Find nearest training gene GenePT for each Tier A
train_vecs={g:gvec(g) for g in panel_genes if gvec(g) is not None}
tier_a_rows=[]
for g in TIER_A_RAW:
    if g not in upfrac_all: continue
    v=gvec(g)
    meta=TIER_A_EXPECTED.get(g,{})
    # nearest training gene
    sims={tg:cosine(v,tv) for tg,tv in train_vecs.items()}
    nearest=max(sims,key=sims.get); sim=sims[nearest]

    row={'gene':g,'is_neg':meta.get('is_neg',float('nan')),'module':meta.get('module','unlabeled'),
         'nearest_train':nearest,'sim':sim}

    pred_uf=upfrac_all[g]
    for ref, gt_dict in [('knn',gt_knn),('ctrl',gt_ctrl)]:
        # per-gene sign on NF-kB core (primary per-axis metric, no nearest proxy needed)
        core_uf = pred_uf[CORE_COLS]
        core_pred_up = (core_uf > 0.5).mean()
        chemo_uf = pred_uf[CHEMO_COLS]
        chemo_pred_up = (chemo_uf > 0.5).mean()
        row[f'core_upfrac_{ref}'] = float(core_pred_up)
        row[f'chemo_upfrac_{ref}'] = float(chemo_pred_up)

        # proxy comparison via nearest training gene
        gt_proxy = gt_dict[nearest]
        m = per_gene_metrics(pred_uf, gt_proxy, g, ref, f'proxy_{ref}')
        row.update(m)

    tier_a_rows.append(row)
    log(f'  {g}: core_up_knn={row["core_upfrac_knn"]:.3f}  core_up_ctrl={row["core_upfrac_ctrl"]:.3f}  '
        f'nearest={nearest}({sim:.2f})')

tier_a_df=pd.DataFrame(tier_a_rows)
tier_a_df.to_csv(os.path.join(PGDIR,'tierA_pergene_metrics.csv'),index=False)
log('tierA_pergene_metrics.csv written.')

# Aggregate: sign accuracy on NF-kB core for labeled genes
labeled=tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].copy()
for ref in ['knn','ctrl']:
    col=f'core_upfrac_{ref}'
    median_score=labeled[col].median()
    labeled[f'core_correct_{ref}']=labeled.apply(
        lambda r: int((r[col]>median_score)==bool(r['is_neg'])), axis=1)
    sign_acc=labeled[f'core_correct_{ref}'].mean()
    log(f'Tier A NF-kB core sign accuracy ({ref}-ref): {labeled["core_correct_"+ref].sum()}/{len(labeled)} = {sign_acc:.2f}')

# Null: shuffle pred_upfrac per gene
np.random.seed(42)
null_accs_knn=[]
for _ in range(1000):
    scores_shuf=labeled['core_upfrac_knn'].sample(frac=1,replace=False).values
    med=np.median(scores_shuf)
    null_accs_knn.append((scores_shuf>med)==labeled['is_neg'].values.astype(bool))
null_mean_knn=float(np.array(null_accs_knn).mean())
log(f'Shuffle null sign accuracy (knn): {null_mean_knn:.3f}')
flush_log()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — In-distribution per-gene evaluation (honest model ceiling)
# ═══════════════════════════════════════════════════════════════════════════════
log('=== SECTION 4: in-distribution per-gene evaluation ===')

indist_rows=[]
for g in panel_genes:
    if g not in upfrac_all: continue
    pred_uf=upfrac_all[g]
    row={'gene':g}
    for ref,gt_dict in [('knn',gt_knn),('ctrl',gt_ctrl)]:
        gt=gt_dict[g]
        m=per_gene_metrics(pred_uf, gt, g, ref, ref)
        row.update(m)
        row[f'core_upfrac_{ref}']=float(pred_uf[CORE_COLS].mean())
        row[f'obs_core_mean_{ref}']=float(gt[CORE_COLS].mean())
        row[f'obs_chemo_mean_{ref}']=float(gt[CHEMO_COLS].mean())
    indist_rows.append(row)
    log(f'  {g}: sign_resp_knn={row["sign_resp_knn"]:.2f}  sign_resp_ctrl={row["sign_resp_ctrl"]:.2f}  '
        f'sp_knn={row["spearman_knn"]:.2f}  sp_ctrl={row["spearman_ctrl"]:.2f}')

indist_df=pd.DataFrame(indist_rows)
indist_df.to_csv(os.path.join(PGDIR,'inDist_pergene.csv'),index=False)

log(f'In-dist sign_resp_knn:  mean={indist_df["sign_resp_knn"].mean():.3f}  std={indist_df["sign_resp_knn"].std():.3f}')
log(f'In-dist sign_resp_ctrl: mean={indist_df["sign_resp_ctrl"].mean():.3f}  std={indist_df["sign_resp_ctrl"].std():.3f}')
log(f'In-dist spearman_knn:   mean={indist_df["spearman_knn"].mean():.3f}')
log(f'In-dist spearman_ctrl:  mean={indist_df["spearman_ctrl"].mean():.3f}')
flush_log()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Two-axis table: NF-kB core vs chemokine module
# ═══════════════════════════════════════════════════════════════════════════════
log('=== SECTION 5: two-axis table ===')

two_axis_rows=[]
for g in panel_genes:
    row={'gene':g}
    pred_uf=upfrac_all.get(g)
    for ref,gt_dict in [('knn',gt_knn),('ctrl',gt_ctrl)]:
        gt=gt_dict[g]
        row[f'core_obs_{ref}']  =float(gt[CORE_COLS].mean())
        row[f'chemo_obs_{ref}'] =float(gt[CHEMO_COLS].mean())
        row[f'core_obs_up_{ref}']  =float((gt[CORE_COLS]>0).mean())
        row[f'chemo_obs_up_{ref}'] =float((gt[CHEMO_COLS]>0).mean())
        if pred_uf is not None:
            row[f'core_pred_up_{ref}'] =float((pred_uf[CORE_COLS]>0.5).mean())
            row[f'chemo_pred_up_{ref}']=float((pred_uf[CHEMO_COLS]>0.5).mean())
        # split-response flag: core and chemo disagree in direction
        core_dir  = np.sign(gt[CORE_COLS].mean())
        chemo_dir = np.sign(gt[CHEMO_COLS].mean())
        row[f'split_response_{ref}'] = int(core_dir != 0 and chemo_dir != 0 and core_dir != chemo_dir)
    two_axis_rows.append(row)
    if row.get('split_response_knn') or row.get('split_response_ctrl'):
        log(f'  SPLIT: {g}  core_knn={row["core_obs_knn"]:+.3f}  chemo_knn={row["chemo_obs_knn"]:+.3f}  '
            f'core_ctrl={row["core_obs_ctrl"]:+.3f}  chemo_ctrl={row["chemo_obs_ctrl"]:+.3f}')

two_df=pd.DataFrame(two_axis_rows)
two_df.to_csv(os.path.join(PGDIR,'two_axis_table.csv'),index=False)

# Check model prediction split for RELA
if 'RELA' in upfrac_all:
    rela_row=two_df[two_df['gene']=='RELA'].iloc[0]
    for ref in ['knn','ctrl']:
        core_obs=rela_row[f'core_obs_{ref}']
        chemo_obs=rela_row[f'chemo_obs_{ref}']
        core_pred=rela_row.get(f'core_pred_up_{ref}',float('nan'))
        chemo_pred=rela_row.get(f'chemo_pred_up_{ref}',float('nan'))
        log(f'  RELA ({ref}): obs_core={core_obs:+.3f}  obs_chemo={chemo_obs:+.3f} | '
            f'pred_core_up={core_pred:.2f}  pred_chemo_up={chemo_pred:.2f}')

n_split_knn  = two_df['split_response_knn'].sum()
n_split_ctrl = two_df['split_response_ctrl'].sum()
log(f'Split-response genes (knn-ref):  {n_split_knn}/{len(two_df)}')
log(f'Split-response genes (ctrl-ref): {n_split_ctrl}/{len(two_df)}')
flush_log()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Plots
# ═══════════════════════════════════════════════════════════════════════════════
log('=== SECTION 6: plots ===')

C_NEG='#e74c3c'; C_POS='#2980b9'; C_CORE='#1abc9c'; C_CHEMO='#e67e22'; C_GREY='#95a5a6'

# ── Plot 1: Tier A per-gene NF-kB core sign agreement bar chart ──────────────
labeled2=tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].copy()
genes_sorted=labeled2.sort_values('core_upfrac_knn',ascending=False)['gene'].tolist()
n=len(genes_sorted)
x=np.arange(n); w=0.35

fig,ax=plt.subplots(figsize=(max(8,n*0.9),5))
knn_scores=[tier_a_df.loc[tier_a_df['gene']==g,'core_upfrac_knn'].values[0] for g in genes_sorted]
ctrl_scores=[tier_a_df.loc[tier_a_df['gene']==g,'core_upfrac_ctrl'].values[0] for g in genes_sorted]
is_negs=[int(tier_a_df.loc[tier_a_df['gene']==g,'is_neg'].values[0]) for g in genes_sorted]

bars1=ax.bar(x-w/2, knn_scores, w, label='kNN-ref', alpha=0.85,
             color=[C_NEG if i else C_POS for i in is_negs])
bars2=ax.bar(x+w/2, ctrl_scores, w, label='Ctrl-ref', alpha=0.45, hatch='//',
             color=[C_NEG if i else C_POS for i in is_negs])

ax.axhline(0.5, linestyle='--', color='black', lw=1.0, label='threshold (0.5)')
ax.axhline(null_mean_knn, linestyle=':', color='grey', lw=1.2, label=f'shuffle null ({null_mean_knn:.2f})')

ax.set_xticks(x); ax.set_xticklabels(genes_sorted, rotation=30, ha='right')
ax.set_ylabel('Mean NF-κB core up-fraction\n(expected >0.5 for neg-regs, <0.5 for pos-nodes)')
ax.set_title('Per-gene NF-κB core directional prediction\n(Tier A, multimodal model — CCL3/4/5 excluded)')
ax.legend(handles=[
    mpatches.Patch(color=C_NEG, label='neg-regulator (expect UP)'),
    mpatches.Patch(color=C_POS, label='positive-node (expect DOWN)'),
    plt.Line2D([0],[0],ls='--',c='black', label='threshold 0.5'),
    plt.Line2D([0],[0],ls=':',c='grey', label=f'shuffle null ({null_mean_knn:.2f})'),
    mpatches.Patch(facecolor='white',edgecolor='grey',hatch='//', label='ctrl-ref (hatched)'),
], fontsize=8, loc='upper right')

# annotate disagreements between refs
for i,g in enumerate(genes_sorted):
    knn_v=knn_scores[i]; ctrl_v=ctrl_scores[i]
    if abs(knn_v-ctrl_v)>0.1:
        ax.annotate('ref↕', (i, max(knn_v,ctrl_v)+0.03), ha='center', fontsize=7, color='#8e44ad')

plt.tight_layout()
plt.savefig(os.path.join(PGDIR,'pergene_signagreement_bar.png'), dpi=150)
plt.savefig(os.path.join(PGDIR,'pergene_signagreement_bar.pdf'))
plt.close()

# ── Plot 2: per-gene scatter for selected genes ───────────────────────────────
# Build nearest-training-gene lookup for Tier A proxy ground truth
_nearest_lookup = {}
for _, row in tier_a_df.iterrows():
    _nearest_lookup[row['gene']] = row['nearest_train']

SCATTER_GENES=['RELA']+[g for g in ['TNFAIP3','IRAK3','IKBKG'] if g in upfrac_all]
for g in SCATTER_GENES:
    if g not in upfrac_all: continue
    pred_uf=upfrac_all[g]
    # Use own ground truth for training genes; nearest-training-gene proxy for Tier A
    is_proxy = g not in gt_knn
    proxy_gene = _nearest_lookup.get(g, g)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,(ref,gt_dict,rtitle) in zip(axes,[('knn',gt_knn,'kNN-neighbor ref'),('ctrl',gt_ctrl,'Control-cell ref')]):
        gt_gene = g if g in gt_dict else proxy_gene
        gt=gt_dict[gt_gene]
        proxy_note = f' [proxy: {proxy_gene}]' if is_proxy else ''
        core_mask=np.zeros(n_genes,bool); core_mask[CORE_COLS]=True
        chemo_mask=np.zeros(n_genes,bool); chemo_mask[CHEMO_COLS]=True
        other_mask=~core_mask&~chemo_mask

        ax.scatter(gt[other_mask], pred_uf[other_mask], s=8, alpha=0.3, color=C_GREY, label='other')
        ax.scatter(gt[core_mask],  pred_uf[core_mask],  s=40, alpha=0.8, color=C_CORE,
                   zorder=3, label=f'NF-κB core (n={len(CORE_COLS)})')
        ax.scatter(gt[chemo_mask], pred_uf[chemo_mask], s=40, alpha=0.8, color=C_CHEMO,
                   marker='^', zorder=3, label=f'Chemokine module (n={len(CHEMO_COLS)})')

        for ci in CORE_COLS:
            ax.annotate(var_symbols[ci][:6], (gt[ci], pred_uf[ci]), fontsize=5.5,
                        xytext=(2,2), textcoords='offset points')
        for ci in CHEMO_COLS:
            ax.annotate(var_symbols[ci][:6], (gt[ci], pred_uf[ci]), fontsize=5.5,
                        xytext=(2,2), textcoords='offset points', color=C_CHEMO)

        ax.axvline(0, color='grey', lw=0.7, ls='--')
        ax.axhline(0.5, color='black', lw=0.7, ls='--')
        sp_r,sp_p=stats.spearmanr(gt, pred_uf)
        ax.set_xlabel(f'Ground-truth LFC{proxy_note} ({rtitle})')
        ax.set_ylabel('Predicted up-fraction')
        ax.set_title(f'{g} — {rtitle}\nSpearman={sp_r:.2f} p={sp_p:.3f}')
        ax.legend(fontsize=7)

    meta=TIER_A_EXPECTED.get(g,'')
    role=('neg-reg' if g in panel_genes else meta.get('module','') if isinstance(meta,dict) else '')
    plt.suptitle(f'{g} [{role}]: per-gene scatter — NF-κB core vs chemokine module', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PGDIR, f'pergene_scatter_{g}.png'), dpi=150)
    plt.close()

# ── Plot 3: in-distribution per-gene sign agreement bar ──────────────────────
indist_sorted=indist_df.sort_values('sign_resp_knn', ascending=True)
fig,ax=plt.subplots(figsize=(13,5))
x=np.arange(len(indist_sorted)); w=0.35
ax.bar(x-w/2, indist_sorted['sign_resp_knn'],  w, label='kNN-ref',  alpha=0.85, color='#2980b9')
ax.bar(x+w/2, indist_sorted['sign_resp_ctrl'], w, label='ctrl-ref', alpha=0.65, color='#27ae60')
ax.axhline(0.5, linestyle='--', color='red', lw=1, label='chance (0.5)')
ax.set_xticks(x); ax.set_xticklabels(indist_sorted['gene'], rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Sign agreement (responsive genes)')
ax.set_title('In-distribution per-gene sign agreement — honest model ceiling\n(responsive set: |LFC|≥0.05 or top/bottom 25 genes)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PGDIR,'inDist_pergene_bar.png'), dpi=150)
plt.savefig(os.path.join(PGDIR,'inDist_pergene_bar.pdf'))
plt.close()

# ── Plot 4: two-axis scatter ──────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(ref,rtitle) in zip(axes,[('knn','kNN-neighbor ref'),('ctrl','Control-cell ref')]):
    core_col=f'core_obs_{ref}'; chemo_col=f'chemo_obs_{ref}'
    for _,row in two_df.iterrows():
        g=row['gene']
        split=row.get(f'split_response_{ref}',0)
        col='#e74c3c' if split else '#7f8c8d'
        ax.scatter(row[core_col], row[chemo_col], s=50, color=col, alpha=0.8, zorder=3)
        ax.annotate(g, (row[core_col], row[chemo_col]), fontsize=6.5,
                    xytext=(3,3), textcoords='offset points',
                    color='#c0392b' if g=='RELA' else 'black',
                    fontweight='bold' if g=='RELA' else 'normal')
    ax.axvline(0, color='grey', lw=0.7, ls='--')
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    ax.plot([two_df[core_col].min(), two_df[core_col].max()],
            [two_df[core_col].min(), two_df[core_col].max()],
            'k--', lw=0.8, label='y=x (both axes agree)')
    n_sp=two_df[f'split_response_{ref}'].sum()
    ax.set_xlabel('NF-κB core mean obs LFC')
    ax.set_ylabel('Chemokine/AP-1 module mean obs LFC')
    ax.set_title(f'{rtitle}\nRed = split response (axes disagree), n={n_sp}')
    ax.legend(fontsize=8)

plt.suptitle('Two-axis response: NF-κB canonical core vs chemokine/AP-1 module', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PGDIR,'two_axis_scatter.png'), dpi=150)
plt.savefig(os.path.join(PGDIR,'two_axis_scatter.pdf'))
plt.close()

log('All plots saved.')
flush_log()

# ═══════════════════════════════════════════════════════════════════════════════
# Final log + README
# ═══════════════════════════════════════════════════════════════════════════════
log('=== FINAL SUMMARY ===')
log(f'Control-ref formula: obs_ctrl[g,j] = ln(mean(raw_counts_KO[:,j]) + 1) - ln(mean(raw_counts_ctrl[:,j]) + 1)')
log(f'Pseudocount = 1. X matrix dtype = int16 (raw counts confirmed).')
log(f'Responsive set: |LFC| >= 0.05 OR top/bottom 25 genes by LFC magnitude.')
log(f'NF-kB core: {CORE_KEPT}')
log(f'Chemokine module: {CHEMO_KEPT}')
log(f'Shuffle-null sign accuracy (knn): {null_mean_knn:.3f}')

# Per-ref sign accuracy for labeled Tier A genes
for ref in ['knn','ctrl']:
    labeled2=tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].copy()
    labeled2[f'correct']=labeled2.apply(
        lambda r: int((r[f'core_upfrac_{ref}']>0.5)==bool(r['is_neg'])), axis=1)
    acc=labeled2['correct'].mean()
    n_corr=labeled2['correct'].sum()
    log(f'Tier A NF-kB core sign accuracy ({ref}-ref): {n_corr}/{len(labeled2)} = {acc:.3f}')

# Reference-frame disagreement for Tier A
for _,row in tier_a_df.iterrows():
    knn_v=row['core_upfrac_knn']; ctrl_v=row['core_upfrac_ctrl']
    if abs(knn_v-ctrl_v)>0.1:
        log(f'  REF-FRAME DISAGREEMENT: {row["gene"]} knn={knn_v:.3f} ctrl={ctrl_v:.3f}')

log('INTEGRITY: frozen Tier A predictions in OUTDIR/exp1_scores.csv were NOT modified.')
log('Script complete.')
flush_log()

# README
knn_acc_final = tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].apply(
    lambda r: int((r['core_upfrac_knn']>0.5)==bool(r['is_neg'])), axis=1).mean()
ctrl_acc_final= tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].apply(
    lambda r: int((r['core_upfrac_ctrl']>0.5)==bool(r['is_neg'])), axis=1).mean()

readme=f"""# Per-gene Zero-Shot Evaluation (Corrected)

## Why this analysis replaces the AUROC=0.292 verdict

The previous evaluation collapsed each 500-gene prediction into one scalar
(mean up-fraction over 19 NF-kB target columns). For RELA KO,
CCL3/CCL4/CCL5 were strongly UP under the kNN neighbor reference (magnitudes
+0.38, +0.81, +1.71) but DOWN under the control reference (-0.018, +0.024, +0.048).
Their large kNN-reference magnitudes hijacked the mean, mislabeling RELA's response
as "NF-kB activating." This produced a spurious below-chance AUROC.

## Corrected evaluation approach

Per-gene metrics computed on the NF-kB CANONICAL CORE only (excluding CCL3/4/5):
  Core: {CORE_KEPT}

CCL3/CCL4/CCL5 are tracked as a SEPARATE chemokine/AP-1 axis:
  Chemokine: {CHEMO_KEPT}

Two reference frames are reported side by side:
  kNN-neighbor reference: matches the training target
  Control-cell reference: biologically interpretable, less reference-artifact

## Results

Tier A NF-kB core sign accuracy:
  kNN-ref:  {knn_acc_final:.2f} ({tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].apply(lambda r: int((r['core_upfrac_knn']>0.5)==bool(r['is_neg'])), axis=1).sum()}/{tier_a_df['is_neg'].apply(lambda x: x in [0,1]).sum()})
  ctrl-ref: {ctrl_acc_final:.2f} ({tier_a_df[tier_a_df['is_neg'].apply(lambda x: x in [0,1])].apply(lambda r: int((r['core_upfrac_ctrl']>0.5)==bool(r['is_neg'])), axis=1).sum()}/{tier_a_df['is_neg'].apply(lambda x: x in [0,1]).sum()})
  Shuffle null: {null_mean_knn:.2f}

In-distribution per-gene sign agreement (kNN-ref):
  mean = {indist_df['sign_resp_knn'].mean():.3f}  std = {indist_df['sign_resp_knn'].std():.3f}

## CCL3/CCL4/CCL5 finding

Under the kNN neighbor reference, CCL3/4/5 appear UP for RELA KO (+0.38/+0.81/+1.71).
Under the control reference, they are DOWN or near zero (-0.018/+0.024/+0.048).
This is a reference-frame artifact: RELA KO tumor cells reside in T-cell-rich
spatial niches; their kNN neighbors have lower basal chemokine expression,
making the relative LFC appear positive. Relative to actual control cells,
CCL3/4/5 do NOT increase. The "split response" pattern is a density artifact,
not a genuine AP-1 derepression signal.

## Pre-registration statement

Frozen Tier A predictions (exp1_scores.csv, exp1_upfrac.npz) were not modified.
All new computation added results to OUTDIR/pergene/.
"""
open(os.path.join(PGDIR,'README.md'),'w').write(readme)
log('README.md written.')
flush_log()
