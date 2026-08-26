import scanpy as sc
import os
import torch
import sys
sys.path.append('/fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial')
from stRoamer.utils.data_build import initial_pyG_graph
from stRoamer.inference.run_inference import load_checkpoint, run_inference

adata=sc.read('/fs/ess/PAS1475/yzhong/sf_project/datasets/xenium/GSE250346/GSE250346.h5ad')

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
model_paras = load_checkpoint(path='/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_317M_8_2026-05-23_10-24-41_balanced_quartz/step_150000.pt', 
                              device=device)

save_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/Xenium_lung_benchmark/spaGFM_dir/05_19_26'
os.makedirs(save_path+f'/corpus_full_317M_05_19_wl', exist_ok=True)

for sample in adata.obs['sample'].unique():

    tmp_adata = adata[adata.obs['sample'] == sample]
    tmp_adata=initial_pyG_graph(
    tmp_adata,
    model_dir='/fs/ess/PAS1475/yzhong/sf_project/base_model/scFM/scGPT_human',
    gene_col='index',
    batch_size=64,
    coord_x='x_centroid',
    coord_y='y_centroid',
    )

    for wl in [8, 64]:
        model_paras['params']['walk_length']=wl
        tmp_adata = run_inference(
                    tmp_adata.copy(),
                    model=model_paras,
                    device=device,
                    batch_size=16,
                    use_amp=True
                )
    tmp_adata.write(save_path+f'/corpus_full_317M_05_19_wl/{sample}.h5ad')
