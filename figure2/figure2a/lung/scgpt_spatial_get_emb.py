import sys
sys.path.append("/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/software/scGPT-spatial")
import scgpt_spatial
import scanpy as sc
import numpy as np
import os

save_path='/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/scripts/CosMx/CosMx_Lung_benchmark/baseline_dir/scgpt_spatial_emb'
os.makedirs(save_path, exist_ok=True)

adata=sc.read("/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/datasets/CosMx/CosMx_Human_Lung.h5ad")

model_dir='/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/datasets/scGPT_spatial_v1'
gene_col='index'

for subset_name in adata.obs["Run_Tissue_name"].unique():
    print(f"Processing subset: {subset_name}")
    adata_subset=adata[adata.obs["Run_Tissue_name"] == subset_name].copy()
    adata_subset.X=np.asarray(adata_subset.X.todense())

    embed_adata = scgpt_spatial.tasks.embed_data(
                                                adata_subset,
                                                model_dir,
                                                gene_col=gene_col,
                                                batch_size=64,
                                                return_new_adata=False,
                                            )
    embed_adata.write_h5ad(save_path+ f"/{os.path.basename(subset_name)}.h5ad")                                        