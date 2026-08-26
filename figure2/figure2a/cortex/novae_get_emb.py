import scanpy as sc
import novae
import os
import numpy as np

save_path='/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/scripts/CosMx/CosMx_Cortex_benchmark/baseline_dir/novae_emb/'
os.makedirs(save_path, exist_ok=True)
adata=sc.read("/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/datasets/CosMx/CosMx_Human_FrontalCortex.h5ad")

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

for subset_name in adata.obs["Run_Tissue_name"].unique():
    print(f"Processing subset: {subset_name}")
    adata_subset=adata[adata.obs["Run_Tissue_name"] == subset_name].copy()

    adata_coord = adata_subset.obs[['x_slide_mm', 'y_slide_mm']].values
    adata_coord_trans = adata_coord - np.min(adata_coord, axis=0)
    adata_coord_trans = adata_coord_trans * 1000
    adata_subset.obsm["spatial"]=adata_coord_trans

    novae.spatial_neighbors(adata_subset, radius=80)
    model.compute_representations(adata_subset, zero_shot=True)
    print(adata_subset)
    adata_subset.write_h5ad(save_path+ f"/{os.path.basename(subset_name)}.h5ad")