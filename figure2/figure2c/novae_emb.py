import novae
import os
import glob
import scanpy as sc

file_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/Xenium_lung_benchmark/spaGFM_dir/05_19_26/corpus_full_317M_05_19_wl'

adata_files = glob.glob(os.path.join(file_path, '*.h5ad'))

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

for adata_file in adata_files:
    adata = sc.read_h5ad(adata_file)
    adata.obsm["spatial"]=adata.obs[['x_centroid', 'y_centroid']].values
    novae.spatial_neighbors(adata, radius=80)
    model.compute_representations(adata, zero_shot=True)
    adata.write_h5ad(file_path+ f"/{os.path.basename(adata_file)}")