import scanpy as sc
import os.path as Path
import anndata as ad
import os


ref_data = sc.read_h5ad(r'/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/KC/train/train.h5ad')
file_folder = r'/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/KC/test'
predict_binary_map = {0:0, 1:0, 2:2}
for file_name in os.listdir(file_folder):
    if not file_name.endswith('.h5ad'):
        continue
    query_adata = sc.read_h5ad(Path.join(file_folder, file_name))

    # 1. Find the intersection of var_names
    var_names = ref_data.var_names.intersection(query_adata.var_names)
    print(f"Number of common variables: {len(var_names)}")

    # 2. Subset both objects to the common variables
    adata_ref = ref_data[:, var_names].copy()
    adata = query_adata[:, var_names].copy()

    # Ensure the variables are in the same order (though intersection often handles this, it's good practice)
    adata = adata[:, adata_ref.var_names]

    sc.pp.neighbors(adata_ref)

    transfer_adata = sc.tl.ingest(adata, adata_ref, embedding_method='pca', obs="label_id",inplace= False)
    adata.obs['PCA_predict'] = transfer_adata.obs['label_id']
    # map PCA prediction into binary

    adata.obs['PCA_binary_predict'] = adata.obs['PCA_predict'].map(predict_binary_map)
    adata.write_h5ad(Path.join(file_folder, file_name))
    del transfer_adata, adata_ref,adata