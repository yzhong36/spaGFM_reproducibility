import scanpy as sc
import os.path as Path
import torch


folder = r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data'

dataset_names = ['LC1', 'LC2', 'LC3', 'LC4', 'LC5']

for dataset_name in dataset_names:


    original_data = sc.read_h5ad(Path.join(folder, f"{dataset_name}.h5ad"))
    finetune_data = sc.read_h5ad(Path.join(folder, "LC_finetune.h5ad"))
    finetune_data = finetune_data[finetune_data.obs["batch"] == dataset_name]
    finetune_data.obs_names = finetune_data.obs_names.str.removesuffix(f'-{dataset_name}')

    original_data.obsm['X_finetune_scGPT'] = finetune_data[original_data.obs_names].obsm['X_scGPT']
    original_data.write_h5ad(Path.join(folder, f"{dataset_name}.h5ad"))
    del finetune_data

    # read corresponding pt to replace with finetuned scGPT and then save it back
    original_pt = torch.load(Path.join(folder, f"{dataset_name}.pt"), map_location='cpu', weights_only=False)

    original_pt.x = torch.tensor(original_data.obsm['X_finetune_scGPT'] , dtype=torch.float)

    torch.save(original_pt, Path.join(folder, f"{dataset_name}_finetune_scGPT.pt"))

    del original_data, original_pt

