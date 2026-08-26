import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import anndata as ad
from typing import Optional, Dict, Any
from tqdm import tqdm
import pandas as pd
from nicheformer.models import Nicheformer
from nicheformer.data import NicheformerDataset
import scanpy as sc
import pandas as pd

save_path='/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/scripts/CosMx/CosMx_Lung_benchmark/baseline_dir/nicheformer_emb'
os.makedirs(save_path, exist_ok=True)
adata=sc.read("/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/datasets/CosMx/CosMx_Human_Lung.h5ad")

model_means = ad.read_h5ad('/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/software/nicheformer/data/model_means/model.h5ad')

for subset_name in adata.obs["Run_Tissue_name"].unique():
    print(f"Processing subset: {subset_name}")
    adata_subset=adata[adata.obs["Run_Tissue_name"] == subset_name].copy()

    config = {
    'technology_mean_path': '/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/software/nicheformer/data/model_means/cosmx_mean_script.npy', #'path/to/technology_mean.npy',  # Path to technology mean file
    'checkpoint_path': '/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/datasets/nicheformer/nicheformer.ckpt',  # Path to model checkpoint
    'output_path': save_path+ f"/{os.path.basename(subset_name)}.h5ad",  # Where to save the result, it is a new h5ad
    'output_dir': save_path,  # Directory for any intermediate outputs
    'batch_size': 32,
    'max_seq_len': 1500, 
    'aux_tokens': 30, 
    'chunk_size': 1000, # to prevent OOM
    'num_workers': 4,
    'precision': 32,
    'embedding_layer': -1,  # Which layer to extract embeddings from (-1 for last layer)
    'embedding_name': 'embeddings'  # Name suffix for the embedding key in adata.obsm
    }

    pl.seed_everything(42)

    gene2symbol = pd.read_csv("/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/datasets/scGPT_human/gene_info.csv")
    technology_mean = np.load(config['technology_mean_path'])

    assert {'feature_name', 'feature_id'}.issubset(gene2symbol.columns), (
    f"gene2symbol must have columns 'feature_name' and 'feature_id', got: {list(gene2symbol.columns)}"
    )

    # Clean up mapping table
    _map_df = (
        gene2symbol[['feature_name', 'feature_id']]
        .dropna()
        .astype({'feature_name': str, 'feature_id': str})
    )
    _map_df['feature_name'] = _map_df['feature_name'].str.strip()
    _map_df['feature_id'] = _map_df['feature_id'].str.strip()

    # Record duplicate symbols present in mapping (for info)
    _dup_count = _map_df.duplicated(subset='feature_name', keep=False).sum()
    # Deduplicate by keeping the first occurrence of each symbol
    _map_df = _map_df.drop_duplicates(subset='feature_name', keep='first')

    sym_to_ensg = dict(zip(_map_df['feature_name'], _map_df['feature_id']))

    # Preserve original symbols
    orig_symbols = adata_subset.var_names.copy()
    adata_subset.var['feature_name'] = orig_symbols

    # Map symbols to ENSG
    mapped = orig_symbols.to_series().map(sym_to_ensg)
    mask = mapped.notna()
    kept = int(mask.sum())
    dropped = int(len(orig_symbols) - kept)

    if kept == 0:
        raise ValueError("No genes could be mapped to ENSG IDs. Check that adata.var_names contain gene symbols matching gene2symbol.feature_name.")

    # Subset to mapped genes only
    adata_subset = adata_subset[:, mask.values].copy()

    # Assign ENSG IDs as var_names for the retained genes
    ensg_names = mapped[mask].astype(str)
    adata_subset.var_names = ensg_names.to_numpy()

    # Ensure uniqueness if needed
    if not pd.Index(adata_subset.var_names).is_unique:
        adata_subset.var_names_make_unique()

    # Store ENSG ids in var for convenience
    adata_subset.var['feature_id'] = adata_subset.var_names

    print(f"Kept {kept} genes with ENSG mapping; dropped {dropped} unmapped.")
    print(f"Duplicate 'feature_name' entries in mapping table: {_dup_count} (kept first occurrence).")
    # Quick peek
    adata_subset.var[['feature_name', 'feature_id']].head()

    # format data properly with the model
    adata_subset = ad.concat([model_means, adata_subset], join='outer', axis=0)
    # dropping the first observation
    adata_subset = adata_subset[1:].copy()

    # Change accordingly

    adata_subset.obs['modality'] = 4 # spatial
    adata_subset.obs['specie'] = 5 # human
    adata_subset.obs['assay'] = 8 #cosmx

    adata_subset.obs['nicheformer_split'] = 'train'

    # Create dataset
    dataset = NicheformerDataset(
        adata=adata_subset,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=1500,
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000),
        metadata_fields={'obs': ['modality', 'specie', 'assay']}
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

        # Load pre-trained model
    model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'], strict=False)
    model.eval()  # Set to evaluation mode

    # Configure trainer
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        default_root_dir=config['output_dir'],
        precision=config.get('precision', 32)
    )

    print("Extracting embeddings...")
    embeddings = []
    device = model.embeddings.weight.device

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get embeddings from the model
            emb = model.get_embeddings(
                batch=batch,
                layer=config.get('embedding_layer', -1)  # Default to last layer
            )
            embeddings.append(emb.cpu().numpy())


    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)

    # Store embeddings in AnnData object
    embedding_key = f"X_niche_{config.get('embedding_name', 'embeddings')}"
    adata_subset.obsm[embedding_key] = embeddings

    # Save updated AnnData
    adata_subset.write_h5ad(config['output_path'])

    print(f"Embeddings saved to {config['output_path']} in obsm['{embedding_key}']")
