# %%
import os
import scanpy as sc
import torch
# import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
import sys
script_dir='/fs/ess/PAS1475/Fei/code/G2PM_finetune/stRoamer/utils'
sys.path.append(script_dir)
from graph_build import filter_edge, edge_index_from_delaunay, pyg_obj
# import scgpt.tasks as sct

import argparse

# Create the ArgumentParser object with a description
parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")

# Add a positional argument (required)
parser.add_argument("file_path", help="The absolute path of the input file.")


# Parse the arguments from the command line
args = parser.parse_args()


# %%
file_path = args.file_path
# %%
threshold = 0.99
# %%
# model_dir='/fs/ess/PAS1475/Fei/code/scPEFT-main/tutorial_peft/save/finetune_KC-False-Mar22-05-20'
# gene_col='index'

# adata_scgpt_graph_l = []

adata = sc.read_h5ad(file_path)

# Node features
# adata = sct.embed_data(
#                     adata,
#                     model_dir,
#                     gene_col=gene_col,
#                     batch_size=64,
#                     return_new_adata=False
#                     )
X_scgpt = adata.obsm["X_scGPT"]
adata_scgpt_f = torch.tensor(X_scgpt, dtype=torch.float)

# Build edges from Delaunay + filter
tri = Delaunay(adata.obs[['x_pixel', 'y_pixel']])
adata_edge = filter_edge(tri, threshold=threshold).simplices
edge_index = edge_index_from_delaunay(adata_edge)  # must output [2, E]

# Build pyg graph
adata_scgpt_graph = pyg_obj(adata_scgpt_f, edge_index)

# preprocess the labels into binary class



# # Add labels

adata_scgpt_graph.niche = torch.tensor(
    pd.Categorical(adata.obs['binary_tls']).codes,
    dtype=torch.long
)
adata_scgpt_graph.tls_type = torch.tensor(
    pd.Categorical(adata.obs['manual_anno_tls']).codes,
    dtype=torch.long
)



# %%
torch.save(adata_scgpt_graph, os.path.join('/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data', os.path.basename(file_path)[:-5]+'_finetune_scGPT.pt'))
adata.write_h5ad(file_path[:-5]+'_finetune_scGPT.h5ad')