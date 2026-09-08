import os.path as Path
# import sys
import scanpy as sc
# from g2pm_finetune_train import file_path
from stRoamer.inference.run_inference_ft import run_inference_ft
import torch
import numpy as np
import argparse
import pandas as pd
import anndata as ad


# Create the ArgumentParser object with a description
parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")

# Add a positional argument (required)
parser.add_argument("test_file_folder", type=str, help="The absolute folder of the test data files.")
parser.add_argument("checkpoint_folder", type=str, help="The absolute folder saving all the checkpoints.")
parser.add_argument("--model_name", type=str,choices=['3M', '15M','36M', '317M'], default='317M', help="The version name of spaGFM.")
parser.add_argument("--frozen_backbone", action="store_true", help="The absolute folder of the input training pt files.")
parser.add_argument("--batch_size", type= int, default=80, help="The absolute folder of the input training pt files.")
parser.add_argument("--lr", type= float, default=0.005, help="The absolute folder of the input training pt files.")

# parser.add_argument("train_dataset", type=str, help="The name of finetuning dataset.")
#
# parser.add_argument("val_dataset", type=str, help="The name of validation dataset.")

# Parse the arguments from the command line
args = parser.parse_args()
test_file_folder = args.test_file_folder
checkpoint_folder = args.checkpoint_folder
model_name = args.model_name
batch_size = args.batch_size
lr = args.lr
# train_dataset = args.train_dataset
# val_dataset = args.val_dataset

# gpu_id = 0  # Change to your desired GPU index
# device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# adata = sc.read_h5ad(file_path[:-18] + ".h5ad")


if args.frozen_backbone:
    is_frozen = 'frozen'
else:
    is_frozen = 'unfrozen'


# adata = ad.AnnData()

for fold_num in range(0,5):

    test_graph = torch.load(Path.join(test_file_folder, f'fold{fold_num}', 'test.pt'), map_location = device, weights_only = False)

    test_graph.y = test_graph.sub_tls
    adata = sc.read_h5ad(Path.join(test_file_folder, f'fold{fold_num}', 'test.h5ad'))
    # adata.obs['labels'] = test_graph.y.detach().cpu().numpy()
    # adata.obs['is_assigned'] = test_graph.is_assigned.detach().cpu().numpy()



    # Define the label mapping
    # label_map = {0: 'ETLS', 1: 'MTLS', 2: 'non_TLS'}

    for walk_length in range(1, 9):
        model_file= Path.join(checkpoint_folder,f'fold{fold_num}_{is_frozen}_b{batch_size}_lr{lr}_checkpoint', model_name, str(walk_length), 'best_model.pt')
        attn_weights, logits = run_inference_ft(test_graph,
                                                model_file,
                                                device)
        # np.save(f"/fs/ess/PAS1475/Fei/data/spaGFM/TLS_results/att_{str(walk_length)}_{os.path.basename(file_path)[:-3]}.npy",attn_weights.detach().cpu().numpy())

        pred_prob, pred_label = torch.softmax(logits, dim=1).max(dim=1)
        adata.obs[f"{model_name}_{str(walk_length)}_prediction"] = pred_label.detach().cpu().numpy()
        adata.obs[f"{model_name}_{str(walk_length)}_prediction_probability"] = pred_prob.detach().cpu().numpy()

    adata.write_h5ad(Path.join(test_file_folder, f'fold{fold_num}',f'test.h5ad'))
    del adata, test_graph

