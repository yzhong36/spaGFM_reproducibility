#!/bin/bash

#SBATCH -o 05_19_human_wl_8_plot_gpu.out
#SBATCH -e 05_19_human_wl_8_plot_gpu.out
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=150gb
#SBATCH --account PAS1475

# Force NCCL to use PCIe / NVLink, not InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=0


/fs/ess/PAS1475/yzhong/sf_project/random_test/spaGFM_dev/06_29_26/spaGFM_env/bin/python \
/fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/utils/CLI_plot_UMAP.py \
/fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human_wl_8_gpu_plot \
/fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human_wl_8 \
subgraph_emb \
--color-keys tissue platform \
--sample-fraction 0.1 \
--pca-dim 50 \
--save-adata /fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human_wl_8_gpu_plot/05_19_human_wl_8_subgraph_emb_adata.h5ad

/fs/ess/PAS1475/yzhong/sf_project/random_test/spaGFM_dev/06_29_26/spaGFM_env/bin/python \
/fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/utils/CLI_plot_UMAP.py \
/fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human_wl_8_gpu_plot \
/fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human_wl_8 \
node_emb \
--color-keys tissue platform \
--sample-fraction 0.1 \
--pca-dim 50 \
--save-adata /fs/ess/PAS1475/yzhong/sf_project/datasets/global_tissue_benchmark/05_19_human_wl_8_gpu_plot/05_19_human_wl_8_node_emb_adata.h5ad