#!/bin/bash

#SBATCH -o corpus_full_05_19_wl_8_segmentation_noise.out
#SBATCH -e corpus_full_05_19_wl_8_segmentation_noise.out
#SBATCH --cpus-per-gpu=5
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --account PAS1475

# Force NCCL to use PCIe / NVLink, not InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Enable NCCL debug for detailed logging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Optional: force NCCL to use only GPUs visible to Slurm
export CUDA_VISIBLE_DEVICES=0

file_path='/fs/ess/PAS1475/yzhong/sf_project/datasets/segmentation_noise_new'
model_path='/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26'
output_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Lung_noise_benchmark/g2pm_dir/05_19_26'

wl=8

echo "Processing corpus_full_317M_05_19_seg_wl_${wl} with noise type: segmentation_noise"

/fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
/fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/inference/CLI_inference.py \
    --model-pattern $model_path/run_scgpt_f_human_full_317M*/step_150000.pt \
    --graph-paths \
    $file_path/CosMx_Human_Lung5_Rep1_noise_processed_og_graph_scgpt_test.pt \
    $file_path/CosMx_Human_Lung9_Rep1_noise_processed_og_graph_scgpt_test.pt \
    $file_path/CosMx_Human_Lung6_noise_processed_og_graph_scgpt_test.pt \
    --save-path $output_path/corpus_full_317M_05_19_wl_${wl}_segmentation_noise \
    --gpu-id 0 \
    --walk-length $wl \
    --batch-size 256 \
    --use-amp

