#!/bin/bash

#SBATCH -o corpus_full_05_19_wl_8_graph_noise.out
#SBATCH -e corpus_full_05_19_wl_8_graph_noise.out
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

file_path='/fs/ess/PAS1475/yzhong/sf_project/datasets/graph_noise'
model_path='/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26'
output_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Lung_noise_benchmark/g2pm_dir/05_19_26'

wl=8

for noise_type in node_shift node_feat_shuffle
do

    echo "Processing corpus_full_317M_05_19_wl_${wl} with noise type: ${noise_type}"

    /fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
    /fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/inference/CLI_inference.py \
    --model-pattern $model_path/run_scgpt_f_human_full_317M*/step_150000.pt \
    --graph-paths \
    $file_path/*${noise_type}*0_Lung5_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0.1_Lung5_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0.3_Lung5_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0.5_Lung5_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0_Lung9_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0.1_Lung9_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0.3_Lung9_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0.5_Lung9_Rep1_scgpt.pt \
    $file_path/*${noise_type}*0_Lung6_scgpt.pt \
    $file_path/*${noise_type}*0.1_Lung6_scgpt.pt \
    $file_path/*${noise_type}*0.3_Lung6_scgpt.pt \
    $file_path/*${noise_type}*0.5_Lung6_scgpt.pt \
    --save-path $output_path/corpus_full_317M_05_19_wl_${wl}_${noise_type} \
    --gpu-id 0 \
    --walk-length $wl \
    --batch-size 256 \
    --use-amp

done

