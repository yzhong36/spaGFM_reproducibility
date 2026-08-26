#!/bin/bash

#SBATCH -o corpus_full_05_19_model_scale_wl.out
#SBATCH -e corpus_full_05_19_model_scale_wl.out
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

file_path='/fs/ess/PAS1475/yzhong/sf_project/backup/ccv/datasets/CosMx'
model_path='/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26'
output_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Cortex_benchmark/g2pm_dir/05_19_26'

for wl in 8 16 32 64; do
    echo "Processing walk length ${wl}"

    for i in 3M 15M 36M 317M; do
    echo "Processing corpus_full_${i}_05_19_wl_${wl}"

    /fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
    /fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/inference/CLI_inference.py \
    --model-pattern $model_path/run_scgpt_f_human_full_${i}*/step_150000.pt \
    --graph-paths \
    $file_path/CosMx_Human_FrontalCortex_processed_scgpt_test.pt \
    --save-path $output_path/corpus_full_${i}_05_19_wl_${wl} \
    --gpu-id 0 \
    --walk-length $wl \
    --batch-size 128 \
    --use-amp
    done

done



