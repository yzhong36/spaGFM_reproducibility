#!/bin/bash

#SBATCH -o corpus_full_05_19_wl_8_segmentation_noise_LP_niche.out
#SBATCH -e corpus_full_05_19_wl_8_segmentation_noise_LP_niche.out
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --account PAS1475

file_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Lung_noise_benchmark/g2pm_dir/05_19_26'

wl=8

echo "Processing corpus_full_317M_05_19_wl_${wl} with segmentation noise"

/fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
/fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/evaluation/CLI_evaluation.py \
--embedding-dir $file_path/corpus_full_317M_05_19_wl_${wl}_segmentation_noise \
--save-path $file_path/corpus_full_317M_05_19_wl_${wl}_segmentation_noise_LP_model_niche \
--output-suffix spaGFM.pt \
--method linear_probe_node \
--label-attr niche \
--embedding-attr subgraph_emb \
