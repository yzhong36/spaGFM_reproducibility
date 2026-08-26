#!/bin/bash

#SBATCH -o corpus_full_05_19_model_scale_wl_KNN.out
#SBATCH -e corpus_full_05_19_model_scale_wl_KNN.out
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --account PAS1475

file_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Liver_benchmark/g2pm_dir/05_19_26'

for wl in 8 16 32 64
do

    for i in 317M
    do
        echo "Processing corpus_full_${i}_05_19_wl_${wl}"

        /fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
        /fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/evaluation/CLI_evaluation.py \
        --embedding-dir $file_path/corpus_full_${i}_05_19_wl_${wl} \
        --save-path $file_path/corpus_full_${i}_05_19_wl_${wl}_KNN_model_niche \
        --output-suffix spaGFM.pt \
        --method knn_node \
        --label-attr niche \
        --embedding-attr subgraph_emb \
        --metric acc_macro,f1_macro,mcc \

    done

done