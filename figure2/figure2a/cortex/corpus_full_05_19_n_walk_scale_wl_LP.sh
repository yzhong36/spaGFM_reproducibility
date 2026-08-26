#!/bin/bash

#SBATCH -o corpus_full_05_19_n_walk_scale_wl_LP.out
#SBATCH -e corpus_full_05_19_n_walk_scale_wl_LP.out
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --account PAS1475

file_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Cortex_benchmark/g2pm_dir/05_19_26'

wl=8
for n_walk in 16 32 64 128
do

    for i in 317M
    do
        echo "Processing corpus_full_${i}_05_19_wl_${wl} with n_walks=${n_walk}"

        /fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
        /fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/evaluation/CLI_evaluation.py \
        --embedding-dir $file_path/corpus_full_${i}_05_19_wl_${wl}_n_walk_${n_walk} \
        --save-path $file_path/corpus_full_${i}_05_19_wl_${wl}_n_walk_${n_walk}_LP_model_niche \
        --output-suffix spaGFM.pt \
        --method linear_probe_node \
        --label-attr niche \
        --embedding-attr subgraph_emb \

    done

done

for n_walk in 16 32 64 128
do

    for i in 317M
    do
        echo "Processing corpus_full_${i}_05_19_wl_${wl} with n_walks=${n_walk}"

        /fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
        /fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/evaluation/CLI_evaluation.py \
        --embedding-dir $file_path/corpus_full_${i}_05_19_wl_${wl}_n_walk_${n_walk} \
        --save-path $file_path/corpus_full_${i}_05_19_wl_${wl}_n_walk_${n_walk}_LP_model_ct \
        --output-suffix spaGFM.pt \
        --method linear_probe_node \
        --label-attr cell_type \
        --embedding-attr node_emb \

    done

done