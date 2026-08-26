#!/bin/bash

#SBATCH -o baseline_LP_ct.out
#SBATCH -e baseline_LP_ct.out
#SBATCH --cpus-per-task=30
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --account PAS1475

file_path='/fs/ess/PAS1475/yzhong/sf_project/backup/ccv/scripts/CosMx/CosMx_Liver_benchmark/baseline_dir'
save_path='/fs/ess/PAS1475/yzhong/sf_project/scripts/CosMx_Liver_benchmark/baseline_dir/05_19_26'

for base_model in nicheformer_emb novae_emb scgpt_spatial_emb
do
    if [ "$base_model" == "nicheformer_emb" ]; then
        echo "Processing nicheformer_emb"
        embedding_attr='X_niche_embeddings'
    elif [ "$base_model" == "novae_emb" ]; then
        echo "Processing novae_emb"
        embedding_attr='novae_latent'
    elif [ "$base_model" == "scgpt_spatial_emb" ]; then
        echo "Processing scgpt_spatial_emb"
        embedding_attr='X_scGPT'
    fi

    /fs/ess/PAS1475/yzhong/sf_project/conda_env/G2PM/bin/python \
    /fs/ess/PAS1475/yzhong/sf_project/codebase/G2PM_spatial/stRoamer/evaluation/CLI_evaluation.py \
    --embedding-dir $file_path/${base_model} \
    --save-path $save_path/${base_model}_LP_model_ct \
    --output-suffix .h5ad \
    --method linear_probe_node \
    --dataset-attr Run_Tissue_name \
    --label-attr cellType \
    --embedding-attr $embedding_attr \

done