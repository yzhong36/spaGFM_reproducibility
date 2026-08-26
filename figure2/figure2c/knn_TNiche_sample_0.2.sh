#!/bin/bash

#SBATCH -o knn_TNiche_sample_0.2.out
#SBATCH -e knn_TNiche_sample_0.2.out
#SBATCH --cpus-per-task=30
#SBATCH --time=15:00:00
#SBATCH --mem=100gb
#SBATCH --account PAS1475

/fs/ess/PAS1475/yzhong/sf_project/conda_env/spaGFM_dev/bin/python knn_model.py \
 --input-dir /fs/ess/PAS1475/yzhong/sf_project/scripts/Xenium_lung_benchmark/spaGFM_dir/05_19_26/corpus_full_317M_05_19_wl \
 --output-dir /fs/ess/PAS1475/yzhong/sf_project/scripts/Xenium_lung_benchmark/spaGFM_dir/05_19_26/knn_TNiche_sample_0.2 \
 --file-pattern '*.h5ad' \
 --split-name sample \
 --label-name TNiche \
 --train-sample-ratio 0.2
