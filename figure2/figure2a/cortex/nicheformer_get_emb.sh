#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50G
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o nicheformer_get_emb.out
#SBATCH -e nicheformer_get_emb.out

/usr/bin/env bash
# Optional: prevent Lightning from picking up cluster env plugins if not desired
export LIGHTNING_DISABLE_ENV_PLUGINS=1

/users/yzhong36/data_wfairbro/yzhong36/spatial_foundation/nicheformer_env/bin/python ./nicheformer_get_emb.py