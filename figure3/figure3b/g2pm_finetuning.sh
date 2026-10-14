#!/bin/bash

#SBATCH -o g2pm_finetuning_%j.out
#SBATCH -e g2pm_finetuning_%j.err
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=150gb


set -euo pipefail

# Workflow:
# scGPT_spot_finetune.py -> g2pm_finetune_train.py
# -> g2pm_finetune_inference.py -> cross_validation_result.py
# -> plot_cross_validation_benchmark.py
#
# This is an orchestration template for the existing Python scripts. Before use:
# 1. Set the paths below and check model_zoo and ft_params['save_folder'] in
#    g2pm_finetune_train.py. CHECKPOINT_DIR must match that save_folder's parent.
# 2. Prepare fold0 through fold4 under CV_DATA_DIR. Each fold needs train.pt,
#    val.pt, test.pt, and test.h5ad, using the finetuned scGPT spot features.
#    All three graphs need a consistently encoded sub_tls attribute, and the
#    test.h5ad observations must match the test.pt node order. Stage 1 writes
#    an embedding h5ad and a graph; it does not create these CV splits or sub_tls.
#    Prepare the folds separately before running this template, or insert your
#    fold preparation command between stages 1 and 2. Regenerate existing folds
#    there if they must use the embeddings produced by this particular run.
# 3. Adapt corss_validation_result.py (the current filename has a typo) to read
#    CV_DATA_DIR/fold{0..4}/test.h5ad, using sub_tls as the target and
#    {model_name}_{walk_length}_prediction as the prediction column. Its current
#    implementation instead reads a hard-coded, older aggregate h5ad file.
#    Choose the model/walk length using validation data before test evaluation.
# 4. Update the metric dictionaries in plot_cross_validation_benchmark.py from
#    stage 4's results and set its plt.savefig path to your desired PNG path.
#    The current plot script does not automatically consume evaluation output.
#
# Run locally: bash /path/to/figure3/figure3b/g2pm_finetuning.sh
# Submit to Slurm from figure3/figure3b: sbatch g2pm_finetuning.sh
# Activate an environment containing scGPT (including its PEFT utilities),
# stRoamer, and the other Python dependencies, or set PYTHON_BIN below.

# Force NCCL to use PCIe / NVLink, not InfiniBand.
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
# Preserve the device allocation supplied by Slurm when present.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND=Agg

# Under sbatch, BASH_SOURCE points to a spool copy; use the submission directory.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-/fs/ess/PAS1475/yzhong/sf_project/random_test/spaGFM_dev/06_29_26/spaGFM_env/bin/python}"
DATA_DIR="${DATA_DIR:-/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-/fs/ess/PAS1475/Fei/code/G2PM_finetune/scgpt/save/scGPT_human}"
TF_FILE="${TF_FILE:-/fs/ess/PAS1475/Fei/code/G2PM_finetune/human_transcription_factors.txt}"
GRAPH_UTILS_DIR="${GRAPH_UTILS_DIR:-/fs/ess/PAS1475/Fei/code/G2PM_finetune/stRoamer/utils}"

DATASET_NAME="${DATASET_NAME:-LC}"
ADAPTER="${ADAPTER:-HYBRID}" # False, HYBRID, ENCODER, TOKEN, PREFIX, or LORA
GRAPH_OUTPUT_DIR="${GRAPH_OUTPUT_DIR:-${DATA_DIR}/finetune_scGPT}"
CV_DATA_DIR="${CV_DATA_DIR:-${DATA_DIR}/LC_5_fold}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/fs/ess/PAS1475/Fei/data/spaGFM/TLS_results/5fold_CV}"
BATCH_SIZE="${BATCH_SIZE:-80}"
LR="${LR:-0.005}"
FROZEN_BACKBONE="${FROZEN_BACKBONE:-false}"
# Training currently runs all three models, each with walk lengths 1 through 8.
MODEL_NAMES=(3M 15M 36M)

BACKBONE_ARGS=()
case "${FROZEN_BACKBONE}" in
    true) BACKBONE_ARGS+=(--frozen_backbone) ;;
    false) ;;
    *) echo "FROZEN_BACKBONE must be true or false." >&2; exit 1 ;;
esac

cd -- "${SCRIPT_DIR}"

# 1. Fine-tune scGPT, extract X_scGPT, and build the spatial graph.
# Input: DATA_DIR/DATASET_NAME.h5ad (raw counts and required obs columns).
# Output: ./save/finetune_*/ plus GRAPH_OUTPUT_DIR/DATASET_NAME_finetune_scGPT.{pt,h5ad}.
echo "[1/5] Fine-tune scGPT spot embeddings and build graph"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scGPT_spot_finetune.py" \
    "${ADAPTER}" "${DATASET_NAME}" \
    --data-dir "${DATA_DIR}" \
    --load-model "${SCGPT_MODEL_DIR}" \
    --tf-file "${TF_FILE}" \
    --graph-utils-dir "${GRAPH_UTILS_DIR}" \
    --graph-output-dir "${GRAPH_OUTPUT_DIR}" \
    --x-key x_pixel \
    --y-key y_pixel \
    --niche-key binary_tls \
    --tls-type-key manual_anno_tls

# Insert project-specific fold preparation here if needed; see prerequisite 2.

# 2. Fine-tune spaGFM on each prepared training/validation fold.
# Output: CHECKPOINT_DIR/foldN_{frozen,unfrozen}_b80_lr0.005_checkpoint/MODEL/WALK/.
echo "[2/5] Train spaGFM on five folds"
for fold in {0..4}; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/g2pm_finetune_train.py" \
        "${CV_DATA_DIR}/fold${fold}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        "${BACKBONE_ARGS[@]}"
done

# 3. Inference handles all five test folds and all eight walk lengths per call.
# Output: prediction and prediction_probability columns added to each test.h5ad.
# The inference script currently runs on CPU.
echo "[3/5] Predict test labels for each model"
for model_name in "${MODEL_NAMES[@]}"; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/g2pm_finetune_inference.py" \
        "${CV_DATA_DIR}" "${CHECKPOINT_DIR}" \
        --model_name "${model_name}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        "${BACKBONE_ARGS[@]}"
done

# 4. Summarize precision, recall, MCC, and macro F1 across folds.
# Supports the requested spelling if the existing file is renamed later.
echo "[4/5] Summarize cross-validation results"
RESULT_SCRIPT="${SCRIPT_DIR}/cross_validation_result.py"
if [[ ! -f "${RESULT_SCRIPT}" ]]; then
    RESULT_SCRIPT="${SCRIPT_DIR}/corss_validation_result.py"
fi
"${PYTHON_BIN}" "${RESULT_SCRIPT}"

# 5. Plot the benchmark with the scores and output path configured beforehand.
echo "[5/5] Plot cross-validation benchmark"
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_cross_validation_benchmark.py"
