#!/bin/bash

#SBATCH -o g2pm_independent_finetuning_%j.out
#SBATCH -e g2pm_independent_finetuning_%j.err
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=150gb

set -euo pipefail

# Independent-test workflow:
# scGPT_spot_finetune.py -> g2pm_finetune_train.py
# -> g2pm_finetune_inference.py -> independent_validation_cross_samples.py
# -> plot_independent_benchmark.py
#
# This is an orchestration TEMPLATE: the current Python scripts require the
# following preparation/adaptations before this becomes an end-to-end run.
# 1. Fine-tune scGPT on the development/training dataset only. Prepare train.pt
#    and val.pt in TRAIN_DATA_DIR, with consistently encoded graph.sub_tls labels.
#    Extract independent-test embeddings using the SAME fitted scGPT model and
#    preprocessing; do not fine-tune scGPT separately on the test samples.
#    Stage 1 does not create train/validation splits or graph.sub_tls, and its
#    CLI does not provide inference-only embedding extraction for new samples.
#    Insert that preparation between stages 1 and 2, or provide prepared files.
# 2. Store each independent sample as TEST_DATA_DIR/SAMPLE.pt and SAMPLE.h5ad.
#    Graph nodes and h5ad observations must have the same order. Keep all test
#    samples separate from training/validation and select hyperparameters using
#    validation data. The example uses KC paths; replace them for your experiment.
# 3. Check model_zoo and ft_params['save_folder'] in g2pm_finetune_train.py.
#    Its checkpoint root is still hard-coded to TLS_results/5fold_CV; change it
#    to CHECKPOINT_DIR below. The training command is run ONCE on TRAIN_DATA_DIR.
# 4. Adapt g2pm_finetune_inference.py to iterate SAMPLE.pt/SAMPLE.h5ad under its
#    test_file_folder argument and reuse the SAME trained checkpoint directory:
#    CHECKPOINT_DIR/<basename(TRAIN_DATA_DIR)>_<backbone>_b<BATCH>_lr<LR>_checkpoint/
#    <MODEL>/<WALK>/best_model.pt. Retain its existing CLI used below.
#    The current implementation instead loops over fold0..fold4 and loads a
#    different fold checkpoint each time; it cannot run this layout as shipped.
# 5. Configure the evaluation script's folder, key_to_label, and keys_to_predict.
#    The existing indepedent_evaluation_cross_samples.py expects obs['label'] and
#    columns such as obs['spaGFM_binary_predict']; inference currently writes
#    <MODEL>_<WALK>_prediction and <MODEL>_<WALK>_prediction_probability.
#    Map the validation-selected predictions to the evaluation columns and use
#    consistent binary label encoding. Generate any baseline prediction columns
#    separately, or remove unavailable baselines from keys_to_predict.
# 6. Transfer stage 4's per-sample scores into the plotting script's metric
#    dictionaries and update its plt.savefig path. Its current scores and
#    Windows output path are hard-coded; stage 5 does not read stage 4's output.
#
# Run locally: bash /path/to/figure3/figure3d/g2pm_finetuning.sh
# Submit to Slurm from figure3/figure3d: sbatch g2pm_finetuning.sh
# Set PYTHON_BIN to an environment with scGPT PEFT utilities, stRoamer, and the
# remaining dependencies. Add an account directive if your cluster requires it.

export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND=Agg

# Under sbatch, use the submission directory rather than the spool copy's path.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-/fs/ess/PAS1475/yzhong/sf_project/random_test/spaGFM_dev/06_29_26/spaGFM_env/bin/python}"
DATA_DIR="${DATA_DIR:-/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-/fs/ess/PAS1475/Fei/code/G2PM_finetune/scgpt/save/scGPT_human}"
TF_FILE="${TF_FILE:-/fs/ess/PAS1475/Fei/code/G2PM_finetune/human_transcription_factors.txt}"
GRAPH_UTILS_DIR="${GRAPH_UTILS_DIR:-/fs/ess/PAS1475/Fei/code/G2PM_finetune/stRoamer/utils}"

# DATA_DIR/DATASET_NAME.h5ad must contain only the development/training data.
DATASET_NAME="${DATASET_NAME:-KC}"
ADAPTER="${ADAPTER:-HYBRID}" # False, HYBRID, ENCODER, TOKEN, PREFIX, or LORA
GRAPH_OUTPUT_DIR="${GRAPH_OUTPUT_DIR:-${DATA_DIR}/independent_finetune_scGPT}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${DATA_DIR}/KC/train}"
TEST_DATA_DIR="${TEST_DATA_DIR:-${DATA_DIR}/KC/test}"
# This shell variable does not override the training script's hard-coded path.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/fs/ess/PAS1475/Fei/data/spaGFM/TLS_results/independent}"
BATCH_SIZE="${BATCH_SIZE:-80}"
LR="${LR:-0.005}"
FROZEN_BACKBONE="${FROZEN_BACKBONE:-false}"
# Match the active model_zoo entries; training runs walk lengths 1 through 8.
MODEL_NAMES=(3M 15M 36M)

BACKBONE_ARGS=()
case "${FROZEN_BACKBONE}" in
    true) BACKBONE_ARGS+=(--frozen_backbone) ;;
    false) ;;
    *) echo "FROZEN_BACKBONE must be true or false." >&2; exit 1 ;;
esac

cd -- "${SCRIPT_DIR}"

# Resolve the requested names, falling back to the filenames in this repository.
RESULT_SCRIPT="${SCRIPT_DIR}/independent_validation_cross_samples.py"
if [[ ! -f "${RESULT_SCRIPT}" ]]; then
    RESULT_SCRIPT="${SCRIPT_DIR}/indepedent_evaluation_cross_samples.py"
fi
PLOT_SCRIPT="${SCRIPT_DIR}/plot_independent_benchmark.py"
if [[ ! -f "${PLOT_SCRIPT}" ]]; then
    PLOT_SCRIPT="${SCRIPT_DIR}/plot_indepedent_benchmark.py"
fi

# 1. Fine-tune scGPT and export training spot embeddings and a spatial graph.
# Output: ./save/finetune_*/ and GRAPH_OUTPUT_DIR/DATASET_NAME_finetune_scGPT.{pt,h5ad}.
echo "[1/5] Fine-tune scGPT on development/training data"
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

# Insert training/validation graph preparation and independent-test embedding
# extraction here if needed. Use stage 1's fitted model for every test sample.

# 2. Fit spaGFM using the single training/validation split.
# Input: TRAIN_DATA_DIR/train.pt and TRAIN_DATA_DIR/val.pt.
echo "[2/5] Train spaGFM on the development split"
"${PYTHON_BIN}" "${SCRIPT_DIR}/g2pm_finetune_train.py" \
    "${TRAIN_DATA_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    "${BACKBONE_ARGS[@]}"

# 3. Apply each fitted model to every independent test sample.
# Requires the independent-sample adaptation described in prerequisite 4.
# The existing inference code runs on CPU and writes predictions into h5ad files.
echo "[3/5] Run independent-test inference"
for model_name in "${MODEL_NAMES[@]}"; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/g2pm_finetune_inference.py" \
        "${TEST_DATA_DIR}" "${CHECKPOINT_DIR}" \
        --model_name "${model_name}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        "${BACKBONE_ARGS[@]}"
done

# Insert the prediction-column/binary-label preparation from prerequisite 5 here.

# 4. Report precision, recall, MCC, and macro F1 across independent samples.
# Configure the Python script's input folder and columns before running it.
echo "[4/5] Evaluate predictions across independent samples"
"${PYTHON_BIN}" "${RESULT_SCRIPT}"

# 5. Plot the independent benchmark using the configured per-sample scores.
echo "[5/5] Plot independent-test benchmark"
"${PYTHON_BIN}" "${PLOT_SCRIPT}"
