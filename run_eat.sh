#!/usr/bin/env bash
# EffiDec3D - train / test on EAT binary segmentation dataset
# Dataset root: /Users/xiaohg/ai_class/model/EffiDec3D/EffiDec3D_work
#   imagesTr/, labelsTr/  (train)
#   imagesVal/, labelsVal/  (validation; also used for test metrics)
#   imagesTs/ (symlink of val images, used for inference only)
#
# Usage:
#   bash run_eat.sh smoke     # tiny run to verify pipeline (default)
#   bash run_eat.sh train     # longer training
#   bash run_eat.sh test      # run final validation with best ckpt

set -euo pipefail

# ---------- Activate conda env ----------
CONDA_BASE="$(conda info --base 2>/dev/null || echo /Users/xiaohg/miniconda3)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate effidec3d

# ---------- Paths ----------
PROJ_DIR="/Users/xiaohg/ai_class/model/EffiDec3D/EffiDec3D-main"
DATA_ROOT="/Users/xiaohg/ai_class/model/EffiDec3D/EffiDec3D_work"
OUTPUT_DIR="${PROJ_DIR}/output_folder/eat_run1"

cd "${PROJ_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ---------- Common hyper-params ----------
DATASET="EAT"
NETWORK="3DUXNET_EffiDec3D"
IMG_SIZE="96 96 96"
N_CHANNELS=1
CHANNELS="48 96 192 384"
N_DEC=48

MODE_ARG="${1:-smoke}"

case "${MODE_ARG}" in
  smoke)
    MODE="train"
    MAX_ITER=40
    EVAL_STEP=20
    BATCH=1
    CROP=1
    LR=0.001
    CACHE=0.0
    WORKERS=0
    OVERLAP=0.5
    ;;
  train)
    MODE="train"
    MAX_ITER=2000
    EVAL_STEP=200
    BATCH=1
    CROP=2
    LR=0.001
    CACHE=0.0
    WORKERS=0
    OVERLAP=0.5
    ;;
  test)
    MODE="validation"
    MAX_ITER=1         # not used in test; loop skipped when mode != train
    EVAL_STEP=1
    BATCH=1
    CROP=1
    LR=0.001
    CACHE=0.0
    WORKERS=0
    OVERLAP=0.5
    ;;
  *)
    echo "Unknown mode: ${MODE_ARG} (expected: smoke | train | test)"
    exit 1
    ;;
esac

echo "[run_eat.sh] mode=${MODE_ARG} python=$(python --version)"
echo "[run_eat.sh] DATA_ROOT=${DATA_ROOT}"
echo "[run_eat.sh] OUTPUT_DIR=${OUTPUT_DIR}"

# PYTORCH_ENABLE_MPS_FALLBACK: fall back to CPU for any ops unsupported on MPS.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# PYTHONUNBUFFERED: unbuffered stdout for live progress.
export PYTHONUNBUFFERED=1

python main_train_BTCV_TU.py \
    --root "${DATA_ROOT}" \
    --output "${OUTPUT_DIR}" \
    --dataset "${DATASET}" \
    --img_size ${IMG_SIZE} \
    --n_channels ${N_CHANNELS} \
    --network "${NETWORK}" \
    --channels ${CHANNELS} \
    --n_decoder_channels ${N_DEC} \
    --ds False \
    --mode "${MODE}" \
    --pretrain False \
    --batch_size ${BATCH} \
    --crop_sample ${CROP} \
    --lr ${LR} \
    --optim AdamW \
    --max_iter ${MAX_ITER} \
    --eval_step ${EVAL_STEP} \
    --val_batch 1 \
    --gpu 0 \
    --cache_rate ${CACHE} \
    --num_workers ${WORKERS} \
    --overlap ${OVERLAP} \
    --skip_aggregation addition \
    --resolution_factor 2

echo "[run_eat.sh] Done."
