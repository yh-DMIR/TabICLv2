#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabicl_regression_amd.py}
MODEL_PATH=${MODEL_PATH:-ckpt/TabICLv2/tabicl-regressor-v2-20260212.ckpt}
OUT_DIR=${OUT_DIR:-result/TabICLv2_official_regression}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

${PYTHON} ${SCRIPT} \
  --model-path "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --batch-size 4 \
  --n-estimators 32 \
  --norm-methods none,power \
  --feat-shuffle latin \
  --kv-cache kv \
  --verbose
