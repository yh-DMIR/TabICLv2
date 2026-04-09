#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabicl_classification_amd.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-openml_cc18_csv=dataset/openml_cc18_72,tabarena_cls=dataset/tabarena/cls,tabzilla_csv=dataset/tabzilla35,talent_cls=dataset/talent_cls}
MODEL_PATH=${MODEL_PATH:-ckpt/TabICLv2/tabicl-classifier-v1-0208.ckpt}
OUT_DIR=${OUT_DIR:-result/TabICLv1_official_classification}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

${PYTHON} ${SCRIPT} \
  --root "${ROOT}" \
  --benchmarks "${BENCHMARKS}" \
  --model-path "${MODEL_PATH}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --batch-size 2 \
  --n-estimators 32 \
  --norm-methods none,power \
  --feat-shuffle latin \
  --kv-cache none \
  --offload-mode auto \
  --disk-offload-dir "${OUT_DIR}/_disk_offload" \
  --softmax-temp 0.9 \
  --test-size 0.2 \
  --verbose
