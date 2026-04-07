#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabicl_classification_amd.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-openml_cc18_csv=../limix/openml_cc18_csv,tabarena_cls=dataset/tabarena/cls,tabzilla_csv=../limix/tabzilla_csv,talent_csv=../limix/talent_csv}
MODEL_PATH=${MODEL_PATH:-ckpt/TabICLv2/tabicl-classifier-v2-20260212.ckpt}
OUT_DIR=${OUT_DIR:-result/TabICLv2_official_classification}
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
  --kv-cache kv \
  --softmax-temp 0.9 \
  --test-size 0.2 \
  --verbose
