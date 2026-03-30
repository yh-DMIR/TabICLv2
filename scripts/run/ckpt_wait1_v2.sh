#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

# ============================================================
# Config
# ============================================================
PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabicl_dynamic_v2.py}

CKPT_DIR=${CKPT_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/lyh/TabICLv2/TabICLv2/ckpt/TabICLv2}
OUT_ROOT=${OUT_ROOT:-result/tabiclv2}

# 新增：官方预训练权重版本设置
OFFICIAL_VER=${OFFICIAL_VER:-"v2"}

MIN_STEP="${MIN_STEP:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

WORKERS=${WORKERS:-8}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

SLEEP_SECS="${SLEEP_SECS:-60}"
STABLE_ROUNDS="${STABLE_ROUNDS:-3}"
STABLE_INTERVAL_SECS="${STABLE_INTERVAL_SECS:-5}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================
# 定义要跑的所有 Benchmark 队列
# 格式为 "任务别名:任务类型:数据集相对路径"
# ============================================================
BENCHMARKS=(
  "talent_cls:classification:dataset/talent_cls"
  "talent_reg:regression:dataset/talent_reg"
  "tabarena_cls:classification:dataset/tabarena/cls"
  "tabarena_reg:regression:dataset/tabarena/reg"
  "ctr23:regression:dataset/ctr23"
)

# 基础参数 (所有任务通用)
COMMON_ARGS="
  --workers ${WORKERS}
  --gpus ${GPUS}
  --device cuda:0
  --batch-size 4
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --verbose
"

mkdir -p "${OUT_ROOT}"

# ============================================================
# 初始化每个 Benchmark 的主 CSV 文件
# ============================================================
for b in "${BENCHMARKS[@]}"; do
  IFS=':' read -r B_NAME B_TASK B_PATH <<< "${b}"
  
  MASTER_CSV="${OUT_ROOT}/summary_all_ckpts_${B_NAME}.csv"
  
  if [[ "${B_TASK}" == "classification" ]]; then
    MAIN_METRIC_NAME="${B_NAME}_avg_acc"
  else
    MAIN_METRIC_NAME="${B_NAME}_avg_r2"
  fi

  if [[ ! -f "${MASTER_CSV}" ]]; then
    cat > "${MASTER_CSV}" <<CSV
ckpt,ckpt_path,started_at,finished_at,total_wall_seconds,${MAIN_METRIC_NAME},wall_seconds,discovered_datasets,processed_datasets,failed_count
CSV
    echo "✅ Created master CSV: ${MASTER_CSV}"
  fi
done

# ============================================================
# Helpers
# ============================================================
parse_summary_field () {
  local summary_txt="$1"
  local key="$2"
  awk -F': ' -v k="${key}" '$1==k {print $2; found=1} END{if(!found) print ""}' "${summary_txt}"
}

extract_step () {
  local base="$1"
  if [[ "${base}" =~ step-([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "-1"
  fi
}

wait_for_ckpt_ready () {
  local f="$1"
  local rounds="${STABLE_ROUNDS}"
  local interval="${STABLE_INTERVAL_SECS}"
  local last_size="-1"
  local stable=0
  while true; do
    if [[ ! -f "${f}" ]]; then
      stable=0
      last_size="-1"
      sleep "${interval}"
      continue
    fi
    local sz
    if ! sz="$(stat -c %s "${f}" 2>/dev/null)"; then
      sleep "${interval}"
      continue
    fi
    if [[ "${sz}" == "${last_size}" ]]; then
      stable=$((stable + 1))
    else
      stable=0
      last_size="${sz}"
    fi
    if (( stable >= rounds )); then
      return 0
    fi
    sleep "${interval}"
  done
}

# ============================================================
# Pending queue
# ============================================================
declare -a PENDING=()
declare -A IN_PENDING=()
declare -A IGNORED=()

refill_pending_from_dir () {
  local tmp
  tmp="$(mktemp)"

  while IFS= read -r f; do
    if [[ -n "${IGNORED[${f}]+x}" ]]; then continue; fi

    local base step
    base="$(basename "${f}")"
    step="$(extract_step "${base}")"
    (( step < 0 )) && continue
    (( step < MIN_STEP )) && continue

    if [[ -n "${IN_PENDING[${f}]+x}" ]]; then continue; fi
    printf '%s\t%s\n' "${step}" "${f}" >> "${tmp}"
  done < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.ckpt" 2>/dev/null)

  if [[ -s "${tmp}" ]]; then
    while IFS=$'\t' read -r _step f; do
      PENDING+=("${f}")
      IN_PENDING["${f}"]=1
    done < <(sort -n -k1,1 "${tmp}")
  fi
  rm -f "${tmp}"
}

pop_pending () {
  local f="${PENDING[0]:-}"
  [[ -z "${f}" ]] && { echo ""; return 0; }
  PENDING=("${PENDING[@]:1}")
  unset "IN_PENDING[${f}]"
  echo "${f}"
}

wait_until_dir_changes () {
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -q -e create -e close_write -e moved_to "${CKPT_DIR}" >/dev/null 2>&1 || true
  else
    sleep "${SLEEP_SECS}"
  fi
}

# ============================================================
# [新增] Official Checkpoint Fallback
# ============================================================
NUM_CKPTS=$(find "${CKPT_DIR}" -maxdepth 1 -name "*.ckpt" 2>/dev/null | wc -l || echo 0)

if [[ "${NUM_CKPTS}" -eq 0 ]]; then
  echo "⚠️  No local checkpoints found in ${CKPT_DIR}."
  echo "⬇️  Running official baseline (version: ${OFFICIAL_VER}) first..."
  
  ckpt_base="official_${OFFICIAL_VER}"
  ckpt_abs="downloaded_automatically"
  ckpt_out_base="${OUT_ROOT}/${ckpt_base}"

  for b in "${BENCHMARKS[@]}"; do
    IFS=':' read -r B_NAME B_TASK B_PATH <<< "${b}"
    
    ckpt_out="${ckpt_out_base}/${B_NAME}"
    summary_txt="${ckpt_out}/tabicl_${B_NAME}.summary.txt"
    all_out="${ckpt_out}/tabicl_${B_NAME}.ALL.csv"
    run_log="${ckpt_out}/tabicl_${B_NAME}.run.log"
    MASTER_CSV="${OUT_ROOT}/summary_all_ckpts_${B_NAME}.csv"
    LOCK_FILE="${MASTER_CSV}.lock"

    if [[ "${SKIP_DONE}" == "1" && -f "${summary_txt}" ]]; then
      echo "   ⏭️  Skip Sub-Task [${B_NAME}] - Already done."
      continue
    fi

    echo "===== Running [${B_NAME}] (${B_TASK}) with Official Weights =====" >&2

    mkdir -p "${ckpt_out}"
    started_at="$(date '+%Y-%m-%d %H:%M:%S')"

    CURRENT_ARGS="${COMMON_ARGS}"
    if [[ "${B_TASK}" == "classification" ]]; then
      CURRENT_ARGS="${CURRENT_ARGS} --softmax-temp 0.9"
    fi

    ${PYTHON} ${SCRIPT} \
      --root "${B_PATH}" \
      --out-dir "${ckpt_out}" \
      --task "${B_TASK}" \
      --all-out "${all_out}" \
      --summary-txt "${summary_txt}" \
      --checkpoint-version "${OFFICIAL_VER}" \
      ${CURRENT_ARGS} \
      > "${run_log}" 2>&1

    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

    if [[ "${B_TASK}" == "classification" ]]; then
      main_metric="$(parse_summary_field "${summary_txt}" "avg_accuracy_ok")"
    else
      main_metric="$(parse_summary_field "${summary_txt}" "avg_r2_ok")"
    fi

    wall_seconds="$(parse_summary_field "${summary_txt}" "wall_seconds")"
    discovered_datasets="$(parse_summary_field "${summary_txt}" "discovered_datasets")"
    processed_datasets="$(parse_summary_field "${summary_txt}" "processed_datasets")"
    failed_count="$(parse_summary_field "${summary_txt}" "failed_count")"

    total_wall_seconds="$(${PYTHON} - <<PY
def f(x):
    try:
        return float(x)
    except:
        return 0.0
print(f"{f('${wall_seconds}'):.6f}")
PY
)"

    {
      flock 200
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${ckpt_base}" \
        "${ckpt_abs}" \
        "${started_at}" \
        "${finished_at}" \
        "${total_wall_seconds}" \
        "${main_metric}" \
        "${wall_seconds}" \
        "${discovered_datasets}" \
        "${processed_datasets}" \
        "${failed_count}"
    } 200>"${LOCK_FILE}" >> "${MASTER_CSV}"

    echo "   ✅ Done [${B_NAME}]. Metric: ${main_metric}"
  done
  
  echo "✅ Official baseline evaluation complete. Now entering monitor mode..."
fi


# ============================================================
# Main Loop (Waiting for new CKPTs)
# ============================================================
echo "✅ Multi-Benchmark Online ckpt consumer started."
echo "   - MIN_STEP=${MIN_STEP}"
echo "   - SKIP_DONE=${SKIP_DONE}"
echo "   - Registered Benchmarks: ${#BENCHMARKS[@]}"

while true; do
  refill_pending_from_dir

  if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "⏳ Queue empty. Waiting for new ckpt in ${CKPT_DIR} ..."
    wait_until_dir_changes
    continue
  fi

  ckpt_abs="$(pop_pending)"
  ckpt_base="$(basename "${ckpt_abs}")"
  ckpt_stem="${ckpt_base%.ckpt}"
  ckpt_out_base="${OUT_ROOT}/${ckpt_stem}"

  # 检查是否所有子任务都已完成
  all_done=1
  if [[ "${SKIP_DONE}" == "1" ]]; then
    for b in "${BENCHMARKS[@]}"; do
      IFS=':' read -r B_NAME B_TASK B_PATH <<< "${b}"
      if [[ ! -f "${ckpt_out_base}/${B_NAME}/tabicl_${B_NAME}.summary.txt" ]]; then
        all_done=0
        break
      fi
    done
    if [[ "${all_done}" == "1" ]]; then
      echo "⏭️  Skip (All benchmarks done for ckpt): ${ckpt_base}"
      IGNORED["${ckpt_abs}"]=1
      continue
    fi
  fi

  echo
  echo "#################################################################"
  echo "### NEXT CKPT (by step order): ${ckpt_base}"
  echo "### PATH: ${ckpt_abs}"
  echo "#################################################################"

  echo "⏱️  Waiting ckpt ready (size stable): ${ckpt_abs}"
  wait_for_ckpt_ready "${ckpt_abs}"
  echo "✅ CKPT ready: ${ckpt_abs}"

  # 遍历执行配置好的所有 Benchmark
  for b in "${BENCHMARKS[@]}"; do
    IFS=':' read -r B_NAME B_TASK B_PATH <<< "${b}"

    ckpt_out="${ckpt_out_base}/${B_NAME}"
    summary_txt="${ckpt_out}/tabicl_${B_NAME}.summary.txt"
    all_out="${ckpt_out}/tabicl_${B_NAME}.ALL.csv"
    run_log="${ckpt_out}/tabicl_${B_NAME}.run.log"
    MASTER_CSV="${OUT_ROOT}/summary_all_ckpts_${B_NAME}.csv"
    LOCK_FILE="${MASTER_CSV}.lock"

    if [[ "${SKIP_DONE}" == "1" && -f "${summary_txt}" ]]; then
      echo "   ⏭️  Skip Sub-Task [${B_NAME}] - Already done."
      continue
    fi

    echo "===== Running [${B_NAME}] (${B_TASK}) =====" >&2

    mkdir -p "${ckpt_out}"
    started_at="$(date '+%Y-%m-%d %H:%M:%S')"

    CURRENT_ARGS="${COMMON_ARGS}"
    if [[ "${B_TASK}" == "classification" ]]; then
      CURRENT_ARGS="${CURRENT_ARGS} --softmax-temp 0.9"
    fi

    # 启动 Python 脚本
    ${PYTHON} ${SCRIPT} \
      --root "${B_PATH}" \
      --out-dir "${ckpt_out}" \
      --task "${B_TASK}" \
      --all-out "${all_out}" \
      --summary-txt "${summary_txt}" \
      --model-path "${ckpt_abs}" \
      ${CURRENT_ARGS} \
      > "${run_log}" 2>&1

    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

    # 动态提取指标
    if [[ "${B_TASK}" == "classification" ]]; then
      main_metric="$(parse_summary_field "${summary_txt}" "avg_accuracy_ok")"
    else
      main_metric="$(parse_summary_field "${summary_txt}" "avg_r2_ok")"
    fi

    wall_seconds="$(parse_summary_field "${summary_txt}" "wall_seconds")"
    discovered_datasets="$(parse_summary_field "${summary_txt}" "discovered_datasets")"
    processed_datasets="$(parse_summary_field "${summary_txt}" "processed_datasets")"
    failed_count="$(parse_summary_field "${summary_txt}" "failed_count")"

    total_wall_seconds="$(${PYTHON} - <<PY
def f(x):
    try:
        return float(x)
    except:
        return 0.0
print(f"{f('${wall_seconds}'):.6f}")
PY
)"

    # 加锁写入对应的主 CSV
    {
      flock 200
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${ckpt_base}" \
        "${ckpt_abs}" \
        "${started_at}" \
        "${finished_at}" \
        "${total_wall_seconds}" \
        "${main_metric}" \
        "${wall_seconds}" \
        "${discovered_datasets}" \
        "${processed_datasets}" \
        "${failed_count}"
    } 200>"${LOCK_FILE}" >> "${MASTER_CSV}"

    echo "   ✅ Done [${B_NAME}]. Metric: ${main_metric}"

  done # 结束当前 Ckpt 的 Benchmark 循环

  # 当所有的子任务跑完后，把这个 Checkpoint 加入忽略列表
  IGNORED["${ckpt_abs}"]=1

done