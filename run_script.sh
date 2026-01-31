#!/usr/bin/env bash
set -Eeuo pipefail

# =======================
# Settings
# =======================
CONFIG_DIR="configs/navix"
ALGO="ppo"
NUM_SEEDS=20

# 是否遇到错误就停止：1=停止，0=继续跑下一个
STOP_ON_ERROR=1

# 日志目录
LOG_DIR="logs_navix"
mkdir -p "$LOG_DIR"

# 断点续跑：每个(config, job)完成后写一个done文件
DONE_DIR=".done_navix"
mkdir -p "$DONE_DIR"

# =======================
# Config list
# =======================
configs=(
  "distshift2.yaml"
  "doorkey_16x16.yaml"
  "doorkey_8x8.yaml"
  "dynamic_obstacles_6x6_random.yaml"
  "empty_16x16.yaml"
  "empty_6x6.yaml"
  "empty_random_8x8.yaml"
  "fourrooms.yaml"
  "gotodoor_6x6.yaml"
  "keycorridors4r4.yaml"
  "lavagaps6.yaml"
  "simplecrossings9n1.yaml"
)

# =======================
# Helpers
# =======================
run_one () {
  local module="$1"      # train_highway 或 train_ppo
  local cfgfile="$2"     # 例如 distshift2.yaml
  local cfgpath="${CONFIG_DIR}/${cfgfile}"

  local tag="${module}__${cfgfile}"
  local done_flag="${DONE_DIR}/${tag}.done"
  local log_path="${LOG_DIR}/${tag}.log"

  # 跳过已完成
  if [[ -f "$done_flag" ]]; then
    echo "[SKIP] $tag (done flag exists)"
    return 0
  fi

  echo "============================================================"
  echo "[RUN ] python -m ${module} --config ${cfgpath} --algorithm ${ALGO} --num-seeds ${NUM_SEEDS}"
  echo "[LOG ] ${log_path}"
  echo "============================================================"

  # 真实执行 + 同时输出到终端和日志
  if python -m "${module}" \
      --config "${cfgpath}" \
      --algorithm "${ALGO}" \
      --num-seeds "${NUM_SEEDS}" 2>&1 | tee "${log_path}"
  then
    touch "$done_flag"
    echo "[OK  ] $tag"
    return 0
  else
    echo "[FAIL] $tag"
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      echo "[STOP] STOP_ON_ERROR=1, exiting."
      exit 1
    else
      echo "[CONT] STOP_ON_ERROR=0, continue to next."
      return 1
    fi
  fi
}

# =======================
# Main loop
# =======================
for cfg in "${configs[@]}"; do
  # 可选：检查文件是否存在
  if [[ ! -f "${CONFIG_DIR}/${cfg}" ]]; then
    echo "[WARN] missing config: ${CONFIG_DIR}/${cfg}"
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      exit 1
    else
      continue
    fi
  fi

  # 先跑 train_highway 再跑 train_ppo（按你需求）
  run_one "train_highway" "$cfg"
  run_one "train_ppo" "$cfg"
done

echo "All jobs finished."
