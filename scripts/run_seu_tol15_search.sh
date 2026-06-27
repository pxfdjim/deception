#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="experiments/component_ablation_logs"
mkdir -p "${LOG_DIR}"

run_one() {
  local gpu="$1"
  local suffix="$2"
  shift 2

  local log_file="${LOG_DIR}/seumld_${suffix}.log"
  echo "Starting ${suffix} on GPU ${gpu}; log=${log_file}"
  setsid bash -c "CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 python main_seumld.py $*" \
    > "${log_file}" 2>&1 < /dev/null &
  echo "pid=$!"
}

COMMON_ARGS="--use-cluster-topk-mean-pooling --cluster-topk-mean-ratio 0.5 \
--use-visual-self-attn --visual-self-attn-heads 4 --visual-self-attn-layers 1 \
--no-instance-loss \
--use-eval-threshold-search --eval-threshold-min 0.20 --eval-threshold-max 0.90 --eval-threshold-step 0.01 \
--eval-threshold-objective acc_tolerant_f1 --eval-threshold-acc-tolerance 0.015 \
--best-epoch-objective acc_tolerant_f1 --best-epoch-acc-tolerance 0.015 \
--eval-every 2 --epochs 50 --batch-size 8 --lr 0.05 --num-workers 2 --disable-tqdm"

run_one 5 "hardcut_cluster_vsa_tol15_besttol15_eval2_t20_90_ep50" \
  "${COMMON_ARGS} --exp-suffix seu_hardcut_cluster_vsa_tol15_besttol15_eval2_t20_90_ep50"

run_one 7 "hardcut_cluster_weighted_vsa_tol15_besttol15_eval2_t20_90_ep50" \
  "${COMMON_ARGS} --use-cluster-topk-weighted-pooling --cluster-topk-weight-temperature 0.5 --cluster-topk-weight-blend 0.3 \
--exp-suffix seu_hardcut_cluster_weighted_vsa_tol15_besttol15_eval2_t20_90_ep50"
