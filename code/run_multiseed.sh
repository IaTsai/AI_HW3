#!/usr/bin/env bash
# Multi-seed runs:
#   LunarLander (cuda:1): 2 extra seeds for Double DQN and Dueling DQN (the
#     core LunarLander comparison). Already have seed=42; add seed=0 and seed=7.
#   Pong (cuda:0): 1 extra seed for each variant at 1.5M steps (shorter budget
#     to fit time). Already have seed=42; add seed=0.
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

lander_run() {
  local VARIANT=$1 SEED=$2 OUT=$3
  python3 "$CODE/train_dqn.py" \
    --env LunarLander-v3 \
    --variant "$VARIANT" \
    --total-steps 500000 \
    --buffer-size 100000 \
    --learning-starts 5000 \
    --batch-size 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --target-update-freq 1000 \
    --train-freq 4 \
    --eps-start 1.0 --eps-end 0.05 --eps-decay-fraction 0.10 \
    --eval-every 100000 \
    --log-every 10000 \
    --seed "$SEED" \
    --device cuda:1 \
    --out-dir "$OUT"
}

pong_retry() {
  local VARIANT=$1 SEED=$2 OUT=$3 LOG=$4
  local tries=0
  while [ $tries -lt 30 ]; do
    python3 "$CODE/train_dqn.py" \
      --env ALE/Pong-v5 \
      --variant "$VARIANT" \
      --total-steps 1500000 \
      --buffer-size 100000 \
      --learning-starts 50000 \
      --batch-size 32 \
      --lr 1e-4 \
      --gamma 0.99 \
      --target-update-freq 1000 \
      --train-freq 4 \
      --eps-start 1.0 --eps-end 0.01 --eps-decay-fraction 0.13 \
      --eval-every 100000 \
      --log-every 10000 \
      --seed "$SEED" \
      --resume \
      --device cuda:0 \
      --out-dir "$OUT" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && return 0
    tries=$((tries+1))
    sleep 3
  done
}

# Lander seeds on GPU 1 (fast, ~3 min each)
(
  lander_run double  0 "$RESULTS/lander_double_s0"  > "$LOGS/lander_double_s0.log"  2>&1
  echo "lander_double_s0_done $(date +%s)" >> "$LOGS/orchestrator.log"
  lander_run double  7 "$RESULTS/lander_double_s7"  > "$LOGS/lander_double_s7.log"  2>&1
  echo "lander_double_s7_done $(date +%s)" >> "$LOGS/orchestrator.log"
  lander_run dueling 0 "$RESULTS/lander_dueling_s0" > "$LOGS/lander_dueling_s0.log" 2>&1
  echo "lander_dueling_s0_done $(date +%s)" >> "$LOGS/orchestrator.log"
  lander_run dueling 7 "$RESULTS/lander_dueling_s7" > "$LOGS/lander_dueling_s7.log" 2>&1
  echo "lander_dueling_s7_done $(date +%s)" >> "$LOGS/orchestrator.log"
  echo "all_lander_seeds_done $(date +%s)" >> "$LOGS/orchestrator.log"
) &
PID1=$!

# Pong seeds on GPU 0 (slower, ~25 min each at 1.5M)
(
  pong_retry vanilla 0 "$RESULTS/pong_vanilla_s0" "$LOGS/pong_vanilla_s0.log"
  echo "pong_vanilla_s0_done $(date +%s)" >> "$LOGS/orchestrator.log"
  pong_retry double  0 "$RESULTS/pong_double_s0"  "$LOGS/pong_double_s0.log"
  echo "pong_double_s0_done $(date +%s)" >> "$LOGS/orchestrator.log"
  pong_retry dueling 0 "$RESULTS/pong_dueling_s0" "$LOGS/pong_dueling_s0.log"
  echo "pong_dueling_s0_done $(date +%s)" >> "$LOGS/orchestrator.log"
  echo "all_pong_seeds_done $(date +%s)" >> "$LOGS/orchestrator.log"
) &
PID0=$!

wait $PID0 $PID1
echo "all_multiseed_done $(date +%s)" >> "$LOGS/orchestrator.log"
