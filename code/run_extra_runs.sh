#!/usr/bin/env bash
# Extra experiments to firm up RQ1 (multi-seed Pong) and RQ2 (PPO on Atari).
#   GPU 0: PPO on Pong, 1M steps, seed=42 (one seed sufficient for "PPO on Atari" evidence).
#   GPU 1: Pong DQN seed=7, three variants (vanilla, double, dueling) at 1.5M each,
#          matching the existing seed=0 budget. Already have seed=42 (2M) and seed=0 (1.5M).
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

mkdir -p "$RESULTS" "$LOGS"

# ------- GPU 0: PPO on Pong (CnnPolicy, atari wrappers) -------
ppo_pong() {
  local OUT=$1 LOG=$2
  python3 "$CODE/train_ppo.py" \
    --env ALE/Pong-v5 \
    --total-steps 1000000 \
    --n-envs 8 \
    --n-steps 128 \
    --batch-size 256 \
    --n-epochs 4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ent-coef 0.01 \
    --vf-coef 0.5 \
    --clip-range 0.1 \
    --clip-schedule linear \
    --lr 2.5e-4 \
    --lr-schedule linear \
    --seed 42 \
    --device cuda:0 \
    --eval-every 50000 \
    --out-dir "$OUT" > "$LOG" 2>&1
}

# ------- GPU 1: DQN Pong seed=7 (retry-resume on crash, same as run_multiseed.sh) -------
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
      --device cuda:1 \
      --out-dir "$OUT" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && return 0
    tries=$((tries+1))
    sleep 3
  done
}

(
  ppo_pong "$RESULTS/pong_ppo_s42" "$LOGS/pong_ppo_s42.log"
  echo "pong_ppo_s42_done $(date +%s)" >> "$LOGS/orchestrator.log"
) &
PID0=$!

(
  pong_retry vanilla 7 "$RESULTS/pong_vanilla_s7" "$LOGS/pong_vanilla_s7.log"
  echo "pong_vanilla_s7_done $(date +%s)" >> "$LOGS/orchestrator.log"
  pong_retry double  7 "$RESULTS/pong_double_s7"  "$LOGS/pong_double_s7.log"
  echo "pong_double_s7_done $(date +%s)" >> "$LOGS/orchestrator.log"
  pong_retry dueling 7 "$RESULTS/pong_dueling_s7" "$LOGS/pong_dueling_s7.log"
  echo "pong_dueling_s7_done $(date +%s)" >> "$LOGS/orchestrator.log"
  echo "all_pong_s7_done $(date +%s)" >> "$LOGS/orchestrator.log"
) &
PID1=$!

wait $PID0 $PID1
echo "all_extra_runs_done $(date +%s)" >> "$LOGS/orchestrator.log"
