#!/usr/bin/env bash
# Launch all LunarLander experiments: DQN-double, DQN-dueling, PPO, and a
# learning-rate sweep (3 values). All run on cuda:1 since LunarLander is
# small and won't compete with Pong-on-GPU0.
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

DEV="${1:-cuda:1}"

cd "$CODE"

dqn_lander() {
  local VARIANT=$1
  local OUT=$2
  local LR=$3
  python3 train_dqn.py \
    --env LunarLander-v3 \
    --variant "$VARIANT" \
    --total-steps 500000 \
    --buffer-size 100000 \
    --learning-starts 5000 \
    --batch-size 64 \
    --lr "$LR" \
    --gamma 0.99 \
    --target-update-freq 1000 \
    --train-freq 4 \
    --eps-start 1.0 --eps-end 0.05 --eps-decay-fraction 0.10 \
    --eval-every 25000 \
    --log-every 5000 \
    --seed 42 \
    --device "$DEV" \
    --out-dir "$OUT"
}

# DQN runs (algorithm comparison)
dqn_lander double  "$RESULTS/lander_double"  3e-4 > "$LOGS/lander_double.log"  2>&1
dqn_lander dueling "$RESULTS/lander_dueling" 3e-4 > "$LOGS/lander_dueling.log" 2>&1

# Learning-rate ablation (Double DQN, only LR changes)
dqn_lander double "$RESULTS/lander_lr1e-3" 1e-3 > "$LOGS/lander_lr1e-3.log" 2>&1
# lr3e-4 is the same config as lander_double; symlink to reuse.
ln -sfn "$RESULTS/lander_double" "$RESULTS/lander_lr3e-4"
dqn_lander double "$RESULTS/lander_lr1e-4" 1e-4 > "$LOGS/lander_lr1e-4.log" 2>&1

# PPO
python3 train_ppo.py \
  --env LunarLander-v3 \
  --total-steps 500000 \
  --n-envs 8 \
  --eval-every 25000 \
  --seed 42 \
  --device cpu \
  --out-dir "$RESULTS/lander_ppo" > "$LOGS/lander_ppo.log" 2>&1

echo "all_lander_done $(date +%s)" >> "$LOGS/orchestrator.log"
