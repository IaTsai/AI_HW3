#!/usr/bin/env bash
# Run the remaining LunarLander experiments after lander_double finishes.
# (lander_double is the current run; we wait for it to finish, then go.)
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

cd "$CODE"

# 1. Wait for lander_double (which is already running) to finish.
until ! pgrep -af "train_dqn.py.*lander_double" >/dev/null; do
  sleep 30
done
echo "lander_double_done $(date +%s)" >> "$LOGS/orchestrator.log"

dqn_lander() {
  local VARIANT=$1 OUT=$2 LR=$3
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
    --device cuda:1 \
    --out-dir "$OUT"
}

# 2. Sequentially: dueling, then LR ablations, then PPO.
dqn_lander dueling "$RESULTS/lander_dueling" 3e-4 > "$LOGS/lander_dueling.log" 2>&1
echo "lander_dueling_done $(date +%s)" >> "$LOGS/orchestrator.log"

dqn_lander double "$RESULTS/lander_lr1e-3" 1e-3 > "$LOGS/lander_lr1e-3.log" 2>&1
echo "lander_lr1e-3_done $(date +%s)" >> "$LOGS/orchestrator.log"

dqn_lander double "$RESULTS/lander_lr1e-4" 1e-4 > "$LOGS/lander_lr1e-4.log" 2>&1
echo "lander_lr1e-4_done $(date +%s)" >> "$LOGS/orchestrator.log"

# 3. PPO on CPU.
python3 train_ppo.py --env LunarLander-v3 --total-steps 500000 \
  --n-envs 8 --eval-every 25000 --seed 42 --device cpu \
  --out-dir "$RESULTS/lander_ppo" > "$LOGS/lander_ppo.log" 2>&1
echo "lander_ppo_done $(date +%s)" >> "$LOGS/orchestrator.log"

echo "all_lander_done $(date +%s)" >> "$LOGS/orchestrator.log"
