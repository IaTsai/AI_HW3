#!/usr/bin/env bash
# Post-training evaluation: 100 deterministic episodes per trained agent.
# Run sequentially because each eval boots an ALE/Box2D env, and we want
# clean per-run stderr.
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

cd "$CODE"

# Pong: per-checkpoint eval (for the learning curve plot), then 100-ep final.
for tup in "vanilla:pong_vanilla" "double:pong_double" "dueling:pong_dueling"; do
  VARIANT=${tup%%:*}; SUB=${tup#*:}
  if [ -d "$RESULTS/$SUB" ]; then
    python3 evaluate_checkpoints.py --env ALE/Pong-v5 --variant "$VARIANT" \
      --results-dir "$RESULTS/$SUB" --episodes 5 --device cuda:0 \
      > "$LOGS/eval_ckpt_$SUB.log" 2>&1 || true
    python3 evaluate.py --env ALE/Pong-v5 --variant "$VARIANT" \
      --model "$RESULTS/$SUB/model.pt" --n-episodes 100 --device cuda:0 \
      --out "$RESULTS/$SUB/eval100.json" \
      > "$LOGS/eval100_$SUB.log" 2>&1 || true
  fi
done

# LunarLander DQN runs
for tup in "double:lander_double" "dueling:lander_dueling" \
           "double:lander_lr1e-3" "double:lander_lr1e-4"; do
  VARIANT=${tup%%:*}; SUB=${tup#*:}
  if [ -d "$RESULTS/$SUB" ]; then
    python3 evaluate_checkpoints.py --env LunarLander-v3 --variant "$VARIANT" \
      --results-dir "$RESULTS/$SUB" --episodes 10 --device cuda:0 \
      > "$LOGS/eval_ckpt_$SUB.log" 2>&1 || true
    python3 evaluate.py --env LunarLander-v3 --variant "$VARIANT" \
      --model "$RESULTS/$SUB/model.pt" --n-episodes 100 --device cuda:0 \
      --out "$RESULTS/$SUB/eval100.json" \
      > "$LOGS/eval100_$SUB.log" 2>&1 || true
  fi
done

# LunarLander PPO
if [ -d "$RESULTS/lander_ppo" ]; then
  python3 evaluate_ppo.py --env LunarLander-v3 \
    --model "$RESULTS/lander_ppo/model.zip" --n-episodes 100 \
    --out "$RESULTS/lander_ppo/eval100.json" \
    > "$LOGS/eval100_lander_ppo.log" 2>&1 || true
fi

echo "all_eval_done $(date +%s)" >> "$LOGS/orchestrator.log"
