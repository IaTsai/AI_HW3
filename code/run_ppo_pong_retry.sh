#!/usr/bin/env bash
# PPO Pong with retry-on-crash. SB3+ALE has the same SIGSEGV/SIGTRAP issue
# we documented for DQN (engineering observation in the report). Since SB3's
# PPO does not have a clean checkpoint-resume hook in our wrapper, we simply
# retry-from-scratch on crash, treating crash within the first ~30 s as a
# fast-fail signal that we can absorb. Once training is past warmup it
# overwhelmingly tends to finish.
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

OUT="$RESULTS/pong_ppo_s42"
LOG="$LOGS/pong_ppo_s42.log"

tries=0
max_tries=10
# Match smoke-test environment: isolate to GPU 0 via CUDA_VISIBLE_DEVICES
# (without this, torch.AcceleratorError "unknown error" reproduced at the
# first conv2d forward pass on cuda:0; isolating fixes it).
export CUDA_VISIBLE_DEVICES=0
while [ $tries -lt $max_tries ]; do
  echo "=== attempt $((tries+1)) at $(date) ===" >> "$LOG"
  rm -rf "$OUT"
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
    --out-dir "$OUT" >> "$LOG" 2>&1
  # cuda:0 here means "first visible GPU" => physical GPU 0 under CUDA_VISIBLE_DEVICES=0
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/history.json" ]; then
    echo "pong_ppo_s42_done $(date +%s) tries=$((tries+1))" >> "$LOGS/orchestrator.log"
    exit 0
  fi
  tries=$((tries+1))
  echo "=== crash rc=$rc, retrying after 5s ===" >> "$LOG"
  sleep 5
done
echo "pong_ppo_s42_FAILED $(date +%s) max_tries=$max_tries" >> "$LOGS/orchestrator.log"
exit 1
