#!/usr/bin/env bash
# PPO Pong 1M on CPU. GPU runs (both cuda:0 and cuda:1) reproduce the
# same SIGSEGV/CUDA-unknown-error documented in the report's engineering-
# observation paragraph -- the ALE/SB3/PyTorch stack is unstable on this
# install. CPU sidesteps CUDA entirely and is reliable for ~1M steps.
set -u

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

OUT="$RESULTS/pong_ppo_s42"
LOG="$LOGS/pong_ppo_s42.log"

tries=0
max_tries=5
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
    --device cpu \
    --eval-every 50000 \
    --out-dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/history.json" ]; then
    echo "pong_ppo_s42_done $(date +%s) tries=$((tries+1)) device=cpu" >> "$LOGS/orchestrator.log"
    exit 0
  fi
  tries=$((tries+1))
  echo "=== crash rc=$rc, retrying after 5s ===" >> "$LOG"
  sleep 5
done
echo "pong_ppo_s42_FAILED_cpu $(date +%s) max_tries=$max_tries" >> "$LOGS/orchestrator.log"
exit 1
