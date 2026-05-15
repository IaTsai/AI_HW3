#!/usr/bin/env bash
# Orchestrate Pong training with crash-resume. On GPU 0 run vanilla, then
# dueling sequentially. On GPU 1 run double (then yield to LunarLander).
# The training script saves a checkpoint every eval-every steps, so a crash
# costs at most that many env steps.
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

train_one() {
  local VARIANT=$1 DEV=$2 OUTDIR=$3
  # Retry loop: if the script crashes (which it can on this setup --- see
  # report), resume from the last checkpoint and keep going. Bail after 6 tries.
  local tries=0
  while [ $tries -lt 6 ]; do
    python3 "$CODE/train_dqn.py" \
      --env ALE/Pong-v5 \
      --variant "$VARIANT" \
      --total-steps 2000000 \
      --buffer-size 100000 \
      --learning-starts 50000 \
      --batch-size 32 \
      --lr 1e-4 \
      --gamma 0.99 \
      --target-update-freq 1000 \
      --train-freq 4 \
      --eps-start 1.0 --eps-end 0.01 --eps-decay-fraction 0.10 \
      --eval-every 100000 \
      --log-every 5000 \
      --seed 42 \
      --resume \
      --device "$DEV" \
      --out-dir "$OUTDIR"
    rc=$?
    if [ $rc -eq 0 ]; then return 0; fi
    echo "[orch] $VARIANT crashed rc=$rc, restart $((tries+1))" >> "$LOGS/orchestrator.log"
    tries=$((tries+1))
    sleep 3
  done
  return 1
}

cd "$CODE"

# GPU 0: vanilla -> dueling
(
  train_one vanilla cuda:0 "$RESULTS/pong_vanilla" >> "$LOGS/pong_vanilla.log" 2>&1
  echo "vanilla_done $(date +%s)" >> "$LOGS/orchestrator.log"
  train_one dueling cuda:0 "$RESULTS/pong_dueling" >> "$LOGS/pong_dueling.log" 2>&1
  echo "dueling_done $(date +%s)" >> "$LOGS/orchestrator.log"
) &
PID0=$!

# GPU 1: double
(
  train_one double  cuda:1 "$RESULTS/pong_double"  >> "$LOGS/pong_double.log" 2>&1
  echo "double_done $(date +%s)" >> "$LOGS/orchestrator.log"
) &
PID1=$!

wait "$PID0" "$PID1"
echo "all_pong_done $(date +%s)" >> "$LOGS/orchestrator.log"
