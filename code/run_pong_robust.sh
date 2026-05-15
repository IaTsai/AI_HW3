#!/usr/bin/env bash
# Same as run_pong_all.sh but with a much higher retry budget and a small
# garbage-collect grace pause between retries. We're seeing sporadic SIGSEGV
# from the ALE C bindings on extended runs; the only mitigation we found is
# to checkpoint often and restart.
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
  local tries=0
  while [ $tries -lt 40 ]; do
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
    echo "[robust] $VARIANT crashed rc=$rc try=$tries $(date +%s)" >> "$LOGS/orchestrator.log"
    tries=$((tries+1))
    sleep 5
  done
  echo "[robust] $VARIANT GAVE UP after $tries tries" >> "$LOGS/orchestrator.log"
  return 1
}

cd "$CODE"

case "${1:-all}" in
  vanilla) train_one vanilla cuda:0 "$RESULTS/pong_vanilla" >> "$LOGS/pong_vanilla.log" 2>&1 ;;
  dueling) train_one dueling cuda:0 "$RESULTS/pong_dueling" >> "$LOGS/pong_dueling.log" 2>&1 ;;
  double)  train_one double  cuda:1 "$RESULTS/pong_double"  >> "$LOGS/pong_double.log"  2>&1 ;;
  all)
    (
      train_one vanilla cuda:0 "$RESULTS/pong_vanilla" >> "$LOGS/pong_vanilla.log" 2>&1
      echo "vanilla_done $(date +%s)" >> "$LOGS/orchestrator.log"
      train_one dueling cuda:0 "$RESULTS/pong_dueling" >> "$LOGS/pong_dueling.log" 2>&1
      echo "dueling_done $(date +%s)" >> "$LOGS/orchestrator.log"
    ) &
    P0=$!
    (
      train_one double cuda:1 "$RESULTS/pong_double" >> "$LOGS/pong_double.log" 2>&1
      echo "double_done $(date +%s)" >> "$LOGS/orchestrator.log"
    ) &
    P1=$!
    wait $P0 $P1
    echo "all_pong_done $(date +%s)" >> "$LOGS/orchestrator.log"
    ;;
esac
