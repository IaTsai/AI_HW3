#!/usr/bin/env bash
# Safety net: after the main orchestrator gives up on a variant, this
# watchdog resumes it from the last checkpoint. Polls every 30 s.
set -u

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

resume_until_done() {
  local VARIANT=$1 DEV=$2 OUTDIR=$3
  local LOG=$4
  # Read current step from history; bail if at or beyond target.
  local target=2000000
  while true; do
    local cur
    cur=$(python3 -c "
import json
try:
  h=json.load(open('$OUTDIR/history.json'))
  print(h['step'][-1] if h.get('step') else 0)
except Exception:
  print(0)
")
    if [ "$cur" -ge "$target" ]; then return 0; fi
    if [ "$cur" = "" ]; then cur=0; fi
    echo "[watchdog] resuming $VARIANT at step $cur $(date +%s)" >> "$LOGS/orchestrator.log"
    python3 "$CODE/train_dqn.py" \
      --env ALE/Pong-v5 \
      --variant "$VARIANT" \
      --total-steps "$target" \
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
      --out-dir "$OUTDIR" >> "$LOG" 2>&1
    sleep 5
  done
}

# Wait for the original orchestrator to give up or finish before kicking in.
# Concretely: check whether a train_dqn.py for each variant is still running.
need_run() {
  local outdir=$1
  # If history shows step < 2M and no train_dqn.py points at this outdir, we need to start it.
  local cur
  cur=$(python3 -c "
import json
try:
  print(json.load(open('$outdir/history.json'))['step'][-1])
except Exception:
  print(0)
")
  if [ "$cur" -ge 2000000 ]; then echo "done"; return; fi
  if pgrep -af "train_dqn.py.*--out-dir $outdir" >/dev/null; then echo "running"; return; fi
  echo "need"
}

# Loop forever, kick in only when a variant is paused.
while true; do
  for tup in "vanilla:cuda:0:pong_vanilla" "double:cuda:1:pong_double" "dueling:cuda:0:pong_dueling"; do
    IFS=: read -r V D1 D2 SUB <<<"$tup"
    DEV="$D1:$D2"
    state=$(need_run "$RESULTS/$SUB")
    if [ "$state" = "need" ]; then
      resume_until_done "$V" "$DEV" "$RESULTS/$SUB" "$LOGS/${SUB}.log" &
    fi
    if [ "$state" = "done" ]; then
      :
    fi
  done
  # Check completion: if all three are at target steps, exit.
  done_count=0
  for SUB in pong_vanilla pong_double pong_dueling; do
    cur=$(python3 -c "
import json
try:
  print(json.load(open('$RESULTS/$SUB/history.json'))['step'][-1])
except Exception:
  print(0)
")
    if [ "$cur" -ge 2000000 ]; then done_count=$((done_count+1)); fi
  done
  if [ "$done_count" -ge 3 ]; then
    echo "watchdog_all_pong_done $(date +%s)" >> "$LOGS/orchestrator.log"
    break
  fi
  sleep 30
done
