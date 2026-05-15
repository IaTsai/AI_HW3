#!/usr/bin/env bash
# Wait until pong_double has finished on GPU 1, then launch all LunarLander
# experiments on GPU 1 so the two GPUs do not sit idle.
set -u

LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs
CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code

until grep -q "double_done" "$LOGS/orchestrator.log" 2>/dev/null; do
  sleep 30
done
echo "watcher: pong_double finished, launching LunarLander" >> "$LOGS/orchestrator.log"
bash "$CODE/run_lander_all.sh" cuda:1
echo "watcher: lander pipeline done" >> "$LOGS/orchestrator.log"
