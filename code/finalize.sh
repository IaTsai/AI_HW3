#!/usr/bin/env bash
# Run the entire post-training pipeline: clean histories, evaluate, plot, tables.
set -u

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
RESULTS=/mnt/nvme0n1/ia313553058/Others/AI_3/results
REPORT=/mnt/nvme0n1/ia313553058/Others/AI_3/report
LOGS=/mnt/nvme0n1/ia313553058/Others/AI_3/logs

cd "$CODE"

# 1. Clean history files (dedupe step entries from crash/resume).
python3 clean_history.py \
  "$RESULTS"/pong_vanilla/history.json \
  "$RESULTS"/pong_double/history.json \
  "$RESULTS"/pong_dueling/history.json \
  "$RESULTS"/lander_double/history.json \
  "$RESULTS"/lander_dueling/history.json \
  "$RESULTS"/lander_lr1e-3/history.json \
  "$RESULTS"/lander_lr1e-4/history.json \
  "$RESULTS"/lander_ppo/history.json 2>&1 | tee "$LOGS/finalize_clean.log"

# 2. Per-checkpoint evaluations (for the greedy-eval learning curves).
bash "$CODE/run_eval_all.sh"

# 3. Plots and tables.
python3 plot_results.py
python3 make_tables.py

# 4. Compile LaTeX
cd "$REPORT"
PATH=/home/ia313553058/.local/bin:$PATH pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
PATH=/home/ia313553058/.local/bin:$PATH bibtex main >/dev/null 2>&1
PATH=/home/ia313553058/.local/bin:$PATH pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
PATH=/home/ia313553058/.local/bin:$PATH pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1

echo "finalize_done $(date +%s)" >> "$LOGS/orchestrator.log"
