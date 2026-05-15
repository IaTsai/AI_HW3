#!/usr/bin/env bash
# Bundle the report and source code into 313553058_AI_HW3.zip for Overleaf.
set -eu

REPORT=/mnt/nvme0n1/ia313553058/Others/AI_3/report
ZIP=/mnt/nvme0n1/ia313553058/Others/AI_3/313553058_AI_HW3.zip

# Refresh code listings from the latest source
for f in wrappers.py replay_buffer.py networks.py dqn_agent.py train_dqn.py \
         train_ppo.py evaluate.py evaluate_ppo.py evaluate_checkpoints.py \
         plot_results.py make_tables.py clean_history.py \
         plot_multiseed.py make_multiseed_table.py make_multiseed_combined.py \
         render_gameplay.py; do
  cp /mnt/nvme0n1/ia313553058/Others/AI_3/code/$f "$REPORT/code_listings/" 2>/dev/null || true
done

rm -f "$ZIP"
cd "$REPORT"
# Include everything Overleaf needs to compile: main.tex, references.bib,
# figures/, tables/, code_listings/.
zip -r "$ZIP" \
  main.tex \
  references.bib \
  figures/ \
  tables/ \
  code_listings/ \
  -x "*/.DS_Store" -x "*/__pycache__/*"
echo "---"
ls -la "$ZIP"
unzip -l "$ZIP" | tail -20
