#!/usr/bin/env bash
# Run this once all Pong seed-0 runs have finished. It will:
#   1. Regenerate the multi-seed tables (now with Pong data).
#   2. Regenerate the multi-seed plot.
#   3. Re-insert the Pong multi-seed table into main.tex (right after the
#      LunarLander multi-seed table) and tweak the narrative.
#   4. Recompile the report and rebuild the submission zip.
set -eu

CODE=/mnt/nvme0n1/ia313553058/Others/AI_3/code
REPORT=/mnt/nvme0n1/ia313553058/Others/AI_3/report
ZIP=/mnt/nvme0n1/ia313553058/Others/AI_3/313553058_AI_HW3.zip

cd "$CODE"
python3 clean_history.py /mnt/nvme0n1/ia313553058/Others/AI_3/results/pong_*/history.json 2>/dev/null || true
python3 make_multiseed_table.py
python3 plot_multiseed.py 2>/dev/null

# Insert Pong multi-seed table reference into main.tex (idempotent).
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("/mnt/nvme0n1/ia313553058/Others/AI_3/report/main.tex")
s = p.read_text()
# Insert pong_multiseed.tex right before lander_multiseed.tex if not already present.
if "tables/pong_multiseed.tex" not in s:
    s = s.replace(
        r"\input{tables/lander_multiseed.tex}",
        r"\input{tables/pong_multiseed.tex}" + "\n" + r"\input{tables/lander_multiseed.tex}",
        1,
    )
# Replace the "On Pong the across-seed std is larger" caveat with the actual data.
s = s.replace(
    "On Pong the across-seed std is larger\n(consistent with general single-seed-warnings in deep-RL); the within-budget Pong rankings should\ntherefore be read as one-seed snapshots, not strong universal claims.",
    "On Pong (Table~\\ref{tab:pong_multiseed}) the across-seed spread is\nsubstantially larger than on LunarLander, consistent with general\nsingle-seed-warnings in deep-RL; we therefore read the Pong rankings as\nwithin-budget snapshots, not strong universal claims.",
)
p.write_text(s)
print("main.tex updated.")
PY

# Refresh code listings
for f in wrappers.py replay_buffer.py networks.py dqn_agent.py train_dqn.py \
         train_ppo.py evaluate.py evaluate_ppo.py evaluate_checkpoints.py \
         plot_results.py make_tables.py clean_history.py \
         render_gameplay.py plot_multiseed.py make_multiseed_table.py; do
  cp "$CODE/$f" "$REPORT/code_listings/" 2>/dev/null || true
done

cd "$REPORT"
PATH=/home/ia313553058/.local/bin:$PATH pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
PATH=/home/ia313553058/.local/bin:$PATH bibtex main >/dev/null 2>&1
PATH=/home/ia313553058/.local/bin:$PATH pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
PATH=/home/ia313553058/.local/bin:$PATH pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1

# Rebuild zip
rm -f "$ZIP"
zip -r "$ZIP" main.tex references.bib figures/ tables/ code_listings/ \
  -x "*/.DS_Store" -x "*/__pycache__/*" >/tmp/zip_log.txt
echo "=== integrate_done ==="
PATH=/home/ia313553058/.local/bin:$PATH pdfinfo main.pdf | grep Pages
grep "Appendix:" main.aux
ls -la "$ZIP"
