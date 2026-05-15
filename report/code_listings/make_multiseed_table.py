"""Aggregate multi-seed results into a LaTeX table.

For each algorithm, compute mean ± std across seeds of:
  - best-100-episode training average (within each run, then averaged)
  - final 30/100-episode eval (for the run that was evaluated)
"""
import json
import os
import numpy as np


RES = "/mnt/nvme0n1/ia313553058/Others/AI_3/results"


def load_history(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def best_train_ma(history, k=100):
    if history is None or "ep_return" not in history:
        return float("nan")
    r = np.asarray(history["ep_return"], dtype=float)
    if len(r) < k:
        return float("nan")
    ma = np.convolve(r, np.ones(k) / k, mode="valid")
    return float(ma.max())


def final_train_ma(history, k=100):
    if history is None or "ep_return" not in history:
        return float("nan")
    r = np.asarray(history["ep_return"], dtype=float)
    if len(r) < k:
        return float("nan")
    ma = np.convolve(r, np.ones(k) / k, mode="valid")
    return float(ma[-1])


def aggregate(runs, k=100):
    """runs: list of history dicts.  Returns (mean_best, std_best, mean_final, std_final)."""
    bests = [best_train_ma(h, k) for h in runs if h is not None]
    finals = [final_train_ma(h, k) for h in runs if h is not None]
    bests = [b for b in bests if not np.isnan(b)]
    finals = [b for b in finals if not np.isnan(b)]
    return (np.mean(bests), np.std(bests), np.mean(finals), np.std(finals), len(bests))


def write_pong_table(out_path):
    """Pong: 3 variants, 2 seeds each.
    The 'seed=0' runs are 1.5M-step shorter budget; use the best they
    achieved as a robust summary across seeds."""
    rows = []
    for label, sub_main, sub_extra in [
        ("Vanilla DQN", "pong_vanilla", "pong_vanilla_s0"),
        ("Double DQN", "pong_double", "pong_double_s0"),
        ("Dueling Double DQN", "pong_dueling", "pong_dueling_s0"),
    ]:
        h_main = load_history(f"{RES}/{sub_main}/history.json")
        h_extra = load_history(f"{RES}/{sub_extra}/history.json")
        mean_b, std_b, mean_f, std_f, n = aggregate([h_main, h_extra], k=100)
        rows.append((label, mean_b, std_b, mean_f, std_f, n))

    s = []
    s.append(r"\begin{table}[h]\centering\small")
    s.append(r"\caption{Pong multi-seed summary. Seed~42 trained for 2\,M env "
             r"steps; seed~0 for 1.5\,M (to fit our compute budget). Reported "
             r"values are mean$\pm$std \emph{across seeds} of (a) the best "
             r"100-episode moving-average return during training and (b) the "
             r"final 100-episode MA. $n$ is the number of seeds with $\geq 100$ "
             r"completed episodes.}")
    s.append(r"\label{tab:pong_multiseed}")
    s.append(r"\begin{tabular}{lccccc}")
    s.append(r"\toprule")
    s.append(r"Variant & Best train (mean$\pm$std) & Final train (mean$\pm$std) & $n$ seeds \\")
    s.append(r"\midrule")
    for label, mb, sb, mf, sf, n in rows:
        s.append(f"{label} & ${mb:.2f}\\pm{sb:.2f}$ & ${mf:.2f}\\pm{sf:.2f}$ & {n} \\\\")
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out_path}")


def write_lander_table(out_path):
    rows = []
    for label, subs in [
        ("DQN (Double)", ["lander_double", "lander_double_s0", "lander_double_s7"]),
        ("DQN (Dueling Double)", ["lander_dueling", "lander_dueling_s0", "lander_dueling_s7"]),
    ]:
        runs = [load_history(f"{RES}/{s}/history.json") for s in subs]
        mean_b, std_b, mean_f, std_f, n = aggregate(runs, k=100)
        rows.append((label, mean_b, std_b, mean_f, std_f, n))

    s = []
    s.append(r"\begin{table}[h]\centering\small")
    s.append(r"\caption{LunarLander multi-seed summary (3 seeds per variant: 42, 0, 7). "
             r"All runs use 500\,K env steps and identical hyperparameters. "
             r"Solved threshold is mean return $\geq 200$.}")
    s.append(r"\label{tab:lander_multiseed}")
    s.append(r"\begin{tabular}{lccc}")
    s.append(r"\toprule")
    s.append(r"Algorithm & Best train (mean$\pm$std across seeds) & Final train (mean$\pm$std) & $n$ seeds \\")
    s.append(r"\midrule")
    for label, mb, sb, mf, sf, n in rows:
        bold_b = mb >= 200
        if bold_b:
            line = (f"\\textbf{{{label}}} & $\\mathbf{{{mb:.2f}\\pm{sb:.2f}}}$ & "
                    f"${mf:.2f}\\pm{sf:.2f}$ & {n} \\\\")
        else:
            line = f"{label} & ${mb:.2f}\\pm{sb:.2f}$ & ${mf:.2f}\\pm{sf:.2f}$ & {n} \\\\"
        s.append(line)
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = "/mnt/nvme0n1/ia313553058/Others/AI_3/report/tables"
    os.makedirs(out_dir, exist_ok=True)
    write_pong_table(os.path.join(out_dir, "pong_multiseed.tex"))
    write_lander_table(os.path.join(out_dir, "lander_multiseed.tex"))
