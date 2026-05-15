"""Generate LaTeX tables for the report from training/evaluation JSONs."""
import argparse
import json
import os

import numpy as np


def load(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pong_final_table(results_dir, out_path):
    """Compare final eval scores across DQN variants on Pong."""
    rows = []
    for label, sub in [
        ("Vanilla DQN", "pong_vanilla"),
        ("Double DQN", "pong_double"),
        ("Dueling Double DQN", "pong_dueling"),
    ]:
        h = load(os.path.join(results_dir, sub, "history.json"))
        e = load(os.path.join(results_dir, sub, "eval100.json"))
        if h is None and e is None:
            continue
        # Use 100-ep eval if available, else final_eval from history
        if e is not None:
            mean, std = e["mean"], e["std"]
            minv, maxv, n = e["min"], e["max"], e["n_episodes"]
        else:
            mean, std = h.get("final_eval_mean", float("nan")), h.get("final_eval_std", float("nan"))
            minv = maxv = float("nan"); n = 20
        # Compute training-time best from 100-ep moving avg
        rets = np.asarray(h["ep_return"]) if h else np.array([])
        if len(rets) >= 100:
            ma = np.convolve(rets, np.ones(100) / 100, mode="valid")
            best_train = float(np.max(ma))
        else:
            best_train = float("nan")
        rows.append((label, mean, std, minv, maxv, n, best_train))

    s = []
    s.append(r"\begin{table}[h]\centering\small")
    s.append(r"\caption{Pong: final greedy evaluation (100 episodes) and best 100-episode "
             r"training-time average across DQN variants. All variants share identical "
             r"hyperparameters, replay buffer, and seed.}")
    s.append(r"\label{tab:pong_final}")
    s.append(r"\begin{tabular}{lrrrrrr}")
    s.append(r"\toprule")
    s.append(r"Variant & Mean & Std & Min & Max & $n$ & Best train (100-ep MA) \\")
    s.append(r"\midrule")
    for label, mean, std, mn, mx, n, bt in rows:
        s.append(f"{label} & {mean:.2f} & {std:.2f} & {mn:.1f} & {mx:.1f} & {n} & {bt:.2f} \\\\")
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out_path}")


def lander_final_table(results_dir, out_path):
    """Compare LunarLander DQN variants and PPO."""
    rows = []
    for label, sub, kind in [
        ("DQN (Double)", "lander_double", "dqn"),
        ("DQN (Dueling Double)", "lander_dueling", "dqn"),
        ("PPO", "lander_ppo", "ppo"),
    ]:
        h = load(os.path.join(results_dir, sub, "history.json"))
        e = load(os.path.join(results_dir, sub, "eval100.json"))
        if e is None:
            e = load(os.path.join(results_dir, sub, "eval100_partial.json"))
        if h is None and e is None:
            continue
        if e is not None:
            mean, std = e["mean"], e["std"]
            mn, mx, n = e["min"], e["max"], e["n_episodes"]
        else:
            # Fallback to training-loop's final small eval (10 eps for PPO).
            mean = h.get("final_eval_mean", float("nan"))
            std = h.get("final_eval_std", float("nan"))
            mn = mx = float("nan"); n = 10
        rets = np.asarray(h["ep_return"]) if h else np.array([])
        if len(rets) >= 100:
            ma = np.convolve(rets, np.ones(100) / 100, mode="valid")
            best_train = float(np.max(ma))
        else:
            best_train = float("nan")
        wt = h.get("wall_time_sec", float("nan")) if h else float("nan")
        rows.append((label, mean, std, mn, mx, n, best_train, wt))

    s = []
    s.append(r"\begin{table}[h]\centering\small")
    s.append(r"\caption{LunarLander: final greedy/deterministic evaluation. "
             r"DQN runs are 100 greedy ($\varepsilon=0.01$) episodes from a 500\,K-step model; "
             r"PPO is 30 deterministic episodes from a 250\,K-timestep model (longer SB3 runs "
             r"crashed; see Discussion). Solved threshold is mean return $\geq 200$ (bold).}")
    s.append(r"\label{tab:lander_final}")
    s.append(r"\begin{tabular}{lrrrrrr}")
    s.append(r"\toprule")
    s.append(r"Algorithm & Mean & Std & Min & Max & $n$ & Best train (100-ep MA) \\")
    s.append(r"\midrule")
    for label, mean, std, mn, mx, n, bt, _ in rows:
        bold = mean >= 200
        if bold:
            line = f"\\textbf{{{label}}} & \\textbf{{{mean:.2f}}} & {std:.2f} & {mn:.1f} & {mx:.1f} & {n} & {bt:.2f} \\\\"
        else:
            line = f"{label} & {mean:.2f} & {std:.2f} & {mn:.1f} & {mx:.1f} & {n} & {bt:.2f} \\\\"
        s.append(line)
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out_path}")


def lr_ablation_table(results_dir, out_path):
    rows = []
    for label, sub in [
        ("$1\\times 10^{-3}$", "lander_lr1e-3"),
        ("$3\\times 10^{-4}$ (default)", "lander_lr3e-4"),
        ("$1\\times 10^{-4}$", "lander_lr1e-4"),
    ]:
        h = load(os.path.join(results_dir, sub, "history.json"))
        e = load(os.path.join(results_dir, sub, "eval100.json"))
        if h is None and e is None:
            continue
        if e is not None:
            mean, std = e["mean"], e["std"]
            n = e["n_episodes"]
        else:
            mean = h.get("final_eval_mean", float("nan"))
            std = h.get("final_eval_std", float("nan"))
            n = 10
        rets = np.asarray(h["ep_return"]) if h else np.array([])
        if len(rets) >= 100:
            ma = np.convolve(rets, np.ones(100) / 100, mode="valid")
            best_train = float(np.max(ma))
        else:
            best_train = float("nan")
        rows.append((label, mean, std, best_train, n))

    s = []
    s.append(r"\begin{table}[h]\centering\small")
    s.append(r"\caption{LunarLander DQN (Double): learning-rate sensitivity. All other "
             r"hyperparameters held fixed; same seed.}")
    s.append(r"\label{tab:lr_ablation}")
    s.append(r"\begin{tabular}{lrrrr}")
    s.append(r"\toprule")
    s.append(r"Learning rate & Final eval mean & Std & Best train (100-ep MA) & $n$ \\")
    s.append(r"\midrule")
    for label, mean, std, bt, n in rows:
        s.append(f"{label} & {mean:.2f} & {std:.2f} & {bt:.2f} & {n} \\\\")
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out_path}")


def hyperparam_table(out_path):
    """Static hyperparameter table."""
    s = []
    s.append(r"\begin{table}[h]\centering\small")
    s.append(r"\caption{Key hyperparameters. DQN settings follow \citet{mnih2015human} "
             r"with a slightly shorter exploration ramp and replay buffer to fit our compute budget. "
             r"PPO settings follow the SB3 LunarLander tuning.}")
    s.append(r"\label{tab:hyperparams}")
    s.append(r"\begin{tabular}{lll}")
    s.append(r"\toprule")
    s.append(r"Hyperparameter & Pong DQN family & LunarLander DQN \\")
    s.append(r"\midrule")
    s.append(r"Total environment steps & 2{,}000{,}000 & 500{,}000 \\")
    s.append(r"Optimizer & Adam ($\beta_1=0.9,\beta_2=0.999$) & Adam \\")
    s.append(r"Learning rate & $1\!\times\!10^{-4}$ & $3\!\times\!10^{-4}$ \\")
    s.append(r"Discount $\gamma$ & 0.99 & 0.99 \\")
    s.append(r"Replay buffer size & 100{,}000 & 100{,}000 \\")
    s.append(r"Batch size & 32 & 64 \\")
    s.append(r"Learning starts at step & 50{,}000 & 5{,}000 \\")
    s.append(r"Train freq (env steps) & 4 & 4 \\")
    s.append(r"Target net sync (steps) & 1{,}000 & 1{,}000 \\")
    s.append(r"Exploration $\varepsilon$ start $\to$ end & $1.0\to 0.01$ & $1.0\to 0.05$ \\")
    s.append(r"Exploration linear-decay fraction & $0.10$ & $0.10$ \\")
    s.append(r"Loss & Huber & Huber \\")
    s.append(r"Gradient clipping & 10.0 & 10.0 \\")
    s.append(r"\midrule")
    s.append(r"\textbf{PPO (LunarLander)} & \multicolumn{2}{l}{}\\")
    s.append(r"Parallel envs & \multicolumn{2}{l}{8} \\")
    s.append(r"Rollout length & \multicolumn{2}{l}{1024 steps / env} \\")
    s.append(r"Mini-batch size & \multicolumn{2}{l}{64} \\")
    s.append(r"Update epochs & \multicolumn{2}{l}{4} \\")
    s.append(r"GAE $\lambda$ & \multicolumn{2}{l}{0.98} \\")
    s.append(r"$\gamma$ & \multicolumn{2}{l}{0.999} \\")
    s.append(r"Entropy coefficient & \multicolumn{2}{l}{0.01} \\")
    s.append(r"Clip $\epsilon$ & \multicolumn{2}{l}{0.2 (SB3 default)} \\")
    s.append(r"Learning rate & \multicolumn{2}{l}{$3\!\times\!10^{-4}$} \\")
    s.append(r"\bottomrule")
    s.append(r"\end{tabular}")
    s.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(s) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/mnt/nvme0n1/ia313553058/Others/AI_3/results")
    parser.add_argument("--out-dir", default="/mnt/nvme0n1/ia313553058/Others/AI_3/report/tables")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    hyperparam_table(os.path.join(args.out_dir, "hyperparams.tex"))
    pong_final_table(args.results_dir, os.path.join(args.out_dir, "pong_final.tex"))
    lander_final_table(args.results_dir, os.path.join(args.out_dir, "lander_final.tex"))
    lr_ablation_table(args.results_dir, os.path.join(args.out_dir, "lr_ablation.tex"))
