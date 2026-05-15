"""Generate all plots and LaTeX tables for the report from training histories."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 130,
})


def load(path):
    with open(path) as f:
        return json.load(f)


def smooth(x, k=20):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    pad = np.full(k - 1, x[0])
    return np.convolve(np.concatenate([pad, x]), np.ones(k) / k, mode="valid")


def plot_learning_curves(runs, out_path, title, ylabel="Episode return", x_scale=1.0):
    """runs: list of (label, history_dict, color).
    x_scale=1e-6 to display steps in millions, etc."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, h, color in runs:
        steps = np.asarray(h["step"]) * x_scale
        rets = np.asarray(h["ep_return"])
        if len(rets) == 0:
            continue
        smoothed = smooth(rets, k=30)
        ax.plot(steps, smoothed, label=label, color=color, linewidth=1.6)
        # Light raw scatter
        ax.scatter(steps, rets, s=2, color=color, alpha=0.15)
    ax.set_xlabel("Environment steps" if x_scale == 1.0 else "Environment steps (millions)" if x_scale == 1e-6 else "Environment steps (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_eval_curves(runs, out_path, title, x_scale=1.0):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, h, color in runs:
        if "eval_step" not in h or not h["eval_step"]:
            continue
        s = np.asarray(h["eval_step"]) * x_scale
        m = np.asarray(h["eval_mean"])
        d = np.asarray(h["eval_std"])
        ax.plot(s, m, label=label, color=color, marker="o", markersize=4, linewidth=1.6)
        ax.fill_between(s, m - d, m + d, color=color, alpha=0.15)
    ax.set_xlabel("Environment steps" if x_scale == 1.0 else "Environment steps (millions)" if x_scale == 1e-6 else "Environment steps (thousands)")
    ax.set_ylabel("Greedy eval mean return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_q_mean(runs, out_path, title):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, h, color in runs:
        if "loss_step" not in h or not h["loss_step"]:
            continue
        s = np.asarray(h["loss_step"]) * 1e-6
        q = np.asarray(h["q_mean"])
        ax.plot(s, smooth(q, k=50), label=label, color=color, linewidth=1.4)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Mean predicted Q (batch avg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def bar_final(rows, out_path, title, ylabel="Mean return (100 eps)"):
    """rows: list of (label, mean, std, color)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    xs = np.arange(len(rows))
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    ax.bar(xs, means, yerr=stds, color=colors, capsize=6, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/mnt/nvme0n1/ia313553058/Others/AI_3/results")
    parser.add_argument("--out-dir", default="/mnt/nvme0n1/ia313553058/Others/AI_3/report/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    R = args.results_dir

    # ------- Pong (DQN variants only) -------
    try:
        runs = []
        for label, sub, color in [
            ("Vanilla DQN", "pong_vanilla", "tab:blue"),
            ("Double DQN", "pong_double", "tab:orange"),
            ("Dueling Double DQN", "pong_dueling", "tab:green"),
        ]:
            p = f"{R}/{sub}/history.json"
            if os.path.exists(p):
                runs.append((label, load(p), color))
        if runs:
            plot_learning_curves(runs, f"{args.out_dir}/pong_learning.pdf",
                                 "Pong: training-time episode return (30-ep MA)",
                                 x_scale=1e-6)
            plot_eval_curves(runs, f"{args.out_dir}/pong_eval.pdf",
                             "Pong: greedy evaluation (5 eps/checkpoint)",
                             x_scale=1e-6)
            plot_q_mean(runs, f"{args.out_dir}/pong_qmean.pdf",
                        "Pong: mean predicted Q-value over training")
    except FileNotFoundError as e:
        print(f"Skipping Pong plots (missing history): {e}")

    # ------- Pong: DQN vs PPO comparison at matched 1M budget -------
    try:
        runs = []
        for label, sub, color in [
            ("Vanilla DQN", "pong_vanilla", "tab:blue"),
            ("Double DQN", "pong_double", "tab:orange"),
            ("Dueling Double DQN", "pong_dueling", "tab:green"),
            ("PPO (CnnPolicy)", "pong_ppo_s42", "tab:red"),
        ]:
            p = f"{R}/{sub}/history.json"
            if os.path.exists(p):
                runs.append((label, load(p), color))
        if any(r[0].startswith("PPO") for r in runs):
            plot_learning_curves(
                runs, f"{args.out_dir}/pong_dqn_vs_ppo.pdf",
                "Pong: DQN variants vs PPO (training-time return, 30-ep MA)",
                x_scale=1e-6,
            )
    except FileNotFoundError:
        pass

    # ------- LunarLander DQN vs PPO -------
    try:
        ll_runs = []
        for label, sub, color in [
            ("DQN (Double)", "lander_double", "tab:orange"),
            ("DQN (Dueling)", "lander_dueling", "tab:green"),
            ("PPO", "lander_ppo", "tab:red"),
        ]:
            p = f"{R}/{sub}/history.json"
            if os.path.exists(p):
                ll_runs.append((label, load(p), color))
        if ll_runs:
            plot_learning_curves(ll_runs, f"{args.out_dir}/lander_learning.pdf",
                                 "LunarLander: training-time episode return (30-ep MA)",
                                 x_scale=1e-3)
            plot_eval_curves(ll_runs, f"{args.out_dir}/lander_eval.pdf",
                             "LunarLander: greedy evaluation curve",
                             x_scale=1e-3)
    except FileNotFoundError as e:
        print(f"Skipping LunarLander plots: {e}")

    # ------- LunarLander hyperparameter ablation (learning rate) -------
    try:
        rows = []
        for label, sub, color in [
            (r"lr = $1\times 10^{-3}$", "lander_lr1e-3", "tab:purple"),
            (r"lr = $3\times 10^{-4}$ (default)", "lander_double", "tab:gray"),
            (r"lr = $1\times 10^{-4}$", "lander_lr1e-4", "tab:brown"),
        ]:
            p = f"{R}/{sub}/history.json"
            if os.path.exists(p):
                rows.append((label, load(p), color))
        if rows:
            plot_learning_curves(rows, f"{args.out_dir}/lander_lr_ablation.pdf",
                                 "LunarLander Double DQN: learning-rate sensitivity",
                                 x_scale=1e-3)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
