"""Plot learning curves with seed averaging (mean line + per-seed shaded band)."""
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

RES = "/mnt/nvme0n1/ia313553058/Others/AI_3/results"


def load(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def smoothed_curve_on_grid(history, k=30, grid=None):
    """Project (step, return) to common grid; return smoothed values."""
    if history is None or not history.get("step"):
        return None
    steps = np.asarray(history["step"], dtype=float)
    rets = np.asarray(history["ep_return"], dtype=float)
    if len(rets) < k:
        return None
    # 30-ep moving average on the time-series of returns
    ma = np.convolve(rets, np.ones(k) / k, mode="valid")
    ma_steps = steps[k - 1:]
    # Linear interpolate onto grid
    if grid is None:
        return ma_steps, ma
    return np.interp(grid, ma_steps, ma, left=np.nan, right=np.nan)


def plot_seeds(runs_dict, out_path, title, ylabel="Episode return (30-ep MA)", x_scale=1e-6):
    """runs_dict: dict {label: (color, [history,...])} where the list spans seeds."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, (color, runs) in runs_dict.items():
        if not runs:
            continue
        # Common grid across the longest seed.
        max_step = 0
        for h in runs:
            if h and h.get("step"):
                max_step = max(max_step, max(h["step"]))
        grid = np.linspace(0, max_step, 400)
        ys = []
        for h in runs:
            r = smoothed_curve_on_grid(h, k=30, grid=grid)
            if r is not None:
                ys.append(r)
        if not ys:
            continue
        ys = np.asarray(ys)
        mean = np.nanmean(ys, axis=0)
        std = np.nanstd(ys, axis=0)
        ax.plot(grid * x_scale, mean, label=f"{label} (n={len(ys)})",
                color=color, linewidth=1.6)
        ax.fill_between(grid * x_scale, mean - std, mean + std,
                        color=color, alpha=0.18)
    ax.set_xlabel("Environment steps (millions)" if x_scale == 1e-6 else "Environment steps (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    out_dir = "/mnt/nvme0n1/ia313553058/Others/AI_3/report/figures"
    os.makedirs(out_dir, exist_ok=True)

    # Pong (2 seeds where available)
    pong = {
        "Vanilla DQN": ("tab:blue", [load(f"{RES}/pong_vanilla/history.json"),
                                      load(f"{RES}/pong_vanilla_s0/history.json")]),
        "Double DQN": ("tab:orange", [load(f"{RES}/pong_double/history.json"),
                                       load(f"{RES}/pong_double_s0/history.json")]),
        "Dueling Double DQN": ("tab:green", [load(f"{RES}/pong_dueling/history.json"),
                                              load(f"{RES}/pong_dueling_s0/history.json")]),
    }
    # Filter Nones
    pong = {k: (c, [h for h in runs if h is not None]) for k, (c, runs) in pong.items()}
    plot_seeds(pong, f"{out_dir}/pong_multiseed.pdf",
               "Pong: training-time return, multi-seed (mean $\\pm$ std)",
               x_scale=1e-6)

    # LunarLander (3 seeds where available)
    lander = {
        "DQN (Double)": ("tab:orange", [load(f"{RES}/lander_double/history.json"),
                                         load(f"{RES}/lander_double_s0/history.json"),
                                         load(f"{RES}/lander_double_s7/history.json")]),
        "DQN (Dueling)": ("tab:green", [load(f"{RES}/lander_dueling/history.json"),
                                         load(f"{RES}/lander_dueling_s0/history.json"),
                                         load(f"{RES}/lander_dueling_s7/history.json")]),
    }
    lander = {k: (c, [h for h in runs if h is not None]) for k, (c, runs) in lander.items()}
    plot_seeds(lander, f"{out_dir}/lander_multiseed.pdf",
               "LunarLander: training-time return, multi-seed (mean $\\pm$ std)",
               x_scale=1e-3)


if __name__ == "__main__":
    main()
