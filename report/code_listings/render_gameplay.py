"""Render trained agents into a figure for the report.

For Pong: collect a strip of frames spanning one rally (random, then key
contact, then scored point). For LunarLander: collect a strip of frames
showing approach, descent, and landing, plus a trajectory plot.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym

from dqn_agent import DQNAgent
from wrappers import make_atari, _ensure_ale_registered


def render_pong(model_path, variant, out_path, device="cuda:0", seed=20251):
    """Run trained Pong agent, capture rgb_array frames around the start of
    the episode (one full rally + opening shot of next), build a horizontal
    strip figure."""
    _ensure_ale_registered()
    env_render = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env_render.reset(seed=seed)

    env_obs = make_atari("ALE/Pong-v5", seed=seed)
    obs, _ = env_obs.reset(seed=seed)
    env_render.reset(seed=seed)

    agent = DQNAgent(env_obs.observation_space, env_obs.action_space.n,
                     variant=variant, device=device)
    agent.load(model_path)

    # Run the episode and remember (step_idx, reward, rgb) so we can pick
    # frames around the first score event.
    frames, rewards = [], []
    total = 0.0
    done = False
    steps = 0
    while not done and steps < 2000:
        a = agent.act(obs, epsilon=0.01)
        obs, r, term, trunc, _ = env_obs.step(a)
        # Preprocessed env uses frame-skip 4 -- step the raw env 4 times for
        # parity, recording only the final RGB.
        for _ in range(4):
            rgb, r2, t2, tr2, _ = env_render.step(a)
            if t2 or tr2:
                break
        frames.append(rgb)
        rewards.append(r)
        total += r
        steps += 1
        done = term or trunc
    env_obs.close()
    env_render.close()

    # Find the first nonzero reward (= someone scored). Center the strip on
    # the rally that led to it so the ball is actually visible.
    score_idx = next((i for i, r in enumerate(rewards) if r != 0), len(frames) // 2)
    span = 30  # ~30 agent-frames around the score event
    start = max(0, score_idx - span)
    end = min(len(frames) - 1, score_idx + 5)
    n_panels = 6
    idxs = np.linspace(start, end, n_panels, dtype=int)

    fig, axes = plt.subplots(1, n_panels, figsize=(12, 3.4))
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        ax.set_title(f"step {i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Pong: trained {variant} DQN, one rally (final episode return: {total:+.0f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path} (episode score {total:+.0f}, {len(frames)} frames; "
          f"showing steps {idxs[0]}..{idxs[-1]} around score event at step {score_idx})")


def render_lunarlander(model_path, variant, out_path, device="cuda:0", seed=20251):
    """Run trained LunarLander agent, capture (a) a trajectory plot and
    (b) 4 key frames from one episode."""
    # Single env in rgb_array mode; obs is the 8-dim state, render() gives RGB
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    agent = DQNAgent(env.observation_space, env.action_space.n,
                     variant=variant, device=device)
    agent.load(model_path)

    frames, xs, ys = [], [], []
    total = 0.0
    done = False
    steps = 0
    while not done and steps < 1000:
        a = agent.act(obs, epsilon=0.01)
        obs, r, term, trunc, _ = env.step(a)
        frames.append(env.render())
        xs.append(float(obs[0])); ys.append(float(obs[1]))
        total += r
        steps += 1
        done = term or trunc
    env.close()

    n_panels = 4
    if len(frames) < n_panels:
        return
    idxs = np.linspace(0, len(frames) - 1, n_panels, dtype=int)

    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.3])
    for col, i in enumerate(idxs):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames[i])
        ax.set_title(f"t={i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, n_panels])
    ax.plot(xs, ys, "-", linewidth=1.5, color="tab:blue")
    ax.scatter([xs[0]], [ys[0]], color="green", marker="o", s=50, label="start", zorder=5)
    ax.scatter([xs[-1]], [ys[-1]], color="red", marker="X", s=80, label="end", zorder=5)
    # Mark landing pad approximately at (0,0)
    ax.axhline(0, color="brown", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Lander trajectory", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"LunarLander, trained {variant} DQN (episode return: {total:+.0f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path} (episode return {total:+.0f}, {len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["pong", "lander"])
    parser.add_argument("--variant", default="dueling")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20251)
    args = parser.parse_args()
    if args.env == "pong":
        render_pong(args.model, args.variant, args.out, args.device, args.seed)
    else:
        render_lunarlander(args.model, args.variant, args.out, args.device, args.seed)
