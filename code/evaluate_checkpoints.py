"""Run greedy evaluation on every saved checkpoint (model_step*.pt) and
inject the results back into history.json under eval_step/eval_mean/eval_std.

This decouples evaluation from training -- if eval crashes (which we saw on
this setup) we don't lose training progress.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import gymnasium as gym

from dqn_agent import DQNAgent
from wrappers import make_atari


def make_env(env_id, seed):
    if env_id.startswith("ALE/"):
        return make_atari(env_id, seed=seed)
    return gym.make(env_id)


def evaluate(agent, env_id, episodes, seed):
    env = make_env(env_id, seed)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            a = agent.act(obs, epsilon=0.01)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
        rets.append(float(total))
    env.close()
    return rets


def step_from_filename(p):
    m = re.search(r"model_step(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--variant", choices=["vanilla", "double", "dueling"], required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.results_dir, "model_step*.pt")), key=step_from_filename)
    if not ckpts:
        print("No model_step*.pt checkpoints found in", args.results_dir)
        return

    # Build agent from any checkpoint to get obs_space; load history.
    env = make_env(args.env, args.seed)
    agent = DQNAgent(env.observation_space, env.action_space.n, variant=args.variant, device=args.device)
    env.close()

    hist_path = os.path.join(args.results_dir, "history.json")
    with open(hist_path) as f:
        history = json.load(f)

    eval_step = list(history.get("eval_step", []))
    eval_mean = list(history.get("eval_mean", []))
    eval_std = list(history.get("eval_std", []))

    done_steps = set(eval_step)
    for ckpt in ckpts:
        step = step_from_filename(ckpt)
        if step in done_steps:
            continue
        agent.load(ckpt)
        rets = evaluate(agent, args.env, args.episodes, args.seed)
        m, s = float(np.mean(rets)), float(np.std(rets))
        eval_step.append(step); eval_mean.append(m); eval_std.append(s)
        print(f"step={step:>8d} mean={m:>7.2f} std={s:>5.2f} ({args.episodes} eps)")

    # sort by step
    order = np.argsort(eval_step)
    history["eval_step"] = [int(eval_step[i]) for i in order]
    history["eval_mean"] = [float(eval_mean[i]) for i in order]
    history["eval_std"] = [float(eval_std[i]) for i in order]
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"Updated {hist_path} with {len(history['eval_step'])} eval points")


if __name__ == "__main__":
    main()
