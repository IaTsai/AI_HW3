"""Run final greedy evaluation on a trained model. Saves a JSON of per-episode
returns and prints summary statistics."""
import argparse
import json
import os

import numpy as np
import torch
import gymnasium as gym

from dqn_agent import DQNAgent
from wrappers import make_atari


def make_env(env_id, seed):
    if env_id.startswith("ALE/"):
        return make_atari(env_id, seed=seed)
    return gym.make(env_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--variant", choices=["vanilla", "double", "dueling"], default="vanilla")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = make_env(args.env, args.seed)
    agent = DQNAgent(env.observation_space, env.action_space.n, variant=args.variant, device=args.device)
    agent.load(args.model)

    returns = []
    lengths = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done:
            a = agent.act(obs, epsilon=args.epsilon)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
            steps += 1
        returns.append(float(total))
        lengths.append(int(steps))

    summary = {
        "env": args.env,
        "variant": args.variant,
        "n_episodes": args.n_episodes,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "median": float(np.median(returns)),
        "returns": returns,
        "lengths": lengths,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.variant}] {args.env} | mean={summary['mean']:.2f} std={summary['std']:.2f} "
          f"min={summary['min']:.1f} max={summary['max']:.1f} (n={args.n_episodes})")


if __name__ == "__main__":
    main()
