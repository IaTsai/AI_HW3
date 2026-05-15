"""Greedy eval for a trained SB3 PPO model. Supports MLP envs (LunarLander)
and Atari envs (CnnPolicy + the standard atari wrappers + frame stack)."""
import argparse
import json

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def _is_atari(env_id: str) -> bool:
    return env_id.startswith("ALE/") or "NoFrameskip" in env_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    atari = _is_atari(args.env)
    model = PPO.load(args.model, device="cpu")

    returns, lengths = [], []
    if atari:
        import ale_py
        gym.register_envs(ale_py)
        # Single-env VecEnv that auto-resets after `done`. Build n_episodes
        # rollouts by reading the Monitor info dict, which carries the
        # episode return *before* any reward clipping (raw game score).
        env = make_atari_env(args.env, n_envs=1, seed=args.seed)
        env = VecFrameStack(env, n_stack=args.n_stack)
        obs = env.reset()
        steps = 0
        while len(returns) < args.n_episodes:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(a)
            steps += 1
            if dones[0]:
                raw_return = float(infos[0]["episode"]["r"]) if "episode" in infos[0] else float(r[0])
                returns.append(raw_return)
                lengths.append(int(infos[0]["episode"]["l"]) if "episode" in infos[0] else steps)
                steps = 0
        env.close()
    else:
        env = gym.make(args.env)
        for ep in range(args.n_episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            total, steps = 0.0, 0
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                total += r
                steps += 1
            returns.append(float(total))
            lengths.append(int(steps))

    summary = {
        "env": args.env,
        "algo": "PPO",
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
    print(f"[PPO] {args.env} | mean={summary['mean']:.2f} std={summary['std']:.2f}")


if __name__ == "__main__":
    main()
