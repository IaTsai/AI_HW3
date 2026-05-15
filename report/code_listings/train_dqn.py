"""Generic DQN training loop used by both Pong and LunarLander runs.

Usage:
    python train_dqn.py --env ALE/Pong-v5 --variant double --total-steps 1500000 \
        --out-dir ../results/pong_double --device cuda:0
"""
import argparse
import faulthandler
import gc
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import gymnasium as gym

# Print Python tracebacks if the process crashes (SIGSEGV, SIGFPE, SIGBUS,
# SIGABRT, SIGILL). Helps localize the rare-but-real native crashes we see.
faulthandler.enable(file=sys.stderr, all_threads=True)

from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from wrappers import make_atari


def make_env(env_id, seed):
    if env_id.startswith("ALE/"):
        env = make_atari(env_id, seed=seed)
    else:
        env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def linear_schedule(start, end, fraction, current, total):
    """Linear epsilon decay over `fraction` of `total` steps, then constant."""
    progress = min(1.0, current / (fraction * total))
    return start + progress * (end - start)


def evaluate(agent, env_id, episodes=5, seed=10000):
    """Greedy evaluation. Returns mean episode reward over `episodes` runs."""
    env = make_env(env_id, seed)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            a = agent.act(obs, epsilon=0.01)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--variant", choices=["vanilla", "double", "dueling"], default="vanilla")
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--resume", action="store_true",
                        help="If set and out-dir contains model.pt + history.json, resume from the "
                             "last eval checkpoint instead of starting from scratch.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.env, args.seed)
    obs_space = env.observation_space
    num_actions = env.action_space.n
    print(f"[{args.variant}] env={args.env} obs={obs_space.shape} actions={num_actions} device={args.device}")

    agent = DQNAgent(
        obs_space=obs_space,
        num_actions=num_actions,
        variant=args.variant,
        device=args.device,
        lr=args.lr,
        gamma=args.gamma,
    )

    obs_dtype = np.uint8 if obs_space.dtype == np.uint8 else np.float32
    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_space.shape,
        obs_dtype=obs_dtype,
        device=args.device,
    )

    history = {
        "step": [], "ep_return": [], "ep_len": [],
        "loss_step": [], "loss": [], "q_mean": [],
        "eval_step": [], "eval_mean": [], "eval_std": [],
    }

    # Optional resume: load model + history, skip to last eval step.
    # Importantly, truncate any entries in history whose step is greater than
    # the resume step. Otherwise crash-restarted runs leave stale episodes
    # (from the post-checkpoint, pre-crash window) in the file and the plotted
    # learning curve shows non-monotone artifacts at the join.
    start_step = 1
    if args.resume:
        ckpt = os.path.join(args.out_dir, "model.pt")
        hist_path = os.path.join(args.out_dir, "history.json")
        if os.path.exists(ckpt) and os.path.exists(hist_path):
            agent.load(ckpt)
            with open(hist_path) as f:
                history = json.load(f)
            # Determine resume point from last checkpoint
            if history.get("step"):
                # Use the highest step <= last_eval as the resume step (since
                # model.pt was saved at the last eval).
                if history.get("eval_step"):
                    start_step = int(history["eval_step"][-1]) + 1
                else:
                    start_step = int(max(history["step"])) + 1

                def _truncate(lst, step_lst, keep_to):
                    """Keep entries whose corresponding step is <= keep_to."""
                    return [v for v, s in zip(lst, step_lst) if s < keep_to]

                # Truncate per-episode arrays so they end exactly at the resume step.
                steps = history.get("step", [])
                history["ep_return"] = _truncate(history.get("ep_return", []), steps, start_step)
                history["ep_len"] = _truncate(history.get("ep_len", []), steps, start_step)
                history["step"] = [s for s in steps if s < start_step]
                # Same for training stats keyed on loss_step.
                loss_steps = history.get("loss_step", [])
                history["loss"] = _truncate(history.get("loss", []), loss_steps, start_step)
                history["q_mean"] = _truncate(history.get("q_mean", []), loss_steps, start_step)
                history["loss_step"] = [s for s in loss_steps if s < start_step]
                last_eval = history["eval_mean"][-1] if history.get("eval_mean") else float("nan")
                print(f"[{args.variant}] RESUMING from step {start_step} "
                      f"(eval={last_eval:.2f}, kept {len(history['ep_return'])} eps in history)",
                      flush=True)

    return_buf = deque(maxlen=100)
    len_buf = deque(maxlen=100)

    obs, _ = env.reset(seed=args.seed + start_step)
    ep_return = 0.0
    ep_len = 0
    start_time = time.time()
    last_log = start_time

    for step in range(start_step, args.total_steps + 1):
        eps = linear_schedule(args.eps_start, args.eps_end, args.eps_decay_fraction, step, args.total_steps)
        action = agent.act(obs, epsilon=eps)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, terminated)  # only "real" terminations bootstrap-zero
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done:
            return_buf.append(ep_return)
            len_buf.append(ep_len)
            history["step"].append(step)
            history["ep_return"].append(float(ep_return))
            history["ep_len"].append(int(ep_len))
            ep_return = 0.0
            ep_len = 0
            obs, _ = env.reset()

        # Train
        if step >= args.learning_starts and step % args.train_freq == 0 and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            loss, q_mean = agent.update(batch)
            if step % args.log_every == 0:
                history["loss_step"].append(step)
                history["loss"].append(loss)
                history["q_mean"].append(q_mean)

        # Sync target net
        if step % args.target_update_freq == 0:
            agent.sync_target()

        # Periodic logging
        if step % args.log_every == 0:
            now = time.time()
            fps = args.log_every / max(now - last_log, 1e-6)
            last_log = now
            mean_ret = np.mean(return_buf) if return_buf else float("nan")
            mean_len = np.mean(len_buf) if len_buf else float("nan")
            print(
                f"[{args.variant}] step={step:>7d} "
                f"eps={eps:.3f} "
                f"avg_return(100)={mean_ret:>7.2f} "
                f"avg_len(100)={mean_len:>6.1f} "
                f"fps={fps:>5.0f} "
                f"elapsed={(now-start_time)/60:.1f}min",
                flush=True,
            )

        # Periodic checkpoint. We deliberately do NOT run inline evaluation during
        # training because spinning up a fresh ALE env every few hundred-thousand
        # steps proved unstable on our setup (sporadic SIGSEGV in the ALE C
        # bindings after extended use). Instead, we (a) track training-time
        # episode return -- which closely tracks greedy performance once
        # epsilon decays to its final value -- and (b) run a separate 100-episode
        # greedy evaluation from a clean process after training.
        if step % args.eval_every == 0 and step >= args.learning_starts:
            agent.save(os.path.join(args.out_dir, "model.pt"))
            # also keep periodic snapshots so we can plot eval-by-checkpoint later
            agent.save(os.path.join(args.out_dir, f"model_step{step}.pt"))
            with open(os.path.join(args.out_dir, "history.json"), "w") as f:
                json.dump(history, f)

    # Final save. Final greedy evaluation is done by a separate process
    # (evaluate.py) on the saved checkpoint, so that an eval-time crash cannot
    # destroy the training artifacts.
    agent.save(os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"[{args.variant}] TRAIN DONE (run evaluate.py for 100-ep greedy eval)")
    env.close()


if __name__ == "__main__":
    main()
