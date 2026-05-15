"""PPO training, using Stable-Baselines3.

Supports both classic-control / Box2D envs (LunarLander, default MlpPolicy) and
Atari (CnnPolicy + the standard Atari wrappers via SB3's ``make_atari_env``).
We use SB3 [Raffin et al. 2021] as a strong, well-tested baseline so that the
DQN-vs-PPO comparison is fair: both are reasonable off-the-shelf settings, and
neither implementation gives the other an unfair speed advantage.

References:
    [Raffin21] A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement
               Learning Implementations", JMLR 2021.
    [Schulman17] J. Schulman et al., "Proximal Policy Optimization Algorithms",
                 arXiv:1707.06347, 2017.
"""
import argparse
import json
import os
import time

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def _is_atari(env_id: str) -> bool:
    return env_id.startswith("ALE/") or "NoFrameskip" in env_id


_ALE_REGISTERED = False


def _ensure_ale_registered():
    """SB3's atari wrappers call gym.make, which needs the ALE namespace
    registered. Gymnasium >=1.0 does not auto-register ale_py, so we do it
    explicitly in the main process before vec env construction."""
    global _ALE_REGISTERED
    if not _ALE_REGISTERED:
        import ale_py  # noqa: F401
        gym.register_envs(ale_py)
        _ALE_REGISTERED = True


def _make_eval_env(env_id, seed, atari, n_stack=4):
    """Build a 1-env vec eval env that matches the training preprocessing."""
    if atari:
        _ensure_ale_registered()
        env = make_atari_env(env_id, n_envs=1, seed=seed + 10_000)
        env = VecFrameStack(env, n_stack=n_stack)
        return env, True
    return gym.make(env_id), False


class HistoryCallback(BaseCallback):
    """Record episode returns / lengths and periodic evaluations."""

    def __init__(self, eval_env_id, eval_every, n_eval=10, seed=0, atari=False):
        super().__init__()
        self.eval_env_id = eval_env_id
        self.eval_every = eval_every
        self.n_eval = n_eval
        self.seed = seed
        self.atari = atari
        self.history = {
            "step": [], "ep_return": [], "ep_len": [],
            "eval_step": [], "eval_mean": [], "eval_std": [],
        }

    def _on_step(self):
        # Episode info comes from Monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.history["step"].append(int(self.num_timesteps))
                self.history["ep_return"].append(float(info["episode"]["r"]))
                self.history["ep_len"].append(int(info["episode"]["l"]))
        if self.num_timesteps and self.num_timesteps % self.eval_every == 0:
            mean_r, std_r = self._eval()
            self.history["eval_step"].append(int(self.num_timesteps))
            self.history["eval_mean"].append(mean_r)
            self.history["eval_std"].append(std_r)
            if self.verbose:
                print(f"[PPO] EVAL step={self.num_timesteps} mean={mean_r:.2f} +/- {std_r:.2f}", flush=True)
        return True

    def _eval(self):
        env, is_vec = _make_eval_env(self.eval_env_id, self.seed, self.atari)
        returns = []
        if is_vec:
            # Vectorized eval env (1 env). Run n_eval episodes by detecting `done`.
            for ep in range(self.n_eval):
                obs = env.reset()
                done = False
                total = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, dones, infos = env.step(action)
                    total += float(r[0])
                    if dones[0]:
                        # SB3 atari env auto-resets internally; episode return
                        # is also in info["episode"]["r"] (raw env return after EpisodicLife wrapper).
                        if "episode" in infos[0]:
                            total = float(infos[0]["episode"]["r"])
                        done = True
                returns.append(total)
            env.close()
        else:
            for ep in range(self.n_eval):
                obs, _ = env.reset(seed=self.seed + 10_000 + ep)
                done = False
                total = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = env.step(action)
                    done = term or trunc
                    total += r
                returns.append(total)
            env.close()
        return float(np.mean(returns)), float(np.std(returns))


def _linear_schedule(initial: float):
    """Return a callable lr/clip schedule that linearly decays initial -> 0."""
    def _f(progress_remaining: float) -> float:
        return progress_remaining * initial
    return _f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--clip-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--n-stack", type=int, default=4, help="Frame stack (Atari only)")
    parser.add_argument("--atari", action="store_true",
                        help="Force Atari mode (CnnPolicy + atari wrappers). "
                             "Auto-detected from --env when it starts with ALE/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-every", type=int, default=20_000)
    args = parser.parse_args()

    atari = args.atari or _is_atari(args.env)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = dict(vars(args))
    cfg["atari_detected"] = atari
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Build training env
    if atari:
        _ensure_ale_registered()
        # SB3's make_atari_env applies the canonical preprocessing
        # (NoopReset, MaxAndSkip(4), EpisodicLife, FireReset, WarpFrame(84x84),
        # ClipReward). VecFrameStack adds the 4-frame stack.
        vec_env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed)
        vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)
        policy = "CnnPolicy"
    else:
        from stable_baselines3.common.monitor import Monitor

        def make_one(rank):
            def _thunk():
                env = gym.make(args.env)
                env = Monitor(env)
                env.reset(seed=args.seed + rank)
                return env
            return _thunk

        vec_env = DummyVecEnv([make_one(i) for i in range(args.n_envs)])
        policy = "MlpPolicy"

    lr = _linear_schedule(args.lr) if args.lr_schedule == "linear" else args.lr
    clip = _linear_schedule(args.clip_range) if args.clip_schedule == "linear" else args.clip_range

    model = PPO(
        policy,
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=clip,
        learning_rate=lr,
        seed=args.seed,
        device=args.device,
        verbose=0,
    )

    cb = HistoryCallback(args.env, eval_every=args.eval_every, n_eval=10,
                         seed=args.seed, atari=atari)
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    # Final eval
    mean_r, std_r = cb._eval()
    cb.history["final_eval_mean"] = mean_r
    cb.history["final_eval_std"] = std_r
    cb.history["wall_time_sec"] = elapsed
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(cb.history, f)
    model.save(os.path.join(args.out_dir, "model.zip"))
    print(f"[PPO] DONE elapsed={elapsed/60:.1f}min final={mean_r:.2f}+/-{std_r:.2f}")


if __name__ == "__main__":
    main()
