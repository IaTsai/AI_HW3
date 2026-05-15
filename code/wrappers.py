"""
Atari preprocessing wrappers for Gymnasium.
Implements the standard "DQN preprocessing": frame skip, max-pool over last
two raw frames, grayscale, resize to 84x84, frame stacking, reward clipping,
and episodic life signal. These wrappers are the same pipeline used in the
DeepMind "Human-level control through deep reinforcement learning" paper.
"""
import numpy as np
import cv2
import gymnasium as gym
from gymnasium.spaces import Box

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class NoopResetEnv(gym.Wrapper):
    """Sample a random number of no-ops on env.reset to add stochasticity."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that require it (e.g., Breakout)."""

    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[1] == "FIRE"
        assert len(meanings) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        else:
            info = {}
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """End an episode after each life lost, but only reset env after lives==0.
    Helps value estimation by making "losing a life" a terminal signal."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Skip k frames, returning max over last 2 raw frames."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    """Convert frame to grayscale and resize to 84x84."""

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(low=0, high=255, shape=(height, width), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, 1} as in the original DQN paper."""

    def reward(self, reward):
        return float(np.sign(reward))


class FrameStack(gym.Wrapper):
    """Stack k consecutive frames along channel axis. Outputs uint8 (k, H, W)."""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = np.zeros((k,) + env.observation_space.shape, dtype=np.uint8)
        self.observation_space = Box(
            low=0, high=255, shape=(k,) + env.observation_space.shape, dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.k):
            self.frames[i] = obs
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # In-place slide instead of np.roll (which allocates a new array each
        # step). Reduces per-step memory churn.
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames.copy(), reward, terminated, truncated, info


_ALE_REGISTERED = False


def _ensure_ale_registered():
    global _ALE_REGISTERED
    if not _ALE_REGISTERED:
        import ale_py
        gym.register_envs(ale_py)
        _ALE_REGISTERED = True


def make_atari(env_id, clip_rewards=True, episode_life=True, frame_stack=4, seed=None):
    """Build a fully-preprocessed Atari env. env_id e.g. 'ALE/Pong-v5'."""
    _ensure_ale_registered()
    env = gym.make(env_id, frameskip=1)  # frameskip handled below
    if seed is not None:
        env.action_space.seed(seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, frame_stack)
    return env
