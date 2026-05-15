"""Uniform replay buffer. Stores observations as uint8 to save memory."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, obs_dtype=np.uint8, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.obs_dtype = obs_dtype
        self.obs = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.from_numpy(self.obs[idx]).to(self.device)
        next_obs = torch.from_numpy(self.next_obs[idx]).to(self.device)
        if self.obs_dtype == np.uint8:
            obs = obs.float() / 255.0
            next_obs = next_obs.float() / 255.0
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self.size
