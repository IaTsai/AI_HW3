"""DQN agent supporting three variants:
    - 'vanilla' : DQN (Mnih et al. 2015)
    - 'double'  : Double DQN (van Hasselt et al. 2016)
    - 'dueling' : Dueling Double DQN (Wang et al. 2016, dueling head + Double target)

The variant only changes (a) the network architecture (dueling vs single-stream)
and (b) how the bootstrap target is computed (Double uses online net to argmax,
target net to evaluate; vanilla uses target net for both).
"""
import numpy as np
import torch
import torch.nn.functional as F
from networks import build_qnet


class DQNAgent:
    def __init__(
        self,
        obs_space,
        num_actions,
        variant="vanilla",
        device="cuda",
        lr=1e-4,
        gamma=0.99,
        head_hidden=None,
        grad_clip=10.0,
    ):
        assert variant in ("vanilla", "double", "dueling")
        self.variant = variant
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device
        self.grad_clip = grad_clip

        dueling = (variant == "dueling")
        self.qnet = build_qnet(obs_space, num_actions, dueling=dueling, head_hidden=head_hidden).to(device)
        self.target = build_qnet(obs_space, num_actions, dueling=dueling, head_hidden=head_hidden).to(device)
        self.target.load_state_dict(self.qnet.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, epsilon):
        """Epsilon-greedy action selection. obs: numpy array, single observation."""
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()
        q = self.qnet(x)
        return int(q.argmax(dim=1).item())

    def update(self, batch):
        """One gradient step. batch: tuple from ReplayBuffer.sample()."""
        obs, actions, rewards, next_obs, dones = batch
        # Current Q(s, a)
        q_pred = self.qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Bootstrap target
        with torch.no_grad():
            if self.variant == "vanilla":
                q_next = self.target(next_obs).max(dim=1)[0]
            else:  # double or dueling (both use Double-style target)
                next_actions = self.qnet(next_obs).argmax(dim=1, keepdim=True)
                q_next = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * q_next * (1.0 - dones)
        loss = F.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item()), float(q_pred.mean().item())

    def sync_target(self):
        self.target.load_state_dict(self.qnet.state_dict())

    def save(self, path):
        torch.save({"qnet": self.qnet.state_dict(), "variant": self.variant}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.target.load_state_dict(self.qnet.state_dict())
