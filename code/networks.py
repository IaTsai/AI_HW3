"""Q-network architectures for image (Atari) and vector (LunarLander) inputs.
Supports both standard ("single-stream") and dueling architectures.
"""
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """The CNN from Mnih et al. 2015 (Nature DQN). Input: 4x84x84 uint8 frames."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.out_dim = 64 * 7 * 7  # 3136

    def forward(self, x):
        return self.conv(x)


class MLPTrunk(nn.Module):
    """Two-hidden-layer MLP used for low-dim observations (LunarLander)."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    """Single-stream Q-network: trunk -> linear -> Q(s,a)."""

    def __init__(self, trunk, num_actions, head_hidden=512):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Sequential(
            nn.Linear(trunk.out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, num_actions),
        )

    def forward(self, x):
        return self.head(self.trunk(x))


class DuelingQNetwork(nn.Module):
    """Dueling architecture (Wang et al. 2016): split into V(s) and A(s,a),
    then combine: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, trunk, num_actions, head_hidden=512):
        super().__init__()
        self.trunk = trunk
        self.value = nn.Sequential(
            nn.Linear(trunk.out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(trunk.out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, num_actions),
        )

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


def build_qnet(obs_space, num_actions, dueling=False, head_hidden=None):
    """Factory: picks CNN or MLP trunk based on observation shape."""
    shape = obs_space.shape
    if len(shape) == 3:  # image: (C, H, W)
        trunk = NatureCNN(in_channels=shape[0])
        head_hidden = head_hidden or 512
    else:  # vector
        trunk = MLPTrunk(obs_dim=shape[0])
        head_hidden = head_hidden or 128
    if dueling:
        return DuelingQNetwork(trunk, num_actions, head_hidden=head_hidden)
    return QNetwork(trunk, num_actions, head_hidden=head_hidden)
