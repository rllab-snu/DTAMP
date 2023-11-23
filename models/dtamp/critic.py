import torch
import torch.nn as nn
from .perception import Perception
from .encoder import Encoder


class Critic(nn.Module):
    def __init__(self, state_dim, act_dim, goal_dim, hidden_size=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + act_dim + goal_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs, act, goal):
        x = torch.cat([obs, act, goal], dim=-1)
        return self.layers(x)


class Critics(nn.Module):
    def __init__(self, state_dim, act_dim, goal_dim, n_critics, visual_perception=True, hidden_size=512):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim, hidden_size=hidden_size) if visual_perception \
            else lambda x: x
        self.goal_encoder = Encoder(state_dim, goal_dim, hidden_size)
        self.critics = nn.ModuleList(
            [Critic(state_dim, act_dim, goal_dim, hidden_size) for _ in range(n_critics)]
        )

    def forward(self, obs, act, goal):
        emb_state = self.perception(obs)
        outputs = []
        for critic in self.critics:
            outputs.append(critic(emb_state, act, goal))
        return torch.cat(outputs, dim=-1)

    def encode(self, obs):
        emb_state = self.perception(obs)
        enc_goal = self.goal_encoder(emb_state)
        return enc_goal

    def min_q(self, obs, act, goal):
        q = self.forward(obs, act, goal)
        return torch.min(q, dim=-1, keepdim=True)[0]
