import torch
import torch.nn as nn
from .perception import Perception
from .encoder import Encoder

MIN_LOG_STD = -20.
MAX_LOG_STD = 2.


class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, goal_dim,
                 visual_perception=True, hidden_size=512):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim, hidden_size=hidden_size) \
            if visual_perception else lambda x: x
        self.goal_encoder = Encoder(state_dim, goal_dim, hidden_size)

        self.layers = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs, goal):
        emb_state = self.perception(obs)
        x = torch.cat([emb_state, goal], dim=-1)
        x = self.layers(x)
        return torch.tanh(x)

    def encode(self, obs):
        emb_state = self.perception(obs)
        enc_goal = self.goal_encoder(emb_state)
        return enc_goal
