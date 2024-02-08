import torch
import torch.nn as nn
from .perception import Perception
from .encoder import Encoder
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, goal_dim,
                 visual_perception=True, hidden_size=512, deterministic=True):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim, hidden_size=hidden_size) \
            if visual_perception else lambda x: x
        self.goal_encoder = Encoder(state_dim, goal_dim, hidden_size, deterministic)

        self.layers = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.deterministic = deterministic

    def forward(self, obs, goal):
        emb_state = self.perception(obs)
        x = torch.cat([emb_state, goal], dim=-1)
        x = self.layers(x)
        return torch.tanh(x)

    def encode(self, obs, eval=False):
        emb_state = self.perception(obs)
        if self.deterministic:
            enc_goal = self.goal_encoder(emb_state)
            return enc_goal, 0
        else:
            goal_dist = self.goal_encoder(emb_state)
            enc_goal = goal_dist.mean if eval else goal_dist.rsample()
            prior = Normal(torch.zeros_like(enc_goal), torch.ones_like(enc_goal))
            kl_loss = kl_divergence(goal_dist, prior).mean()
            return enc_goal, kl_loss
