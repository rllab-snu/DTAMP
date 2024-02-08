import torch
import torch.nn as nn
from .perception import Perception
from .encoder import Encoder
from .distributions import SquashedNormal

MIN_LOG_STD = -20.
MAX_LOG_STD = 2.


class SkillProposal(nn.Module):
    def __init__(self, state_dim, goal_dim, skill_dim, hidden_size=256):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim, hidden_size=hidden_size)
        self.goal_encoder = Encoder(state_dim, goal_dim, hidden_size)

        self.layers = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * skill_dim)
        )
        self.skill_dim = skill_dim

    def forward(self, obs, goal):
        emb_state = self.perception(obs)
        emb_goal = self.perception(goal)
        enc_goal = self.goal_encoder(emb_goal)
        return self.get_proposal_from_embeddings(emb_state, enc_goal)

    def encode(self, obs):
        emb_state = self.perception(obs)
        enc_goal = self.goal_encoder(emb_state)
        return emb_state, enc_goal

    def get_proposal_from_embeddings(self, emb_state, enc_goal):
        x = torch.cat([emb_state, enc_goal], dim=-1)
        x = self.layers(x)
        mean, logstd = torch.split(x, self.skill_dim, dim=-1)
        std = logstd.clamp(min=MIN_LOG_STD, max=MAX_LOG_STD).exp()
        return SquashedNormal(mean, std)


class SkillRecognition(nn.Module):
    def __init__(self, state_dim, act_dim, skill_dim, hidden_size=256):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim, hidden_size=hidden_size)

        self.gru = nn.GRU(
            input_size=state_dim + act_dim,
            hidden_size=hidden_size,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(2 * hidden_size, 2 * skill_dim)
        self.skill_dim = skill_dim
        self.hidden_size = hidden_size

    def forward(self, obs, act):
        emb_state = self.perception(obs)
        x = torch.cat([emb_state, act], dim=-1)
        x, _ = self.gru(x)
        x = torch.cat([x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]], dim=-1)
        x = self.fc(x)
        mean, logstd = torch.split(x, self.skill_dim, dim=-1)
        std = logstd.clamp(min=MIN_LOG_STD, max=MAX_LOG_STD).exp()
        return SquashedNormal(mean, std)

