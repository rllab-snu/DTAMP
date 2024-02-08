import torch.nn as nn
import torch.nn.functional as F
from .distributions import NormalizedNormal


MIN_LOG_STD = -20.
MAX_LOG_STD = 2.


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=512, deterministic=True):
        super().__init__()
        output_dim = output_dim if deterministic else 2 * output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        self.deterministic = deterministic
        self.output_dim = output_dim

    def forward(self, x):
        x = self.layers(x)
        if self.deterministic:
            return F.normalize(x, p=2.0, dim=-1)
        else:
            mean, logstd = x.split(self.output_dim // 2, dim=-1)
            stddev = logstd.clamp(min=MIN_LOG_STD, max=MAX_LOG_STD).exp()
            return NormalizedNormal(mean, stddev)
