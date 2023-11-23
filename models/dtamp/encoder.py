import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=512, normalize=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        self.normalize = normalize

    def forward(self, x):
        x = self.layers(x)
        if self.normalize:
            return F.normalize(x, p=2.0, dim=-1)
        else:
            return x