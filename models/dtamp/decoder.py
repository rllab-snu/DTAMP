import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 16, 6, stride=2)
        self.conv5 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, state):
        batch_shape = state.shape
        state = state.reshape(-1, *batch_shape[-1:])
        hidden = self.fc1(state)
        hidden = hidden.view(-1, 1024, 1, 1)
        hidden = F.relu(self.conv1(hidden))
        hidden = F.relu(self.conv2(hidden))
        hidden = F.relu(self.conv3(hidden))
        hidden = F.relu(self.conv4(hidden))
        observation = torch.tanh(self.conv5(hidden))
        observation = observation.reshape(*batch_shape[:-1], 3, 128, 128)
        return observation