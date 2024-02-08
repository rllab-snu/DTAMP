import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    def __init__(self, channel, height, width):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.device = torch.device('cuda')
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height, device=self.device),
            torch.linspace(-1.0, 1.0, self.width, device=self.device)
        )
        pos_x = pos_x.reshape(self.height * self.width)
        pos_y = pos_y.reshape(self.height * self.width)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        x = x.reshape(-1, self.height * self.width)
        softmax_c = F.softmax(x, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_c, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_c, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class Perception(nn.Module):
    def __init__(self, input_w, input_h, out_channels=64, out_dim=32, hidden_size=256):
        super().__init__()
        w, h = self.calc_out_wh(input_w, input_h, 8, 0, 4)
        w, h = self.calc_out_wh(w, h, 4, 0, 2)
        w, h = self.calc_out_wh(w, h, 3, 0, 1)

        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, out_channels, 3, 1)
        self.spatial_softmax = SpatialSoftmax(out_channels, w, h)
        self.fc1 = nn.Linear(2 * out_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_dim)
        self.out_channels = out_channels
        self.out_dim = out_dim

    def forward(self, obs):
        batch_shape = obs.shape
        x_visual = obs.reshape(-1, *batch_shape[-3:])
        x_visual = F.relu(self.conv1(x_visual))
        x_visual = F.relu(self.conv2(x_visual))
        x_visual = F.relu(self.conv3(x_visual))
        x_visual = self.spatial_softmax(x_visual)
        x_visual = x_visual.reshape(*batch_shape[:-3], 2 * self.out_channels)
        x = F.relu(self.fc1(x_visual))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, p=2.0, dim=-1)

    def calc_out_wh(self, w, h, k, p, s):
        w_out = int((w + 2 * p - k) / s + 1)
        h_out = int((h + 2 * p - k) / s + 1)
        return w_out, h_out
