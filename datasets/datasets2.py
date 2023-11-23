import numpy as np
import torch
from torch.utils.data import Dataset


class D4RLDataset(Dataset):
    def __init__(self, env, max_interval, horizon):
        data = env.get_dataset()
        data['timeouts'][-1] = True
        self.episodes = {key: [] for key in data.keys()}
        self.episode_lengths = []
        episode_ends = np.where(data['timeouts'])[0]
        start = 0
        for end in episode_ends:
            if end - start + 1 > horizon:
                for key in data.keys():
                    self.episodes[key].append(data[key][start:end + 1])
                self.episode_lengths.append(end - start + 1)
            start = end + 1
        self.max_interval = max_interval
        self.horizon = horizon

    def __len__(self):
        # return int(1e6)
        return int(1e4)

    def __getitem__(self, item):
        epi_i = np.random.randint(len(self.episode_lengths))
        length = self.episode_lengths[epi_i]
        max_interval = np.minimum(self.max_interval, length // (self.horizon - 1))
        interval = np.random.randint(max_interval) + 1
        t = np.random.randint(length - interval * (self.horizon - 1))
        timesteps = t + interval * np.arange(self.horizon)
        batch = {
            'observations': torch.as_tensor(self.episodes['observations'][epi_i][timesteps], dtype=torch.float32),
            'actions': torch.as_tensor(self.episodes['actions'][epi_i][timesteps], dtype=torch.float32)
        }
        return batch