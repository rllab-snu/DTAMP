import numpy as np
import torch
from torchvision.transforms.functional import affine
from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset
from tqdm import tqdm
import joblib



class CalvinDataContainer:
    def __init__(self, data_pths, modularities=('observations', 'actions')):
        self.data = dict()
        for key in modularities:
            self.data[key] = []
        self.epi_lengths = []
        ptr = 0
        for pth in tqdm(data_pths):
            episode = joblib.load(open(pth, 'rb'))
            epi_length = len(episode[modularities[0]])
            self.epi_lengths.append(epi_length)
            for key in modularities:
                self.data[key].append(episode[key])
            ptr += epi_length

    def __getitem__(self, idx):
        epi_idx, timestep = idx
        return {k: self.data[k][epi_idx][timestep] for k in self.data.keys()}


class PlayDataset(Dataset):
    def __init__(self, data_container, min_skill_length, max_skill_length, use_padding=True):
        self.data_container = data_container
        self.min_skill_length, self.max_skill_length = min_skill_length, max_skill_length
        self.indices = []
        for i, epi_length in enumerate(self.data_container.epi_lengths):
            self.indices += [(i, t) for t in range(epi_length - min_skill_length)]
        self.indices = np.array(self.indices)
        self.use_padding = use_padding

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        epi_idx, t = self.indices[idx]
        max_skill_length = np.minimum(
            self.max_skill_length, self.data_container.epi_lengths[epi_idx] - t - 1
        )
        skill_length = np.random.randint(self.min_skill_length, max_skill_length + 1)
        trajectory = self.data_container[epi_idx, t:t + skill_length]
        trajectory['goals'] = self.data_container[epi_idx, t + skill_length]['observations']
        if skill_length < self.max_skill_length and self.use_padding:
            trajectory = self.padding(trajectory)
        return trajectory

    def padding(self, trajectory):
        skill_length = len(trajectory['observations'])
        pad_length = self.max_skill_length - skill_length
        observation_pad = np.repeat(trajectory['goals'][np.newaxis, ...], pad_length, axis=0)
        action_pad = np.repeat(trajectory['actions'][-1:], pad_length, axis=0)
        action_pad[:, :-1] = 0.     # preserve gripper control at axis -1.
        trajectory['observations'] = np.concatenate(
            [trajectory['observations'], observation_pad], axis=0
        )
        trajectory['actions'] = np.concatenate(
            [trajectory['actions'], action_pad], axis=0
        )
        return trajectory


class CalvinSkillDataset(Dataset):
    def __init__(self, data_container, max_interval, min_skill_length, max_skill_length, horizon):
        self.data_container = data_container
        self.max_interval = max_interval
        self.min_skill_length = min_skill_length
        self.max_skill_length = max_skill_length
        self.horizon = horizon
        self.epi_idxs_n_lengths = []
        for i, length in enumerate(data_container.epi_lengths):
            if length >= horizon:
                self.epi_idxs_n_lengths.append((i, length))

    def __len__(self):
        return int(1e6)

    def __getitem__(self, item):
        epi_i = np.random.randint(len(self.epi_idxs_n_lengths))
        epi_idx, epi_length = self.epi_idxs_n_lengths[epi_i]
        max_interval = np.minimum(self.max_interval, epi_length // (self.horizon - 1))
        interval = np.random.randint(self.min_skill_length, max_interval + 1)
        t = np.random.randint(epi_length - interval * (self.horizon - 1))
        timesteps = t + interval * np.arange(self.horizon)
        returns = interval / self.max_interval
        skill_idx = np.minimum(interval - self.min_skill_length, self.max_skill_length - self.min_skill_length + 1)
        batch = {
            'observations': torch.as_tensor(self.data_container[epi_idx, timesteps]['observations']),
            'actions': torch.as_tensor(self.data_container[epi_idx, timesteps]['skills'][:, skill_idx], dtype=torch.float32),
            'returns': torch.as_tensor(returns, dtype=torch.float32)
        }
        return batch