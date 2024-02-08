from models.play_lmp import PlayLMP
from datasets.augmentation import Transform
import numpy as np
import torch
import yaml
import joblib

import argparse
import os
from tqdm import tqdm

transform = Transform()

@torch.no_grad()
def generate_skills(episode, model, min_skill_length, max_skill_length, device):
    epi_length = len(episode['observations'])
    skills = np.zeros([epi_length, max_skill_length - min_skill_length + 1, model.skill_dim], dtype=np.float32)
    for t in range(epi_length - min_skill_length - 1):
        max_length = np.minimum(epi_length - t, max_skill_length + 1)
        batch = {'observations': [], 'actions': []}
        for skill_length in range(min_skill_length, max_length):
            observations = episode['observations'][t:t + skill_length]
            actions = episode['actions'][t:t + skill_length]
            if skill_length < max_skill_length:
                pad_length = max_skill_length - skill_length
                observation_pad = np.repeat(
                    episode['observations'][t + skill_length][np.newaxis, ...],
                    pad_length, axis=0
                )
                action_pad = np.repeat(
                    episode['actions'][t + skill_length - 1][np.newaxis, ...],
                    pad_length, axis=0
                )
                action_pad[:, :-1] = 0.
                observations = np.concatenate([observations, observation_pad], axis=0)
                actions = np.concatenate([actions, action_pad], axis=0)
            batch['observations'].append(torch.as_tensor(observations))
            batch['actions'].append(torch.as_tensor(actions))
        for k in batch.keys():
            batch[k] = torch.stack(batch[k], dim=0).to(device)
        batch['observations'], _ = transform(batch['observations'], eval=True)
        skill_posterior = model.skill_recognition(batch['observations'], batch['actions'])
        skills[t, :max_length - min_skill_length] = skill_posterior.mean.cpu().detach().numpy()
    return skills


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='calvin')
parser.add_argument('--data_dir', type=str, default=None)
args = parser.parse_args()

config = yaml.load(open(f'config/calvin/calvin.yml'), Loader=yaml.FullLoader)['lmp_cfg']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = PlayLMP(
    state_dim=config['goal_dim'],
    act_dim=config['action_dim'],
    goal_dim=config['goal_dim'],
    skill_dim=config['skill_dim'],
    kl_coeff=config['kl_coeff'],
    kl_balance_coeff=config['kl_balance_coeff']
).to(device)
checkpoint = torch.load(f'checkpoints/lmp_calvin/checkpoint.pt')
model.load_state_dict(checkpoint['model'])

train_data_pths = [os.path.join(args.data_dir, 'train_%d.pkl' % i) for i in range(35)]
val_data_pths = [os.path.join(args.data_dir, 'validation_%d.pkl' % i) for i in range(6)]

for pth in tqdm(train_data_pths):
    episode = joblib.load(pth)
    skills = generate_skills(
        episode, model,
        min_skill_length=config['min_skill_length'],
        max_skill_length=config['max_skill_length'],
        device=device
    )
    episode['skills'] = skills
    joblib.dump(episode, pth)

for pth in tqdm(val_data_pths):
    episode = joblib.load(pth)
    skills = generate_skills(
        episode, model,
        min_skill_length=config['min_skill_length'],
        max_skill_length=config['max_skill_length'],
        device=device
    )
    episode['skills'] = skills
    joblib.dump(episode, pth)
