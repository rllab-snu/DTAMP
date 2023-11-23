import os
import shutil

from datasets.datasets import D4RLDataset
from torch.utils.data import DataLoader

import argparse
import torch
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
from models.dtamp import DTAMP
from tqdm import tqdm

import gym
import d4rl

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-medium-play-v2')
    parser.add_argument('--goal_dim', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--max_interval', type=int, default=16)
    parser.add_argument('--n_critics', type=int, default=4)
    parser.add_argument('--rl_coeff', type=float, default=2.5)
    parser.add_argument('--decoder_coeff', type=float, default=0)
    parser.add_argument('--diffuser_coeff', type=float, default=1e-3)
    parser.add_argument('--predict_epsilon', action='store_true', dest='predict_epsilon', default=True)
    parser.add_argument('--diffuser_timesteps', type=int, default=300)
    parser.add_argument('--returns_condition', action='store_true', dest='returns_condition', default=False)
    parser.add_argument('--condition_guidance_w', type=float, default=0)
    parser.add_argument('--hidden_size', type=int, default=1024)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_updates', type=int, default=50000)
    parser.add_argument('--updates_per_epoch', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    exp_name = f'dtamp_{args.env}'
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    env = gym.make(args.env)
    dataset = D4RLDataset(env, args.max_interval, args.horizon)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda')
    model = DTAMP(
        state_dim=state_dim,
        act_dim=action_dim,
        goal_dim=args.goal_dim,
        visual_perception=False,
        horizon=args.horizon,
        n_critics=args.n_critics,
        rl_coeff=args.rl_coeff,
        decoder_coeff=args.decoder_coeff,
        diffuser_coeff=args.diffuser_coeff,
        predict_epsilon=args.predict_epsilon,
        diffuser_timesteps=args.diffuser_timesteps,
        returns_condition=args.returns_condition,
        condition_guidance_w=args.condition_guidance_w,
        hidden_size=args.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    n_updates, epoch = 0, 0
    total_updates = args.updates_per_epoch * args.epochs
    pbar = tqdm(total=args.updates_per_epoch, desc=f'Epoch {epoch}')
    while n_updates < total_updates:
        for batch in data_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            loss, logs = model.loss(batch, n_updates < args.warmup_updates)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tag = 'dtamp/bc' if n_updates < args.warmup_updates else 'dtamp/rl'
            for key, val in logs.items():
                writer.add_scalar(f'{tag}/{key}', val, n_updates)

            pbar.update(1)
            n_updates += 1

            if n_updates % args.updates_per_epoch == 0:
                pbar.close()
                epoch += 1
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'n_updates': n_updates
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint'))
                if n_updates == total_updates:
                    break
                pbar = tqdm(total=args.updates_per_epoch, desc=f'Epoch {epoch}')


if __name__ == '__main__':
    train()