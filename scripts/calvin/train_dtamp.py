import os
import shutil

import yaml

from datasets.calvin_dataset import CalvinSkillDataset, CalvinDataContainer
from datasets.augmentation import Transform
from torch.utils.data import DataLoader

import argparse
import torch
from torch.nn.utils import clip_grad_norm_

from models.dtamp import DTAMP
from tqdm import tqdm
from glob import glob


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--lmp_dir', type=str, default=None)
    parser.add_argument('--epochs_per_save', type=int, default=50)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

    exp_name = f'dtamp_calvin'
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    config = yaml.load(open(f'config/calvin/calvin.yml'), Loader=yaml.FullLoader)

    data_pths = glob(os.path.join(args.data_dir, 'train_*'))
    data_container = CalvinDataContainer(data_pths, modularities=('observations', 'skills'))
    dataset = CalvinSkillDataset(
        data_container,
        config['max_interval'],
        config['lmp_cfg']['min_skill_length'],
        config['lmp_cfg']['max_skill_length'],
        config['horizon']
    )
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    transform = Transform()

    device = torch.device('cuda')

    model = DTAMP(
        state_dim=config['goal_dim'] // 2,
        act_dim=config['lmp_cfg']['skill_dim'],
        goal_dim=config['goal_dim'],
        visual_perception=True,
        horizon=config['horizon'],
        n_critics=config['n_critics'],
        rl_coeff=config['rl_coeff'],
        kl_coeff=config['kl_coeff'],
        decoder_coeff=config['decoder_coeff'],
        diffuser_coeff=config['diffuser_coeff'],
        predict_epsilon=config['predict_epsilon'],
        diffuser_timesteps=config['diffuser_timesteps'],
        returns_condition=config['returns_condition'],
        condition_guidance_w=config['condition_guidance_w'],
        hidden_size=config['hidden_size']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    lmp_model = torch.load(os.path.join(args.lmp_dir, 'checkpoint.pt'))['model']

    model.train()
    n_updates, epoch = 0, 0
    total_updates = config['updates_per_epoch'] * config['epochs']
    pbar = tqdm(total=config['updates_per_epoch'], desc=f'Epoch {epoch}')
    while n_updates < total_updates:
        for batch in data_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            batch['observations'], batch['target_observations'] = transform(batch['observations'])
            loss, logs = model.loss(batch, warmup=n_updates < config['warmup_updates'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tag = 'dtamp/bc' if n_updates < config['warmup_updates'] else 'dtamp/rl'
            for key, val in logs.items():
                writer.add_scalar(f'{tag}/{key}', val, n_updates)

            pbar.update(1)
            n_updates += 1

            if n_updates % config['updates_per_epoch'] == 0:
                pbar.close()
                epoch += 1
                model.eval()

                checkpoint = {
                    'model': model.state_dict(),
                    'lmp_model': lmp_model,
                    'optimizer': optimizer.state_dict(),
                    'n_updates': n_updates
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))

                if epoch % args.epochs_per_save == 0:
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_%d.pt' % epoch))

                if n_updates == total_updates:
                    break
                model.train()
                pbar = tqdm(total=config['updates_per_epoch'], desc=f'Epoch {epoch}')


if __name__ == '__main__':
    train()
