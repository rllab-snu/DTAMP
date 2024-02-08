import os
import shutil

import yaml

from datasets.calvin_dataset import CalvinDataContainer, PlayDataset
from datasets.augmentation import Transform
from torch.utils.data import DataLoader

import argparse
import torch
from torch.nn.utils import clip_grad_norm_

from models.play_lmp import PlayLMP
from tqdm import tqdm
from glob import glob


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs_per_save', type=int, default=50)
    args = parser.parse_args()

    if args.data_dir is None:
        raise Exception('Please specify data_dir')

    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

    exp_name = 'lmp_calvin'
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    config = yaml.load(open('config/calvin/calvin.yml'), Loader=yaml.FullLoader)
    config = config['lmp_cfg']

    data_pths = glob(os.path.join(args.data_dir, 'train_*'))
    data_container = CalvinDataContainer(data_pths)
    dataset = PlayDataset(data_container, config['min_skill_length'], config['max_skill_length'], use_padding=True)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    transform = Transform()

    state_dim = config['goal_dim']
    action_dim = config['action_dim']
    device = torch.device('cuda')

    model = PlayLMP(
        state_dim=state_dim,
        act_dim=action_dim,
        goal_dim=config['goal_dim'],
        skill_dim=config['skill_dim'],
        kl_coeff=config['kl_coeff'],
        kl_balance_coeff=config['kl_balance_coeff']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    model.train()
    n_updates, epoch = 0, 0
    total_updates = config['updates_per_epoch'] * config['epochs']
    pbar = tqdm(total=config['updates_per_epoch'], desc=f'Epoch {epoch}')
    while n_updates < total_updates:
        for batch in data_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            batch['observations'], _ = transform(batch['observations'])
            batch['goals'], _ = transform(batch['goals'])
            loss, logs = model.loss(batch)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tag = 'lmp'
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
