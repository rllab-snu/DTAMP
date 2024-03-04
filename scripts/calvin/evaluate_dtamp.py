import numpy as np
import torch

from models.dtamp import DTAMP
from models.play_lmp import PlayLMP
from envs.calvin_env import CalvinEnv

import yaml
from tqdm import tqdm


def evaluate(env, model, lmp, eval_episodes, threshold, time_limit, skill_duration, device):
    ep_returns = []
    pbar = tqdm(total=eval_episodes, desc=f'Evaluation')
    for task_id in range(eval_episodes):
        obs, goal = env.reset(task_id=task_id)
        obs_list = [obs]
        done = False
        ep_return = 0
        milestones = model.planning(obs, goal, target_returns=None)
        timestep = 0
        len_milestones = len(milestones)
        while not done:
            if timestep % skill_duration == 0:
                skill, milestones = model.get_action(obs, milestones, threshold=threshold)
                obs_list = obs_list[-1:]
            timestep += 1
            if len_milestones != len(milestones):
                len_milestones = len(milestones)
                timestep = 0
            if timestep > time_limit and len(milestones) > 1:
                milestones = milestones[1:]
                timestep = 0
            act = lmp.get_action(obs_list, skill)
            obs, rew, done, _ = env.step(act)
            obs_list.append(obs)
            ep_return += rew
        ep_returns.append(ep_return)
        pbar.set_description(f'Evaluation - Avg return: {np.mean(ep_returns):.3f}')
        pbar.update(1)
    pbar.close()
    return np.mean(ep_returns)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=100)
    parser.add_argument('--calvin_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='config/calvin')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--tasks_per_rollout', type=int, default=1)
    parser.add_argument('--render', action='store_true', dest='render', default=False)
    args = parser.parse_args()

    if args.render:
        print('Warning! When CALVIN environment runs with GUI for rendering, the style of observations slightly changes from the non-rendering version,\n'
              + 'which causes performance degradation.')

    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = os.path.join('checkpoints', 'dtamp_calvin')

    config = yaml.load(open('config/calvin/calvin.yml'), Loader=yaml.FullLoader)

    env = CalvinEnv(args.calvin_dir, args.data_dir, args.config_dir, show_gui=args.render)

    action_dim = env.act_dim
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
    lmp = PlayLMP(
        state_dim=config['lmp_cfg']['goal_dim'],
        act_dim=config['lmp_cfg']['action_dim'],
        goal_dim=config['lmp_cfg']['goal_dim'],
        skill_dim=config['lmp_cfg']['skill_dim'],
        kl_coeff=config['lmp_cfg']['kl_coeff'],
        kl_balance_coeff=config['lmp_cfg']['kl_balance_coeff']
    ).to(device)
    if args.checkpoint_epoch:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint_%d.pt' % args.checkpoint_epoch))
    else:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    lmp.load_state_dict(checkpoint['lmp_model'])
    lmp.eval()

    env.prepare_tasks(args.tasks_per_rollout)
    env.order_rollouts()

    avg_return = evaluate(
        env, model, lmp, args.eval_episodes,
        config['threshold'], config['time_limit'], config['skill_duration'], device
    )
    print(avg_return)
