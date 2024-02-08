import numpy as np
import cv2
import joblib
import json
import os
from distutils.dir_util import copy_tree
from tqdm import tqdm


'''
This code is for preprocessing CALVIN dataset.
Note that the code is for augmented data provided by 
Erick Rosete-Beas et. al., "Latent Plans for Task-Agnostic Offline Reinforcement Learning", CoRL 2022
Please download the dataset from: http://tacorl.cs.uni-freiburg.de/dataset/taco_rl_calvin.zip
'''


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source_data_dir', type=str, default=None)
parser.add_argument('--target_data_dir', type=str, default=None)
args = parser.parse_args()

if args.source_data_dir is None:
    raise Exception('Please specify the location of CALVIN dataset (--source_data_dir {tacorl_data_dir})')
if args.target_data_dir is None:
    raise Exception('Please specify the location to store processed dataset (--target_data_dir {dir_to_store_data})')

os.makedirs(args.target_data_dir, exist_ok=True)
start_end_tasks = json.load(open(os.path.join(args.source_data_dir, 'start_end_tasks.json'), 'r'))
new_start_end_tasks = dict()
hard_start_end_tasks = json.load(open(os.path.join(args.source_data_dir, 'hard_start_end_tasks.json'), 'r'))
new_hard_start_end_tasks = dict()

split = json.load(open(os.path.join(args.source_data_dir, 'split.json'), 'r'))
os.makedirs(args.target_data_dir, exist_ok=True)

copy_tree(os.path.join(args.source_data_dir, '.hydra'), os.path.join(args.target_data_dir, '.hydra'))

for field, start_end_idxs in split.items():
    for i, (start_idx, end_idx) in enumerate(tqdm(start_end_idxs, desc=f'Processing {field} data...')):
        epi_length = end_idx - start_idx + 1
        data = dict(
            observations=np.zeros([epi_length, 3, 128, 128], dtype=np.uint8),
            actions=np.zeros([epi_length, 7], dtype=np.float32),
            robot_obs=np.zeros([epi_length, 15], dtype=np.float32),
            scene_obs=np.zeros([epi_length, 24], dtype=np.float32)
        )
        for t in range(start_idx, end_idx + 1):
            source = np.load(os.path.join(args.source_data_dir, f'episode_{t:07d}.npz'))
            img = source['rgb_static']
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            data['observations'][t - start_idx, :, :, :] = np.transpose(img, (2, 0, 1))
            data['actions'][t - start_idx, :] = source['rel_actions_world']
            data['robot_obs'][t - start_idx, :] = source['robot_obs']
            data['scene_obs'][t - start_idx, :] = source['scene_obs']
        joblib.dump(data, os.path.join(args.target_data_dir, f'{field}_{i}.pkl'))

        if field == 'validation':
            for task_start_idx in start_end_tasks.keys():
                if start_idx <= int(task_start_idx) < end_idx:
                    new_start_end_tasks[(i, int(task_start_idx) - start_idx)] = dict()
                    for task_end_idx, completed_tasks in start_end_tasks[task_start_idx].items():
                        new_start_end_tasks[(i, int(task_start_idx) - start_idx)][int(task_end_idx) - start_idx] \
                            = completed_tasks
            joblib.dump(new_start_end_tasks, os.path.join(args.target_data_dir, 'start_end_tasks.pkl'))

            for task_start_idx in hard_start_end_tasks.keys():
                if start_idx <= int(task_start_idx) < end_idx:
                    new_hard_start_end_tasks[(i, int(task_start_idx) - start_idx)] = dict()
                    for task_end_idx, completed_tasks in hard_start_end_tasks[task_start_idx].items():
                        new_hard_start_end_tasks[(i, int(task_start_idx) - start_idx)][int(task_end_idx) - start_idx] \
                            = completed_tasks
            joblib.dump(new_hard_start_end_tasks, os.path.join(args.target_data_dir, 'hard_start_end_tasks.json'))
