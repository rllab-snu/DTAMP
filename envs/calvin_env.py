from omegaconf import OmegaConf

import numpy as np
import cv2
import os
import sys
import hydra
import joblib
import json


class CalvinEnv:
    def __init__(self, calvin_dir, data_dir, config_dir, show_gui=False, max_step=None):
        sys.path.append(os.path.join(calvin_dir, 'calvin_env'))
        sys.path.append(os.path.join(calvin_dir, 'calvin_models'))
        self.task_cfg = OmegaConf.load(
            os.path.join(calvin_dir, 'calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml')
        )
        self.task_oracle = hydra.utils.instantiate(self.task_cfg)

        from calvin_env.envs.play_table_env import get_env
        self.env = get_env(config_dir, show_gui=show_gui)
        self.obs_shape = (3, 128, 128)
        self.act_dim = 7
        self.goal_state = None
        self.data_dir = data_dir

        self.max_step = max_step
        self._step = 0
        self.rollout_tasks = []

    def prepare_tasks(self, tasks_per_rollout, min_seq_length=10):
        self.tasks_per_rollout = tasks_per_rollout
        self.rollout_tasks = []
        start_end_tasks = joblib.load(os.path.join(self.data_dir, 'start_end_tasks.pkl'))
        for (epi_idx, start_idx), end_tasks in start_end_tasks.items():
            for end_idx, completed_tasks in end_tasks.items():
                if len(completed_tasks) == tasks_per_rollout and end_idx - start_idx > min_seq_length:
                    rel_task = {
                        'episode_idx': epi_idx,
                        'start_step': start_idx,
                        'end_step': end_idx,
                        'seq_len': end_idx - start_idx,
                        'completed_tasks': completed_tasks
                    }
                    self.rollout_tasks.append(rel_task)
        if self.max_step is None:
            self.max_step = 450 if self.tasks_per_rollout > 2 else 300

    def prepare_hard_tasks(self, min_seq_length=10):
        self.tasks_per_rollout = 1
        self.rollout_tasks = []
        start_end_tasks = joblib.load(os.path.join(self.data_dir, 'hard_start_end_tasks.pkl'))
        for (epi_idx, start_idx), end_tasks in start_end_tasks.items():
            for end_idx, completed_tasks in end_tasks.items():
                if len(completed_tasks) == 1 and end_idx - start_idx > min_seq_length:
                    rel_task = {
                        'episode_idx': epi_idx,
                        'start_step': start_idx,
                        'end_step': end_idx,
                        'seq_len': end_idx - start_idx,
                        'completed_tasks': completed_tasks
                    }
                    self.rollout_tasks.append(rel_task)
        self.max_step = 300

    def order_rollouts(self):
        self.rollout_tasks = sorted(self.rollout_tasks, key=lambda d: d['seq_len'])

    def reset(self, task_id=None):
        task_id = np.random.randint(len(self.rollout_tasks)) if task_id is None else task_id
        task = self.rollout_tasks[task_id]
        target_episode = joblib.load(os.path.join(self.data_dir, f"validation_{task['episode_idx']}.pkl"))
        goal = target_episode['observations'][task['end_step']]
        goal = 2. * goal / 255. - 1.

        observation = self.env.reset(robot_obs=target_episode['robot_obs'][task['start_step']],
                                     scene_obs=target_episode['scene_obs'][task['start_step']])
        observation = cv2.resize(observation['rgb_obs']['rgb_static'], (128, 128))
        observation = np.transpose(observation, (2, 0, 1))
        observation = 2. * observation / 255. - 1.
        self.start_info = self.env.get_info()
        self._step = 0
        self.curr_task = task['completed_tasks']
        return observation, goal

    def step(self, action):
        action[-1] = -1 if action[-1] < 0 else 1
        self._step += 1
        observation, _, _, _ = self.env.step(action)
        observation = cv2.resize(observation['rgb_obs']['rgb_static'], (128, 128))
        observation = np.transpose(observation, (2, 0, 1))
        observation = 2 * observation / 255. - 1
        curr_info = self.env.get_info()
        task_info = self.task_oracle.get_task_info_for_set(self.start_info, curr_info, {*self.curr_task})
        reward = float(len(task_info) == self.tasks_per_rollout)
        done = (self._step == self.max_step) or reward == 1
        return observation, reward, done, task_info
