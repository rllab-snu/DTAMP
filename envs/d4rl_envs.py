import numpy as np
from gym.spaces import Box


class AntmazeEnvWrapper:
    def __init__(self, env, normalize_obs=True):
        self.env = env
        self.dataset = env.get_dataset()
        self.dataset['episode_ends'] = np.where(self.dataset['timeouts'])[0]
        self.normalize_obs = normalize_obs
        self.max_obs = np.max(self.dataset['observations'], axis=0)
        self.min_obs = np.min(self.dataset['observations'], axis=0)
        if normalize_obs:
            max_obs = np.reshape(self.max_obs, (1, -1))
            min_obs = np.reshape(self.min_obs, (1, -1))
            self.dataset['observations'] = 2. * (self.dataset['observations'] - min_obs) / (max_obs - min_obs) - 1.

    def get_dataset(self):
        return self.dataset

    def normalize(self, obs):
        return 2. * (obs - self.min_obs) / (self.max_obs - self.min_obs) - 1.

    def reset(self):
        obs = self.env.reset()
        obs = self.normalize(obs) if self.normalize_obs else obs
        goal = self.get_goal()
        return obs, goal

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        obs = self.normalize(obs) if self.normalize_obs else obs
        return obs, rew, done, info

    def get_goal(self):
        goal = np.zeros(self.env.observation_space.shape)
        goal[:2] = self.env.target_goal
        goal = self.normalize(goal) if self.normalize_obs else goal
        goal[2:] = 0.
        return goal

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, mode='human'):
        self.env.render(mode)


class KitchenEnvWrapper:
    def __init__(self, env, normalize_obs=True):
        self.env = env
        self.dataset = env.get_dataset()
        self.dataset['episode_ends'] = np.where(
            np.logical_or(self.dataset['timeouts'], self.dataset['terminals'])
        )[0]
        '''
        Kitchen environments provide observations concatenated with the goal (index: 30 ~59).
        We only consider current state (index: 0 ~ 29).
        '''
        self.dataset['observations'] = self.dataset['observations'][:, :30]
        self.normalize_obs = normalize_obs
        self.max_obs = np.max(self.dataset['observations'], axis=0)
        self.min_obs = np.min(self.dataset['observations'], axis=0)
        if normalize_obs:
            max_obs = np.reshape(self.max_obs, (1, -1))
            min_obs = np.reshape(self.min_obs, (1, -1))
            self.dataset['observations'] = 2. * (self.dataset['observations'] - min_obs) / (max_obs - min_obs) - 1.

    def get_dataset(self):
        return self.dataset

    def normalize(self, obs):
        return 2. * (obs - self.min_obs) / (self.max_obs - self.min_obs) - 1.

    def reset(self):
        obs = self.env.reset()
        obs = self.normalize(obs[:30]) if self.normalize_obs else obs[:30]
        goal = self.get_goal()
        return obs, goal

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        obs = self.normalize(obs[:30]) if self.normalize_obs else obs[:30]
        return obs, rew, done, info

    def get_goal(self):
        goal = self.env.goal
        goal = self.normalize(goal) if self.normalize_obs else goal
        goal[:9] = 0
        return goal

    @property
    def observation_space(self):
        return Box(-np.ones(30), np.ones(30), shape=(30,))

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, mode='human'):
        mode = 'display' if mode == 'human' else mode
        self.env.render(mode)