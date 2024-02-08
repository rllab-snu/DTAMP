import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from .distributions import DiscreteMixLogistic, SquashedNormal
from .perception import Perception

import numpy as np

MIN_LOG_STD = -20.
MAX_LOG_STD = 2.


class SkillDecoder(nn.Module):
    def __init__(self, state_dim, skill_dim, act_dim, hidden_size=256):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim, hidden_size=hidden_size)

        self.layers = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * act_dim)
        )
        self.act_dim = act_dim

    def forward(self, obs, skill):
        emb_state = self.perception(obs)
        if len(skill.size()) != len(emb_state.size()):
            skill = skill.unsqueeze(1).repeat(1, emb_state.size(1), 1)
        x = torch.cat([emb_state, skill], dim=-1)
        x = self.layers(x)
        mean, logstd = torch.split(x, self.act_dim, dim=-1)
        std = logstd.clamp(min=MIN_LOG_STD, max=MAX_LOG_STD).exp()
        return SquashedNormal(mean, std)

    def sample(self, obs, skill, deterministic=True):
        dist = self.forward(obs, skill)
        if deterministic:
            return dist.mean
        else:
            return dist.sample()

    def log_prob(self, obs, act, skill):
        dist = self.forward(obs, skill)
        log_prob = dist.log_prob(act)
        return log_prob

    def log_prob_with_sample(self, obs, act, skill):
        dist = self.forward(obs, skill)
        log_prob = dist.log_prob(act)
        pi = dist.rsample()
        return log_prob, pi


class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=False):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures, self._dist_size = n_mixtures, torch.Size((out_dim, n_mixtures))
        self._mu = nn.Linear(in_dim, out_dim * n_mixtures)
        if const_var:
            ln_scale = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        else:
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None

    def forward(self, x):
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)

        logit_prob = self._logit_prob(x).reshape((x.shape[:-1] + self._dist_size)) if self._n_mixtures > 1 else torch.ones_like(mu)
        return mu, ln_scale, logit_prob


class ManipulationSkillDecoder(nn.Module):
    def __init__(self, state_dim, act_dim, skill_dim, n_mixtures=10, hidden_size=2048):
        super().__init__()
        self.perception = Perception(128, 128, out_dim=state_dim)
        self.gru = nn.GRU(
            input_size=state_dim + skill_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.fc_eef = _DiscreteLogHead(hidden_size, act_dim - 1, n_mixtures)
        self.fc_gripper = nn.Linear(hidden_size, 1)
        self.n_mixtures = n_mixtures
        self.act_dim = act_dim

    def forward(self, obs, skill):
        emb_state = self.perception(obs)
        if len(skill.size()) != len(emb_state.size()):
            skill = skill.unsqueeze(1).repeat(1, emb_state.size(1), 1)
        x = torch.cat([emb_state, skill], dim=-1)
        x, _ = self.gru(x)
        mu, ln_scale, logit_prob = self.fc_eef(x)
        eef_dist = DiscreteMixLogistic(mu, ln_scale, logit_prob, self.n_mixtures)
        gripper_logit = self.fc_gripper(x)
        gripper_dist = Bernoulli(logits=gripper_logit)
        return eef_dist, gripper_dist

    def sample(self, obs, skill, deterministic=True):
        eef_dist, gripper_dist = self.forward(obs, skill)
        if deterministic:
            eef = eef_dist.mean
            gripper = torch.where(gripper_dist.logits > 0, 1., -1.)
        else:
            eef = eef_dist.sample()
            gripper = torch.where(gripper_dist.sample() == 1., 1., -1.)
        act = torch.cat([eef, gripper], dim=-1)
        return act

    def log_prob(self, obs, act, skill):
        eef, gripper = torch.split(act, (self.act_dim - 1, 1), dim=-1)
        gripper = torch.where(gripper == 1., 1., 0.)
        eef_dist, gripper_dist = self.forward(obs, skill)
        logp_eef = eef_dist.log_prob(eef)
        logp_gripper = gripper_dist.log_prob(gripper)
        log_prob = logp_eef.mean(-1) + logp_gripper.mean(-1)
        return log_prob

    def log_prob_with_sample(self, obs, act, skill):
        eef, gripper = torch.split(act, (self.act_dim - 1, 1), dim=-1)
        gripper = torch.where(gripper == 1., 1., 0.)
        eef_dist, gripper_dist = self.forward(obs, skill)
        logp_eef = eef_dist.log_prob(eef)
        logp_gripper = gripper_dist.log_prob(gripper)
        log_prob = logp_eef.mean(-1) + logp_gripper.mean(-1)

        pi_eef = eef_dist.sample()
        pi_gripper = gripper_dist.sample()
        pi_gripper = torch.where(pi_gripper == 1., 1., -1.)
        pi = torch.cat([pi_eef, pi_gripper], dim=-1)
        return log_prob, pi