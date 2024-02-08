import numpy as np
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

from networks import SkillRecognition, SkillProposal, ManipulationSkillDecoder


class PlayLMP(nn.Module):
    def __init__(
            self,
            state_dim,
            act_dim,
            goal_dim,
            skill_dim,
            kl_coeff,
            kl_balance_coeff,
    ):
        super().__init__()
        self.skill_recognition = SkillRecognition(
            state_dim, act_dim, skill_dim
        )
        self.skill_proposal = SkillProposal(
            state_dim, goal_dim, skill_dim
        )
        self.skill_decoder = ManipulationSkillDecoder(
            state_dim, act_dim, skill_dim
        )

        self.state_dim, self.skill_dim, self.act_dim, self.goal_dim = state_dim, skill_dim, act_dim, goal_dim

        self.kl_coeff = kl_coeff
        self.kl_balance_coeff = kl_balance_coeff

    def loss(self, batch, **kwargs):
        """
        observations: [B, H, C, H, W]
        actions: [B, H, Da]
        """
        skill_posterior = self.skill_recognition(batch['observations'], batch['actions'])
        skill_prior = self.skill_proposal(batch['observations'][:, 0], batch['goals'])
        kl_loss = self.kl_balance_coeff * kl_divergence(skill_posterior, skill_prior.detach()).mean() \
                  + (1. - self.kl_balance_coeff) * kl_divergence(skill_posterior.detach(), skill_prior).mean()

        skill = skill_posterior.rsample()
        log_prob, pi = self.skill_decoder.log_prob_with_sample(batch['observations'], batch['actions'], skill)
        decoder_loss = -log_prob.mean()

        loss = decoder_loss + self.kl_coeff * kl_loss
        logs = {
            'train/loss': loss,
            'train/decoder_loss': decoder_loss,
            'train/kl_loss': kl_loss
        }
        return loss, logs

    @torch.no_grad()
    def get_action(self, obs_list, skill, device=torch.device('cuda')):
        obs = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device).unsqueeze(0)
        skill = torch.as_tensor(skill, dtype=torch.float32, device=device).unsqueeze(0)
        act = self.skill_decoder.sample(obs, skill)
        return act[0, -1].cpu().detach().numpy()