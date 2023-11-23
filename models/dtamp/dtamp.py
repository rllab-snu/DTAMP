import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

from .actor import Actor
from .critic import Critics
from .decoder import Decoder
from models.diffuser import LatentDiffusion, TemporalUnet

from copy import deepcopy


class DTAMP(nn.Module):
    def __init__(
            self,
            state_dim,
            act_dim,
            goal_dim,
            visual_perception,
            horizon,
            n_critics,
            rl_coeff,
            decoder_coeff,
            diffuser_coeff,
            predict_epsilon,
            diffuser_timesteps,
            returns_condition,
            condition_guidance_w,
            hidden_size,
    ):
        super().__init__()
        self.actor = Actor(
            state_dim, act_dim, goal_dim // 2, visual_perception, hidden_size
        )
        self.critic = Critics(
            state_dim, act_dim, goal_dim // 2, n_critics, visual_perception, hidden_size
        )
        self.diffuser = LatentDiffusion(
            model=TemporalUnet(horizon, goal_dim, returns_condition=returns_condition),
            horizon=horizon,
            latent_dim=goal_dim,
            n_timesteps=diffuser_timesteps,
            predict_epsilon=predict_epsilon,
            returns_condition=returns_condition,
            condition_guidance_w=condition_guidance_w
        )
        if visual_perception:
            self.decoder = Decoder(goal_dim)

        self.goal_dim = goal_dim
        self.visual_perception = visual_perception
        self.rl_coeff = rl_coeff
        self.decoder_coeff = decoder_coeff
        self.diffuser_coeff = diffuser_coeff

    def loss(self, batch, warmup=False):
        """
        observations: [B, H, Do]
        actions: [B, H, Da]
        """
        batch['g_actor'] = self.actor.encode(batch['observations'])
        batch['g_critic'] = self.critic.encode(batch['observations'])
        batch['g'] = torch.cat([batch['g_actor'], batch['g_critic']], dim=-1)

        actor_loss = self.compute_actor_loss(batch, warmup)
        critic_loss = self.compute_critic_loss(batch)
        diffuser_loss = self.compute_diffuser_loss(batch)
        loss = actor_loss + critic_loss + self.diffuser_coeff * diffuser_loss
        if self.visual_perception:
            decoder_loss = self.compute_decoder_loss(batch)
            loss += self.decoder_coeff * decoder_loss

        logs = {
            'loss': loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'diffuser_loss': diffuser_loss,
        }
        if self.visual_perception:
            logs['decoder_loss'] = decoder_loss
        return loss, logs

    def compute_actor_loss(self, batch, warmup):
        goals = batch['g_actor'][:, 1:]
        observations = batch['observations'][:, :-1]
        pi = self.actor(observations, goals)
        bc_loss = (pi - batch['actions'][:, :-1, :]).pow(2).mean()
        if warmup:
            return bc_loss
        else:
            critic = deepcopy(self.critic)
            min_q_pi = critic.min_q(batch['observations'], pi, batch['g_critic'])
            actor_loss = -self.rl_coeff * min_q_pi.mean() / min_q_pi.abs().mean().detach() + bc_loss
            return actor_loss
        
    def compute_actor_loss2(self, batch, warmup):
        goals = batch['g_actor'][:, 1:]
        observations = batch['observations'][:, :-1]
        pi = self.actor(observations, goals)
        bc_loss = (pi - batch['actions'][:, :-1, :]).pow(2).mean()
        if warmup:
            return bc_loss
        else:
            critic = deepcopy(self.critic)
            min_q_pi = critic.min_q(batch['observations'], pi, batch['g_critic'])
            actor_loss = -self.rl_coeff * min_q_pi.mean() / min_q_pi.abs().mean().detach() + bc_loss
            return actor_loss

    def compute_critic_loss(self, batch):
        g_neg = self.sample_negative_goals(batch)
        q = self.critic(batch['observations'], batch['actions'], batch['g_critic'])
        q_neg = self.critic(batch['observations'], batch['actions'], g_neg)
        critic_loss = (bce_loss(q, torch.ones_like(q), reduction='mean')
                       + bce_loss(q_neg, torch.zeros_like(q_neg), reduction='mean'))
        return critic_loss

    def compute_diffuser_loss(self, batch):
        cond = {0: batch['g'][:, 0], -1: batch['g'][:, -1]}
        return self.diffuser.loss(batch['g'], cond)

    def compute_decoder_loss(self, batch):
        obs_recon = self.decoder(batch['g'])
        decoder_loss = (obs_recon - batch['target_observations']).pow(2).mean()
        return decoder_loss

    def sample_negative_goals(self, batch):
        neg_goals = torch.cat([batch['g_critic'][1:], batch['g_critic'][:1]], dim=0)
        neg_goals = neg_goals[:, torch.randperm(neg_goals.size(1)), :]
        return neg_goals
