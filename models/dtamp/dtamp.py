import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

from networks import Actor, Critics, Decoder
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
            kl_coeff,
            decoder_coeff,
            diffuser_coeff,
            predict_epsilon,
            diffuser_timesteps,
            returns_condition,
            condition_guidance_w,
            hidden_size
    ):
        super().__init__()
        self.deterministic_enc = (kl_coeff == 0)
        self.actor = Actor(
            state_dim, act_dim, goal_dim // 2, visual_perception, hidden_size, self.deterministic_enc
        )
        self.critic = Critics(
            state_dim, act_dim, goal_dim // 2, n_critics, visual_perception, hidden_size, self.deterministic_enc
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
        self.kl_coeff = kl_coeff
        self.returns_condition = returns_condition
        self.decoder_coeff = decoder_coeff
        self.diffuser_coeff = diffuser_coeff

    def loss(self, batch, warmup=False):
        """
        observations: [B, H, Do]
        actions: [B, H, Da]
        """
        batch['g_actor'], batch['kl_actor'] = self.actor.encode(batch['observations'])
        batch['g_critic'], batch['kl_critic'] = self.critic.encode(batch['observations'])
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
            actor_loss = bc_loss
        else:
            critic = deepcopy(self.critic)
            min_q_pi = critic.min_q(
                observations, pi, batch['g_critic'][:, 1:].detach()
            )
            actor_loss = -self.rl_coeff * min_q_pi.mean() / min_q_pi.abs().mean().detach() + bc_loss
        actor_loss += self.kl_coeff * batch['kl_actor']
        return actor_loss

    def compute_critic_loss(self, batch):
        g_neg = self.sample_negative_goals(batch)
        q = self.critic(
            batch['observations'][:, :-1], batch['actions'][:, :-1], batch['g_critic'][:, 1:]
        )
        q_neg = self.critic(
            batch['observations'][:, :-1], batch['actions'][:, :-1], g_neg[:, 1:])
        critic_loss = (bce_loss(q, torch.ones_like(q), reduction='mean')
                       + bce_loss(q_neg, torch.zeros_like(q_neg), reduction='mean'))
        critic_loss += self.kl_coeff * batch['kl_critic']
        return critic_loss

    def compute_diffuser_loss(self, batch):
        g = batch['g']
        cond = {0: g[:, 0], -1: g[:, -1]}
        
        returns = batch['returns'].reshape(-1, 1) if self.returns_condition else None
        diffuser_loss = self.diffuser.loss(g, cond, returns)
        return diffuser_loss

    def compute_decoder_loss(self, batch):
        obs_recon = self.decoder(batch['g'])
        decoder_loss = (obs_recon - batch['target_observations']).pow(2).mean()
        return decoder_loss

    def sample_negative_goals(self, batch):
        if self.visual_perception:
            neg_goals = torch.zeros_like(batch['g_critic'])
            for i in range(len(batch['g_critic'])):
                candidates = torch.cat([batch['g_critic'][:i], batch['g_critic'][i + 1:]], dim=0)
                candidates = candidates.reshape(-1, self.goal_dim // 2)
                distance = (batch['g_critic'][i].unsqueeze(1) - candidates.unsqueeze(0)).pow(2).sum(-1)
                neg_goal_idx = torch.argmin(distance, dim=-1)
                neg_goals[i] = candidates[neg_goal_idx]
        else:
            neg_goals = torch.cat([batch['g_critic'][1:], batch['g_critic'][:1]], dim=0)
            neg_goals = neg_goals[:, torch.randperm(neg_goals.size(1)), :]
        return neg_goals

    @torch.no_grad()
    def planning(self, obs, goal, target_returns=None, device=torch.device('cuda')):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=device)
        if target_returns is not None and self.returns_condition:
            target_returns = torch.as_tensor(target_returns, dtype=torch.float32, device=device).view(1, 1)
        else:
            target_returns=None
        obs_goal = torch.stack([obs, goal], dim=0)
        g_actor, _ = self.actor.encode(obs_goal, eval=True)
        g_critic, _ = self.critic.encode(obs_goal, eval=True)
        g = torch.cat([g_actor, g_critic], dim=-1)
        cond = {0: g[:1, :], -1: g[1:, :]}
        g_pred = self.diffuser(cond, returns=target_returns)
        g_pred_actor, g_pred_critic = torch.split(g_pred, self.goal_dim // 2, dim=-1)
        g_pred_actor = F.normalize(g_pred_actor, p=2.0, dim=-1)
        g_pred_critic = F.normalize(g_pred_critic, p=2.0, dim=-1)
        milestones = torch.cat([g_pred_actor, g_pred_critic], dim=-1)
        return milestones[:, 1:, :].squeeze()


    @torch.no_grad()
    def get_action(self, obs, milestones, threshold=0.1, device=torch.device('cuda')):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        g = torch.as_tensor(milestones, dtype=torch.float32, device=device)
        
        g_obs = torch.cat([self.actor.encode(obs, eval=True)[0], self.critic.encode(obs, eval=True)[0]], dim=-1)
        distance = (g_obs - g).pow(2).sum(-1)
        reached_idx = torch.where(distance[:-1] < threshold)[0]
        if len(reached_idx) > 0:
            g = g[reached_idx[-1]:]

        act = self.actor(obs, g[:1, :self.goal_dim // 2])
        return act.squeeze().cpu().detach().numpy(), g

    @torch.no_grad()
    def encode(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda'))
        g_actor, _ = self.actor.encode(obs, eval=True)
        g_critic, _ = self.critic.encode(obs, eval=True)
        g = torch.cat([g_actor, g_critic], dim=-1)
        return g
