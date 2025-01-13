import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.sac import SACAgent
from .diayn import DIAYN


class DIAYN_SACAgent(SACAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale  # 1.0
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        # sdpbrl
        self.solved_meta = None

        # create actor and critic
        super().__init__(**kwargs)

        # create diayn (self.obs_dim = obs_dim + skill_dim)
        self.diayn = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)

        self.diayn.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def _get_random_skill(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        return skill

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_diayn(self, skill, next_obs, step):
        metrics = dict()

        loss, df_accuracy = self.compute_diayn_loss(next_obs, skill)

        self.diayn_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, next_obs, step):
        z_hat = torch.argmax(skill, dim=1) # true skill
        d_pred = self.diayn(next_obs)  # predicted skill
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)

        return reward * self.diayn_scale

    def compute_diayn_loss(self, next_state, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)  # true skill
        d_pred = self.diayn(next_state)  # predicted skill
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_diayn(skill, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obs, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def update_without_skill(self, replay_iter, step, reward_model):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if len(batch) > 5:
            batch = batch[:5]  # remove skills
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        skill = self.use_maxR_skill(replay_iter, step, reward_model).reshape(
            1, -1).repeat(obs.shape[0], 1)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)
        
        reward = extr_reward  # must be not reward_free

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(self.update_critic(obs.detach(), 
            action, reward, discount, next_obs.detach(), step))
        # update actor
        metrics.update(self.update_actor(obs.detach(), step))
        # update alpha
        metrics.update(self.update_alpha(obs, step))
        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def use_maxR_skill(self, replay_iter, step, reward_model):
        if self.solved_meta is None or step % 500 == 0:
            self.maxR_skill(reward_model)
        return torch.as_tensor(self.solved_meta['skill']).to(self.device)
    
    @torch.no_grad()
    def maxR_skill(self, reward_model):
        random_skills = np.stack([self._get_random_skill() for _ in range(50)], axis=0)
        z_returns = reward_model.get_skill_return(  # shape: (50,)
            torch.from_numpy(random_skills).float().to(self.device))[0].detach().cpu().numpy()
        best_skill = random_skills[np.argmax(z_returns)]
        meta = OrderedDict()
        meta['skill'] = best_skill
        # save for evaluation
        self.solved_meta = meta
        return meta


