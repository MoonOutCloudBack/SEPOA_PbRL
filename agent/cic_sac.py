import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.sac import SACAgent
from .cic import CIC, RMS, APTArgs


class CIC_SACAgent(SACAgent):
    # Contrastive Intrinsic Control (CIC)
    def __init__(self, update_skill_every_step, skill_dim, scale, 
                    project_skill, rew_type, update_rep, temp, **kwargs):
        self.temp = temp
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.scale = scale  # 1.0
        self.project_skill = project_skill
        self.rew_type = rew_type
        self.update_rep = update_rep
        self.device = kwargs['device']
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        # sdpbrl
        self.solved_meta = None
        

        super().__init__(**kwargs)

        # create rms
        self.rms = RMS(device=self.device)

        # create cic first (self.obs_dim = obs_dim + skill_dim)
        self.cic = CIC(self.obs_dim - skill_dim, skill_dim,
                           kwargs['hidden_dim'], project_skill).to(kwargs['device'])

        # optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(),
                                                lr=self.lr)

        self.cic.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)
    
    def _get_random_skill(self):
        return np.random.uniform(0, 1, self.skill_dim).astype(np.float32)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        if not self.reward_free:
            # selects mean skill of 0.5 (to select skill automatically use CEM or Grid Sweep
            # procedures described in the CIC paper)
            skill = np.ones(self.skill_dim).astype(np.float32) * 0.5
        else:
            skill = np.random.uniform(0, 1, self.skill_dim).astype(np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, step, time_step):
        if step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.temp
        eps = 1e-6
        query, key = self.cic.forward(obs, next_obs, skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T) # (b,b)
        sim = torch.exp(cov / temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        return loss, cov / temperature

    def update_cic(self, obs, skill, next_obs, step):
        metrics = dict()

        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()
        self.cic_optimizer.zero_grad()
        loss.backward()
        self.cic_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['cic_loss'] = loss.item()
            metrics['cic_logits'] = logits.norm()

        return metrics

    def compute_intr_reward(self, obs, skill, next_obs, step):  # haven't used (?)
        
        with torch.no_grad():
            loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
      
        reward = loss
        reward = reward.clone().detach().unsqueeze(-1)

        return reward * self.scale

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        # reward = compute_apt_reward(source, target, args) # (b,)

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
        reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not args.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if args.rms:
                moving_mean, moving_std = self.rms(reward)
                reward = reward / moving_std
            reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(self.device))  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if args.rms:
                moving_mean, moving_std = self.rms(reward)
                reward = reward / moving_std
            reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(self.device))
            reward = reward.reshape((b1, args.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward.unsqueeze(-1) # (b,1) -> (b,)
    

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        with torch.no_grad():
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            if self.update_rep:
                metrics.update(self.update_cic(obs, skill, next_obs, step))

            intr_reward = self.compute_apt_reward(next_obs, next_obs)
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            if self.reward_free:
                metrics['extr_reward'] = extr_reward.mean().item()
                # metrics['intr_reward'] = apt_reward.mean().item()
                metrics['intr_reward'] = intr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, step))

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
        with torch.no_grad():
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
        
        reward = extr_reward  # must be not reward_free

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
        # update actor
        metrics.update(self.update_actor(obs, step))
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
