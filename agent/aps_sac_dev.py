import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.sac_dev import SACAgent


from .aps import CriticSF_Dev, APS


class APSSACAgent(SACAgent):
    def __init__(self, update_task_every_step, sf_dim, knn_rms, knn_k, knn_avg,
                 knn_clip, num_init_steps, lstsq_batch_size, update_encoder,
                 lr_aps, feature_dim, hidden_dim,
                 # SD + PbRL
                 sd_pbrl=False, reward_lamda: float = 0.1, meta_variance: float = 0.1,
                 ent_reward_ratio: float = 3,
                 **kwargs):
        self.sf_dim = sf_dim
        self.update_task_every_step = update_task_every_step
        self.num_init_steps = num_init_steps
        self.lstsq_batch_size = lstsq_batch_size
        self.update_encoder = update_encoder
        self.solved_meta = None
        # SD + PbRL
        self.sd_pbrl = sd_pbrl
        self.reward_lamda = reward_lamda
        self.meta_variance = meta_variance
        self.ent_reward_ratio = ent_reward_ratio

        self.lr_aps = lr_aps
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # increase obs shape to include task dim
        kwargs["meta_dim"] = self.sf_dim

        # create actor and critic
        super().__init__(**kwargs)

        # overwrite critic with critic sf
        self.critic = CriticSF_Dev(self.obs_type, self.obs_dim, self.action_dim,
                                   self.feature_dim, self.hidden_dim,
                                   self.sf_dim).to(self.device)
        self.critic_target = CriticSF_Dev(self.obs_type, self.obs_dim,
                                          self.action_dim, self.feature_dim,
                                          self.hidden_dim,
                                          self.sf_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.lr_aps)

        self.aps = APS(self.obs_dim - self.sf_dim, self.sf_dim,
                       self.hidden_dim).to(kwargs['device'])
        #    kwargs['hidden_dim']).to(kwargs['device'])

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

        # optimizers
        self.aps_opt = torch.optim.Adam(self.aps.parameters(), lr=self.lr_aps)

        self.train()
        self.critic_target.train()

        self.aps.train()

    def get_meta_specs(self):
        return (specs.Array((self.sf_dim,), np.float32, 'task'),)

    def _get_random_skill(self):
        task = torch.randn(self.sf_dim).to(self.device)
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        return task

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        meta = OrderedDict()
        meta['task'] = self._get_random_skill()
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def update_aps(self, task, next_obs, step):
        metrics = dict()

        loss = self.compute_aps_loss(next_obs, task)

        self.aps_opt.zero_grad(set_to_none=True)
        # if self.encoder_opt is not None:
        #     self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.aps_opt.step()
        # if self.encoder_opt is not None:
        #     self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['aps_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, task, next_obs, step):
        # maxent reward
        with torch.no_grad():
            rep = self.aps(next_obs, norm=False)
        reward = self.pbe(rep)
        intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", task, rep).reshape(-1, 1)

        return intr_ent_reward, intr_sf_reward

    def compute_aps_loss(self, next_obs, task):
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", task, self.aps(next_obs)).mean()
        return loss

    def ordinary_sac_dev_update(self, replay_iter, step, gradient_update=1):
        metrics = dict()
        for index in range(gradient_update):
            batch = next(replay_iter)
            if len(batch) > 5:  # if has skills
                batch = batch[:5]
            obs, action, extr_reward, discount, next_obs, task = utils.to_torch(
                batch, self.device)

            if self.reward_free:
                metrics.update(self.update_aps(task, next_obs, step))

                with torch.no_grad():
                    intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                        task, next_obs, step)
                    intr_reward = self.ent_reward_ratio * intr_ent_reward + intr_sf_reward

                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()
                    metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                    metrics['intr_sf_reward'] = intr_sf_reward.mean().item()

                reward = intr_ent_reward
            elif self.sd_pbrl:  # skill discovery + pbrl
                metrics.update(self.update_aps(task, next_obs, step))
                with torch.no_grad():
                    intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                        task, next_obs, step)
                    intr_reward = intr_ent_reward + intr_sf_reward
                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()
                    metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                    metrics['intr_sf_reward'] = intr_sf_reward.mean().item()
                reward = extr_reward + self.reward_lamda * intr_reward

            else:
                reward = extr_reward

            if index == gradient_update - 1:
                metrics['extr_reward'] = extr_reward.mean().item()
                metrics['batch_reward'] = reward.mean().item()

            m = self.update_critic(obs, action, reward, discount, next_obs, step)
            metrics.update(m)

            if step % self.actor_update_frequency == 0:
                m = self.update_actor_and_alpha(obs.detach(), step)
                metrics.update(m)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        # print(f'self.sd_pbrl: {self.sd_pbrl}')

        # if step % self.update_every_steps != 0:
        #     return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, task = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            # freeze successor features at finetuning phase
            metrics.update(self.update_aps(task, next_obs, step))

            with torch.no_grad():
                intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                    task, next_obs, step)
                intr_reward = self.ent_reward_ratio * intr_ent_reward + intr_sf_reward

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                metrics['intr_sf_reward'] = intr_sf_reward.mean().item()

            reward = intr_reward
        elif self.sd_pbrl:  # skill discovery + pbrl
            metrics.update(self.update_aps(task, next_obs, step))
            with torch.no_grad():
                intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                    task, next_obs, step)
                intr_reward = intr_ent_reward + intr_sf_reward
            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                metrics['intr_sf_reward'] = intr_sf_reward.mean().item()
            reward = extr_reward + self.reward_lamda * intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with task
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), task, step))

        # update actor
        if step % self.actor_update_frequency == 0:  # seem actor_update_frequency = 1
            metrics.update(self.update_actor(obs.detach(), task, step))
            metrics.update(self.update_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_tau)

        self.meta_variance = max(self.meta_variance * 0.99999, 0.001)
        return metrics

    def update_without_skill_old(self, replay_iter, step):
        metrics = dict()
        # print(f'self.sd_pbrl: {self.sd_pbrl}')

        # if step % self.update_every_steps != 0:
        #     return metrics

        batch = next(replay_iter)
        if len(batch) > 5:
            batch = batch[:5]
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        task = self.use_regressed_meta(replay_iter, step).reshape(1, -1).repeat(obs.shape[0], 1)

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

        # extend observations with task
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        # update critic
        metrics.update(self.update_critic(obs.detach(), action,
                       reward, discount, next_obs.detach(), task, step))
        # update actor
        if step % self.actor_update_frequency == 0:  # seem actor_update_frequency = 1
            metrics.update(self.update_actor(obs.detach(), task, step))
            metrics.update(self.update_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_tau)

        self.meta_variance = max(self.meta_variance * 0.99999, 0.001)
        return metrics

    def use_regressed_meta(self, replay_iter, step):
        if self.solved_meta is None or step % 500 == 0:
            self.regress_meta(replay_iter, step)
        return torch.as_tensor(self.solved_meta['task']).to(self.device)

    @torch.no_grad()
    def _regress_meta(self, replay_iter, step):
        obs, reward = [], []
        batch_size = 0
        while batch_size < self.lstsq_batch_size:  # sample a batch
            batch = next(replay_iter)
            batch_obs, _, batch_reward, *_ = utils.to_torch(batch, self.device)
            obs.append(batch_obs)
            reward.append(batch_reward)
            batch_size += batch_obs.size(0)
        obs, reward = torch.cat(obs, 0), torch.cat(reward, 0)

        obs = self.aug_and_encode(obs)
        rep = self.aps(obs)
        task = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]  # least square
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task

        # save for evaluation
        self.solved_meta = meta
        return meta

    def regress_meta(self, replay_iter, step):
        ''' task sampled from a gaussian distribution with mean = _regress_meta()'''
        return self._regress_meta(replay_iter, step)
        regressed_task = self._regress_meta(replay_iter, step)['task']
        gaussian = torch.distributions.MultivariateNormal(
            loc=torch.as_tensor(regressed_task), covariance_matrix=self.meta_variance * torch.eye(self.sf_dim))
        task = gaussian.sample()
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task
        return meta

    def update_without_skill(self, replay_iter, step, reward_model):
        metrics = dict()
        # print(f'self.sd_pbrl: {self.sd_pbrl}')

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if len(batch) > 5:
            batch = batch[:5]
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        task = self.use_maxR_skill(replay_iter, step, reward_model).reshape(
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

        # extend observations with task
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        # update critic
        metrics.update(self.update_critic(obs.detach(), action, 
                       reward, discount, next_obs.detach(), task, step))
        # update actor
        metrics.update(self.update_actor(obs.detach(), task, step))
        # update alpha
        metrics.update(self.update_alpha(obs.detach(), step))
        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        
        self.meta_variance = max(self.meta_variance * 0.99999, 0.001)
        return metrics

    def use_maxR_skill(self, replay_iter, step, reward_model):
        if self.solved_meta is None or step % 500 == 0:
            self.maxR_skill(reward_model)
        return torch.as_tensor(self.solved_meta['task']).to(self.device)

    @torch.no_grad()
    def maxR_skill(self, reward_model):
        random_skills = np.stack([self._get_random_skill() for _ in range(50)], axis=0)
        z_returns = reward_model.get_skill_return(  # shape: (50,)
            torch.from_numpy(random_skills).float().to(self.device))[0].detach().cpu().numpy()
        best_skill = random_skills[np.argmax(z_returns)]
        meta = OrderedDict()
        meta['task'] = best_skill
        # save for evaluation
        self.solved_meta = meta
        return meta

    def update_critic(self, obs, action, reward, discount, next_obs, task,
                      step):
        """diff is critic takes task as input"""
        metrics = dict()

        with torch.no_grad():
            # stddev = utils.schedule(self.stddev_schedule, step)
            # dist = self.actor(obs, stddev)
            dist = self.actor(next_obs)
            # next_action = dist.sample(clip=self.stddev_clip)
            next_action = dist.rsample()
            next_log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, task)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, task, step):
        """diff is critic takes task as input"""
        metrics = dict()

        # stddev = utils.schedule(self.stddev_schedule, step)
        # dist = self.actor(obs, stddev)
        dist = self.actor(obs)
        # action = dist.sample(clip=self.stddev_clip)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, task)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def update_alpha(self, obs, step):
        metrics = dict()

        with torch.no_grad():
            # stddev = utils.schedule(self.stddev_schedule, step)
            # dist = self.actor(obs, stddev)
            dist = self.actor(obs)
            # action = dist.sample(clip=self.stddev_clip)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()
        # print(
        #     f"alpha_loss:{alpha_loss}, alpha: {self.log_alpha.exp().detach().item()}, log_prob: {log_prob.mean().detach().item()}, target_entropy: {self.target_entropy}")

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        # self.alpha = self.log_alpha.exp().item()

        metrics["alpha_loss"] = alpha_loss.mean().item()
        metrics["alpha"] = self.alpha.mean().item()
        return metrics
