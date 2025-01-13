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


def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :],
                              dim=-1, p=2)
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)


class RuneSACAgent(SACAgent):
    def __init__(self, knn_rms, knn_k, knn_avg,
                 knn_clip, num_init_steps, update_encoder,
                 # rune
                 intr_reward_beta=0.05, one_minus_tau=0.9999,  # 1 - 1e-4
                 **kwargs):
        self.num_init_steps = num_init_steps
        self.update_encoder = update_encoder
        self.intr_reward_beta = intr_reward_beta
        self.one_minus_tau = one_minus_tau

        # create actor and critic
        super().__init__(**kwargs)

        self.train()
        self.critic_target.train()

    def borrow_reward_model(self, reward_model):
        self.temp_reward_model = reward_model

    def del_reward_model(self):
        del self.temp_reward_model

    def compute_intr_reward_std(self, obs, action):
        x = torch.cat([obs, action], dim=1)  # (1024, 23)
        _, intr_ent_reward = self.temp_reward_model.r_hat_disagreement(x)
        intr_ent_reward = intr_ent_reward.detach().reshape(-1, 1)  # (1024, 1)
        return intr_ent_reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.intr_reward_beta > 1e-7:
            intr_reward = self.compute_intr_reward_std(obs, action)
        else:
            intr_reward = 0
        # print(f'intr_reward: {intr_reward.mean().item()}, {intr_reward.shape}')
        # print(f'extr_reward: {extr_reward.mean().item()}, {extr_reward.shape}')
        # print(f'self.intr_reward_beta: {self.intr_reward_beta}')
        reward = self.intr_reward_beta * intr_reward + extr_reward
        if self.use_tb or self.use_wandb:
            if self.intr_reward_beta > 1e-7:
                metrics['rune_intr_reward'] = intr_reward.mean().item()
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            metrics['rune_beta'] = self.intr_reward_beta

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        # update beta for intrinsic reward
        self.intr_reward_beta *= self.one_minus_tau

        return metrics
