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


class APTSACAgent(SACAgent):
    def __init__(self, knn_rms, knn_k, knn_avg,
                 knn_clip, num_init_steps, update_encoder,
                 **kwargs):
        self.num_init_steps = num_init_steps
        self.update_encoder = update_encoder

        # create actor and critic
        super().__init__(**kwargs)

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=self.device)

        self.train()
        self.critic_target.train()

    def compute_intr_reward(self, obs):
        # maxent reward
        reward = self.pbe(obs)
        intr_ent_reward = reward.reshape(-1, 1)
        return intr_ent_reward

    def update(self, replay_iter, step, gradient_update=1):
        metrics = dict()
        # for index in range(gradient_update):  # gradient update = 1
        batch = next(replay_iter)
        if len(batch) > 5:  # if has skills
            batch = batch[:5]
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.reward_free:
            intr_ent_reward = self.compute_intr_reward(obs)
            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_ent_reward.mean().item()

            reward = intr_ent_reward
        else:
            reward = extr_reward

        # if index == gradient_update - 1:
        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount,
                               next_obs, step))
        # update actor
        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor_and_alpha(obs.detach(), step))

        # update critic target
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        return metrics
