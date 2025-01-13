import random
import numpy as np
import torch
import utils
from typing import Tuple, Iterator
from dmc import ExtendedTimeStep
import pickle
import torch.utils.data


class ReplayBuffer(torch.utils.data.IterableDataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, meta_shape_list, capacity, device,
                 batch_size, nstep, discount,
                 window=1, max_episode_len=1001, use_preference: bool = False):
        capacity = capacity // max_episode_len
        self.capacity = capacity
        self.device = device
        self.max_episode_len = max_episode_len
        self.window = window
        self.nstep = nstep
        self.gamma = discount

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.zeros((capacity, max_episode_len, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, max_episode_len, *action_shape), dtype=np.float32)
        self.metas_list = [np.zeros((capacity, max_episode_len, *meta_shape), dtype=np.float32)
                           for meta_shape in meta_shape_list]
        self.rewards = np.zeros((capacity, max_episode_len, 1), dtype=np.float32)
        self.predicted_rewards = np.zeros((capacity, max_episode_len, 1), dtype=np.float32)

        self.batch_size = batch_size
        self.episode_idx = 0
        self.idx = 0
        self.last_save = 0
        self.full = False

        self.use_preference = use_preference

    def __len__(self):
        return (self.capacity if self.full else self.episode_idx) * self.max_episode_len

    def add(self, time_step: ExtendedTimeStep, meta, predicted_reward=None):
        next_obs = time_step.observation
        action = time_step.action  # is zero when reset (action=None)
        reward = time_step.reward
        discount = time_step.discount 
        done = time_step.last()
        done_no_max = False
        meta = list(meta.values()) 
        self._add(next_obs, action, reward, meta, predicted_reward)

    def _add(self, obs, action, reward, meta, predicted_reward=None):
        np.copyto(self.obses[self.episode_idx, self.idx], obs)
        np.copyto(self.actions[self.episode_idx, self.idx], action)
        np.copyto(self.rewards[self.episode_idx, self.idx], reward)
        if predicted_reward is not None:
            np.copyto(self.predicted_rewards[self.episode_idx, self.idx], predicted_reward)
        for i in range(len(self.metas_list)):
            np.copyto(self.metas_list[i][self.episode_idx, self.idx], meta[i])

        self.idx = (self.idx + 1) % self.max_episode_len
        if self.idx == 0:
            self.episode_idx = (self.episode_idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0 and self.episode_idx == 0)

    @torch.no_grad()
    def relabel_with_predictor_meta(self, predictor_rew, predictor_meta):
        episode_num = self.capacity if self.full else self.episode_idx
        for e_i in range(episode_num):
            obses = self.obses[e_i, :-1, :]
            actions = self.actions[e_i, 1:, :]  # take the second recorded action
            inputs = np.concatenate([obses, actions], axis=-1)
            pred_reward = predictor_rew.r_hat_batch(inputs)
            self.predicted_rewards[e_i, 1:, :] = pred_reward

            pred_meta = list(predictor_meta(obses, pred_reward).values())
            for i in range(len(self.metas_list)):
                np.copyto(self.metas_list[i][e_i, 1:, :], pred_meta[i])
                self.metas_list[i][e_i, 0] = self.metas_list[i][e_i, 1].copy()

    @torch.no_grad()
    def relabel_with_predictor(self, predictor):
        episode_num = self.capacity if self.full else self.episode_idx
        for e_i in range(episode_num):
            obses = self.obses[e_i, :-1, :]
            actions = self.actions[e_i, 1:, :]  # take the second recorded action
            inputs = np.concatenate([obses, actions], axis=-1)
            pred_reward = predictor.r_hat_batch(inputs)
            self.predicted_rewards[e_i, 1:, :] = pred_reward

    def sample(self, batch_size, set_device=True):
        device_modifier = dict(device=self.device) if set_device else {}

        if (not self.full) and self.episode_idx == 0:  # no episode completed
            episode_idxs = np.zeros((batch_size, ), dtype=int)
            idxs = np.random.randint(0, self.idx - self.nstep, size=batch_size) + 1
        else:
            episode_idxs = np.random.randint(0,
                                             self.capacity if self.full else self.episode_idx,
                                             size=batch_size)
            idxs = np.random.randint(0, self.max_episode_len - self.nstep, size=batch_size) + 1  # [1, 1001)

        # if random.random() < 0.0001:
        #     print(f'episode_idxs: {episode_idxs[0]}, idxs: {idxs[0]}')

        obses = torch.as_tensor(self.obses[episode_idxs, idxs - 1], **device_modifier).float()
        actions = torch.as_tensor(self.actions[episode_idxs, idxs], **device_modifier)
        next_obses = torch.as_tensor(self.obses[episode_idxs, idxs + self.nstep - 1],
                                     **device_modifier).float()
        metas = []
        for i in range(len(self.metas_list)):
            metas.append(torch.as_tensor(self.metas_list[i][episode_idxs, idxs], **device_modifier))

        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        discounts = np.ones((batch_size, 1), dtype=np.float32)
        used_rewards = self.predicted_rewards if self.use_preference else self.rewards
        for i in range(self.nstep):
            rewards += discounts * used_rewards[episode_idxs, idxs + i]
            discounts *= self.gamma
        rewards = torch.as_tensor(rewards, **device_modifier)
        discounts = torch.as_tensor(discounts, **device_modifier)

        return obses, actions, rewards, discounts, next_obses, *metas  # , not_dones, not_dones_no_max

    def __iter__(self):
        while True:
            # yield self.sample1()
            yield self.sample(self.batch_size)

    def save_pickle(self, path='./models/states/walker/aps/1/replay_buffer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_pickle(self, path='./models/states/walker/aps/1/replay_buffer.pkl'):
        with open(path, 'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer


def make_replay_loader(train_env, meta_specs, replay_buffer_size, device, batch_size, num_workers, nstep, discount, use_preference
                       ) -> Tuple[ReplayBuffer, Iterator[ReplayBuffer]]:
    replay_buffer = ReplayBuffer(obs_shape=train_env.observation_spec().shape,
                                 action_shape=train_env.action_spec().shape,
                                 meta_shape_list=[meta_spec.shape for meta_spec in meta_specs],
                                 capacity=replay_buffer_size,
                                 device=device,
                                 batch_size=batch_size,
                                 nstep=nstep,
                                 discount=discount,
                                 use_preference=use_preference,
                                 )
    loader = replay_buffer
    return replay_buffer, loader


if __name__ == '__main__':
    replay_buffer_hx = ReplayBuffer(obs_shape=(3, ),
                                    action_shape=(1, ),
                                    meta_shape_list=[(2,), ],
                                    capacity=10000,
                                    device='cpu',
                                    batch_size=32,
                                    nstep=1,
                                    discount=0.99, )
    for _ in range(2):
        replay_buffer_hx.add(
            time_step=ExtendedTimeStep(step_type='FIRST', observation=np.random.rand(3),
                                       action=np.random.random(), reward=0.0, discount=0.99),
            meta={'task': np.random.rand(2)}
        )
    print(replay_buffer_hx.sample(1))
    replay_buffer_hx.save_pickle(path='./123123123.pkl')
    del replay_buffer_hx
    replay_buffer_hx = ReplayBuffer(obs_shape=(3, ),
                                    action_shape=(1, ),
                                    meta_shape_list=[(2,), ],
                                    capacity=100,
                                    device='cpu',
                                    batch_size=32,
                                    nstep=1,
                                    discount=0.99, )
    replay_buffer_hx = replay_buffer_hx.load_pickle(path='./123123123.pkl')
    print(replay_buffer_hx.sample(1))
