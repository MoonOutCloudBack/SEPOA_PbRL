import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import dmc
from copy import deepcopy
import time

from replay_buffer_hx_new import ReplayBuffer


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def KCenterGreedy(obs, full_obs, num_new_sample, device):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs, device)
        max_index = torch.argmax(dist)
        max_index = max_index.item()

        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index + 1:]

        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs,
            obs[selected_index]],
            axis=0)
    return selected_index


def compute_smallest_dist(obs, full_obs, device):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)

        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, ds, da, device,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 # teacher
                 label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 skip_impl_tuple=(),
                 # dont know
                 mu=1,
                 weight_factor=1.0,
                 adv_mu=2,
                 path=None,
                 # surf
                 data_aug_ratio=20,
                 data_aug_window=5,
                 threshold_u=0.99,  # surf use 0.99
                 lambda_u=1,
                 large_batch=10,
                 # skill space query selection
                 skill_diff_epsilon=0.1,  # previous 0.05
                 dz: int = 10,
                 disagree_base=1.0,
                 ):

        # train data is trajectories, must process to sa and s..
        self.ds = ds
        self.da = da
        self.device = device
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size  # max number of trajectories
        self.activation = activation
        self.size_segment = size_segment
        self.path = path
        self.count = 0

        self.capacity = int(capacity)  # max number of segments
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_mask = np.ones((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.inputs = []
        self.skill_list = []  # skill list
        self.targets = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.UCELoss = nn.CrossEntropyLoss(reduction='none')
        self.mu = mu
        self.weight_factor = weight_factor
        self.adv_mu = adv_mu
        self.obs_l = 0
        self.action_l = 0

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.skip_impl_tuple = skip_impl_tuple
        self.skip_use_error = 'error' in skip_impl_tuple
        self.skip_use_div = 'div' in skip_impl_tuple
        self.skip_use_traj = 'traj' in skip_impl_tuple
        print(
            f"skip_impl_tuple: {skip_impl_tuple}, div: {self.skip_use_div}, error: {self.skip_use_error}, traj: {self.skip_use_traj}")
        self.disagree_base = disagree_base

        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

        # surf
        self.u_buffer_seg1 = np.empty((self.capacity, self.size_segment, self.ds + self.da), dtype=np.float32)
        self.u_buffer_seg2 = np.empty((self.capacity, self.size_segment, self.ds + self.da), dtype=np.float32)
        self.u_buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.u_buffer_index = 0
        self.u_buffer_full = False

        self.data_aug_ratio = data_aug_ratio
        self.data_aug_window = data_aug_window
        self.large_batch = large_batch
        self.threshold_u = threshold_u
        self.lambda_u = lambda_u

        # skill space query selection
        self.dz = dz
        self.skill_diff_epsilon = skill_diff_epsilon
        self.construct_skill_return_estimator()

        # for predicting skill when querying
        self.predictor_meta = None
        self.state_dim = None
        self.action_dim = None

        # for recording
        self.df = pd.DataFrame(columns=['step', 'en0', 'start0', 'return0',
                                        'en1', 'start1', 'return1', 'can_comp', ])
        self.df_count = 0
        self.episode_counter = 0
        self.episode_num_list = []

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        if self.skip_use_traj:
            self.teacher_thres_skip *= (1000 // 50)  # for the whole trajectory

    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds + self.da,
                                           out_size=1, H=256, n_layers=3,
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew

        flat_input = sa_t.reshape(1, self.da + self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.episode_num_list.append(self.episode_counter)
            self.episode_counter += 1
            # print(f'self.inputs[-1]: {self.inputs[-1]}')
            # print(f'self.inputs[-1]: {self.inputs[-1].shape}')
            # print(f'self.targets[-1]: {self.targets[-1]}')
            # print(f'self.targets[-1]: {self.targets[-1].shape}')
            # FIFO
            if len(self.inputs) > self.max_size:
                # self.inputs = self.inputs[1:]
                # self.targets = self.targets[1:]
                self.inputs = self.inputs[:10] + self.inputs[11:]  # reserve the first 10 random skills
                self.targets = self.targets[:10] + self.targets[11:]
                self.episode_num_list = self.episode_num_list[:10] + self.episode_num_list[11:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])

    def add_data_skills_batch(self, obses, skills, rewards):
        '''
        obses: (batch_size, segment_len, dim_s + dim_a)
        skills: (batch_size, dim_z)
        rewards: (batch_size, segment_len, 1)
        '''
        num_episode = obses.shape[0]
        for index in range(num_episode):
            self.inputs.append(obses[index])
            self.skill_list.append(skills[index])
            self.targets.append(rewards[index])
            self.episode_num_list.append(self.episode_counter)
            self.episode_counter += 1

    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)

        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_rank_discriminator(self, x_1, x_2, disc):
        r_1 = self.r_hat_batch(x_1)
        r_2 = self.r_hat_batch(x_2)
        r_1 = torch.from_numpy(np.sum(r_1, axis=1)).float().to(self.device)
        r_2 = torch.from_numpy(np.sum(r_2, axis=1)).float().to(self.device)
        # r_hat = torch.cat([r_1, r_2], axis=-1)
        labels = 1 * (r_1 < r_2)
        snip1 = x_1.reshape(x_1.shape[0], -1)
        snip1 = torch.from_numpy(snip1).float().to(self.device)
        snip2 = x_2.reshape(x_2.shape[0], -1)
        snip2 = torch.from_numpy(snip2).float().to(self.device)
        p = disc(snip1, snip2, labels)

        return p.cpu().detach().numpy()

    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:, 0]

    def get_p_value(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.abs(np.mean(probs, axis=0) - 0.5)

    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat_member_ndarray(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](x)

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def r_hat_disagreement(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member_ndarray(x, member=member).detach())
        r_hats = torch.cat(r_hats, axis=-1)

        return torch.mean(r_hats, axis=-1), torch.std(r_hats, axis=-1)

    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )

    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )

    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))

        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if (epoch + 1) * batch_size > max_len:
                last_index = max_len

            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    def get_queries(self, mb_size=20):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2]  # Batch x T x 1

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        if self.skip_use_traj:  # 要求 batch_index_1 和 batch_index_2 不同
            while np.any(batch_index_1 == batch_index_2):
                same_index = np.where(batch_index_1 == batch_index_2)[0]
                batch_index_1[same_index] = np.random.choice(max_len, size=len(same_index), replace=True)
        sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1]  # Batch x T x 1

        # tr_1, tr_2 are the sum of reward of the whole trajectory
        tr_1 = np.sum(r_t_1, axis=1).reshape(-1, 1)
        tr_2 = np.sum(r_t_2, axis=1).reshape(-1, 1)

        # sa_t_1_path = self.path + f'{self.count}_sa_t_1.npy'
        # r_t_1_path = self.path + f'{self.count}_r_t_1.npy'
        # sa_t_2_path = self.path + f'{self.count}_sa_t_2.npy'
        # r_t_2_path = self.path + f'{self.count}_r_t_2.npy'
        # batch_1_path = self.path + f'{self.count}_batch_1.npy'
        # batch_2_path = self.path + f'{self.count}_batch_2.npy'
        # np.save(sa_t_1_path, sa_t_1)
        # np.save(r_t_1_path, r_t_1)
        # np.save(sa_t_2_path, sa_t_2)
        # np.save(r_t_2_path, r_t_2)
        # np.save(batch_1_path, batch_index_1)
        # np.save(batch_2_path, batch_index_2)

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])  # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])  # (Batch x T) x 1

        # Generate time index
        # time_index = np.array([list(range(i*len_traj,
        #                                     i*len_traj+self.size_segment)) for i in range(mb_size)])  # (batch_size, segment_len)
        time_index = np.arange(0, self.size_segment).reshape(1, -1).repeat(mb_size, axis=0) + \
            np.arange(0, mb_size).reshape(-1, 1).repeat(self.size_segment, axis=1) * len_traj
        time_index_2 = time_index + np.random.choice(len_traj - self.size_segment,
                                                     size=mb_size, replace=True).reshape(-1, 1)
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment,
                                                     size=mb_size, replace=True).reshape(-1, 1)

        # snippet_1_path = self.path + f'{self.count}_snippet_1.npy'
        # snippet_2_path = self.path + f'{self.count}_snippet_2.npy'
        # np.save(snippet_1_path, time_index_1)
        # np.save(snippet_2_path, time_index_2)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)  # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)  # Batch x size_seg x 1

        print(f'sa_t_1: {sa_t_1.shape}, sa_t_2: {sa_t_2.shape}, r_t_1: {r_t_1.shape}, r_t_2: {r_t_2.shape}')
        print(f'tr_1: {tr_1.shape}, tr_2: {tr_2.shape}')
        return sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2

    # skill space query selection
    def surf_data_aug_process(self):
        # get queries
        u_sa_t_1, u_sa_t_2, _, _, _, _ = self.get_queries(mb_size=self.mb_size * self.large_batch)
        self.put_unlabeled_queries(u_sa_t_1, u_sa_t_2)

    def get_queries_skills(self, mb_size=20):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
        z_2 = np.array(self.skill_list)[batch_index_2]  # Batch x dim of z
        r_t_2 = train_targets[batch_index_2]  # Batch x T x 1

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        if self.skip_use_traj:  # batch_index_1 and batch_index_2 must be different
            while np.any(batch_index_1 == batch_index_2):
                same_index = np.where(batch_index_1 == batch_index_2)[0]
                batch_index_1[same_index] = np.random.choice(max_len, size=len(same_index), replace=True)
        sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
        z_1 = np.array(self.skill_list)[batch_index_1]  # Batch x dim of z
        r_t_1 = train_targets[batch_index_1]  # Batch x T x 1

        ep_num_1 = np.array(self.episode_num_list)[batch_index_1].reshape(-1, 1)  # Batch x 1
        ep_num_2 = np.array(self.episode_num_list)[batch_index_2].reshape(-1, 1)

        # tr_1, tr_2 are the sum of reward of the whole trajectory
        tr_1 = np.sum(r_t_1, axis=1).reshape(-1, 1)
        tr_2 = np.sum(r_t_2, axis=1).reshape(-1, 1)

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])  # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])  # (Batch x T) x 1

        # Generate time index
        # time_index = np.array([list(range(i*len_traj,
        #                                     i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index = np.arange(0, self.size_segment).reshape(1, -1).repeat(mb_size, axis=0) + \
            np.arange(0, mb_size).reshape(-1, 1).repeat(self.size_segment, axis=1) * len_traj
        ep_start_1 = np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        ep_start_2 = np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        time_index_1 = time_index + ep_start_1
        time_index_2 = time_index + ep_start_2

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)  # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)  # Batch x size_seg x 1

        # print(f'sa_t_1: {sa_t_1.shape}, sa_t_2: {sa_t_2.shape}, r_t_1: {r_t_1.shape}, r_t_2: {r_t_2.shape}')
        # print(f'z_1: {z_1.shape}, z_2: {z_2.shape}, tr_1: {tr_1.shape}, tr_2: {tr_2.shape}')
        # (500, 50, 30), (500, 50, 1), (500, 10), (500, 1)
        return sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2, \
            ep_num_1, ep_num_2, ep_start_1, ep_start_2

    # 3
    def skill_diff_sampling(self):
        ''' sample different skills '''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * 5)
        # test shape
        print(f'sa_t_1: {sa_t_1.shape}, sa_t_2: {sa_t_2.shape}')
        print(f'z_1: {z_1.shape}, z_2: {z_2.shape}')
        print(f'r_t_1: {r_t_1.shape}, r_t_2: {r_t_2.shape}')

        # select queies with different skills
        mask = np.linalg.norm(z_1 - z_2, axis=1) > self.skill_diff_epsilon
        print(f'diff skill ration: {sum(mask)/len(mask)}')
        metrics = {'diff_skill_ratio': sum(mask) / len(mask)}

        sa_t_1, sa_t_2, r_t_1, r_t_2 = sa_t_1[mask], sa_t_2[mask], r_t_1[mask], r_t_2[mask]
        z_1, z_2, tr_1, tr_2 = z_1[mask], z_2[mask], tr_1[mask], tr_2[mask]
        sa_t_1, sa_t_2 = sa_t_1[:self.mb_size], sa_t_2[:self.mb_size]
        r_t_1, r_t_2 = r_t_1[:self.mb_size], r_t_2[:self.mb_size]
        z_1, z_2 = z_1[:self.mb_size], z_2[:self.mb_size]
        tr_1, tr_2 = tr_1[:self.mb_size], tr_2[:self.mb_size]

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    # 4
    def skill_diff_disagreement_sampling(self):
        ''' sample different skills, then use disagreement '''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch * 5)

        # select queies with different skills
        mask = np.linalg.norm(z_1 - z_2, axis=1) > self.skill_diff_epsilon
        print(f'diff skill ration: {sum(mask)/len(mask)}')
        metrics = {'diff_skill_ratio': sum(mask) / len(mask)}
        sa_t_1, sa_t_2, r_t_1, r_t_2 = sa_t_1[mask], sa_t_2[mask], r_t_1[mask], r_t_2[mask]
        z_1, z_2 = z_1[mask], z_2[mask]

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]
        for i in range(5):
            print(f'z1: {z_1[i, :3]}, z2: {z_2[i, :3]}')

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    def skill_diff_disagreement_sampling2(self):
        ''' sample different skills, then use disagreement '''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch * 5)

        # pick top disagreement queries
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:5 * self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]

        # select queies with different skills
        z_11, z_22 = z_1[self.mb_size // 2:], z_2[self.mb_size // 2:]
        top_k_index = (-np.linalg.norm(z_11 - z_22, axis=1)).argsort()[:self.mb_size - self.mb_size // 2]
        top_k_index = np.concatenate([np.arange(self.mb_size // 2), top_k_index + self.mb_size // 2])
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]
        for i in range(5):
            print(f'z1: {z_1[i, :3]}, z2: {z_2[i, :3]}')

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics2

    # R(z)
    def construct_skill_return_estimator(self):
        self.skill_return_estimator, self.skill_return_paramlst = [], []
        for i in range(self.de):
            model = nn.Sequential(
                *gen_net(in_size=self.dz, out_size=1, H=256, n_layers=3, activation=self.activation)
            ).float().to(self.device)
            self.skill_return_estimator.append(model)
            self.skill_return_paramlst.extend(model.parameters())
        self.skill_return_opt = torch.optim.Adam(self.skill_return_paramlst, lr=self.lr)
        self.skill_return_loss = nn.MSELoss()

    def get_skill_return(self, skills: torch.Tensor):
        Rz = []
        with torch.no_grad():
            for member in range(self.de):
                Rz.append(self.skill_return_estimator[member](skills))
        Rz = torch.cat(Rz, axis=-1)
        print(f'Rz: {Rz.shape}')
        return torch.mean(Rz, axis=-1), torch.std(Rz, axis=-1)

    def train_skill_return_estimator(self, replay_buffer: ReplayBuffer, num_epochs=50):
        max_len = replay_buffer.capacity if replay_buffer.full else replay_buffer.episode_idx
        skills = replay_buffer.metas_list[0][:max_len, 0, :]
        skills = torch.from_numpy(skills).float().to(self.device)
        return_labels = np.sum(replay_buffer.predicted_rewards[:max_len], axis=1)
        return_labels = torch.from_numpy(return_labels).float().to(self.device)
        print(f'skills: {skills.shape}, return_labels: {return_labels.shape}')
        print(f'return max: {return_labels.max()}, min: {return_labels.min()}, mean: {return_labels.mean()}')
        # return labels normalized to 0-1
        return_labels = (return_labels - return_labels.min()) / (return_labels.max() - return_labels.min())

        for epoch_i in range(num_epochs):
            loss = 0
            for member in range(self.de):
                predicted_returns = self.skill_return_estimator[member](skills)
                loss += self.skill_return_loss(predicted_returns, return_labels)
            self.skill_return_opt.zero_grad()
            loss.backward()
            self.skill_return_opt.step()
            if epoch_i % 10 == 0:
                print(f'epoch: {epoch_i}, loss: {loss.item()}')

    # 5
    def skill_return_sampling(self):
        ''' sample skills with maximum return diff '''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch)

        # calculate return diff
        z_return_1 = self.get_skill_return(
            torch.from_numpy(z_1).float().to(self.device))[0].detach().cpu().numpy()
        z_return_2 = self.get_skill_return(
            torch.from_numpy(z_2).float().to(self.device))[0].detach().cpu().numpy()
        return_diff = np.abs(z_return_1 - z_return_2).reshape(-1)
        # get max return diff
        top_k_index = (-return_diff).argsort()[:self.mb_size]
        print(
            f'top_k shape: {top_k_index.shape}, top k: {top_k_index[:5]}, kth return diff: {return_diff[top_k_index[-1]]}')
        # print(f'max return diff: {return_diff[top_k_index]}')
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]

        # quantile 0.25 0.5 0.75 1 of return diff
        quant_return_diff = np.quantile(return_diff, [0.25, 0.5, 0.75, 1])
        metrics = {
            'kth_return_diff': return_diff[top_k_index[-1]],
            'mean_return_diff': return_diff.mean(),
            'quant_return_diff_25': quant_return_diff[0], 'quant_return_diff_50': quant_return_diff[1],
            'quant_return_diff_75': quant_return_diff[2], 'quant_return_diff_100': quant_return_diff[3],
        }

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    # 6
    def skill_return_disagreement_sampling(self):
        ''' sample skills with high return diff, maximum disagreement '''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch * self.large_batch)

        # calculate return diff
        z_return_1 = self.get_skill_return(
            torch.from_numpy(z_1).float().to(self.device))[0].detach().cpu().numpy()
        z_return_2 = self.get_skill_return(
            torch.from_numpy(z_2).float().to(self.device))[0].detach().cpu().numpy()
        return_diff = np.abs(z_return_1 - z_return_2).reshape(-1)
        # get max return diff
        top_k_index = (-return_diff).argsort()[:self.mb_size * self.large_batch]
        print(
            f'top_k shape: {top_k_index.shape}, top k: {top_k_index[:5]}, kth return diff: {return_diff[top_k_index[-1]]}')
        metrics = {'kth_return_diff': return_diff[top_k_index[-1]]}
        # print(f'max return diff: {return_diff[top_k_index]}')
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]

        # disagreement
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]
        for i in range(5):
            print(f'return diff: {return_diff[top_k_index[i]]}')
            print(f'z1: {z_1[i, :3]}, z2: {z_2[i, :3]}')

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    def disagreement_skill_return_sampling(self):
        ''' sample skills with high disagreement, maximum return diff '''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch * 6)

        # disagreement
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size * 3]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]

        # calculate return diff
        z_return_1 = self.get_skill_return(
            torch.from_numpy(z_1).float().to(self.device))[0].detach().cpu().numpy()
        z_return_2 = self.get_skill_return(
            torch.from_numpy(z_2).float().to(self.device))[0].detach().cpu().numpy()
        return_diff = np.abs(z_return_1 - z_return_2).reshape(-1)
        # get max return diff
        # top_k_index = (-return_diff).argsort()[:self.mb_size]
        top_k_index = (return_diff).argsort()[:self.mb_size]
        print(
            f'top_k shape: {top_k_index.shape}, top k: {top_k_index[:5]}, kth return diff: {return_diff[top_k_index[-1]]}')
        metrics = {'kth_return_diff': return_diff[top_k_index[-1]]}
        # print(f'max return diff: {return_diff[top_k_index]}')
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]
        for i in range(5):
            print(f'return diff: {return_diff[top_k_index[i]]}')
            print(f'z1: {z_1[i, :3]}, z2: {z_2[i, :3]}')

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    # 7
    def skill_diff_disagreement_mul_sampling(self):
        ''' add diff R(z) and disagreement, sample top K'''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch)

        # calculate z diff
        z_diff = np.linalg.norm(z_1 - z_2, axis=1)
        z_diff_norm = (z_diff - z_diff.min()) / (z_diff.max() - z_diff.min())
        # calculate disagreement
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        disagree_norm = (disagree - disagree.min()) / (disagree.max() - disagree.min())
        # get top K
        top_k_index = (-(1 + z_diff_norm) * (2 + disagree_norm)).argsort()[:self.mb_size]
        print(f'top_k shape: {top_k_index.shape}, top k: {top_k_index[:5]}, kth z diff: {z_diff[top_k_index[-1]]}')
        # quantile 0.25 0.5 0.75 1 of return diff
        quant_z_diff = np.quantile(z_diff, [0.25, 0.5, 0.75, 1])
        quant_disagree = np.quantile(disagree, [0.25, 0.5, 0.75, 1])
        metrics = {
            'kth_z_diff': z_diff[top_k_index[-1]],
            'kth_disagree': disagree[top_k_index[-1]],
            'mean_return_diff': z_diff.mean(),
            'mean_disagree': disagree.mean(),
            # quantile 0.25 0.5 0.75 1 of return diff
            'quant_z_diff_25': quant_z_diff[0], 'quant_z_diff_50': quant_z_diff[1],
            'quant_z_diff_75': quant_z_diff[2], 'quant_z_diff_100': quant_z_diff[3],
            # quantile 0.25 0.5 0.75 1 of disagreement
            'quant_disagree_25': quant_disagree[0], 'quant_disagree_50': quant_disagree[1],
            'quant_disagree_75': quant_disagree[2], 'quant_disagree_100': quant_disagree[3],
        }
        # print(f'max return diff: {return_diff[top_k_index]}')
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    # 8
    def skill_return_disagreement_mul_sampling(self):
        ''' multiply diff R(z) and disagreement, sample top K'''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2, \
            ep_num_1, ep_num_2, ep_start_1, ep_start_2 = self.get_queries_skills(
                mb_size=self.mb_size * self.large_batch)

        # calculate return diff
        z_return_1 = self.get_skill_return(
            torch.from_numpy(z_1).float().to(self.device))[0].detach().cpu().numpy()
        z_return_2 = self.get_skill_return(
            torch.from_numpy(z_2).float().to(self.device))[0].detach().cpu().numpy()
        return_diff = np.abs(z_return_1 - z_return_2).reshape(-1)
        return_diff_norm = (return_diff - return_diff.min()) / (return_diff.max() - return_diff.min())
        # calculate disagreement
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        disagree_norm = (disagree - disagree.min()) / (disagree.max() - disagree.min())
        # get top K
        top_k_index = (-(1 + return_diff_norm) * (self.disagree_base + disagree_norm)).argsort()[:self.mb_size]
        print(
            f'top_k shape: {top_k_index.shape}, top k: {top_k_index[:5]}, kth return diff: {return_diff[top_k_index[-1]]}')
        # quantile 0.25 0.5 0.75 1 of return diff
        quant_return_diff = np.quantile(return_diff, [0.25, 0.5, 0.75, 1])
        quant_disagree = np.quantile(disagree, [0.25, 0.5, 0.75, 1])
        metrics = {
            'kth_return_diff': return_diff[top_k_index[-1]],
            'kth_disagree': disagree[top_k_index[-1]],
            'mean_return_diff': return_diff.mean(),
            'mean_disagree': disagree.mean(),
            # quantile 0.25 0.5 0.75 1 of return diff
            'quant_return_diff_25': quant_return_diff[0], 'quant_return_diff_50': quant_return_diff[1],
            'quant_return_diff_75': quant_return_diff[2], 'quant_return_diff_100': quant_return_diff[3],
            # quantile 0.25 0.5 0.75 1 of disagreement
            'quant_disagree_25': quant_disagree[0], 'quant_disagree_50': quant_disagree[1],
            'quant_disagree_75': quant_disagree[2], 'quant_disagree_100': quant_disagree[3],
        }
        # print(f'max return diff: {return_diff[top_k_index]}')
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]
        ep_num_1, ep_num_2 = ep_num_1[top_k_index], ep_num_2[top_k_index]
        ep_start_1, ep_start_2 = ep_start_1[top_k_index], ep_start_2[top_k_index]

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2,
                ep_num_1, ep_num_2, ep_start_1, ep_start_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    # 9
    def skill_return_Rdisagreement_sampling(self):
        ''' multiply diff R(z) and R(z) disagreement, sample top K'''
        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries_skills(
            mb_size=self.mb_size * self.large_batch)

        # calculate return diff
        z_return_1, Rz_disagree_1 = self.get_skill_return(
            torch.from_numpy(z_1).float().to(self.device))
        z_return_2, Rz_disagree_2 = self.get_skill_return(
            torch.from_numpy(z_2).float().to(self.device))
        z_return_1, z_return_2 = z_return_1.detach().cpu().numpy(), z_return_2.detach().cpu().numpy()
        return_diff = np.abs(z_return_1 - z_return_2).reshape(-1)
        # R(z) disagreement
        Rz_disagree = (Rz_disagree_1 + Rz_disagree_2).detach().cpu().numpy()
        Rz_disagree_norm = (Rz_disagree - Rz_disagree.min()) / (Rz_disagree.max() - Rz_disagree.min())
        # calculate disagreement
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        disagree_norm = (disagree - disagree.min()) / (disagree.max() - disagree.min())

        # get top K
        top_k_index = (-return_diff * (1 + Rz_disagree_norm) * (1 + disagree_norm)).argsort()[:self.mb_size]
        print(
            f'top_k shape: {top_k_index.shape}, top k: {top_k_index[:5]}, kth return diff: {return_diff[top_k_index[-1]]}')
        # quantile 0.25 0.5 0.75 1 of return diff
        quant_return_diff = np.quantile(return_diff, [0.25, 0.5, 0.75, 1])
        quant_Rz_disagree = np.quantile(Rz_disagree, [0.25, 0.5, 0.75, 1])
        metrics = {
            'kth_return_diff': return_diff[top_k_index[-1]],
            'kth_Rz_disagree': Rz_disagree[top_k_index[-1]],
            'mean_return_diff': return_diff.mean(),
            'mean_disagree': Rz_disagree.mean(),
            # quantile 0.25 0.5 0.75 1 of return diff
            'quant_return_diff_25': quant_return_diff[0], 'quant_return_diff_50': quant_return_diff[1],
            'quant_return_diff_75': quant_return_diff[2], 'quant_return_diff_100': quant_return_diff[3],
            # quantile 0.25 0.5 0.75 1 of Rz_disagree
            'quant_Rz_disagree_25': quant_Rz_disagree[0], 'quant_Rz_disagree_50': quant_Rz_disagree[1],
            'quant_Rz_disagree_75': quant_Rz_disagree[2], 'quant_Rz_disagree_100': quant_Rz_disagree[3],
        }
        # print(f'max return diff: {return_diff[top_k_index]}')
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)
            metrics.update(metrics2)
        else:
            labels = []

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:  # unlabeled part doesn't use skill_sampling (?)
            self.surf_data_aug_process()

        return len(labels), metrics

    # original reward model
    def get_queries_part(self, mb_size=20, part=10):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), part
        img_t_1, img_t_2 = None, None

        # get train traj
        if len(self.inputs[-1]) < len_traj:  # if the last traj is not full
            train_inputs = np.array(self.inputs[-part - 1:-1])
            train_targets = np.array(self.targets[-part - 1:-1])
        else:
            train_inputs = np.array(self.inputs[-part:])
            train_targets = np.array(self.targets[-part:])

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2]  # Batch x T x 1

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1]  # Batch x T x 1

        # sa_t_1_path = self.path + f'{self.count}_sa_t_1.npy'
        # r_t_1_path = self.path + f'{self.count}_r_t_1.npy'
        # sa_t_2_path = self.path + f'{self.count}_sa_t_2.npy'
        # r_t_2_path = self.path + f'{self.count}_r_t_2.npy'
        # batch_1_path = self.path + f'{self.count}_batch_1.npy'
        # batch_2_path = self.path + f'{self.count}_batch_2.npy'
        # np.save(sa_t_1_path, sa_t_1)
        # np.save(r_t_1_path, r_t_1)
        # np.save(sa_t_2_path, sa_t_2)
        # np.save(r_t_2_path, r_t_2)
        # np.save(batch_1_path, batch_index_1)
        # np.save(batch_2_path, batch_index_2)

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])  # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])  # (Batch x T) x 1

        # Generate time index
        time_index = np.array([list(range(i * len_traj, i * len_traj + self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj - self.size_segment,
                                                     size=mb_size, replace=True).reshape(-1, 1)
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment,
                                                     size=mb_size, replace=True).reshape(-1, 1)

        # snippet_1_path = self.path + f'{self.count}_snippet_1.npy'
        # snippet_2_path = self.path + f'{self.count}_snippet_2.npy'
        # np.save(snippet_1_path, time_index_1)
        # np.save(snippet_2_path, time_index_2)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)  # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)  # Batch x size_seg x 1

        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def get_contrast_queries(self, mb_size=20):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        if max_len <= 10:
            batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
            sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
            r_t_2 = train_targets[batch_index_2]  # Batch x T x 1
            batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
            sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
            r_t_1 = train_targets[batch_index_1]  # Batch x T x 1
        else:
            batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
            sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
            r_t_2 = train_targets[batch_index_2]  # Batch x T x 1
            batch_index_1 = max_len - 10 + np.random.choice(10, size=mb_size, replace=True)
            sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
            r_t_1 = train_targets[batch_index_1]  # Batch x T x 1

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])  # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])  # (Batch x T) x 1

        # Generate time index
        time_index = np.array([list(range(i * len_traj,
                                          i * len_traj + self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj - self.size_segment,
                                                     size=mb_size, replace=True).reshape(-1, 1)
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment,
                                                     size=mb_size, replace=True).reshape(-1, 1)
        # time_index_2 = time_index + int((len_traj/3)*2) + np.random.choice(len_traj-int((len_traj/3)*2)-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        # time_index_1 = time_index + int((len_traj/3)*2) + np.random.choice(len_traj-int((len_traj/3)*2)-self.size_segment, size=mb_size, replace=True).reshape(-1,1)

        # snippet_1_path = self.path + f'{self.count}_snippet_1.npy'
        # snippet_2_path = self.path + f'{self.count}_snippet_2.npy'
        # np.save(snippet_1_path, time_index_1)
        # np.save(snippet_2_path, time_index_2)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)  # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)  # Batch x size_seg x 1

        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            if self.buffer_seg1.dtype == 'O':
                for i in range(sa_t_1.shape[0]):
                    self.buffer_seg1[self.buffer_index + i] = sa_t_1[i]
                for i in range(sa_t_2.shape[0]):
                    self.buffer_seg2[self.buffer_index + i] = sa_t_2[i]
            else:
                np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
                np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def put_unlabel_queries(self, sa_t_1, sa_t_2, labels):
        ''' hx version. the surf version is put_unlabeled_queries(self, sa_t_1, sa_t_2) '''
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            self.buffer_mask[self.buffer_index:self.capacity] = 0

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                self.buffer_mask[0:remain] = 0

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_mask[self.buffer_index:next_index] = 0
            self.buffer_index = next_index

    def put_unlabeled_queries(self, sa_t_1, sa_t_2):
        # r_t_1 = self.r_hat_batch(sa_t_1)
        # r_t_2 = self.r_hat_batch(sa_t_2)
        # sum_r_t_1 = np.sum(r_t_1[:, self.data_aug_window: -self.data_aug_window], axis=1)
        # sum_r_t_2 = np.sum(r_t_2[:, self.data_aug_window: -self.data_aug_window], axis=1)
        # labels = 1*(sum_r_t_1 < sum_r_t_2)
        labels = np.zeros((sa_t_1.shape[0], 1))  # (batch_size, 1)

        total_sample = sa_t_1.shape[0]
        next_index = self.u_buffer_index + total_sample
        if next_index >= self.capacity:
            self.u_buffer_full = True
            maximum_index = self.capacity - self.u_buffer_index
            np.copyto(self.u_buffer_seg1[self.u_buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.u_buffer_seg2[self.u_buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.u_buffer_label[self.u_buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.u_buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.u_buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.u_buffer_label[0:remain], labels[maximum_index:])

            self.u_buffer_index = remain
        else:
            np.copyto(self.u_buffer_seg1[self.u_buffer_index:next_index], sa_t_1)
            np.copyto(self.u_buffer_seg2[self.u_buffer_index:next_index], sa_t_2)
            np.copyto(self.u_buffer_label[self.u_buffer_index:next_index], labels)
            self.u_buffer_index = next_index

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1=None, tr_2=None,
                  ep_num_1=None, ep_num_2=None, ep_start_1=None, ep_start_2=None):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        metrics = {}

        # skip the query
        if (not self.skip_use_error) and self.teacher_thres_skip > 0:
            if self.skip_use_div:  # the reserved index
                max_index = np.bitwise_or((sum_r_t_1 / (sum_r_t_2 + 1e-7)) > self.teacher_thres_skip,
                                          (sum_r_t_2 / (sum_r_t_1 + 1e-7)) > self.teacher_thres_skip).reshape(-1)
            elif self.skip_use_traj:
                if tr_1 is None or tr_2 is None:
                    raise ValueError(f'tr_1 and tr_2 should not be None, tr_1: {tr_1}, tr_2: {tr_2}')
                max_index = (np.abs(tr_1 - tr_2) > self.teacher_thres_skip).reshape(-1)
            else:
                max_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_thres_skip).reshape(-1)
            print(f'skip ratio: {sum(max_index)/len(max_index)}')
            metrics = {'skip_ratio': sum(max_index) / len(max_index)}
            if sum(max_index) == 0:
                return None, None, None, None, [], {}

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels

        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1

        # skip -> mistake
        if self.skip_use_error and self.teacher_thres_skip > 0:
            if self.skip_use_div:
                max_index = np.bitwise_and((sum_r_t_1 / (sum_r_t_2 + 1e-7)) < self.teacher_thres_skip,
                                           (sum_r_t_2 / (sum_r_t_1 + 1e-7)) < self.teacher_thres_skip).reshape(-1)
            elif self.skip_use_traj:
                if tr_1 is None or tr_2 is None:
                    raise ValueError(f'tr_1 and tr_2 should not be None, tr_1: {tr_1}, tr_2: {tr_2}')
                max_index = (np.abs(tr_1 - tr_2) < self.teacher_thres_skip).reshape(-1)  # error index
            else:
                max_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_skip).reshape(-1)
            print(f'mistake ratio: {1 - sum(max_index)/len(max_index)}')
            metrics = {'mistake_ratio': 1 - sum(max_index) / len(max_index)}
            index_value = np.where(max_index)[0]
            # if sum(max_index) == 0:
            #     return None, None, None, None, [], {}

            mistake_index = np.random.permutation(index_value)[: len(index_value) // 2]
            # to bool
            mistake_index = np.isin(np.arange(len(max_index)), mistake_index)
            labels[mistake_index] = 1 - labels[mistake_index]

            for i in range(10):  # range(len(sum_r_t_1))
                self.df.loc[len(self.df)] = [
                    self.df_count * 20000,  # step
                    ep_num_1[i][0], ep_start_1[i][0], sum_r_t_1[i][0],  # en0, start0, return0
                    ep_num_2[i][0], ep_start_2[i][0], sum_r_t_2[i][0],  # en1, start1, return1
                    (not max_index[i]),  # can_comp
                ]
            self.df_count += 1
            self.df.to_csv(f'./query.csv', index=False)
            print(f'csv of query returns saved!')

        if self.path:
            sa_t_1_path = self.path + f'{self.count}_sa_t_1.npy'
            r_t_1_path = self.path + f'{self.count}_r_t_1.npy'
            sa_t_2_path = self.path + f'{self.count}_sa_t_2.npy'
            r_t_2_path = self.path + f'{self.count}_r_t_2.npy'
            np.save(sa_t_1_path, sa_t_1)
            np.save(r_t_1_path, r_t_1)
            np.save(sa_t_2_path, sa_t_2)
            np.save(r_t_2_path, r_t_2)

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics

    def kcenter_sampling(self):

        # get queries
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size, self.device)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def kcenter_disagree_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size, self.device)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def kcenter_entropy_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size, self.device)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def uniform_contrast_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_contrast_queries(
            mb_size=self.mb_size)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, None, None)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def uniform_sampling(self, explore=None):
        # get queries
        if True:  # not explore:
            sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2, \
                ep_num_1, ep_num_2, ep_start_1, ep_start_2 = self.get_queries_skills(
                    mb_size=self.mb_size * self.large_batch)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2 = self.get_queries(
                mb_size=int(self.mb_size * explore))
            sa_t_1_, sa_t_2_, r_t_1_, r_t_2_ = self.get_queries_part(
                mb_size=int(self.mb_size * (1 - explore)))
            sa_t_1 = np.concatenate([sa_t_1, sa_t_1_], axis=0)
            sa_t_2 = np.concatenate([sa_t_2, sa_t_2_], axis=0)
            r_t_1 = np.concatenate([r_t_1, r_t_1_], axis=0)
            r_t_2 = np.concatenate([r_t_2, r_t_2_], axis=0)

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2,
                ep_num_1, ep_num_2, ep_start_1, ep_start_2)
        else:
            labels, metrics2 = [], {}

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:
            self.surf_data_aug_process()

        return len(labels), metrics2

    def disagreement_contrast_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_contrast_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, None, None)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def disagreement_sampling(self):

        # get queries
        sa_t_1, sa_t_2, z_1, z_2, r_t_1, r_t_2, tr_1, tr_2, \
            ep_num_1, ep_num_2, ep_start_1, ep_start_2 = self.get_queries_skills(
                mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        z_1, z_2 = z_1[top_k_index], z_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]
        ep_num_1, ep_num_2 = ep_num_1[top_k_index], ep_num_2[top_k_index]
        ep_start_1, ep_start_2 = ep_start_1[top_k_index], ep_start_2[top_k_index]
        for i in range(5):
            print(f'z_1: {z_1[i, :3]}, z_2: {z_2[i, :3]}')

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, tr_1, tr_2,
                ep_num_1, ep_num_2, ep_start_1, ep_start_2)
        else:
            labels, metrics2 = [], {}

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels), metrics2

    def ucb_sampling(self, ucb_lamda):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, _, _ = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        r_1 = self.r_hat_batch(sa_t_1)
        r_2 = self.r_hat_batch(sa_t_2)
        r_1 = np.sum(r_1, axis=1).flatten()
        r_2 = np.sum(r_2, axis=1).flatten()
        r = (r_1 + r_2) / 2
        ucb = r + ucb_lamda * disagree
        top_k_index = (-ucb).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, None, None)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def adv_sampling(self, disc):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, _, _ = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:int(self.mb_size * self.adv_mu)]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # pick the most unconfident using discriminator
        p = self.get_rank_discriminator(sa_t_1, sa_t_2, disc)
        worst_k_index = p.flatten().argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[worst_k_index], sa_t_1[worst_k_index]
        r_t_2, sa_t_2 = r_t_2[worst_k_index], sa_t_2[worst_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, None, None)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def unlabel_sampling(self):

        # get queries
        sa_t_1, sa_t_2, _, _, _, _ = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:int(self.mb_size * self.mu)]
        sa_t_1 = sa_t_1[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]

        r_t_1 = self.r_hat_batch(sa_t_1)
        r_t_2 = self.r_hat_batch(sa_t_2)

        # get labels
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels

        if len(labels) > 0:
            self.put_unlabel_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def entropy_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, _, _ = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)

        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        tr_1, tr_2 = tr_1[top_k_index], tr_2[top_k_index]

        # get labels
        if len(r_t_1) > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2, labels, metrics2 = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, None, None)
        else:
            labels, metrics2 = [], {}

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels), metrics2

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc, np.mean(ensemble_losses)

    def semi_train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        u_max_len = self.capacity if self.u_buffer_full else self.u_buffer_index
        u_total_batch_index = self.shuffle_dataset(u_max_len)

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        mu = u_max_len / max_len
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            u_last_index = int((epoch + 1) * self.train_batch_size * mu)
            if u_last_index > u_max_len:
                u_last_index = u_max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                labels = labels.repeat(self.data_aug_ratio)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)  # (batch size, segment len, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                # shifting & cropping time
                mask_1, mask_2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                r_hat1 = r_hat1.repeat(self.data_aug_ratio, 1, 1)  # (batch size * data_aug_ratio, segment len, 1)
                r_hat2 = r_hat2.repeat(self.data_aug_ratio, 1, 1)
                r_hat1 = (mask_1 * r_hat1).sum(axis=1)
                r_hat2 = (mask_2 * r_hat2).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)

                # get random unlabeled batch
                u_idxs = u_total_batch_index[member][int(epoch * self.train_batch_size * mu): u_last_index]
                u_sa_t_1 = self.u_buffer_seg1[u_idxs]
                u_sa_t_2 = self.u_buffer_seg2[u_idxs]

                # get logits
                u_r_hat1 = self.r_hat_member(u_sa_t_1, member=member)
                u_r_hat2 = self.r_hat_member(u_sa_t_2, member=member)

                # pseudo-labeling
                u_r_hat1_noaug = u_r_hat1[:, self.data_aug_window:-self.data_aug_window]
                u_r_hat2_noaug = u_r_hat2[:, self.data_aug_window:-self.data_aug_window]
                with torch.no_grad():
                    u_r_hat1_noaug = u_r_hat1_noaug.sum(axis=1)
                    u_r_hat2_noaug = u_r_hat2_noaug.sum(axis=1)
                    u_r_hat_noaug = torch.cat([u_r_hat1_noaug, u_r_hat2_noaug], axis=-1)

                    pred = torch.softmax(u_r_hat_noaug, dim=1)
                    pred_max = pred.max(1)
                    mask = (pred_max[0] >= self.threshold_u)
                    pseudo_labels = pred_max[1].detach()
                pseudo_labels = pseudo_labels.repeat(self.data_aug_ratio)
                mask = mask.repeat(self.data_aug_ratio)

                # shifting & cropping time
                u_mask_1, u_mask_2 = self.get_cropping_mask(u_r_hat1, self.data_aug_ratio)
                u_r_hat1 = u_r_hat1.repeat(self.data_aug_ratio, 1, 1)
                u_r_hat2 = u_r_hat2.repeat(self.data_aug_ratio, 1, 1)
                u_r_hat1 = (u_mask_1 * u_r_hat1).sum(axis=1)
                u_r_hat2 = (u_mask_2 * u_r_hat2).sum(axis=1)
                u_r_hat = torch.cat([u_r_hat1, u_r_hat2], axis=-1)

                curr_loss += torch.mean(self.UCELoss(u_r_hat, pseudo_labels) * mask) * self.lambda_u

                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)  # supervised acc
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc, np.mean(ensemble_losses)

    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index

    def get_cropping_mask(self, r_hat1, w):
        mask_1_, mask_2_ = [], []
        B, S, _ = r_hat1.shape  # batch size, segment length
        mask_1 = torch.zeros((B * w, S, 1)).to(self.device)
        mask_2 = torch.zeros((B * w, S, 1)).to(self.device)

        length = np.random.randint(S - 15, S - 5 + 1, size=(B * w, ))  # 60 -> [45, 55]
        start_index_1 = np.random.randint(0, S + 1 - length)  # (B*w, )
        start_index_2 = np.random.randint(0, S + 1 - length)
        end_index_1 = np.array(start_index_1 + length, dtype=int)
        end_index_2 = np.array(start_index_2 + length, dtype=int)
        # print(f'start_index_1: {start_index_1[5:]}, end_index_1: {end_index_1[5:]}')
        indices = np.arange(S).reshape(1, -1)  # (1, segment_length)
        mask_1[(indices >= start_index_1[:, None]) & (indices <= end_index_1[:, None])] = 1
        mask_2[(indices >= start_index_2[:, None]) & (indices <= end_index_2[:, None])] = 1

        return mask_1, mask_2  # (batch_size * w, segment_length, 1)

    def train_reward_revised(self, num_iters):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)

        total = 0

        start_index = 0
        for epoch in range(num_iters):
            self.opt.zero_grad()
            loss = 0.0

            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                if self.data_aug_ratio:
                    labels = labels.repeat(self.data_aug_ratio)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                if self.data_aug_ratio:
                    mask_1, mask_2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                    r_hat1 = r_hat1.repeat(self.data_aug_ratio, 1, 1)
                    r_hat2 = r_hat2.repeat(self.data_aug_ratio, 1, 1)
                    r_hat1 = (mask_1 * r_hat1).sum(axis=1)
                    r_hat2 = (mask_2 * r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break

        ensemble_acc = ensemble_acc / total

        return ensemble_acc, np.mean(ensemble_losses)

    def reshape_input(self, sa_t_1):
        x = []
        for i in range(sa_t_1.shape[0]):
            x.append(sa_t_1[i])
        return np.concatenate(x)

    def compute_r(self, r_hat1, t_len):
        x = []
        index = 0
        for i in range(len(t_len)):
            x.append(r_hat1[index:index + t_len[i]].sum().reshape(1))
            index += t_len[i]
        return torch.cat(x).reshape(-1, 1)

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def train_scl_reward(self, disc):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        num_label_ = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        loss_d_ = []
        ps = []
        masks_ = []

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_r = 0.0
            loss_d = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                masks = self.buffer_mask[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                loss_A = self.CEloss_(r_hat, labels)

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                true_index = np.where(masks == 1)[0]
                correct = (predicted[true_index] == labels[true_index]).sum().item()
                ensemble_acc[member] += correct

                snip1 = sa_t_1.reshape(sa_t_1.shape[0], -1)
                snip1 = torch.from_numpy(snip1).float().to(self.device)
                snip2 = sa_t_2.reshape(sa_t_2.shape[0], -1)
                snip2 = torch.from_numpy(snip2).float().to(self.device)
                p = disc(snip1, snip2, r_hat, loss_A.clone().detach())
                soft_mask = masks
                soft_mask[soft_mask > 0.9] = 0.9
                soft_mask[soft_mask < 0.1] = 0.1
                soft_mask = torch.from_numpy(soft_mask).float().to(self.device)
                loss_B = torch.nn.BCELoss(reduction='mean')(p, soft_mask.to(self.device))

                p1 = p.clone().detach().squeeze()
                ps.append(p1.cpu().numpy())
                masks_.append(masks.squeeze())

                sample_weights = torch.zeros_like(p1, dtype=torch.float, device=self.device)
                num_label = (masks > 0.5).sum()
                num_label_[member] += num_label
                num_unlabel = (masks < 0.5).sum()
                sample_weights[masks.squeeze() > 0.5] = (1 + self.weight_factor / p1[masks.squeeze() > 0.5]) / num_label
                sample_weights[masks.squeeze() < 0.5] = (1 - self.weight_factor * 1 /
                                                         (1 - p1[masks.squeeze() < 0.5])) / num_unlabel
                loss_C = (sample_weights * loss_A).sum() / sample_weights.sum()
                ensemble_losses[member].append(loss_C.item())

                loss_r += loss_C
                loss_d += loss_B

            loss_r.backward()
            self.opt.step()

            disc.disc_optimizer.zero_grad()
            loss_d.backward()
            disc.disc_optimizer.step()

            loss_d_.append(loss_d.item())

        ensemble_acc = ensemble_acc / num_label_

        ps = np.concatenate(ps)
        masks_ = np.concatenate(masks_)
        label_ps = ps[masks_ > 0.5]
        unlabel_ps = ps[masks_ < 0.5]

        return ensemble_acc, np.mean(ensemble_losses), np.mean(loss_d_), np.mean(label_ps), np.mean(unlabel_ps)

    def train_adv_reward(self, disc, num_iters):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        num_label_ = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        loss_d_ = []
        ps = []
        masks_ = []
        start_index = 0

        for epoch in range(num_iters):
            self.opt.zero_grad()
            loss_r = 0.0
            loss_d = 0.0

            last_index = start_index + self.train_batch_size
            # last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                masks = self.buffer_mask[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                true_index = np.where(masks == 1)[0]

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                loss_A = self.CEloss(r_hat[true_index], labels[true_index])
                ensemble_losses[member].append(loss_A.item())
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted[true_index] == labels[true_index]).sum().item()
                ensemble_acc[member] += correct

                snip1 = sa_t_1.reshape(sa_t_1.shape[0], -1)
                snip1 = torch.from_numpy(snip1).float().to(self.device)
                snip2 = sa_t_2.reshape(sa_t_2.shape[0], -1)
                snip2 = torch.from_numpy(snip2).float().to(self.device)
                p = disc(snip1, snip2, r_hat)
                soft_mask = masks
                soft_mask[soft_mask > 0.9] = 1.0
                soft_mask[soft_mask < 0.1] = 0.
                soft_mask = torch.from_numpy(soft_mask).float().to(self.device)
                loss_B = torch.nn.BCELoss(reduction='mean')(p, soft_mask.to(self.device))

                p1 = p.clone().detach().squeeze()
                ps.append(p1.cpu().numpy())
                masks_.append(masks.squeeze())
                num_label = (masks > 0.5).sum()
                num_label_[member] += num_label

                loss_r += loss_A
                loss_d += loss_B

            loss_r.backward()
            self.opt.step()

            if epoch % int(num_iters / 10) == 0:
                disc.disc_optimizer.zero_grad()
                loss_d.backward()
                disc.disc_optimizer.step()

            loss_d_.append(loss_d.item())

            start_index += self.train_batch_size
            if last_index == max_len:
                start_index = 0
                total_batch_index = []
                for _ in range(self.de):
                    total_batch_index.append(np.random.permutation(max_len))

        ensemble_acc = ensemble_acc / num_label_

        ps = np.concatenate(ps)
        masks_ = np.concatenate(masks_)
        label_ps = ps[masks_ > 0.5]
        unlabel_ps = ps[masks_ < 0.5]

        return ensemble_acc, np.mean(ensemble_losses), np.mean(loss_d_), np.mean(label_ps), np.mean(unlabel_ps)

    def train_reward_sp(self, num_iters):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        num_label_ = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        total = 0

        start_index = 0

        for epoch in range(num_iters):
            self.opt.zero_grad()
            loss_r = 0.0

            last_index = start_index + self.train_batch_size
            # last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                masks = self.buffer_mask[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                true_index = np.where(masks == 1)[0]

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                loss_A = self.CEloss(r_hat[true_index], labels[true_index])
                ensemble_losses[member].append(loss_A.item())
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted[true_index] == labels[true_index]).sum().item()
                ensemble_acc[member] += correct

                num_label = (masks > 0.5).sum()
                num_label_[member] += num_label

                loss_r += loss_A

            loss_r.backward()
            self.opt.step()

            start_index += self.train_batch_size
            if last_index == max_len:
                start_index = 0
                total_batch_index = []
                for _ in range(self.de):
                    total_batch_index.append(np.random.permutation(max_len))

        ensemble_acc = ensemble_acc / num_label_

        return ensemble_acc, np.mean(ensemble_losses)

    def train_disc(self, disc, num_iters):

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)

        loss_d_ = []
        ps = []
        masks_ = []
        start_index = 0

        for epoch in range(num_iters):
            loss_d = 0.0

            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            # get random batch
            idxs = total_batch_index[start_index:last_index]
            sa_t_1 = self.buffer_seg1[idxs]
            sa_t_2 = self.buffer_seg2[idxs]
            labels = self.buffer_label[idxs]
            masks = self.buffer_mask[idxs]
            labels = torch.from_numpy(labels).long().to(self.device)

            snip1 = sa_t_1.reshape(sa_t_1.shape[0], -1)
            snip1 = torch.from_numpy(snip1).float().to(self.device)
            snip2 = sa_t_2.reshape(sa_t_2.shape[0], -1)
            snip2 = torch.from_numpy(snip2).float().to(self.device)
            p = disc(snip1, snip2, labels)
            soft_mask = masks
            soft_mask[soft_mask > 0.9] = 1.0
            soft_mask[soft_mask < 0.1] = 0.
            soft_mask = torch.from_numpy(soft_mask).float().to(self.device)
            loss_d = torch.nn.BCELoss(reduction='mean')(p, soft_mask.to(self.device))

            p1 = p.clone().detach().squeeze()
            ps.append(p1.cpu().numpy())
            masks_.append(masks.squeeze())

            disc.disc_optimizer.zero_grad()
            loss_d.backward()
            disc.disc_optimizer.step()

            loss_d_.append(loss_d.item())

            start_index += self.train_batch_size
            if last_index == max_len:
                start_index = 0
                total_batch_index = np.random.permutation(max_len)

        ps = np.concatenate(ps)
        masks_ = np.concatenate(masks_)
        label_ps = ps[masks_ > 0.5]
        unlabel_ps = ps[masks_ < 0.5]

        return np.mean(loss_d_), np.mean(label_ps), np.mean(unlabel_ps)

    def pretrain(self, disc):
        reward_pretrain_epoches = 200
        disc_pretrain_epoches = 150

        max_len = self.capacity if self.buffer_full else self.buffer_index

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoches in range(reward_pretrain_epoches):
            # reward train
            ensemble_losses = [[] for _ in range(self.de)]
            ensemble_acc = np.array([0 for _ in range(self.de)])
            num_label_ = np.array([0 for _ in range(self.de)])

            total_batch_index = []
            for _ in range(self.de):
                total_batch_index.append(np.random.permutation(max_len))

            for epoch in range(num_epochs):
                self.opt.zero_grad()
                loss_r = 0.0

                last_index = (epoch + 1) * self.train_batch_size
                if last_index > max_len:
                    last_index = max_len

                for member in range(self.de):

                    # get random batch
                    idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                    sa_t_1 = self.buffer_seg1[idxs]
                    sa_t_2 = self.buffer_seg2[idxs]
                    labels = self.buffer_label[idxs]
                    masks = self.buffer_mask[idxs]
                    labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                    true_index = np.where(masks == 1)[0]

                    if member == 0:
                        total += labels.size(0)

                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                    r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                    # compute loss
                    loss_A = self.CEloss(r_hat[true_index], labels[true_index])
                    ensemble_losses[member].append(loss_A.item())

                    # compute acc
                    _, predicted = torch.max(r_hat.data, 1)
                    correct = (predicted[true_index] == labels[true_index]).sum().item()
                    ensemble_acc[member] += correct

                    num_label = (masks > 0.5).sum()
                    num_label_[member] += num_label

                    loss_r += loss_A

                loss_r.backward()
                self.opt.step()

            ensemble_acc = ensemble_acc / num_label_
            if np.mean(ensemble_acc) > 0.97:
                break

        for epoches in range(disc_pretrain_epoches):
            # discriminator train
            loss_d_ = []
            ps = []
            masks_ = []

            total_batch_index = np.random.permutation(max_len)

            for epoch in range(num_epochs):
                disc.disc_optimizer.zero_grad()
                loss_d = 0.0

                last_index = (epoch + 1) * self.train_batch_size
                if last_index > max_len:
                    last_index = max_len

                # get random batch
                idxs = total_batch_index[epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                masks = self.buffer_mask[idxs]
                true_index = np.where(masks == 1)[0]

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                snip1 = sa_t_1.reshape(sa_t_1.shape[0], -1)
                snip1 = torch.from_numpy(snip1).float().to(self.device)
                snip2 = sa_t_2.reshape(sa_t_2.shape[0], -1)
                snip2 = torch.from_numpy(snip2).float().to(self.device)
                p = disc(snip1, snip2, r_hat)
                soft_mask = masks
                soft_mask[soft_mask > 0.9] = 1.0
                soft_mask[soft_mask < 0.1] = 0.
                soft_mask = torch.from_numpy(soft_mask).float().to(self.device)
                loss_B = torch.nn.BCELoss(reduction='mean')(p, soft_mask.to(self.device))

                p1 = p.clone().detach().squeeze()
                ps.append(p1.cpu().numpy())
                masks_.append(masks.squeeze())

                loss_d += loss_B

                loss_d.backward()
                disc.disc_optimizer.step()

                loss_d_.append(loss_d.item())

            ps = np.concatenate(ps)
            masks_ = np.concatenate(masks_)
            label_ps = ps[masks_ > 0.5]
            unlabel_ps = ps[masks_ < 0.5]
            if np.mean(label_ps) > 0.9 and np.mean(unlabel_ps) < 0.1:
                break

        return np.mean(ensemble_acc), np.mean(ensemble_losses), np.mean(loss_d_), np.mean(label_ps), np.mean(unlabel_ps)

    def relabel_unlabel(self):
        unlabel_index = np.where(self.buffer_mask[:self.buffer_index] == 0)[0]
        sa_t_1 = self.buffer_seg1[unlabel_index]
        sa_t_2 = self.buffer_seg2[unlabel_index]
        r_1 = self.r_hat_batch(sa_t_1)
        r_2 = self.r_hat_batch(sa_t_2)
        r_1 = np.sum(r_1, axis=1)
        r_2 = np.sum(r_2, axis=1)

        labels = 1 * (r_1 < r_2)

        self.buffer_label[unlabel_index] = labels

    def get_s_a_l(self, index=1):
        # label_index = np.where(self.buffer_mask[:self.buffer_index] == 1)[0]
        # sa_t_1 = self.buffer_seg1[label_index]
        # sa_t_2 = self.buffer_seg2[label_index]
        # sa_t_1 = self.buffer_seg1[label_index]
        # sa_t_2 = self.buffer_seg2[label_index]
        sa_t_1 = self.buffer_seg1.reshape(-1, sa_t_1.shape[-1])
        sa_t_2 = self.buffer_seg2.reshape(-1, sa_t_2.shape[-1])
        obs_l_1, action_l_1 = np.hsplit(sa_t_1, [index])
        obs_l_2, action_l_2 = np.hsplit(sa_t_2, [index])
        self.obs_l = np.concatenate([obs_l_1, obs_l_2], axis=0)
        self.action_l = np.concatenate([action_l_1, action_l_2], axis=0)

    def sample(self, batch_size):
        sa_t_1 = self.buffer_seg1.reshape(-1, sa_t_1.shape[-1])
        sa_t_2 = self.buffer_seg2.reshape(-1, sa_t_2.shape[-1])
        sa_l = np.concatenate([sa_t_1, sa_t_2], axis=0)
        idxs = np.random.randint(0, self.sa_l.shape[0], size=batch_size)

        # obs_ls = torch.as_tensor(self.obs_l[idxs], device=self.device).float()
        # action_ls = torch.as_tensor(self.action_l[idxs], device=self.device)
        sa_l = torch.as_tensor(sa_l[idxs], device=self.device)

        return sa_l


'''
env = dmc.make('walker_run', 'states', 1, 1, 1)

env.observation_spec(): Array(shape=(24,), dtype=dtype('float32'), name='observation')
env.action_spec(): BoundedArray(shape=(6,), dtype=dtype('float32'), name='action', minimum=-1.0, maximum=1.0)

time_step = env.step(action)
time_step = ExtendedTimeStep(observation=time_step.observation,
                             step_type=time_step.step_type,
                             action=action,
                             reward=time_step.reward or 0.0,
                             discount=time_step.discount or 1.0)
time_step.step_type = StepType.FIRST, StepType.MID, StepType.LAST

'''
