import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import wandb

import utils
# from preference_buffer import PreferenceDataset, SegmentReplayBuffer
from reward_model import RewardModel


class PbTrainer:
    Feedtype_Set_Train_R = {
        5, "sr",
        6, "sdr",
        8,
        9,
    }

    def __init__(self, device, train_env,
                 segment_len: int = 50, reward_model_train_epoch: int = 50,
                 reward_batch: int = 100,
                 # query selection
                 feed_type: int = 0, label_margin: float = 0.0,
                 # surf
                 data_aug_ratio: int = 20,
                 # teacher
                 teacher_beta: float = -1.0,
                 teacher_gamma: float = 1.0,
                 teacher_eps_mistake: float = 0.0,
                 teacher_eps_skip: float = 0.0,
                 teacher_eps_equal: float = 0.0,
                 skip_impl_tuple: tuple = (),
                 # skill space query selection
                 sf_dim: int = 10,
                 **kwargs):

        self.device = device
        self.segment_len = segment_len
        self.reward_model_train_epoch = reward_model_train_epoch
        self.feed_type = feed_type
        self.label_margin = label_margin
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_eps_equal = teacher_eps_equal
        self.data_aug_ratio = data_aug_ratio

        # preference
        self.reward_model = RewardModel(
            ds=train_env.observation_spec().shape[0],
            da=train_env.action_spec().shape[0],
            device=self.device,
            # copy pebble's config yaml
            ensemble_size=3,
            size_segment=segment_len,
            activation='tanh',
            lr=0.0003,
            mb_size=reward_batch,
            # surf
            # large_batch=1,
            large_batch=10,
            data_aug_ratio=data_aug_ratio,
            data_aug_window=5,
            # teacher
            label_margin=self.label_margin,
            teacher_beta=self.teacher_beta,
            teacher_gamma=self.teacher_gamma,
            teacher_eps_mistake=self.teacher_eps_mistake,
            teacher_eps_skip=self.teacher_eps_skip,  # use this
            teacher_eps_equal=self.teacher_eps_equal,
            skip_impl_tuple=skip_impl_tuple,  # [mean, diff, error]
            # skill space query selection
            dz=sf_dim,
        )
        self.total_feedback = 0
        self.labeled_feedback = 0

    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries, metrics = self.reward_model.uniform_sampling()
        else:
            if self.feed_type in (0, "u"):
                labeled_queries, metrics = self.reward_model.uniform_sampling()
            elif self.feed_type in (1, "d"):
                labeled_queries, metrics = self.reward_model.disagreement_sampling()
            elif self.feed_type in (2, "e"):
                labeled_queries, metrics = self.reward_model.entropy_sampling()
            # skill space query selection
            elif self.feed_type in (3, "sf"):
                labeled_queries, metrics = self.reward_model.skill_diff_sampling()
            elif self.feed_type in (4, "sfd"):
                labeled_queries, metrics = self.reward_model.skill_diff_disagreement_sampling2()
            elif self.feed_type in (5, "sr"):
                labeled_queries, metrics = self.reward_model.skill_return_sampling()
            elif self.feed_type in (6, "sdr"):
                labeled_queries, metrics = self.reward_model.disagreement_skill_return_sampling()
            elif self.feed_type == 7:
                labeled_queries, metrics = self.reward_model.skill_diff_disagreement_mul_sampling()
            elif self.feed_type == 8:
                labeled_queries, metrics = self.reward_model.skill_return_disagreement_mul_sampling()
            elif self.feed_type == 9:
                labeled_queries, metrics = self.reward_model.skill_return_Rdisagreement_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        # update reward
        if self.labeled_feedback > 0:
            train_acc = 0
            for _ in range(self.reward_model_train_epoch):
                if self.label_margin > 0 or self.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    if self.data_aug_ratio:
                        train_acc, reward_loss = self.reward_model.semi_train_reward()
                    else:
                        train_acc, reward_loss = self.reward_model.train_reward()
                if np.mean(train_acc) > 0.98:
                    break

            # if self.cfg.wandb:
            #     wandb.log({'train_acc':total_acc, 'reward_loss':reward_loss}, step=self.step)

        # add metrics about thresholds of reward_model
        if self.reward_model.skip_use_error:
            metrics["teacher_thres_error"] = self.reward_model.teacher_thres_skip
        else:
            metrics["teacher_thres_skip"] = self.reward_model.teacher_thres_skip
        metrics["teacher_thres_equal"] = self.reward_model.teacher_thres_equal
        return metrics
