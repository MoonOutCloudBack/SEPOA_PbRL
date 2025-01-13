from collections import deque
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
from dmc_benchmark import get_domain
import utils
from logger import Logger
from replay_buffer_hx_new import make_replay_loader, ReplayBuffer
from video import TrainVideoRecorder, VideoRecorder
from pb_trainer import PbTrainer

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    replay_buffer: ReplayBuffer

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.agent.name, cfg.task,  # cfg.obs_type,
                str(cfg.seed), "pbrl" if cfg.snapshot_ts > 0 else "finetune_pb_no_pretrain",
                str(cfg.reward_batch), str(cfg.feed_type), str(cfg.skip_impl), str(cfg.teacher_eps_skip),
                "surf" if cfg.data_aug_ratio != 0 else "",
            ])
            wandb.init(project="urlb", group=cfg.agent.name, config=utils.hydra_config_to_dict(cfg), name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs

        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent,)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.state_dim = self.train_env.observation_spec().shape[0]
        self.action_dim = self.train_env.action_spec().shape[0]

        # create replay buffer
        self.replay_buffer, self.replay_loader = make_replay_loader(
            train_env=self.train_env, meta_specs=meta_specs,
            replay_buffer_size=cfg.replay_buffer_size,
            device=self.device,
            batch_size=cfg.batch_size,
            use_preference=cfg.preference,
            nstep=cfg.nstep,
            discount=cfg.discount,
            num_workers=cfg.replay_buffer_num_workers,
        )
        self._replay_iter = None

        self.skip_impl_tuple = self.cfg.skip_impl.split("_")
        self.skip_ratio_method = self.skip_impl_tuple[0]

        # preference
        if self.cfg.preference:
            self.pb_trainer = PbTrainer(device=self.device, train_env=self.train_env,
                                        segment_len=self.cfg.segment_len,
                                        reward_model_train_epoch=self.cfg.reward_model_train_epoch,
                                        reward_batch=self.cfg.reward_batch,
                                        # skill space query selection
                                        # sf_dim=self.agent.sf_dim,
                                        feed_type=self.cfg.feed_type,
                                        # teacher
                                        teacher_eps_equal=self.cfg.teacher_eps_equal,
                                        teacher_eps_skip=self.cfg.teacher_eps_skip,
                                        teacher_eps_mistake=self.cfg.teacher_eps_mistake,
                                        skip_impl_tuple=self.skip_impl_tuple,
                                        # surf
                                        data_aug_ratio=self.cfg.data_aug_ratio,
                                        )
            self.label_acc = 0

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            is_quadruped=('quadruped' in self.cfg.task),
            use_wandb=self.cfg.upload_train_video)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            use_wandb=self.cfg.upload_train_video)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.log_success = "metaworld" in self.cfg.task
        self.train_video_count = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        success_rate = 0
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            episode_success = 0
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

                if self.log_success:
                    episode_success = max(episode_success, self.eval_env.last_info['success'])

            episode += 1
            success_rate += episode_success
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if self.cfg.preference:
                log('label_acc', self.label_acc)
            if self.log_success:
                log('success_rate', success_rate / episode * 100.0)

    def evaluate_reward_mismatch(self):
        r_true_buffer, r_hat_buffer = [], []
        meta = self.agent.init_meta()
        for i in range(10):
            time_step = self.eval_env.reset()
            obs = time_step.observation
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                sa_t = np.concatenate([obs, action], axis=-1)
                reward_hat = self.pb_trainer.reward_model.r_hat(sa_t)
                time_step = self.eval_env.step(action)
                r_true_buffer.append(time_step.reward)
                r_hat_buffer.append(reward_hat)
                obs = time_step.observation
        r_true_buffer = np.array(r_true_buffer).reshape(10, -1)
        r_hat_buffer = np.array(r_hat_buffer).reshape(10, -1)
        traj_index_1, traj_index_2 = np.random.choice(r_true_buffer.shape[0],
                                                      size=100, replace=True), \
            np.random.choice(r_true_buffer.shape[0],
                             size=100, replace=True)  # 100 random 0-9 number
        index_1, index_2 = np.random.choice(r_true_buffer.shape[1] - self.cfg.segment_len,
                                            size=traj_index_1.shape[0], replace=True), \
            np.random.choice(r_true_buffer.shape[1] - self.cfg.segment_len,
                             size=traj_index_1.shape[0], replace=True)  # 100 random 0-950 number
        time_index_1, time_index_2 = index_1.reshape(-1, 1) + \
            np.tile(np.arange(self.cfg.segment_len), (index_1.shape[0], 1)), \
            index_2.reshape(-1, 1) + \
            np.tile(np.arange(self.cfg.segment_len), (index_2.shape[0], 1))  # (100, 50) index

        r_true_1, r_hat_1, r_true_2, r_hat_2 = [], [], [], []
        for i in range(time_index_1.shape[0]):
            r_true_1.append(r_true_buffer[traj_index_1][i][time_index_1[i]])
            r_hat_1.append(r_hat_buffer[traj_index_1][i][time_index_1[i]])
        for i in range(time_index_2.shape[0]):
            r_true_2.append(r_true_buffer[traj_index_2][i][time_index_2[i]])
            r_hat_2.append(r_hat_buffer[traj_index_2][i][time_index_2[i]])
        r_true_1, r_hat_1 = np.array(r_true_1), np.array(r_hat_1)
        r_true_2, r_hat_2 = np.array(r_true_2), np.array(r_hat_2)
        true_label = 1 * (r_true_1.sum(axis=1) < r_true_2.sum(axis=1))
        pre_label = 1 * (r_hat_1.sum(axis=1) < r_hat_2.sum(axis=1))
        acc = true_label == pre_label

        return acc.sum() / 100

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # init
        episode_step, episode_reward, interact_count = 0, 0, 0
        episode_success = 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_buffer.add(time_step, meta, 0)
        if self.cfg.preference:
            self.pb_trainer.reward_model.skill_list.append(
                np.zeros(self.pb_trainer.reward_model.dz))
        if self.cfg.save_train_video:
            self.video_recorder.init(self.train_env)
            # self.train_video_recorder.init(time_step.observation)
        metrics = None
        # store train returns of recent 10 episodes
        domain = get_domain(self.cfg.task)
        avg_train_true_return = deque([], maxlen=(10 if domain == 'cheetah' else 10))

        # start training
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self.cfg.save_train_video:
                    self.video_recorder.save(f'train_video/{self.train_video_count}.mp4')
                    # self.train_video_recorder.save(f'{self.train_video_count}.mp4')
                    self.train_video_count += 1
                avg_train_true_return.append(episode_reward)
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)
                        if self.cfg.preference:
                            log('label_acc', self.label_acc)
                        if self.log_success:
                            log('episode_success', episode_success)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_buffer.add(time_step, meta, 0)
                if self.cfg.preference:
                    self.pb_trainer.reward_model.skill_list.append(
                        np.zeros(self.pb_trainer.reward_model.dz))
                if self.cfg.save_train_video:
                    self.video_recorder.init(self.train_env)
                    # self.train_video_recorder.init(time_step.observation)
                self._replay_iter = None

                episode_step = 0
                episode_reward = 0
                episode_success = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "_regress_meta"):
                repeat = self.cfg.action_repeat  # 1
                every = self.agent.update_task_every_step // repeat  # aps: 5
                init_step = self.agent.num_init_steps
                if self.global_step > (
                        init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent._regress_meta(self.replay_iter,
                                                    self.global_step)

            # sample action
            obs = time_step.observation
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                # update reward model
                if self.cfg.preference and self.pb_trainer.total_feedback < self.cfg.max_feedback:
                    if interact_count >= self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:  # become less
                            frac = (self.cfg.num_train_frames - self._global_step) / self.cfg.num_train_frames
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:  # become more
                            frac = self.cfg.num_train_frames / (self.cfg.num_train_frames - self._global_step + 1)
                        else:
                            frac = 1
                        self.pb_trainer.reward_model.change_batch(frac)

                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment_len / 1000)
                        self.pb_trainer.reward_model.set_teacher_thres_equal(new_margin)
                        if self.skip_ratio_method == 'diff':
                            max_min_return = (np.max(avg_train_true_return) -
                                              np.min(avg_train_true_return)) * (self.cfg.segment_len / 1000)
                            self.pb_trainer.reward_model.set_teacher_thres_skip(max_min_return)
                        elif self.skip_ratio_method == 'mean':
                            self.pb_trainer.reward_model.set_teacher_thres_skip(new_margin)
                        elif self.skip_ratio_method == 'fix':
                            self.pb_trainer.reward_model.set_teacher_thres_skip(1)
                        else:
                            raise NotImplementedError()

                        # corner case: new total feed > max feed
                        if self.pb_trainer.reward_model.mb_size + self.pb_trainer.total_feedback > self.cfg.max_feedback:
                            self.pb_trainer.reward_model.set_batch(
                                self.cfg.max_feedback - self.pb_trainer.total_feedback)

                        metrics = self.pb_trainer.learn_reward()
                        self.logger.log_metrics(metrics, self.global_frame, ty='train')
                        if self.cfg.log_reward_mismatch:
                            self.label_acc = self.evaluate_reward_mismatch()
                            # print(f'label_acc: {self.label_acc}')
                        self.replay_buffer.relabel_with_predictor(self.pb_trainer.reward_model)
                        interact_count = 0

                if self.global_step >= self.cfg.num_interact:
                    if hasattr(self.agent, "borrow_reward_model"):  # rune
                        self.agent.borrow_reward_model(self.pb_trainer.reward_model)
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    if hasattr(self.agent, "del_reward_model"):
                        self.agent.del_reward_model()
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            if self.cfg.preference:
                predicted_reward = self.pb_trainer.reward_model.r_hat(
                    np.concatenate([obs, action], axis=-1))
                self.replay_buffer.add(time_step, meta, predicted_reward)
                self.pb_trainer.reward_model.add_data(obs, action, time_step.reward, time_step.last())
            else:
                self.replay_buffer.add(time_step, meta, 0)
            if self.cfg.save_train_video:
                self.video_recorder.record(self.train_env)
                # self.train_video_recorder.record(time_step.observation)
            interact_count += 1
            episode_step += 1
            self._global_step += 1
            if self.log_success:
                episode_success = max(episode_success, self.train_env.last_info['success'])

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        print(snapshot_base_dir.absolute())
        domain = get_domain(self.cfg.task)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            # print(f"try load: {snapshot}")
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f, map_location=self.device)
            print(f"load from {snapshot} successfully")
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try all seed
        for seed in range(100):
            payload = try_load(seed)
            if payload is not None:
                return payload
        print("snapshot not found")
        return None


@hydra.main(config_path='.', config_name='finetune_pb')
def main(cfg):
    from finetune_pb import Workspace as W
    root_dir = Path.cwd()
    print(f"root_dir: {root_dir}")

    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
