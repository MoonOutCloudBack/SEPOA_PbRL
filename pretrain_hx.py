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
import utils
from logger import Logger
from replay_buffer_hx_new import make_replay_loader, ReplayBuffer
from video import TrainVideoRecorder, VideoRecorder

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


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
                cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed), "pretrain_hx",
            ])
            wandb.init(project="urlb", group=cfg.agent.name, config=utils.hydra_config_to_dict(cfg), name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = PRIMAL_TASKS[self.cfg.domain]
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create replay buffer
        self.replay_buffer, self.replay_loader = make_replay_loader(
            train_env=self.train_env, meta_specs=meta_specs,
            replay_buffer_size=cfg.replay_buffer_size,
            device=self.device,
            batch_size=cfg.batch_size,
            use_preference=False,
            nstep=cfg.nstep,
            discount=cfg.discount,
            num_workers=cfg.replay_buffer_num_workers,
        )
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.upload_train_video)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.upload_train_video)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.log_success = "metaworld" in PRIMAL_TASKS[self.cfg.domain]

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
            if self.log_success:
                log('success_rate', success_rate / episode * 100.0)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        episode_success = 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_buffer.add(time_step, meta, 0)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        # start training
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
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
                        if self.log_success:
                            log('episode_success', episode_success)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_buffer.add(time_step, meta, 0)
                self.train_video_recorder.init(time_step.observation)
                self._replay_iter = None
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                    self.save_replay_buffer_hx()
                episode_step = 0
                episode_reward = 0
                episode_success = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step, meta, 0)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            if self.log_success:
                episode_success = max(episode_success, self.train_env.last_info['success'])

        # save snapshot and eval finally
        self.save_snapshot()
        self.save_replay_buffer_hx()
        self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
        self.eval()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_replay_buffer_hx(self):
        replay_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        replay_dir.mkdir(exist_ok=True, parents=True)
        replay_dir = replay_dir / f'replay_buffer_{self.global_frame}.pkl'
        self.replay_buffer.save_pickle(replay_dir)
        print(f'saved replay buffer {self.global_frame} to {replay_dir}')


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain_hx import Workspace as W
    root_dir = Path.cwd()
    print(f"snapshots: {cfg['snapshots']}")
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
