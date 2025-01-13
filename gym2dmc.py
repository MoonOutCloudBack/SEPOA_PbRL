from dm_env import specs, Environment, TimeStep, StepType
from collections import OrderedDict
import numpy as np

# https://github.com/NM512/gym2dmc


class Gym2DMC(Environment):
    """Convert a Gym environment to a DMC environment"""

    def __init__(self, gym_env) -> None:
        """Initializes a new Gym2DMC wrapper

        Args:
            gym_env (GymEnv): The Gym environment to convert.
        """
        gym_obs_space = gym_env.observation_space
        self._observation_spec = OrderedDict()
        self._observation_spec['observations'] = specs.BoundedArray(
            shape=gym_obs_space.shape,
            dtype=gym_obs_space.dtype,
            minimum=gym_obs_space.low,
            maximum=gym_obs_space.high,
            name='observation'
        )

        gym_act_space = gym_env.action_space
        self._action_spec = specs.BoundedArray(
            shape=gym_act_space.shape,
            dtype=gym_act_space.dtype,
            minimum=gym_act_space.low,
            maximum=gym_act_space.high,
            name='action'
        )
        self._gym_env = gym_env
        self.last_info = {}

    def step(self, action):
        obs, reward, done, info = self._gym_env.step(action)
        self.last_info = info

        if done:
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID
            discount = 1.0

        return TimeStep(step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=OrderedDict({'observations': obs}))

    def reset(self):
        # to fix
        # metaworld use gym style interface
        obs = self._gym_env.reset()
        if not isinstance(obs, np.ndarray):
            obs = obs[0]
        return TimeStep(step_type=StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=OrderedDict({'observations': obs}))

    def render(self):
        return self._gym_env.render()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
