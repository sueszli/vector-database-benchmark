from typing import Dict
import gym
import numpy as np
from ding.envs import ObsNormWrapper, RewardNormWrapper, DelayRewardWrapper, EvalEpisodeReturnWrapper

def wrap_mujoco(env_id, norm_obs: Dict=dict(use_norm=False), norm_reward: Dict=dict(use_norm=False), delay_reward_step: int=1) -> gym.Env:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Wrap Mujoco Env to preprocess env step\'s return info, e.g. observation normalization, reward normalization, etc.\n    Arguments:\n        - env_id (:obj:`str`): Mujoco environment id, for example "HalfCheetah-v3"\n        - norm_obs (:obj:`EasyDict`): Whether to normalize observation or not\n        - norm_reward (:obj:`EasyDict`): Whether to normalize reward or not. For evaluator, environment\'s reward \\\n            should not be normalized: Either ``norm_reward`` is None or ``norm_reward.use_norm`` is False can do this.\n    Returns:\n        - wrapped_env (:obj:`gym.Env`): The wrapped mujoco environment\n    '
    from . import mujoco_gym_env
    env = gym.make(env_id)
    env = EvalEpisodeReturnWrapper(env)
    if norm_obs is not None and norm_obs.use_norm:
        env = ObsNormWrapper(env)
    if norm_reward is not None and norm_reward.use_norm:
        env = RewardNormWrapper(env, norm_reward.reward_discount)
    if delay_reward_step > 1:
        env = DelayRewardWrapper(env, delay_reward_step)
    return env