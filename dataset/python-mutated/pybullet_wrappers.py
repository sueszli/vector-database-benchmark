import gym
import numpy as np
from ditk import logging
from ding.envs import ObsNormWrapper, RewardNormWrapper
try:
    import pybulletgym
except ImportError:
    logging.warning('not found pybullet env, please install it, refer to https://github.com/benelot/pybullet-gym')

def wrap_pybullet(env_id, norm_obs=True, norm_reward=True, only_info=False) -> gym.Env:
    if False:
        return 10
    '\n    Overview:\n        Wrap Pybullet Env to preprocess env step\'s return info, e.g. observation normalization, reward normalization, etc.\n    Arguments:\n        - env_id (:obj:`str`): Pybullet environment id, for example "HalfCheetah-v3"\n        - norm_obs (:obj:`EasyDict`): Whether to normalize observation or not\n        - norm_reward (:obj:`EasyDict`): Whether to normalize reward or not. For evaluator, environment\'s reward \\\n            should not be normalized: Either ``norm_reward`` is None or ``norm_reward.use_norm`` is False can do this.\n    Returns:\n        - wrapped_env (:obj:`gym.Env`): The wrapped Pybullet environment\n    '
    if not only_info:
        env = gym.make(env_id)
        if norm_obs is not None and norm_obs.use_norm:
            env = ObsNormWrapper(env)
        if norm_reward is not None and norm_reward.use_norm:
            env = RewardNormWrapper(env, norm_reward.reward_discount)
        return env
    else:
        wrapper_info = ''
        if norm_obs is not None and norm_obs.use_norm:
            wrapper_info = ObsNormWrapper.__name__ + '\n'
        if norm_reward is not None and norm_reward.use_norm:
            wrapper_info = RewardNormWrapper.__name__ + '\n'
        return wrapper_info