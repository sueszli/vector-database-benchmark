"""
Helpers for scripts like run_atari.py.
"""
import os
import warnings
import gym
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.misc_util import mpi_rank_or_zero
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_vec_env(env_id, n_envs=1, seed=None, start_index=0, monitor_dir=None, wrapper_class=None, env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a wrapped, monitored `VecEnv`.\n    By default it uses a `DummyVecEnv` which is usually faster\n    than a `SubprocVecEnv`.\n\n    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class\n    :param n_envs: (int) the number of environments you wish to have in parallel\n    :param seed: (int) the initial seed for the random number generator\n    :param start_index: (int) start rank index\n    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.\n        If None, no file will be written, however, the env will still be wrapped\n        in a Monitor wrapper to provide additional information about training.\n    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.\n        This can also be a function with single argument that wraps the environment in many things.\n    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor\n    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.\n    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.\n    :return: (VecEnv) The wrapped environment\n    '
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        if False:
            for i in range(10):
                print('nop')

        def _init():
            if False:
                while True:
                    i = 10
            if isinstance(env_id, str):
                env = gym.make(env_id)
                if len(env_kwargs) > 0:
                    warnings.warn('No environment class was passed (only an env ID) so `env_kwargs` will be ignored')
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init
    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv
    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, allow_early_resets=True, start_method=None, use_subprocess=False):
    if False:
        return 10
    '\n    Create a wrapped, monitored VecEnv for Atari.\n\n    :param env_id: (str) the environment ID\n    :param num_env: (int) the number of environment you wish to have in subprocesses\n    :param seed: (int) the initial seed for RNG\n    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function\n    :param start_index: (int) start rank index\n    :param allow_early_resets: (bool) allows early reset of the environment\n    :param start_method: (str) method used to start the subprocesses.\n        See SubprocVecEnv doc for more information\n    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when\n        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False\n    :return: (VecEnv) The atari environment\n    '
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        if False:
            return 10

        def _thunk():
            if False:
                while True:
                    i = 10
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=allow_early_resets)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], start_method=start_method)

def make_mujoco_env(env_id, seed, allow_early_resets=True):
    if False:
        print('Hello World!')
    '\n    Create a wrapped, monitored gym.Env for MuJoCo.\n\n    :param env_id: (str) the environment ID\n    :param seed: (int) the initial seed for RNG\n    :param allow_early_resets: (bool) allows early reset of the environment\n    :return: (Gym Environment) The mujoco environment\n    '
    set_global_seeds(seed + 10000 * mpi_rank_or_zero())
    env = gym.make(env_id)
    env = Monitor(env, os.path.join(logger.get_dir(), '0'), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env

def make_robotics_env(env_id, seed, rank=0, allow_early_resets=True):
    if False:
        return 10
    '\n    Create a wrapped, monitored gym.Env for MuJoCo.\n\n    :param env_id: (str) the environment ID\n    :param seed: (int) the initial seed for RNG\n    :param rank: (int) the rank of the environment (for logging)\n    :param allow_early_resets: (bool) allows early reset of the environment\n    :return: (Gym Environment) The robotic environment\n    '
    set_global_seeds(seed)
    env = gym.make(env_id)
    keys = ['observation', 'desired_goal']
    try:
        from gym.wrappers import FilterObservation, FlattenObservation
        env = FlattenObservation(FilterObservation(env, keys))
    except ImportError:
        from gym.wrappers import FlattenDictWrapper
        env = FlattenDictWrapper(env, keys)
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), info_keywords=('is_success',), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env

def arg_parser():
    if False:
        while True:
            i = 10
    '\n    Create an empty argparse.ArgumentParser.\n\n    :return: (ArgumentParser)\n    '
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    if False:
        return 10
    "\n    Create an argparse.ArgumentParser for run_atari.py.\n\n    :return: (ArgumentParser) parser {'--env': 'BreakoutNoFrameskip-v4', '--seed': 0, '--num-timesteps': int(1e7)}\n    "
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10000000.0))
    return parser

def mujoco_arg_parser():
    if False:
        i = 10
        return i + 15
    "\n    Create an argparse.ArgumentParser for run_mujoco.py.\n\n    :return:  (ArgumentParser) parser {'--env': 'Reacher-v2', '--seed': 0, '--num-timesteps': int(1e6), '--play': False}\n    "
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1000000.0))
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def robotics_arg_parser():
    if False:
        print('Hello World!')
    "\n    Create an argparse.ArgumentParser for run_mujoco.py.\n\n    :return: (ArgumentParser) parser {'--env': 'FetchReach-v0', '--seed': 0, '--num-timesteps': int(1e6)}\n    "
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1000000.0))
    return parser