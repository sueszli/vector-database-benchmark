"""
This script demonstrates how one can specify custom env APIs in
combination with RLlib's `remote_worker_envs` setting, which
parallelizes individual sub-envs within a vector env by making each
one a ray Actor.

You can access your Env's API via a custom callback as shown below.
"""
import argparse
import gymnasium as gym
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='PPO', help='The RLlib-registered algorithm to use.')
parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--num-envs-per-worker', type=int, default=4)
parser.add_argument('--as-test', action='store_true', help='Whether this script should be run as a test: --stop-reward must be achieved within --stop-timesteps AND --stop-iters.')
parser.add_argument('--stop-iters', type=int, default=50, help='Number of iterations to train.')
parser.add_argument('--stop-timesteps', type=int, default=100000, help='Number of timesteps to train.')
parser.add_argument('--stop-reward', type=float, default=180.0, help='Reward at which we stop training.')
parser.add_argument('--local-mode', action='store_true', help='Init Ray in local mode for easier debugging.')

class NonVectorizedEnvToBeVectorizedIntoRemoteBaseEnv(TaskSettableEnv):
    """Class for a single sub-env to be vectorized into RemoteBaseEnv.

    If you specify this class directly under the "env" config key, RLlib
    will auto-wrap

    Note that you may implement your own custom APIs. Here, we demonstrate
    using RLlib's TaskSettableEnv API (which is a simple sub-class
    of gym.Env).
    """

    def __init__(self, config=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.action_space = gym.spaces.Box(0, 1, shape=(1,))
        self.observation_space = gym.spaces.Box(0, 1, shape=(2,))
        self.task = 1

    def reset(self, *, seed=None, options=None):
        if False:
            while True:
                i = 10
        self.steps = 0
        return (self.observation_space.sample(), {})

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        self.steps += 1
        done = truncated = self.steps > 10
        return (self.observation_space.sample(), 0, done, truncated, {})

    def set_task(self, task) -> None:
        if False:
            print('Hello World!')
        'We can set the task of each sub-env (ray actor)'
        print('Task set to {}'.format(task))
        self.task = task

class TaskSettingCallback(DefaultCallbacks):
    """Custom callback to verify, we can set the task on each remote sub-env."""

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Curriculum learning as seen in Ray docs'
        if result['episode_reward_mean'] > 0.0:
            phase = 0
        else:
            phase = 1
        algorithm.workers.foreach_env(lambda env: env.set_task.remote(phase))
if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_cpus=6, local_mode=args.local_mode)
    config = get_trainable_cls(args.run).get_default_config().environment(NonVectorizedEnvToBeVectorizedIntoRemoteBaseEnv).framework(args.framework).callbacks(TaskSettingCallback).rollouts(remote_worker_envs=True, num_rollout_workers=args.num_workers, num_envs_per_worker=args.num_envs_per_worker).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')))
    stop = {'training_iteration': args.stop_iters, 'timesteps_total': args.stop_timesteps, 'episode_reward_mean': args.stop_reward}
    results = tune.Tuner(args.run, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)).fit()
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()