"""
Example of a curriculum learning setup using the `TaskSettableEnv` API
and the env_task_fn config.

This example shows:
  - Writing your own curriculum-capable environment using gym.Env.
  - Defining a env_task_fn that determines, whether and which new task
    the env(s) should be set to (using the TaskSettableEnv API).
  - Using Tune and RLlib to curriculum-learn this env.

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import numpy as np
import os
import ray
from ray import air, tune
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.curriculum_capable_env import CurriculumCapableEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
(tf1, tf, tfv) = try_import_tf()
(torch, nn) = try_import_torch()
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='PPO', help='The RLlib-registered algorithm to use.')
parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
parser.add_argument('--as-test', action='store_true', help='Whether this script should be run as a test: --stop-reward must be achieved within --stop-timesteps AND --stop-iters.')
parser.add_argument('--stop-iters', type=int, default=50, help='Number of iterations to train.')
parser.add_argument('--stop-timesteps', type=int, default=200000, help='Number of timesteps to train.')
parser.add_argument('--stop-reward', type=float, default=10000.0, help='Reward at which we stop training.')
parser.add_argument('--local-mode', action='store_true', help='Init Ray in local mode for easier debugging.')

def curriculum_fn(train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext) -> TaskType:
    if False:
        while True:
            i = 10
    "Function returning a possibly new task to set `task_settable_env` to.\n\n    Args:\n        train_results: The train results returned by Algorithm.train().\n        task_settable_env: A single TaskSettableEnv object\n            used inside any worker and at any vector position. Use `env_ctx`\n            to get the worker_index, vector_index, and num_workers.\n        env_ctx: The env context object (i.e. env's config dict\n            plus properties worker_index, vector_index and num_workers) used\n            to setup the `task_settable_env`.\n\n    Returns:\n        TaskType: The task to set the env to. This may be the same as the\n            current one.\n    "
    new_task = int(np.log10(train_results['episode_reward_mean']) + 2.1)
    new_task = max(min(new_task, 5), 1)
    print(f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}\nR={train_results['episode_reward_mean']}\nSetting env to task={new_task}")
    return new_task
if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(local_mode=args.local_mode)
    config = get_trainable_cls(args.run).get_default_config().environment(CurriculumCapableEnv, env_config={'start_level': 1}, env_task_fn=curriculum_fn).framework(args.framework).rollouts(num_rollout_workers=2, num_envs_per_worker=5).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')))
    stop = {'training_iteration': args.stop_iters, 'timesteps_total': args.stop_timesteps, 'episode_reward_mean': args.stop_reward}
    tuner = tune.Tuner(args.run, param_space=config.to_dict(), run_config=air.RunConfig(stop=stop, verbose=2))
    results = tuner.fit()
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()