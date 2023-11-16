"""
This script demonstrates how one can specify n (vectorized) envs
as ray remote (actors), such that stepping through these occurs in parallel.
Also, actions for each env step will be calculated on the "main" node.

This can be useful if the "main" node is a GPU machine and we would like to
speed up batched action calculations, similar to DeepMind's SEED
architecture, described here:

https://ai.googleblog.com/2020/03/massively-scaling-reinforcement.html
"""
import argparse
import os
from typing import Union
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from ray.tune import PlacementGroupFactory
from ray.tune.logger import pretty_print

def get_cli_args():
    if False:
        return 10
    'Create CLI parser and return parsed arguments'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs-per-worker', type=int, default=4)
    parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
    parser.add_argument('--as-test', action='store_true', help='Whether this script should be run as a test: --stop-reward must be achieved within --stop-timesteps AND --stop-iters.')
    parser.add_argument('--stop-iters', type=int, default=50, help='Number of iterations to train.')
    parser.add_argument('--stop-timesteps', type=int, default=100000, help='Number of timesteps to train.')
    parser.add_argument('--stop-reward', type=float, default=150.0, help='Reward at which we stop training.')
    parser.add_argument('--no-tune', action='store_true', help='Run without Tune using a manual train loop instead. Here,there is no TensorBoard support.')
    parser.add_argument('--local-mode', action='store_true', help='Init Ray in local mode for easier debugging.')
    args = parser.parse_args()
    print(f'Running with following CLI args: {args}')
    return args

class PPORemoteInference(PPO):

    @classmethod
    @override(Algorithm)
    def default_resource_request(cls, config: Union[AlgorithmConfig, PartialAlgorithmConfigDict]):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(config, AlgorithmConfig):
            cf = config
        else:
            cf = cls.get_default_config().update_from_dict(config)
        return PlacementGroupFactory(bundles=[{'CPU': 1, 'GPU': cf.num_gpus}, {'CPU': cf.num_envs_per_worker}], strategy=cf.placement_strategy)
if __name__ == '__main__':
    args = get_cli_args()
    ray.init(num_cpus=6, local_mode=args.local_mode)
    config = PPOConfig().environment('CartPole-v1').framework(args.framework).rollouts(remote_worker_envs=True, num_envs_per_worker=args.num_envs_per_worker, num_rollout_workers=0).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')), num_cpus_for_local_worker=args.num_envs_per_worker + 1)
    if args.no_tune:
        algo = PPORemoteInference(config=config)
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            if result['timesteps_total'] >= args.stop_timesteps or result['episode_reward_mean'] >= args.stop_reward:
                break
    else:
        stop = {'training_iteration': args.stop_iters, 'timesteps_total': args.stop_timesteps, 'episode_reward_mean': args.stop_reward}
        results = tune.Tuner(PPORemoteInference, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)).fit()
        if args.as_test:
            check_learning_achieved(results, args.stop_reward)
    ray.shutdown()