import argparse
import os
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation-duration', type=lambda v: v if v == 'auto' else int(v), default=13, help="Number of evaluation episodes/timesteps to run each iteration. If 'auto', will run as many as possible during train pass.")
parser.add_argument('--evaluation-duration-unit', type=str, default='episodes', choices=['episodes', 'timesteps'], help='The unit in which to measure the duration (`episodes` or `timesteps`).')
parser.add_argument('--evaluation-num-workers', type=int, default=2, help='The number of evaluation workers to setup. 0 for a single local evaluation worker. Note that for values >0, nolocal evaluation worker will be created (b/c not needed).')
parser.add_argument('--evaluation-interval', type=int, default=2, help='Every how many train iterations should we run an evaluation loop?')
parser.add_argument('--run', type=str, default='PPO', help='The RLlib-registered algorithm to use.')
parser.add_argument('--num-cpus', type=int, default=0)
parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
parser.add_argument('--as-test', action='store_true', help='Whether this script should be run as a test: --stop-reward must be achieved within --stop-timesteps AND --stop-iters.')
parser.add_argument('--stop-iters', type=int, default=200, help='Number of iterations to train.')
parser.add_argument('--stop-timesteps', type=int, default=200000, help='Number of timesteps to train.')
parser.add_argument('--stop-reward', type=float, default=180.0, help='Reward at which we stop training.')
parser.add_argument('--local-mode', action='store_true', help='Init Ray in local mode for easier debugging.')

class AssertEvalCallback(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'evaluation' in result and 'hist_stats' in result['evaluation']:
            hist_stats = result['evaluation']['hist_stats']
            if algorithm.config.evaluation_duration_unit == 'episodes':
                num_episodes_done = len(hist_stats['episode_lengths'])
                if isinstance(algorithm.config.evaluation_duration, int):
                    assert num_episodes_done == algorithm.config.evaluation_duration
                else:
                    assert algorithm.config.evaluation_duration == 'auto'
                    assert num_episodes_done >= algorithm.config.evaluation_num_workers
                print(f'Number of run evaluation episodes: {num_episodes_done} (ok)!')
            else:
                num_timesteps_reported = result['evaluation']['timesteps_this_iter']
                num_timesteps_wanted = algorithm.config.evaluation_duration
                if num_timesteps_wanted != 'auto':
                    delta = num_timesteps_wanted - num_timesteps_reported
                    assert abs(delta) < 20, (delta, num_timesteps_wanted, num_timesteps_reported)
                print(f'Number of run evaluation timesteps: {num_timesteps_reported} (ok)!')
            print(f"R={result['evaluation']['episode_reward_mean']}")
if __name__ == '__main__':
    import ray
    from ray import air, tune
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    config = get_trainable_cls(args.run).get_default_config().environment('CartPole-v1').framework(args.framework).training().evaluation(evaluation_parallel_to_training=True, evaluation_num_workers=args.evaluation_num_workers, evaluation_interval=args.evaluation_interval, evaluation_duration=args.evaluation_duration, evaluation_duration_unit=args.evaluation_duration_unit).callbacks(AssertEvalCallback).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')))
    stop = {'training_iteration': args.stop_iters, 'timesteps_total': args.stop_timesteps, 'episode_reward_mean': args.stop_reward}
    results = tune.Tuner(args.run, param_space=config, run_config=air.RunConfig(stop=stop, verbose=2)).fit()
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()