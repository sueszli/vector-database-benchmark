"""Example of using a custom training workflow.

Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. Both are executed concurrently
via a custom training workflow.
"""
import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED, NUM_TARGET_UPDATES, LAST_TARGET_UPDATE_TS
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_env
parser = argparse.ArgumentParser()
parser.add_argument('--torch', action='store_true')
parser.add_argument('--mixed-torch-tf', action='store_true')
parser.add_argument('--local-mode', action='store_true', help='Init Ray in local mode for easier debugging.')
parser.add_argument('--as-test', action='store_true', help='Whether this script should be run as a test: --stop-reward must be achieved within --stop-timesteps AND --stop-iters.')
parser.add_argument('--stop-iters', type=int, default=600, help='Number of iterations to train.')
parser.add_argument('--stop-timesteps', type=int, default=200000, help='Number of timesteps to train.')
parser.add_argument('--stop-reward', type=float, default=600.0, help='Reward at which we stop training.')

class MyAlgo(Algorithm):

    @override(Algorithm)
    def setup(self, config):
        if False:
            while True:
                i = 10
        super().setup(config)
        self.local_replay_buffer = MultiAgentReplayBuffer(num_shards=1, capacity=50000)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        if False:
            while True:
                i = 10
        ppo_batches = []
        num_env_steps = 0
        while num_env_steps < 200:
            ma_batches = synchronous_parallel_sample(worker_set=self.workers, concat=False)
            for ma_batch in ma_batches:
                self._counters[NUM_ENV_STEPS_SAMPLED] += ma_batch.count
                self._counters[NUM_AGENT_STEPS_SAMPLED] += ma_batch.agent_steps()
                ppo_batch = ma_batch.policy_batches.pop('ppo_policy')
                self.local_replay_buffer.add(ma_batch)
                ppo_batches.append(ppo_batch)
                num_env_steps += ppo_batch.count
        dqn_train_results = {}
        if self._counters[NUM_ENV_STEPS_SAMPLED] > 1000:
            for _ in range(10):
                dqn_train_batch = self.local_replay_buffer.sample(num_items=64)
                dqn_train_results = train_one_step(self, dqn_train_batch, ['dqn_policy'])
                self._counters['agent_steps_trained_DQN'] += dqn_train_batch.agent_steps()
                print('DQN policy learning on samples from', 'agent steps trained', dqn_train_batch.agent_steps())
        if self._counters['agent_steps_trained_DQN'] - self._counters[LAST_TARGET_UPDATE_TS] >= self.get_policy('dqn_policy').config['target_network_update_freq']:
            self.workers.local_worker().get_policy('dqn_policy').update_target()
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters['agent_steps_trained_DQN']
        ppo_train_batch = concat_samples(ppo_batches)
        self._counters['agent_steps_trained_PPO'] += ppo_train_batch.agent_steps()
        ppo_train_batch[Postprocessing.ADVANTAGES] = standardized(ppo_train_batch[Postprocessing.ADVANTAGES])
        print('PPO policy learning on samples from', 'agent steps trained', ppo_train_batch.agent_steps())
        ppo_train_batch = MultiAgentBatch({'ppo_policy': ppo_train_batch}, ppo_train_batch.count)
        ppo_train_results = train_one_step(self, ppo_train_batch, ['ppo_policy'])
        results = dict(ppo_train_results, **dqn_train_results)
        return results
if __name__ == '__main__':
    args = parser.parse_args()
    assert not (args.torch and args.mixed_torch_tf), 'Use either --torch or --mixed-torch-tf, not both!'
    ray.init(local_mode=args.local_mode)
    register_env('multi_agent_cartpole', lambda _: MultiAgentCartPole({'num_agents': 4}))
    policies = {'ppo_policy': (PPOTorchPolicy if args.torch or args.mixed_torch_tf else PPOTF1Policy, None, None, PPOConfig().training(num_sgd_iter=10, sgd_minibatch_size=128).framework('torch' if args.torch or args.mixed_torch_tf else 'tf')), 'dqn_policy': (DQNTorchPolicy if args.torch else DQNTFPolicy, None, None, DQNConfig().training(target_network_update_freq=500).framework('tf'))}

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if False:
            while True:
                i = 10
        if agent_id % 2 == 0:
            return 'ppo_policy'
        else:
            return 'dqn_policy'
    config = AlgorithmConfig().experimental(_enable_new_api_stack=False).environment('multi_agent_cartpole').framework('torch' if args.torch else 'tf').multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn).rollouts(num_rollout_workers=0, rollout_fragment_length=50).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0'))).reporting(metrics_num_episodes_for_smoothing=30)
    stop = {'training_iteration': args.stop_iters, 'timesteps_total': args.stop_timesteps, 'episode_reward_mean': args.stop_reward}
    results = tune.Tuner(MyAlgo, param_space=config.to_dict(), run_config=air.RunConfig(stop=stop)).fit()
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()