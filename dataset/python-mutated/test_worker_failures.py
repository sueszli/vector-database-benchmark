from collections import defaultdict
import gymnasium as gym
import numpy as np
import time
import unittest
import ray
from ray.util.state import list_actors
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.pg.pg_tf_policy import PGTF2Policy
from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import register_env

@ray.remote
class Counter:
    """Remote counter service that survives restarts."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.reset()

    def _key(self, eval, worker_index, vector_index):
        if False:
            print('Hello World!')
        return f'{eval}:{worker_index}:{vector_index}'

    def increment(self, eval, worker_index, vector_index):
        if False:
            i = 10
            return i + 15
        self.counter[self._key(eval, worker_index, vector_index)] += 1

    def get(self, eval, worker_index, vector_index):
        if False:
            return 10
        return self.counter[self._key(eval, worker_index, vector_index)]

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.counter = defaultdict(int)

class FaultInjectEnv(gym.Env):
    """Env that fails upon calling `step()`, but only for some remote worker indices.

    The worker indices that should produce the failure (a ValueError) can be
    provided by a list (of ints) under the "bad_indices" key in the env's
    config.

    .. testcode::
        :skipif: True

        from ray.rllib.env.env_context import EnvContext
        # This env will fail for workers 1 and 2 (not for the local worker
        # or any others with an index != [1|2]).
        bad_env = FaultInjectEnv(
            EnvContext(
                {"bad_indices": [1, 2]},
                worker_index=1,
                num_workers=3,
             )
        )

        from ray.rllib.env.env_context import EnvContext
        # This env will fail only on the first evaluation worker, not on the first
        # regular rollout worker.
        bad_env = FaultInjectEnv(
            EnvContext(
                {"bad_indices": [1], "eval_only": True},
                worker_index=2,
                num_workers=5,
            )
        )
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        self.env = RandomEnv(config)
        self._skip_env_checking = True
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.config = config
        if 'counter' in config:
            self.counter = ray.get_actor(config['counter'])
        else:
            self.counter = None
        if config.get('init_delay', 0) > 0.0 and (not config.get('init_delay_indices', []) or self.config.worker_index in config.get('init_delay_indices', [])) and (self._get_count() > 0):
            time.sleep(config.get('init_delay'))

    def _increment_count(self):
        if False:
            while True:
                i = 10
        if self.counter:
            eval = self.config.get('evaluation', False)
            worker_index = self.config.worker_index
            vector_index = self.config.vector_index
            ray.wait([self.counter.increment.remote(eval, worker_index, vector_index)])

    def _get_count(self):
        if False:
            i = 10
            return i + 15
        if self.counter:
            eval = self.config.get('evaluation', False)
            worker_index = self.config.worker_index
            vector_index = self.config.vector_index
            return ray.get(self.counter.get.remote(eval, worker_index, vector_index))
        return -1

    def _maybe_raise_error(self):
        if False:
            i = 10
            return i + 15
        if self.config.worker_index not in self.config.get('bad_indices', []):
            return
        if self.counter:
            count = self._get_count()
            if self.config.get('failure_start_count', -1) >= 0 and count < self.config.get('failure_start_count'):
                return
            if self.config.get('failure_stop_count', -1) >= 0 and count >= self.config.get('failure_stop_count'):
                return
        raise ValueError(f"This is a simulated error from {('eval-' if self.config.get('evaluation', False) else '')}worker-idx={self.config.worker_index}!")

    def reset(self, *, seed=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        self._increment_count()
        self._maybe_raise_error()
        return self.env.reset()

    def step(self, action):
        if False:
            return 10
        self._increment_count()
        self._maybe_raise_error()
        if self.config.get('step_delay', 0) > 0.0 and (not self.config.get('init_delay_indices', []) or self.config.worker_index in self.config.get('step_delay_indices', [])):
            time.sleep(self.config.get('step_delay'))
        return self.env.step(action)

    def action_space_sample(self):
        if False:
            while True:
                i = 10
        return self.env.action_space.sample()

class ForwardHealthCheckToEnvWorker(RolloutWorker):
    """Configure RolloutWorker to error in specific condition is hard.

    So we take a short-cut, and simply forward ping() to env.sample().
    """

    def ping(self) -> str:
        if False:
            i = 10
            return i + 15
        _ = self.env.step(self.env.action_space_sample())
        return super().ping()

def wait_for_restore(num_restarting_allowed=0):
    if False:
        return 10
    'Wait for Ray actor fault tolerence to restore all failed workers.\n\n    Args:\n        num_restarting_allowed: Number of actors that are allowed to be\n            in "RESTARTING" state. This is because some actors may\n            hang in __init__().\n    '
    while True:
        states = [a['state'] for a in list_actors(filters=[('class_name', '=', 'ForwardHealthCheckToEnvWorker')])]
        finished = True
        for s in states:
            if s not in ['ALIVE', 'DEAD', 'RESTARTING']:
                finished = False
                break
        restarting = [s for s in states if s == 'RESTARTING']
        if len(restarting) > num_restarting_allowed:
            finished = False
        print('waiting ... ', states)
        if finished:
            break
        time.sleep(0.5)

class AddPolicyCallback(DefaultCallbacks):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def on_algorithm_init(self, *, algorithm, **kwargs):
        if False:
            return 10
        algorithm.add_policy(policy_id='test_policy', policy_cls=PGTorchPolicy if algorithm.config.framework_str == 'torch' else PGTF2Policy, observation_space=gym.spaces.Box(low=0, high=1, shape=(8,)), action_space=gym.spaces.Discrete(2), config={}, policy_state=None, evaluation_workers=True)

class TestWorkerFailures(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.init()
        register_env('fault_env', lambda c: FaultInjectEnv(c))
        register_env('multi_agent_fault_env', lambda c: make_multi_agent(FaultInjectEnv)(c))

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def _do_test_fault_ignore(self, config: AlgorithmConfig, fail_eval: bool=False):
        if False:
            print('Hello World!')
        config.num_rollout_workers = 2
        config.ignore_worker_failures = True
        config.recreate_failed_workers = False
        config.env = 'fault_env'
        config.env_config = {'bad_indices': [1]}
        if fail_eval:
            config.evaluation_num_workers = 2
            config.evaluation_interval = 1
            config.evaluation_config = {'ignore_worker_failures': True, 'recreate_failed_workers': False, 'env_config': {'bad_indices': [1], 'evaluation': True}}
        print(config)
        for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
            algo = config.build()
            algo.train()
            self.assertEqual(algo.workers.num_healthy_remote_workers(), 1)
            if fail_eval:
                self.assertEqual(algo.evaluation_workers.num_healthy_remote_workers(), 1)
            algo.stop()

    def _do_test_fault_fatal(self, config, fail_eval=False):
        if False:
            i = 10
            return i + 15
        config.num_rollout_workers = 2
        config.env = 'fault_env'
        config.env_config = {'bad_indices': [1, 2]}
        if fail_eval:
            config.evaluation_num_workers = 2
            config.evaluation_interval = 1
            config.evaluation_config = {'env_config': {'bad_indices': [1], 'evaluation': True}}
        for _ in framework_iterator(config, frameworks=('torch', 'tf')):
            a = config.build()
            self.assertRaises(Exception, lambda : a.train())
            a.stop()

    def _do_test_fault_fatal_but_recreate(self, config, multi_agent=False):
        if False:
            return 10
        COUNTER_NAME = f"_do_test_fault_fatal_but_recreate{('_ma' if multi_agent else '')}"
        counter = Counter.options(name=COUNTER_NAME).remote()
        config.num_rollout_workers = 1
        config.evaluation_num_workers = 1
        config.evaluation_interval = 1
        config.env = 'fault_env' if not multi_agent else 'multi_agent_fault_env'
        config.evaluation_config = AlgorithmConfig.overrides(recreate_failed_workers=True, delay_between_worker_restarts_s=0, env_config={'bad_indices': [1], 'failure_start_count': 3, 'failure_stop_count': 4, 'counter': COUNTER_NAME}, **dict(policy_mapping_fn=lambda aid, episode, worker, **kwargs: 'This is the eval mapping fn' if episode is None else 'main' if episode.episode_id % 2 == aid else 'p{}'.format(np.random.choice([0, 1]))) if multi_agent else {})
        for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
            ray.wait([counter.reset.remote()])
            a = config.build()
            for _ in range(2):
                a.train()
                wait_for_restore()
                a.train()
                self.assertEqual(a.workers.num_healthy_remote_workers(), 1)
                self.assertEqual(a.evaluation_workers.num_healthy_remote_workers(), 1)
                if multi_agent:
                    test = a.evaluation_workers.foreach_worker(lambda w: w.policy_mapping_fn(0, None, None))
                    self.assertEqual(test[0], 'This is the eval mapping fn')
            a.stop()

    def test_fatal(self):
        if False:
            print('Hello World!')
        self._do_test_fault_fatal(PGConfig().training(optimizer={}))

    def test_async_samples(self):
        if False:
            while True:
                i = 10
        self._do_test_fault_ignore(ImpalaConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).resources(num_gpus=0))

    def test_sync_replay(self):
        if False:
            while True:
                i = 10
        self._do_test_fault_ignore(DQNConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).reporting(min_sample_timesteps_per_iteration=1))

    def test_multi_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self._do_test_fault_ignore(PPOConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).training(train_batch_size=10, sgd_minibatch_size=1, num_sgd_iter=1))

    def test_sync_samples(self):
        if False:
            for i in range(10):
                print('nop')
        self._do_test_fault_ignore(PPOConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).training(optimizer={}))

    def test_eval_workers_failing_ignore(self):
        if False:
            return 10
        self._do_test_fault_ignore(PPOConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).training(model={'fcnet_hiddens': [4]}), fail_eval=True)

    def test_recreate_eval_workers_parallel_to_training_w_actor_manager(self):
        if False:
            while True:
                i = 10
        config = PGConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).evaluation(evaluation_num_workers=1, enable_async_evaluation=True, evaluation_parallel_to_training=True, evaluation_duration='auto').training(model={'fcnet_hiddens': [4]})
        self._do_test_fault_fatal_but_recreate(config)

    def test_recreate_eval_workers_parallel_to_training_w_actor_manager_and_multi_agent(self):
        if False:
            for i in range(10):
                print('nop')
        config = PGConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker).multi_agent(policies={'main', 'p0', 'p1'}, policy_mapping_fn=lambda aid, episode, worker, **kwargs: 'main' if episode.episode_id % 2 == aid else 'p{}'.format(np.random.choice([0, 1]))).evaluation(evaluation_num_workers=1, enable_async_evaluation=True, evaluation_parallel_to_training=True, evaluation_duration='auto').training(model={'fcnet_hiddens': [4]})
        self._do_test_fault_fatal_but_recreate(config, multi_agent=True)

    def test_eval_workers_failing_fatal(self):
        if False:
            i = 10
            return i + 15
        self._do_test_fault_fatal(PPOConfig().training(model={'fcnet_hiddens': [4]}), fail_eval=True)

    def test_workers_fatal_but_recover(self):
        if False:
            for i in range(10):
                print('nop')
        COUNTER_NAME = 'test_workers_fatal_but_recover'
        counter = Counter.options(name=COUNTER_NAME).remote()
        config = PGConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker, num_rollout_workers=2, rollout_fragment_length=16).training(train_batch_size=32, model={'fcnet_hiddens': [4]}).environment(env='fault_env', env_config={'bad_indices': [1, 2], 'failure_start_count': 3, 'failure_stop_count': 4, 'counter': COUNTER_NAME}).fault_tolerance(recreate_failed_workers=True, delay_between_worker_restarts_s=0)
        for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
            ray.wait([counter.reset.remote()])
            a = config.build()
            self.assertEqual(a.workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.workers.num_remote_worker_restarts(), 0)
            a.train()
            wait_for_restore()
            a.train()
            self.assertEqual(a.workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.workers.num_remote_worker_restarts(), 2)

    def test_policies_are_restored_on_recovered_worker(self):
        if False:
            i = 10
            return i + 15
        COUNTER_NAME = 'test_policies_are_restored_on_recovered_worker'
        counter = Counter.options(name=COUNTER_NAME).remote()
        config = PGConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker, num_rollout_workers=2, rollout_fragment_length=16).training(train_batch_size=32, model={'fcnet_hiddens': [4]}).environment(env='multi_agent_fault_env', env_config={'bad_indices': [1, 2], 'failure_start_count': 3, 'failure_stop_count': 4, 'counter': COUNTER_NAME}).evaluation(evaluation_num_workers=1, evaluation_interval=1, evaluation_config=PGConfig.overrides(recreate_failed_workers=True, restart_failed_sub_environments=False, env_config={'evaluation': True, 'bad_indices': [1], 'failure_start_count': 3, 'failure_stop_count': 4, 'counter': COUNTER_NAME})).callbacks(AddPolicyCallback).fault_tolerance(recreate_failed_workers=True, delay_between_worker_restarts_s=0)
        for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
            ray.wait([counter.reset.remote()])
            a = config.build()
            self.assertIsNotNone(a.get_policy('test_policy'))
            self.assertEqual(a.workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.workers.num_remote_worker_restarts(), 0)
            self.assertEqual(a.evaluation_workers.num_healthy_remote_workers(), 1)
            self.assertEqual(a.evaluation_workers.num_remote_worker_restarts(), 0)
            a.train()
            wait_for_restore()
            a.train()
            self.assertEqual(a.workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.workers.num_remote_worker_restarts(), 2)
            self.assertEqual(a.evaluation_workers.num_healthy_remote_workers(), 1)
            self.assertEqual(a.evaluation_workers.num_remote_worker_restarts(), 1)

            def has_test_policy(w):
                if False:
                    return 10
                return 'test_policy' in w.policy_map
            self.assertTrue(all(a.workers.foreach_worker(has_test_policy, local_worker=False)))
            self.assertTrue(all(a.evaluation_workers.foreach_worker(has_test_policy, local_worker=False)))

    def test_eval_workers_fault_but_recover(self):
        if False:
            while True:
                i = 10
        COUNTER_NAME = 'test_eval_workers_fault_but_recover'
        counter = Counter.options(name=COUNTER_NAME).remote()
        config = PGConfig().rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker, num_rollout_workers=2, rollout_fragment_length=16).training(train_batch_size=32, model={'fcnet_hiddens': [4]}).environment(env='fault_env').evaluation(evaluation_num_workers=2, evaluation_interval=1, evaluation_config=PGConfig.overrides(env_config={'evaluation': True, 'p_terminated': 0.0, 'max_episode_len': 20, 'bad_indices': [1, 2], 'failure_start_count': 3, 'failure_stop_count': 4, 'counter': COUNTER_NAME})).fault_tolerance(recreate_failed_workers=True, delay_between_worker_restarts_s=0)
        for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
            ray.wait([counter.reset.remote()])
            a = config.build()
            self.assertEqual(a.evaluation_workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.evaluation_workers.num_remote_worker_restarts(), 0)
            a.train()
            wait_for_restore()
            a.train()
            self.assertEqual(a.evaluation_workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.evaluation_workers.num_remote_worker_restarts(), 2)

    def test_worker_recover_with_hanging_workers(self):
        if False:
            return 10
        COUNTER_NAME = 'test_eval_workers_fault_but_recover'
        counter = Counter.options(name=COUNTER_NAME).remote()
        config = ImpalaConfig().resources(num_gpus=0).rollouts(env_runner_cls=ForwardHealthCheckToEnvWorker, num_rollout_workers=3, rollout_fragment_length=16).training(train_batch_size=32, model={'fcnet_hiddens': [4]}).reporting(min_time_s_per_iteration=0.5, metrics_episode_collection_timeout_s=1).environment(env='fault_env', env_config={'evaluation': True, 'p_terminated': 0.0, 'max_episode_len': 20, 'bad_indices': [1, 2], 'failure_start_count': 3, 'failure_stop_count': 4, 'counter': COUNTER_NAME, 'init_delay': 3600, 'init_delay_indices': [2], 'step_delay': 3600, 'step_delay_indices': [3]}).fault_tolerance(recreate_failed_workers=True, worker_health_probe_timeout_s=0.01, worker_restore_timeout_s=5, delay_between_worker_restarts_s=0)
        for _ in framework_iterator(config, frameworks=('tf2', 'torch')):
            ray.wait([counter.reset.remote()])
            a = config.build()
            self.assertEqual(a.workers.num_healthy_remote_workers(), 3)
            self.assertEqual(a.workers.num_remote_worker_restarts(), 0)
            a.train()
            wait_for_restore(num_restarting_allowed=1)
            a.train()
            self.assertEqual(a.workers.num_healthy_remote_workers(), 2)
            self.assertEqual(a.workers.num_remote_worker_restarts(), 1)

    def test_eval_workers_on_infinite_episodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests whether eval workers warn appropriately after some episode timeout.'
        config = PPOConfig().environment(env=RandomEnv, env_config={'p_terminated': 0.0}).reporting(metrics_episode_collection_timeout_s=5.0).evaluation(evaluation_num_workers=2, evaluation_interval=1, evaluation_sample_timeout_s=5.0)
        algo = config.build()
        results = algo.train()
        self.assertTrue(np.isnan(results['evaluation']['episode_reward_mean']))
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))