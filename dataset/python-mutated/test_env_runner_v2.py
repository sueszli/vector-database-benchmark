import unittest
import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.connector import ActionConnector, ConnectorContext
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.examples.env.debug_counter_env import DebugCounterEnv
from ray.rllib.examples.env.multi_agent import BasicMultiAgent, GuessTheNumberGame
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from ray.rllib.utils.test_utils import check
register_env('basic_multiagent', lambda _: BasicMultiAgent(2))

class TestEnvRunnerV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        ray.init()

        class AlternatePolicyMapper:

            def __init__(self):
                if False:
                    return 10
                self.policies = ['one', 'two']
                self.next = 0

            def map(self):
                if False:
                    while True:
                        i = 10
                p = self.policies[self.next]
                self.next = 1 - self.next
                return p
        cls.mapper = AlternatePolicyMapper()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_sample_batch_rollout_single_agent_env(self):
        if False:
            while True:
                i = 10
        config = PPOConfig().environment(DebugCounterEnv).framework('torch').training(train_batch_size=200).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True)
        algo = PPO(config)
        rollout_worker = algo.workers.local_worker()
        sample_batch = rollout_worker.sample()
        sample_batch = convert_ma_batch_to_sample_batch(sample_batch)
        self.assertEqual(sample_batch['t'][0], 0)
        self.assertEqual(sample_batch.env_steps(), 200)
        self.assertEqual(sample_batch.agent_steps(), 200)

    def test_sample_batch_rollout_multi_agent_env(self):
        if False:
            return 10
        config = PPOConfig().environment('basic_multiagent').framework('torch').training(train_batch_size=200).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True)
        algo = PPO(config)
        rollout_worker = algo.workers.local_worker()
        sample_batch = rollout_worker.sample()
        self.assertEqual(sample_batch.env_steps(), 200)
        self.assertEqual(sample_batch.agent_steps(), 400)

    def test_guess_the_number_multi_agent(self):
        if False:
            for i in range(10):
                print('nop')
        'This test will test env runner in the game of GuessTheNumberGame.\n\n        The policies are chosen to be deterministic, so that we can test for an\n        expected reward. Agent 1 will always pick 1, and agent 2 will always guess that\n        the picked number is higher than 1. The game will end when the picked number is\n        1, and agent 1 will win. The reward will be 100 for winning, and 1 for each\n        step that the game is dragged on for. So the expected reward for agent 1 is 100\n        + 19 = 119. 19 is the number of steps that the game will last for agent 1\n        before it wins or loses.\n        '
        register_env('env_under_test', lambda config: GuessTheNumberGame(config))

        def mapping_fn(agent_id, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return 'pol1' if agent_id == 0 else 'pol2'

        class PickOne(RandomPolicy):
            """This policy will always pick 1."""

            def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
                if False:
                    while True:
                        i = 10
                return ([np.array([2, 1])] * len(obs_batch), [], {})

        class GuessHigherThanOne(RandomPolicy):
            """This policy will guess that the picked number is higher than 1."""

            def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                return ([np.array([1, 1])] * len(obs_batch), [], {})
        config = PPOConfig().framework('torch').environment('env_under_test', disable_env_checking=True).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True, rollout_fragment_length=100).multi_agent(policies={'pol1': PolicySpec(policy_class=PickOne), 'pol2': PolicySpec(policy_class=GuessHigherThanOne)}, policy_mapping_fn=mapping_fn).rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs={'pol1': SingleAgentRLModuleSpec(module_class=RandomRLModule), 'pol2': SingleAgentRLModuleSpec(module_class=RandomRLModule)})).debugging(seed=42)
        algo = PPO(config)
        rollout_worker = algo.workers.local_worker()
        sample_batch = rollout_worker.sample()
        pol1_batch = sample_batch.policy_batches['pol1']
        check(pol1_batch['rewards'], 119 * np.ones_like(pol1_batch['rewards']))
        check(len(set(pol1_batch['eps_id'])), len(pol1_batch['eps_id']))
        pol2_batch = sample_batch.policy_batches['pol2']
        check(len(set(pol2_batch['eps_id'])) * 19, len(pol2_batch['eps_id']))

    def test_inference_batches_are_grouped_by_policy(self):
        if False:
            while True:
                i = 10

        class RandomPolicyOne(RandomPolicy):

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(*args, **kwargs)
                self.view_requirements['rewards'].used_for_compute_actions = True
                self.view_requirements['terminateds'].used_for_compute_actions = True

        class RandomPolicyTwo(RandomPolicy):

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                super().__init__(*args, **kwargs)
                self.view_requirements['rewards'].used_for_compute_actions = False
                self.view_requirements['terminateds'].used_for_compute_actions = False
        config = PPOConfig().environment('basic_multiagent').framework('torch').training(train_batch_size=200).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True).multi_agent(policies={'one': PolicySpec(policy_class=RandomPolicyOne), 'two': PolicySpec(policy_class=RandomPolicyTwo)}, policy_mapping_fn=lambda *args, **kwargs: self.mapper.map(), policies_to_train=['one'], count_steps_by='agent_steps').rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs={'one': SingleAgentRLModuleSpec(module_class=RandomRLModule), 'two': SingleAgentRLModuleSpec(module_class=RandomRLModule)}))
        algo = PPO(config)
        local_worker = algo.workers.local_worker()
        env = local_worker.env
        (obs, rewards, terminateds, truncateds, infos) = local_worker.env.step({0: env.action_space.sample(), 1: env.action_space.sample()})
        env_id = 0
        env_runner = local_worker.sampler._env_runner_obj
        env_runner.create_episode(env_id)
        (_, to_eval, _) = env_runner._process_observations({0: obs}, {0: rewards}, {0: terminateds}, {0: truncateds}, {0: infos})
        self.assertTrue('one' in to_eval)
        self.assertEqual(len(to_eval['one']), 1)
        self.assertTrue('two' in to_eval)
        self.assertEqual(len(to_eval['two']), 1)

    def test_action_connector_gets_raw_input_dict(self):
        if False:
            for i in range(10):
                print('nop')

        class CheckInputDictActionConnector(ActionConnector):

            def __call__(self, ac_data):
                if False:
                    for i in range(10):
                        print('nop')
                assert ac_data.input_dict, 'raw input dict should be available'
                return ac_data

        class AddActionConnectorCallbacks(DefaultCallbacks):

            def on_create_policy(self, *, policy_id, policy) -> None:
                if False:
                    print('Hello World!')
                policy.action_connectors.append(CheckInputDictActionConnector(ConnectorContext.from_policy(policy)))
        config = PPOConfig().environment('basic_multiagent').framework('torch').training(train_batch_size=200).callbacks(callbacks_class=AddActionConnectorCallbacks).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True)
        algo = PPO(config)
        rollout_worker = algo.workers.local_worker()
        _ = rollout_worker.sample()

    def test_start_episode(self):
        if False:
            while True:
                i = 10
        config = PPOConfig().environment('basic_multiagent').framework('torch').training(train_batch_size=200).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True).multi_agent(policies={'one': PolicySpec(policy_class=RandomPolicy), 'two': PolicySpec(policy_class=RandomPolicy)}, policy_mapping_fn=lambda *args, **kwargs: self.mapper.map(), policies_to_train=['one'], count_steps_by='agent_steps').rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs={'one': SingleAgentRLModuleSpec(module_class=RandomRLModule), 'two': SingleAgentRLModuleSpec(module_class=RandomRLModule)}))
        algo = PPO(config)
        local_worker = algo.workers.local_worker()
        env_runner = local_worker.sampler._env_runner_obj
        self.assertEqual(env_runner._active_episodes.get(0), None)
        env_runner.step()
        self.assertEqual(env_runner._active_episodes[0].total_env_steps, 0)
        self.assertEqual(env_runner._active_episodes[0].total_agent_steps, 0)
        env_runner.step()
        self.assertEqual(env_runner._active_episodes[0].total_env_steps, 1)
        self.assertEqual(env_runner._active_episodes[0].total_agent_steps, 2)

    def test_env_runner_output(self):
        if False:
            i = 10
            return i + 15
        config = PPOConfig().environment('basic_multiagent').framework('torch').training(train_batch_size=200).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True).multi_agent(policies={'one': PolicySpec(policy_class=RandomPolicy), 'two': PolicySpec(policy_class=RandomPolicy)}, policy_mapping_fn=lambda *args, **kwargs: self.mapper.map(), policies_to_train=['one'], count_steps_by='agent_steps').rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs={'one': SingleAgentRLModuleSpec(module_class=RandomRLModule), 'two': SingleAgentRLModuleSpec(module_class=RandomRLModule)}))
        algo = PPO(config)
        local_worker = algo.workers.local_worker()
        env_runner = local_worker.sampler._env_runner_obj
        outputs = []
        while not outputs:
            outputs = env_runner.step()
        self.assertEqual(len(outputs), 1)
        self.assertTrue(len(list(outputs[0].agent_rewards.keys())) == 2)

    def test_env_error(self):
        if False:
            for i in range(10):
                print('nop')

        class CheckErrorCallbacks(DefaultCallbacks):

            def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
                if False:
                    return 10
                assert isinstance(episode, Exception)
        config = PPOConfig().environment('basic_multiagent').framework('torch').training(train_batch_size=200).rollouts(num_envs_per_worker=1, num_rollout_workers=0, enable_connectors=True).multi_agent(policies={'one': PolicySpec(policy_class=RandomPolicy), 'two': PolicySpec(policy_class=RandomPolicy)}, policy_mapping_fn=lambda *args, **kwargs: self.mapper.map(), policies_to_train=['one'], count_steps_by='agent_steps').rl_module(rl_module_spec=MultiAgentRLModuleSpec(module_specs={'one': SingleAgentRLModuleSpec(module_class=RandomRLModule), 'two': SingleAgentRLModuleSpec(module_class=RandomRLModule)})).callbacks(callbacks_class=CheckErrorCallbacks)
        algo = PPO(config)
        local_worker = algo.workers.local_worker()
        env_runner = local_worker.sampler._env_runner_obj
        env_runner.step()
        env_runner.step()
        (active_envs, to_eval, outputs) = env_runner._process_observations(unfiltered_obs={0: AttributeError('mock error')}, rewards={0: {}}, terminateds={0: {'__all__': True}}, truncateds={0: {'__all__': False}}, infos={0: {}})
        self.assertEqual(active_envs, {0})
        self.assertTrue(to_eval)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(isinstance(outputs[0], RolloutMetrics))
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))