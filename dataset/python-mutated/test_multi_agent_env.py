import gymnasium as gym
import numpy as np
import random
import tree
import unittest
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent, MultiAgentEnvWrapper
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import get_global_worker, RolloutWorker
from ray.rllib.evaluation.tests.test_rollout_worker import MockPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole, BasicMultiAgent, EarlyDoneMultiAgent, FlexAgentsMultiAgent, RoundRobinMultiAgent, SometimesZeroAgentsMultiAgent
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.tests.test_nested_observation_spaces import NestedMultiAgentEnv
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.test_utils import check

class TestMultiAgentEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_basic_mock(self):
        if False:
            return 10
        env = BasicMultiAgent(4)
        (obs, info) = env.reset()
        self.assertEqual(obs, {0: 0, 1: 0, 2: 0, 3: 0})
        for _ in range(24):
            (obs, rew, done, truncated, info) = env.step({0: 0, 1: 0, 2: 0, 3: 0})
            self.assertEqual(obs, {0: 0, 1: 0, 2: 0, 3: 0})
            self.assertEqual(rew, {0: 1, 1: 1, 2: 1, 3: 1})
            self.assertEqual(done, {0: False, 1: False, 2: False, 3: False, '__all__': False})
        (obs, rew, done, truncated, info) = env.step({0: 0, 1: 0, 2: 0, 3: 0})
        self.assertEqual(done, {0: True, 1: True, 2: True, 3: True, '__all__': True})

    def test_round_robin_mock(self):
        if False:
            while True:
                i = 10
        env = RoundRobinMultiAgent(2)
        (obs, info) = env.reset()
        self.assertEqual(obs, {0: 0})
        for _ in range(5):
            (obs, rew, done, truncated, info) = env.step({0: 0})
            self.assertEqual(obs, {1: 0})
            self.assertEqual(done['__all__'], False)
            (obs, rew, done, truncated, info) = env.step({1: 0})
            self.assertEqual(obs, {0: 0})
            self.assertEqual(done['__all__'], False)
        (obs, rew, done, truncated, info) = env.step({0: 0})
        self.assertEqual(done['__all__'], True)

    def test_no_reset_until_poll(self):
        if False:
            return 10
        env = MultiAgentEnvWrapper(lambda v: BasicMultiAgent(2), [], 1)
        self.assertFalse(env.get_sub_environments()[0].resetted)
        env.poll()
        self.assertTrue(env.get_sub_environments()[0].resetted)

    def test_vectorize_basic(self):
        if False:
            print('Hello World!')
        env = MultiAgentEnvWrapper(lambda v: BasicMultiAgent(2), [], 2)
        (obs, rew, terminateds, truncateds, _, _) = env.poll()
        self.assertEqual(obs, {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        self.assertEqual(rew, {0: {}, 1: {}})
        self.assertEqual(terminateds, {0: {'__all__': False}, 1: {'__all__': False}})
        self.assertEqual(truncateds, terminateds)
        for _ in range(24):
            env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
            (obs, rew, terminateds, truncateds, _, _) = env.poll()
            self.assertEqual(obs, {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
            self.assertEqual(rew, {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}})
            self.assertEqual(terminateds, {0: {0: False, 1: False, '__all__': False}, 1: {0: False, 1: False, '__all__': False}})
            self.assertEqual(truncateds, terminateds)
        env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        (obs, rew, terminateds, truncateds, _, _) = env.poll()
        self.assertEqual(terminateds, {0: {0: True, 1: True, '__all__': True}, 1: {0: True, 1: True, '__all__': True}})
        self.assertEqual(truncateds, terminateds)
        self.assertRaises(ValueError, lambda : env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}))
        (init_obs, init_infos) = env.try_reset(0)
        self.assertEqual(init_obs, {0: {0: 0, 1: 0}})
        self.assertEqual(init_infos, {0: {0: {}, 1: {}}})
        (init_obs, init_infos) = env.try_reset(1)
        self.assertEqual(init_obs, {1: {0: 0, 1: 0}})
        self.assertEqual(init_infos, {1: {0: {}, 1: {}}})
        env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        (obs, rew, terminateds, truncateds, _, _) = env.poll()
        self.assertEqual(obs, {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        self.assertEqual(rew, {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}})
        self.assertEqual(terminateds, {0: {0: False, 1: False, '__all__': False}, 1: {0: False, 1: False, '__all__': False}})
        self.assertEqual(truncateds, terminateds)

    def test_vectorize_round_robin(self):
        if False:
            i = 10
            return i + 15
        env = MultiAgentEnvWrapper(lambda v: RoundRobinMultiAgent(2), [], 2)
        (obs, rew, terminateds, truncateds, _, _) = env.poll()
        self.assertEqual(obs, {0: {0: 0}, 1: {0: 0}})
        self.assertEqual(rew, {0: {}, 1: {}})
        self.assertEqual(truncateds, {0: {'__all__': False}, 1: {'__all__': False}})
        env.send_actions({0: {0: 0}, 1: {0: 0}})
        (obs, rew, terminateds, truncateds, _, _) = env.poll()
        self.assertEqual(obs, {0: {1: 0}, 1: {1: 0}})
        self.assertEqual(truncateds, {0: {'__all__': False, 1: False}, 1: {'__all__': False, 1: False}})
        env.send_actions({0: {1: 0}, 1: {1: 0}})
        (obs, rew, terminateds, truncateds, _, _) = env.poll()
        self.assertEqual(obs, {0: {0: 0}, 1: {0: 0}})
        self.assertEqual(truncateds, {0: {'__all__': False, 0: False}, 1: {'__all__': False, 0: False}})

    def test_multi_agent_sample(self):
        if False:
            i = 10
            return i + 15

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if False:
                while True:
                    i = 10
            return 'p{}'.format(agent_id % 2)
        ev = RolloutWorker(env_creator=lambda _: BasicMultiAgent(5), default_policy_class=MockPolicy, config=AlgorithmConfig().rollouts(rollout_fragment_length=50, num_rollout_workers=0).multi_agent(policies={'p0', 'p1'}, policy_mapping_fn=policy_mapping_fn))
        batch = ev.sample()
        self.assertEqual(batch.count, 50)
        self.assertEqual(batch.policy_batches['p0'].count, 150)
        self.assertEqual(batch.policy_batches['p1'].count, 100)
        self.assertEqual(batch.policy_batches['p0']['t'].tolist(), list(range(25)) * 6)

    def test_multi_agent_sample_sync_remote(self):
        if False:
            for i in range(10):
                print('nop')
        ev = RolloutWorker(env_creator=lambda _: BasicMultiAgent(5), default_policy_class=MockPolicy, config=AlgorithmConfig().rollouts(rollout_fragment_length=50, num_rollout_workers=0, num_envs_per_worker=4, remote_worker_envs=True, remote_env_batch_wait_ms=99999999).multi_agent(policies={'p0', 'p1'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'p{}'.format(agent_id % 2)))
        batch = ev.sample()
        self.assertEqual(batch.count, 200)

    def test_multi_agent_sample_async_remote(self):
        if False:
            return 10
        ev = RolloutWorker(env_creator=lambda _: BasicMultiAgent(5), default_policy_class=MockPolicy, config=AlgorithmConfig().rollouts(rollout_fragment_length=50, num_rollout_workers=0, num_envs_per_worker=4, remote_worker_envs=True).multi_agent(policies={'p0', 'p1'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'p{}'.format(agent_id % 2)))
        batch = ev.sample()
        self.assertEqual(batch.count, 200)

    def test_sample_from_early_done_env(self):
        if False:
            return 10
        ev = RolloutWorker(env_creator=lambda _: EarlyDoneMultiAgent(), default_policy_class=MockPolicy, config=AlgorithmConfig().rollouts(rollout_fragment_length=1, num_rollout_workers=0, batch_mode='complete_episodes').multi_agent(policies={'p0', 'p1'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'p{}'.format(agent_id % 2)))
        ma_batch = ev.sample()
        ag0_ts = ma_batch.policy_batches['p0']['t']
        ag1_ts = ma_batch.policy_batches['p1']['t']
        self.assertTrue(np.all(np.abs(ag0_ts[:-1] - ag1_ts[:-1]) == 1.0))
        self.assertTrue(ag0_ts[-1] == ag1_ts[-1])

    def test_multi_agent_with_flex_agents(self):
        if False:
            i = 10
            return i + 15
        register_env('flex_agents_multi_agent', lambda _: FlexAgentsMultiAgent())
        config = PPOConfig().environment('flex_agents_multi_agent').rollouts(num_rollout_workers=0).framework('tf').training(train_batch_size=50, sgd_minibatch_size=50, num_sgd_iter=1)
        algo = config.build()
        for i in range(10):
            result = algo.train()
            print('Iteration {}, reward {}, timesteps {}'.format(i, result['episode_reward_mean'], result['timesteps_total']))
        algo.stop()

    def test_multi_agent_with_sometimes_zero_agents_observing(self):
        if False:
            return 10
        register_env('sometimes_zero_agents', lambda _: SometimesZeroAgentsMultiAgent(num=4))
        config = PPOConfig().environment('sometimes_zero_agents').rollouts(num_rollout_workers=0, enable_connectors=True).framework('tf')
        algo = config.build()
        for i in range(4):
            result = algo.train()
            print('Iteration {}, reward {}, timesteps {}'.format(i, result['episode_reward_mean'], result['timesteps_total']))
        algo.stop()

    def test_multi_agent_sample_round_robin(self):
        if False:
            return 10
        ev = RolloutWorker(env_creator=lambda _: RoundRobinMultiAgent(5, increment_obs=True), default_policy_class=MockPolicy, config=AlgorithmConfig().rollouts(rollout_fragment_length=50, num_rollout_workers=0).multi_agent(policies={'p0'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'p0'))
        batch = ev.sample()
        self.assertEqual(batch.count, 50)
        self.assertEqual(batch.policy_batches['p0'].count, 42)
        check(batch.policy_batches['p0']['obs'][:10], one_hot(np.array([0, 1, 2, 3, 4] * 2), 10))
        check(batch.policy_batches['p0']['new_obs'][:10], one_hot(np.array([1, 2, 3, 4, 5] * 2), 10))
        self.assertEqual(batch.policy_batches['p0']['rewards'].tolist()[:10], [100, 100, 100, 100, 0] * 2)
        self.assertEqual(batch.policy_batches['p0']['terminateds'].tolist()[:10], [False, False, False, False, True] * 2)
        self.assertEqual(batch.policy_batches['p0']['truncateds'].tolist()[:10], [False, False, False, False, True] * 2)
        self.assertEqual(batch.policy_batches['p0']['t'].tolist()[:10], [4, 9, 14, 19, 24, 5, 10, 15, 20, 25])

    def test_custom_rnn_state_values(self):
        if False:
            print('Hello World!')
        h = {'some': {'here': np.array([1.0, 2.0, 3.0])}}

        class StatefulPolicy(RandomPolicy):

            def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, episodes=None, explore=True, timestep=None, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                obs_shape = (len(obs_batch),)
                actions = np.zeros(obs_shape, dtype=np.int32)
                states = tree.map_structure(lambda x: np.ones(obs_shape + x.shape) * x, h)
                return (actions, [states], {})

            def get_initial_state(self):
                if False:
                    while True:
                        i = 10
                return [{}]

            def is_recurrent(self):
                if False:
                    while True:
                        i = 10
                return True
        ev = RolloutWorker(env_creator=lambda _: gym.make('CartPole-v1'), default_policy_class=StatefulPolicy, config=AlgorithmConfig().rollouts(rollout_fragment_length=5, num_rollout_workers=0).training(model={'max_seq_len': 1}))
        batch = ev.sample()
        batch = convert_ma_batch_to_sample_batch(batch)
        self.assertEqual(batch.count, 5)
        check(batch['state_in_0'][0], {})
        check(batch['state_out_0'][0], h)
        for i in range(1, 5):
            check(batch['state_in_0'][i], h)
            check(batch['state_out_0'][i], h)

    def test_returning_model_based_rollouts_data(self):
        if False:
            return 10

        class ModelBasedPolicy(DQNTFPolicy):

            def compute_actions_from_input_dict(self, input_dict, explore=None, timestep=None, episodes=None, **kwargs):
                if False:
                    while True:
                        i = 10
                obs_batch = input_dict['obs']
                if episodes is not None:
                    env_id = episodes[0].env_id
                    fake_eps = Episode(episodes[0].policy_map, episodes[0].policy_mapping_fn, lambda : None, lambda x: None, env_id)
                    builder = get_global_worker().sampler.sample_collector
                    agent_id = 'extra_0'
                    policy_id = 'p1'
                    builder.add_init_obs(episode=fake_eps, agent_id=agent_id, policy_id=policy_id, env_id=env_id, init_obs=obs_batch[0], init_infos={})
                    for t in range(4):
                        builder.add_action_reward_next_obs(episode_id=fake_eps.episode_id, agent_id=agent_id, env_id=env_id, policy_id=policy_id, agent_done=t == 3, values=dict(t=t, actions=0, rewards=0, terminateds=False, truncateds=t == 3, infos={}, new_obs=obs_batch[0]))
                    batch = builder.postprocess_episode(episode=fake_eps, build=True)
                    episodes[0].add_extra_batch(batch)
                return ([0] * len(obs_batch), [], {})
        ev = RolloutWorker(env_creator=lambda _: MultiAgentCartPole({'num_agents': 2}), default_policy_class=ModelBasedPolicy, config=DQNConfig().framework('tf').rollouts(rollout_fragment_length=5, num_rollout_workers=0, enable_connectors=False).multi_agent(policies={'p0', 'p1'}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'p0'))
        batch = ev.sample()
        self.assertEqual(batch.count, 5)
        self.assertEqual(batch.policy_batches['p0'].count, 10)
        self.assertEqual(batch.policy_batches['p1'].count, 20)

    def test_train_multi_agent_cartpole_single_policy(self):
        if False:
            return 10
        n = 10
        register_env('multi_agent_cartpole', lambda _: MultiAgentCartPole({'num_agents': n}))
        config = PPOConfig().environment('multi_agent_cartpole').rollouts(num_rollout_workers=0).framework('tf')
        algo = config.build()
        for i in range(50):
            result = algo.train()
            print('Iteration {}, reward {}, timesteps {}'.format(i, result['episode_reward_mean'], result['timesteps_total']))
            if result['episode_reward_mean'] >= 50 * n:
                algo.stop()
                return
        raise Exception('failed to improve reward')

    def test_train_multi_agent_cartpole_multi_policy(self):
        if False:
            for i in range(10):
                print('nop')
        n = 10
        register_env('multi_agent_cartpole', lambda _: MultiAgentCartPole({'num_agents': n}))

        def gen_policy():
            if False:
                while True:
                    i = 10
            config = PPOConfig.overrides(gamma=random.choice([0.5, 0.8, 0.9, 0.95, 0.99]), lr=random.choice([0.001, 0.002, 0.003]))
            return PolicySpec(config=config)
        config = PPOConfig().environment('multi_agent_cartpole').rollouts(num_rollout_workers=0).multi_agent(policies={'policy_1': gen_policy(), 'policy_2': gen_policy()}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'policy_1').framework('tf').training(train_batch_size=50, sgd_minibatch_size=50, num_sgd_iter=1)
        algo = config.build()
        for i in range(10):
            result = algo.train()
            print('Iteration {}, reward {}, timesteps {}'.format(i, result['episode_reward_mean'], result['timesteps_total']))
        self.assertTrue(algo.compute_single_action([0, 0, 0, 0], policy_id='policy_1') in [0, 1])
        self.assertTrue(algo.compute_single_action([0, 0, 0, 0], policy_id='policy_2') in [0, 1])
        self.assertRaisesRegex(KeyError, 'not found in PolicyMap', lambda : algo.compute_single_action([0, 0, 0, 0], policy_id='policy_3'))

    def test_space_in_preferred_format(self):
        if False:
            return 10
        env = NestedMultiAgentEnv()
        action_space_in_preferred_format = env._check_if_action_space_maps_agent_id_to_sub_space()
        obs_space_in_preferred_format = env._check_if_obs_space_maps_agent_id_to_sub_space()
        assert action_space_in_preferred_format, 'Act space is not in preferred format.'
        assert obs_space_in_preferred_format, 'Obs space is not in preferred format.'
        env2 = make_multi_agent('CartPole-v1')()
        action_spaces_in_preferred_format = env2._check_if_action_space_maps_agent_id_to_sub_space()
        obs_space_in_preferred_format = env2._check_if_obs_space_maps_agent_id_to_sub_space()
        assert not action_spaces_in_preferred_format, 'Action space should not be in preferred format but is.'
        assert not obs_space_in_preferred_format, 'Observation space should not be in preferred format but is.'

    def test_spaces_sample_contain_in_preferred_format(self):
        if False:
            i = 10
            return i + 15
        env = NestedMultiAgentEnv()
        obs = env.observation_space_sample()
        assert env.observation_space_contains(obs), 'Observation space does not contain obs'
        action = env.action_space_sample()
        assert env.action_space_contains(action), 'Action space does not contain action'

    def test_spaces_sample_contain_not_in_preferred_format(self):
        if False:
            return 10
        env = make_multi_agent('CartPole-v1')({'num_agents': 2})
        obs = env.observation_space_sample()
        assert env.observation_space_contains(obs), 'Observation space does not contain obs'
        action = env.action_space_sample()
        assert env.action_space_contains(action), 'Action space does not contain action'
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))