import gymnasium as gym
import numpy as np
import unittest
from gymnasium.core import ActType, ObsType
from typing import Any, Dict, Optional, SupportsFloat, Tuple
import ray
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

class TestEnv(gym.Env):

    def __init__(self):
        if False:
            print('Hello World!')
        self.observation_space = gym.spaces.Discrete(201)
        self.action_space = gym.spaces.Discrete(200)
        self.t = 0

    def reset(self, *, seed: Optional[int]=None, options=Optional[Dict[str, Any]]) -> Tuple[ObsType, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        self.t = 0
        return (0, {})

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if False:
            return 10
        self.t += 1
        if self.t == 200:
            is_terminated = True
        else:
            is_terminated = False
        return (self.t, self.t, is_terminated, False, {})

class TestSingelAgentEpisode(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests initialization of `SingleAgentEpisode`.\n\n        Three cases are tested:\n            1. Empty episode with default starting timestep.\n            2. Empty episode starting at `t_started=10`. This is only interesting\n                for ongoing episodes, where we do not want to carry on the stale\n                entries from the last rollout.\n            3. Initialization with pre-collected data.\n        '
        episode = SingleAgentEpisode()
        self.assertTrue(episode.t_started == episode.t == 0)
        episode = SingleAgentEpisode(t_started=10)
        self.assertTrue(episode.t == episode.t_started == 10)
        env = gym.make('CartPole-v1')
        observations = []
        rewards = []
        actions = []
        infos = []
        extra_model_outputs = []
        states = np.random.random(10)
        (init_obs, init_info) = env.reset()
        observations.append(init_obs)
        infos.append(init_info)
        for _ in range(100):
            action = env.action_space.sample()
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            extra_model_outputs.append({'extra_1': np.random.random()})
        episode = SingleAgentEpisode(observations=observations, actions=actions, rewards=rewards, infos=infos, states=states, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_outputs=extra_model_outputs)
        self.assertTrue(episode.t == episode.t_started == len(observations) - 1)

    def test_add_initial_observation(self):
        if False:
            print('Hello World!')
        'Tests adding initial observations and infos.\n\n        This test ensures that when initial observation and info are provided\n        the length of the lists are correct and the timestep is still at zero,\n        as the agent has not stepped, yet.\n        '
        episode = SingleAgentEpisode()
        env = gym.make('CartPole-v1')
        (obs, info) = env.reset()
        episode.add_initial_observation(initial_observation=obs, initial_info=info)
        self.assertTrue(len(episode.observations) == 1)
        self.assertTrue(len(episode.infos) == 1)
        self.assertTrue(episode.t == episode.t_started == 0)

    def test_add_timestep(self):
        if False:
            while True:
                i = 10
        'Tests if adding timestep data to a `SingleAgentEpisode` works.\n\n        Adding timestep data is the central part of collecting episode\n        dara. Here it is tested if adding to the internal data lists\n        works as intended and the timestep is increased during each step.\n        '
        episode = SingleAgentEpisode()
        env = gym.make('CartPole-v1')
        (obs, info) = env.reset(seed=0)
        episode.add_initial_observation(initial_observation=obs, initial_info=info)
        for i in range(100):
            action = env.action_space.sample()
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            episode.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)})
            if is_terminated or is_truncated:
                break
        self.assertTrue(episode.t == len(episode.observations) - 1 == i + 1)
        self.assertTrue(episode.t_started == 0)
        self.assertTrue(len(episode.actions) == len(episode.rewards) == len(episode.observations) - 1 == len(episode.infos) - 1 == i + 1)
        self.assertTrue(episode.is_terminated == is_terminated)
        self.assertTrue(episode.is_truncated == is_truncated)
        self.assertTrue(episode.is_done == is_terminated or is_truncated)

    def test_create_successor(self):
        if False:
            while True:
                i = 10
        "Tests creation of a scucessor of a `SingleAgentEpisode`.\n\n        This test makes sure that when creating a successor the successor's\n        data is coherent with the episode that should be succeeded.\n        Observation and info are available before each timestep; therefore\n        these data is carried over to the successor.\n        "
        episode_1 = SingleAgentEpisode()
        env = TestEnv()
        (init_obs, init_info) = env.reset()
        episode_1.add_initial_observation(initial_observation=init_obs, initial_info=init_info)
        for i in range(100):
            action = i
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            episode_1.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)})
        self.assertTrue(episode_1.t == 100)
        episode_2 = episode_1.create_successor()
        self.assertTrue(episode_1.id_ == episode_2.id_)
        self.assertTrue(episode_1.t == episode_2.t == episode_2.t_started)
        self.assertTrue(episode_1.observations[-1] == episode_2.observations[0])
        self.assertTrue(episode_1.infos[-1] == episode_2.infos[0])
        action = 100
        (obs, reward, is_terminated, is_truncated, info) = env.step(action)
        episode_2.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)})
        self.assertFalse(len(episode_1.observations) == len(episode_2.observations))

    def test_concat_episode(self):
        if False:
            while True:
                i = 10
        'Tests if concatenation of two `SingleAgentEpisode`s works.\n\n        This test ensures that concatenation of two episodes work. Note that\n        concatenation should only work for two chunks of the same episode, i.e.\n        they have the same `id_` and one should be the successor of the other.\n        It is also tested that concatenation fails, if timesteps do not match or\n        the episode to which we want to concatenate is already terminated.\n        '
        env = TestEnv()
        (init_obs, init_info) = env.reset()
        episode_1 = SingleAgentEpisode()
        episode_1.add_initial_observation(initial_observation=init_obs, initial_info=init_info)
        for i in range(100):
            action = i
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            episode_1.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)})
        episode_2 = episode_1.create_successor()
        for i in range(100, 200):
            action = i
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            episode_2.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)})
        self.assertTrue(episode_1.t == episode_2.t_started)
        self.assertTrue(episode_2.t == 200)
        episode_2.id_ = 'wrong'
        with self.assertRaises(AssertionError):
            episode_1.concat_episode(episode_2)
        episode_2.id_ = episode_1.id_
        episode_2.t += 1
        with self.assertRaises(AssertionError):
            episode_1.concat_episode(episode_2)
        episode_2.t -= 1
        episode_1.is_terminated = True
        with self.assertRaises(AssertionError):
            episode_1.concat_episode(episode_2)
        episode_1.is_terminated = False
        episode_1.concat_episode(episode_2)
        self.assertTrue(episode_1.t_started == 0)
        self.assertTrue(episode_1.t == 200)
        self.assertTrue(len(episode_1.actions) == len(episode_1.rewards) == len(episode_1.observations) - 1 == len(episode_1.infos) - 1 == 200)
        self.assertEqual(episode_2.observations[5], episode_1.observations[105])

    def test_get_and_from_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests, if a `SingleAgentEpisode` can be reconstructed form state.\n\n        This test constructs an episode, stores it to its dictionary state and\n        recreates a new episode form this state. Thereby it ensures that all\n        atttributes are indeed identical to the primer episode and the data is\n        complete.\n        '
        episode = SingleAgentEpisode()
        env = TestEnv()
        (init_obs, init_info) = env.reset()
        episode.add_initial_observation(initial_observation=init_obs, initial_info=init_info)
        for i in range(100):
            action = i
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            episode.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)})
        state = episode.get_state()
        episode_reproduced = SingleAgentEpisode.from_state(state)
        self.assertEqual(episode.id_, episode_reproduced.id_)
        self.assertEqual(episode.t, episode_reproduced.t)
        self.assertEqual(episode.t_started, episode_reproduced.t_started)
        self.assertEqual(episode.is_terminated, episode_reproduced.is_terminated)
        self.assertEqual(episode.is_truncated, episode_reproduced.is_truncated)
        self.assertListEqual(episode.observations, episode_reproduced.observations)
        self.assertListEqual(episode.actions, episode_reproduced.actions)
        self.assertListEqual(episode.rewards, episode_reproduced.rewards)
        self.assertListEqual(episode.infos, episode_reproduced.infos)
        self.assertEqual(episode.is_terminated, episode_reproduced.is_terminated)
        self.assertEqual(episode.is_truncated, episode_reproduced.is_truncated)
        self.assertEqual(episode.states, episode_reproduced.states)
        self.assertListEqual(episode.render_images, episode_reproduced.render_images)
        self.assertDictEqual(episode.extra_model_outputs, episode_reproduced.extra_model_outputs)
        state[1][1].pop()
        with self.assertRaises(AssertionError):
            episode_reproduced = SingleAgentEpisode.from_state(state)

    def test_to_and_from_sample_batch(self):
        if False:
            i = 10
            return i + 15
        'Tests if a `SingelAgentEpisode` can be reconstructed from a `SampleBatch`.\n\n        This tests converst an episode to a `SampleBatch` and reconstructs the\n        episode then from this sample batch. It is then tested, if all data is\n        complete.\n        Note that `extra_model_outputs` are defined by the user and as the format\n        in the episode from which a `SampleBatch` was created is unknown this\n        reconstruction would only work, if the user does take care of it (as a\n        counter example just rempve the index [0] from the `extra_model_output`).\n        '
        episode = SingleAgentEpisode()
        env = TestEnv()
        (init_obs, init_obs) = env.reset()
        episode.add_initial_observation(initial_observation=init_obs, initial_info=init_obs)
        for i in range(100):
            action = i
            (obs, reward, is_terminated, is_truncated, info) = env.step(action)
            episode.add_timestep(observation=obs, action=action, reward=reward, info=info, is_terminated=is_terminated, is_truncated=is_truncated, extra_model_output={'extra': np.random.random(1)[0]})
        batch = episode.to_sample_batch()
        episode_reproduced = SingleAgentEpisode.from_sample_batch(batch)
        self.assertEqual(episode.id_, episode_reproduced.id_)
        self.assertEqual(episode.t, episode_reproduced.t)
        self.assertEqual(episode.t_started, episode_reproduced.t_started)
        self.assertEqual(episode.is_terminated, episode_reproduced.is_terminated)
        self.assertEqual(episode.is_truncated, episode_reproduced.is_truncated)
        self.assertListEqual(episode.observations, episode_reproduced.observations)
        self.assertListEqual(episode.actions, episode_reproduced.actions)
        self.assertListEqual(episode.rewards, episode_reproduced.rewards)
        self.assertEqual(episode.infos, episode_reproduced.infos)
        self.assertEqual(episode.is_terminated, episode_reproduced.is_terminated)
        self.assertEqual(episode.is_truncated, episode_reproduced.is_truncated)
        self.assertEqual(episode.states, episode_reproduced.states)
        self.assertListEqual(episode.render_images, episode_reproduced.render_images)
        self.assertDictEqual(episode.extra_model_outputs, episode_reproduced.extra_model_outputs)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))