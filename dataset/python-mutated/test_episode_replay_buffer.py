import unittest
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

class TestEpisodeReplayBuffer(unittest.TestCase):

    @staticmethod
    def _get_episode(episode_len=None, id_=None):
        if False:
            return 10
        eps = SingleAgentEpisode(id_=id_, observations=[0.0], infos=[{}])
        ts = np.random.randint(1, 200) if episode_len is None else episode_len
        for t in range(ts):
            eps.add_timestep(observation=float(t + 1), action=int(t), reward=0.1 * (t + 1), info={})
        eps.is_terminated = np.random.random() > 0.5
        eps.is_truncated = False if eps.is_terminated else np.random.random() > 0.8
        return eps

    def test_add_and_eviction_logic(self):
        if False:
            while True:
                i = 10
        'Tests batches getting properly added to buffer and cause proper eviction.'
        buffer = EpisodeReplayBuffer(capacity=100)
        episode = self._get_episode(id_='A', episode_len=50)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 1)
        self.assertTrue(buffer.get_num_timesteps() == 50)
        episode = self._get_episode(id_='B', episode_len=25)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 2)
        self.assertTrue(buffer.get_num_timesteps() == 75)
        episode = self._get_episode(id_='C', episode_len=25)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 3)
        self.assertTrue(buffer.get_num_timesteps() == 100)
        episode = self._get_episode(id_='D', episode_len=1)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 3)
        self.assertTrue(buffer.get_num_timesteps() == 51)
        self.assertTrue({eps.id_ for eps in buffer.episodes} == {'B', 'C', 'D'})
        episode = self._get_episode(id_='E', episode_len=200)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 1)
        self.assertTrue(buffer.get_num_timesteps() == 200)
        self.assertTrue({eps.id_ for eps in buffer.episodes} == {'E'})
        episode = self._get_episode(id_='F', episode_len=2)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 1)
        self.assertTrue(buffer.get_num_timesteps() == 2)
        self.assertTrue({eps.id_ for eps in buffer.episodes} == {'F'})
        for i in range(10):
            episode = self._get_episode(id_=str(i), episode_len=10)
            buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 10)
        self.assertTrue(buffer.get_num_timesteps() == 100)
        episode = self._get_episode(id_='G', episode_len=21)
        buffer.add(episode)
        self.assertTrue(buffer.get_num_episodes() == 8)
        self.assertTrue(buffer.get_num_timesteps() == 91)
        self.assertTrue({eps.id_ for eps in buffer.episodes} == {'3', '4', '5', '6', '7', '8', '9', 'G'})

    def test_episode_replay_buffer_sample_logic(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests whether batches are correctly formed when sampling from the buffer.'
        buffer = EpisodeReplayBuffer(capacity=10000)
        for _ in range(200):
            episode = self._get_episode()
            buffer.add(episode)
        for _ in range(1000):
            sample = buffer.sample(batch_size_B=16, batch_length_T=64)
            (obs, actions, rewards, is_first, is_last, is_terminated, is_truncated) = (sample['obs'], sample['actions'], sample['rewards'], sample['is_first'], sample['is_last'], sample['is_terminated'], sample['is_truncated'])
            assert not np.any(np.logical_and(is_truncated, is_terminated))
            assert np.all(is_last[:, -1])
            assert np.all(is_first[:, 0])
            assert obs.shape[:2] == rewards.shape == actions.shape == is_first.shape == is_last.shape == is_terminated.shape
            assert np.all(np.equal(obs * 0.1, rewards))
            assert np.all(np.where(is_last, True, np.equal(obs, actions)))
            assert np.all(np.where(is_terminated[:, 1:], np.equal(actions[:, 1:], actions[:, :-1]), True))
            assert np.all(np.where(is_terminated[:, :-1], rewards[:, 1:] == 0.0, True))
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))