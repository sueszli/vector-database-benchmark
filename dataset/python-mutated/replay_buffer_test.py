"""Tests for open_spiel.python.utils.replay_buffer."""
from absl.testing import absltest
from open_spiel.python.utils.replay_buffer import ReplayBuffer

class ReplayBufferTest(absltest.TestCase):

    def test_replay_buffer_add(self):
        if False:
            while True:
                i = 10
        replay_buffer = ReplayBuffer(replay_buffer_capacity=10)
        self.assertEqual(len(replay_buffer), 0)
        replay_buffer.add('entry1')
        self.assertEqual(len(replay_buffer), 1)
        replay_buffer.add('entry2')
        self.assertEqual(len(replay_buffer), 2)
        self.assertIn('entry1', replay_buffer)
        self.assertIn('entry2', replay_buffer)

    def test_replay_buffer_max_capacity(self):
        if False:
            return 10
        replay_buffer = ReplayBuffer(replay_buffer_capacity=2)
        replay_buffer.add('entry1')
        replay_buffer.add('entry2')
        replay_buffer.add('entry3')
        self.assertEqual(len(replay_buffer), 2)
        self.assertIn('entry2', replay_buffer)
        self.assertIn('entry3', replay_buffer)

    def test_replay_buffer_sample(self):
        if False:
            i = 10
            return i + 15
        replay_buffer = ReplayBuffer(replay_buffer_capacity=3)
        replay_buffer.add('entry1')
        replay_buffer.add('entry2')
        replay_buffer.add('entry3')
        samples = replay_buffer.sample(3)
        self.assertIn('entry1', samples)
        self.assertIn('entry2', samples)
        self.assertIn('entry3', samples)

    def test_replay_buffer_reset(self):
        if False:
            return 10
        replay_buffer = ReplayBuffer(replay_buffer_capacity=3)
        replay_buffer.add('entry1')
        replay_buffer.add('entry2')
        replay_buffer.reset()
        self.assertEmpty(replay_buffer)
if __name__ == '__main__':
    absltest.main()