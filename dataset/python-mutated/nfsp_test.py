"""Tests for open_spiel.python.algorithms.nfsp."""
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp
tf.disable_v2_behavior()

class NFSPTest(tf.test.TestCase):

    def test_run_kuhn(self):
        if False:
            for i in range(10):
                print('nop')
        env = rl_environment.Environment('kuhn_poker')
        state_size = env.observation_spec()['info_state'][0]
        num_actions = env.action_spec()['num_actions']
        with self.session() as sess:
            agents = [nfsp.NFSP(sess, player_id, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=[16], reservoir_buffer_capacity=10, anticipatory_param=0.1) for player_id in [0, 1]]
            sess.run(tf.global_variables_initializer())
            for unused_ep in range(10):
                time_step = env.reset()
                while not time_step.last():
                    current_player = time_step.observations['current_player']
                    current_agent = agents[current_player]
                    agent_output = current_agent.step(time_step)
                    time_step = env.step([agent_output.action])
                for agent in agents:
                    agent.step(time_step)

class ReservoirBufferTest(tf.test.TestCase):

    def test_reservoir_buffer_add(self):
        if False:
            print('Hello World!')
        reservoir_buffer = nfsp.ReservoirBuffer(reservoir_buffer_capacity=10)
        self.assertEmpty(reservoir_buffer)
        reservoir_buffer.add('entry1')
        self.assertLen(reservoir_buffer, 1)
        reservoir_buffer.add('entry2')
        self.assertLen(reservoir_buffer, 2)
        self.assertIn('entry1', reservoir_buffer)
        self.assertIn('entry2', reservoir_buffer)

    def test_reservoir_buffer_max_capacity(self):
        if False:
            for i in range(10):
                print('nop')
        reservoir_buffer = nfsp.ReservoirBuffer(reservoir_buffer_capacity=2)
        reservoir_buffer.add('entry1')
        reservoir_buffer.add('entry2')
        reservoir_buffer.add('entry3')
        self.assertLen(reservoir_buffer, 2)

    def test_reservoir_buffer_sample(self):
        if False:
            i = 10
            return i + 15
        replay_buffer = nfsp.ReservoirBuffer(reservoir_buffer_capacity=3)
        replay_buffer.add('entry1')
        replay_buffer.add('entry2')
        replay_buffer.add('entry3')
        samples = replay_buffer.sample(3)
        self.assertIn('entry1', samples)
        self.assertIn('entry2', samples)
        self.assertIn('entry3', samples)
if __name__ == '__main__':
    tf.test.main()