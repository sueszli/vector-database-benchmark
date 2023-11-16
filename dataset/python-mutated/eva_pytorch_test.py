"""Tests for open_spiel.python.algorithms.eva."""
from absl.testing import absltest
from absl.testing import parameterized
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import eva
SEED = 24984617

class EVATest(parameterized.TestCase):

    @parameterized.parameters('tic_tac_toe', 'kuhn_poker', 'liars_dice')
    def test_run_games(self, game):
        if False:
            i = 10
            return i + 15
        env = rl_environment.Environment(game)
        num_players = env.num_players
        eva_agents = []
        num_actions = env.action_spec()['num_actions']
        state_size = env.observation_spec()['info_state'][0]
        for player in range(num_players):
            eva_agents.append(eva.EVAAgent(env, player, state_size, num_actions, embedding_network_layers=(64, 32), embedding_size=12, learning_rate=0.0001, mixing_parameter=0.5, memory_capacity=int(1000000.0), discount_factor=1.0, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_duration=int(1000000.0)))
        time_step = env.reset()
        while not time_step.last():
            current_player = time_step.observations['current_player']
            current_agent = eva_agents[current_player]
            agent_output = current_agent.step(time_step)
            time_step = env.step([agent_output.action])
        for agent in eva_agents:
            agent.step(time_step)

class QueryableFixedSizeRingBufferTest(absltest.TestCase):

    def test_replay_buffer_add(self):
        if False:
            for i in range(10):
                print('nop')
        replay_buffer = eva.QueryableFixedSizeRingBuffer(replay_buffer_capacity=10)
        self.assertEmpty(replay_buffer)
        replay_buffer.add('entry1')
        self.assertLen(replay_buffer, 1)
        replay_buffer.add('entry2')
        self.assertLen(replay_buffer, 2)
        self.assertIn('entry1', replay_buffer)
        self.assertIn('entry2', replay_buffer)

    def test_replay_buffer_max_capacity(self):
        if False:
            for i in range(10):
                print('nop')
        replay_buffer = eva.QueryableFixedSizeRingBuffer(replay_buffer_capacity=2)
        replay_buffer.add('entry1')
        replay_buffer.add('entry2')
        replay_buffer.add('entry3')
        self.assertLen(replay_buffer, 2)
        self.assertIn('entry2', replay_buffer)
        self.assertIn('entry3', replay_buffer)

    def test_replay_buffer_sample(self):
        if False:
            for i in range(10):
                print('nop')
        replay_buffer = eva.QueryableFixedSizeRingBuffer(replay_buffer_capacity=3)
        replay_buffer.add('entry1')
        replay_buffer.add('entry2')
        replay_buffer.add('entry3')
        samples = replay_buffer.sample(3)
        self.assertIn('entry1', samples)
        self.assertIn('entry2', samples)
        self.assertIn('entry3', samples)
if __name__ == '__main__':
    torch.manual_seed(SEED)
    absltest.main()