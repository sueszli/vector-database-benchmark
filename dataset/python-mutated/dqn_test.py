"""Tests for open_spiel.python.algorithms.dqn."""
from absl.testing import absltest
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
import pyspiel
tf.disable_v2_behavior()
SIMPLE_EFG_DATA = '\n  EFG 2 R "Simple single-agent problem" { "Player 1" } ""\n  p "ROOT" 1 1 "ROOT" { "L" "R" } 0\n    t "L" 1 "Outcome L" { -1.0 }\n    t "R" 2 "Outcome R" { 1.0 }\n'

class DQNTest(tf.test.TestCase):

    def test_simple_game(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
        env = rl_environment.Environment(game=game)
        with self.session() as sess:
            agent = dqn.DQN(sess, 0, state_representation_size=game.information_state_tensor_shape()[0], num_actions=game.num_distinct_actions(), hidden_layers_sizes=[16], replay_buffer_capacity=100, batch_size=5, epsilon_start=0.02, epsilon_end=0.01)
            total_reward = 0
            sess.run(tf.global_variables_initializer())
            for _ in range(100):
                time_step = env.reset()
                while not time_step.last():
                    agent_output = agent.step(time_step)
                    time_step = env.step([agent_output.action])
                    total_reward += time_step.rewards[0]
                agent.step(time_step)
            self.assertGreaterEqual(total_reward, 75)

    def test_run_tic_tac_toe(self):
        if False:
            print('Hello World!')
        env = rl_environment.Environment('tic_tac_toe')
        state_size = env.observation_spec()['info_state'][0]
        num_actions = env.action_spec()['num_actions']
        with self.session() as sess:
            agents = [dqn.DQN(sess, player_id, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=[16], replay_buffer_capacity=10, batch_size=5) for player_id in [0, 1]]
            sess.run(tf.global_variables_initializer())
            time_step = env.reset()
            while not time_step.last():
                current_player = time_step.observations['current_player']
                current_agent = agents[current_player]
                agent_output = current_agent.step(time_step)
                time_step = env.step([agent_output.action])
            for agent in agents:
                agent.step(time_step)

    @absltest.skip('Causing a segmentation fault on wheel tests')
    def test_run_hanabi(self):
        if False:
            for i in range(10):
                print('nop')
        game = 'hanabi'
        if game not in pyspiel.registered_names():
            return
        num_players = 3
        env_configs = {'players': num_players, 'max_life_tokens': 1, 'colors': 2, 'ranks': 3, 'hand_size': 2, 'max_information_tokens': 3, 'discount': 0.0}
        env = rl_environment.Environment(game, **env_configs)
        state_size = env.observation_spec()['info_state'][0]
        num_actions = env.action_spec()['num_actions']
        with self.session() as sess:
            agents = [dqn.DQN(sess, player_id, state_representation_size=state_size, num_actions=num_actions, hidden_layers_sizes=[16], replay_buffer_capacity=10, batch_size=5) for player_id in range(num_players)]
            sess.run(tf.global_variables_initializer())
            time_step = env.reset()
            while not time_step.last():
                current_player = time_step.observations['current_player']
                agent_output = [agent.step(time_step) for agent in agents]
                time_step = env.step([agent_output[current_player].action])
            for agent in agents:
                agent.step(time_step)
if __name__ == '__main__':
    tf.test.main()