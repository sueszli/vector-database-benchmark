"""Tests for deep average-network fictitious play."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn
from open_spiel.python.mfg.algorithms import average_network_fictitious_play
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel
from open_spiel.python.utils import training

class AverageNetworkFictitiousPlayTest(parameterized.TestCase):

    @parameterized.named_parameters(('cpp', 'mfg_crowd_modelling'), ('python', 'python_mfg_crowd_modelling'))
    def test_train(self, name):
        if False:
            while True:
                i = 10
        'Checks that the training works.'
        game = pyspiel.load_game(name)
        assert game.num_players() == 1
        uniform_policy = policy.UniformRandomPolicy(game)
        uniform_dist = distribution.DistributionPolicy(game, uniform_policy)
        env = rl_environment.Environment(game, mfg_distribution=uniform_dist, mfg_population=0)
        info_state_size = env.observation_spec()['info_state'][0]
        num_actions = env.action_spec()['num_actions']
        np.random.seed(0)
        dqn_args = {'batch_size': 32, 'epsilon_end': 0.1, 'epsilon_start': 0.1, 'hidden_layers_sizes': [128], 'learn_every': 32, 'learning_rate': 0.01, 'min_buffer_size_to_learn': 32, 'optimizer_str': 'adam', 'replay_buffer_capacity': 2000, 'update_target_network_every': 32}
        br_agent = dqn.DQN(0, info_state_size, num_actions, **dqn_args)
        args = {'batch_size': 32, 'hidden_layers_sizes': [128], 'reservoir_buffer_capacity': 100000, 'learning_rate': 0.01, 'min_buffer_size_to_learn': 32, 'optimizer_str': 'adam', 'seed': 0, 'tau': 1.0}
        fp = average_network_fictitious_play.AverageNetworkFictitiousPlay(game, [env], [br_agent], num_episodes_per_iteration=50, num_training_steps_per_iteration=10, **args)
        for _ in range(5):
            training.run_episodes([env], [br_agent], num_episodes=50, is_evaluation=False)
            fp.iteration()
        nash_conv_fp = nash_conv.NashConv(game, fp.policy)
        self.assertLessEqual(nash_conv_fp.nash_conv(), 15)
if __name__ == '__main__':
    absltest.main()