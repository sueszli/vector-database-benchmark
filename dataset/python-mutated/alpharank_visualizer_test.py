"""Tests for open_spiel.python.egt.alpharank_visualizer."""
from absl.testing import absltest
import matplotlib
matplotlib.use('agg')
import mock
import numpy as np
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
import pyspiel

class AlpharankVisualizerTest(absltest.TestCase):

    @mock.patch('%s.alpharank_visualizer.plt' % __name__)
    def test_plot_pi_vs_alpha(self, mock_plt):
        if False:
            while True:
                i = 10
        game = pyspiel.load_matrix_game('matrix_rps')
        payoff_tables = utils.game_payoffs_array(game)
        (_, payoff_tables) = utils.is_symmetric_matrix_game(payoff_tables)
        payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
        alpha = 100.0
        (_, _, pi, num_profiles, num_strats_per_population) = alpharank.compute(payoff_tables, alpha=alpha)
        strat_labels = utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)
        num_populations = len(payoff_tables)
        pi_list = np.empty((num_profiles, 0))
        alpha_list = []
        for _ in range(2):
            pi_list = np.append(pi_list, np.reshape(pi, (-1, 1)), axis=1)
            alpha_list.append(alpha)
        alpharank_visualizer.plot_pi_vs_alpha(pi_list.T, alpha_list, num_populations, num_strats_per_population, strat_labels, num_strats_to_label=0)
        self.assertTrue(mock_plt.show.called)
if __name__ == '__main__':
    absltest.main()