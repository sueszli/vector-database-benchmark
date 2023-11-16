"""Tests for open_spiel.python.algorithms.response_graph_ucb."""
import itertools
from absl.testing import absltest
import matplotlib
matplotlib.use('agg')
import numpy as np
from open_spiel.python.algorithms import response_graph_ucb
from open_spiel.python.algorithms import response_graph_ucb_utils

class ResponseGraphUcbTest(absltest.TestCase):

    def get_example_2x2_payoffs(self):
        if False:
            for i in range(10):
                print('nop')
        mean_payoffs = np.random.uniform(-1, 1, size=(2, 2, 2))
        mean_payoffs[0, :, :] = np.asarray([[0.5, 0.85], [0.15, 0.5]])
        mean_payoffs[1, :, :] = 1 - mean_payoffs[0, :, :]
        return mean_payoffs

    def test_sampler(self):
        if False:
            while True:
                i = 10
        mean_payoffs = self.get_example_2x2_payoffs()
        game = response_graph_ucb_utils.BernoulliGameSampler([2, 2], mean_payoffs, payoff_bounds=[-1.0, 1.0])
        game.p_max = mean_payoffs
        game.means = mean_payoffs
        sampling_methods = ['uniform-exhaustive', 'uniform', 'valence-weighted', 'count-weighted']
        conf_methods = ['ucb-standard', 'ucb-standard-relaxed', 'clopper-pearson-ucb', 'clopper-pearson-ucb-relaxed']
        per_payoff_confidence = [True, False]
        time_dependent_delta = [True, False]
        methods = list(itertools.product(sampling_methods, conf_methods, per_payoff_confidence, time_dependent_delta))
        max_total_interactions = 50
        for m in methods:
            r_ucb = response_graph_ucb.ResponseGraphUCB(game, exploration_strategy=m[0], confidence_method=m[1], delta=0.1, ucb_eps=0.1, per_payoff_confidence=m[2], time_dependent_delta=m[3])
            _ = r_ucb.run(max_total_iterations=max_total_interactions)

    def test_soccer_data_import(self):
        if False:
            return 10
        response_graph_ucb_utils.get_soccer_data()
if __name__ == '__main__':
    absltest.main()