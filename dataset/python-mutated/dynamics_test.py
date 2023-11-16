"""Tests for open_spiel.python.egt.dynamics."""
import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

def _sum_j_x_j_ln_x_j_over_x_i(x):
    if False:
        i = 10
        return i + 15
    'Computes \\sum_j x_j ln(x_j / x_i).'
    a = x.reshape([1, -1])
    b = x.reshape([-1, 1])
    return np.sum(a * np.log(np.divide(a, b)), axis=1)

def _q_learning_dynamics(composition, payoff, temperature):
    if False:
        print('Hello World!')
    'An equivalent implementation of `dynamics.boltzmannq`.'
    return 1 / temperature * dynamics.replicator(composition, payoff) + composition * _sum_j_x_j_ln_x_j_over_x_i(composition)

class _InternalTest(absltest.TestCase):

    def test__sum_j_x_j_ln_x_j_over_x_i(self):
        if False:
            while True:
                i = 10
        x = np.asarray([1.0, 2.0, 3.0])
        expected = [sum([x_j * math.log(x_j / x_i) for x_j in x]) for x_i in x]
        log = math.log
        expected_0 = 1.0 * log(1 / 1.0) + 2 * log(2 / 1.0) + 3 * log(3 / 1.0)
        expected_1 = 1.0 * log(1 / 2.0) + 2 * log(2 / 2.0) + 3 * log(3 / 2.0)
        expected_2 = 1.0 * log(1 / 3.0) + 2 * log(2 / 3.0) + 3 * log(3 / 3.0)
        expected_2 = np.asarray([expected_0, expected_1, expected_2])
        np.testing.assert_array_equal(expected, expected_2)
        np.testing.assert_array_almost_equal(expected, _sum_j_x_j_ln_x_j_over_x_i(x))

class DynamicsTest(parameterized.TestCase):

    def test_boltzmann_q(self):
        if False:
            return 10
        x = np.asarray([1 / 2, 1 / 2])
        payoff = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        temperature = 1
        np.testing.assert_array_equal(dynamics.boltzmannq(x, payoff, temperature), _q_learning_dynamics(x, payoff, temperature))

    def test_rd_rps_pure_fixed_points(self):
        if False:
            return 10
        game = pyspiel.load_matrix_game('matrix_rps')
        payoff_matrix = game_payoffs_array(game)
        rd = dynamics.replicator
        dyn = dynamics.SinglePopulationDynamics(payoff_matrix, rd)
        x = np.eye(3)
        np.testing.assert_allclose(dyn(x[0]), np.zeros((3,)))
        np.testing.assert_allclose(dyn(x[1]), np.zeros((3,)))
        np.testing.assert_allclose(dyn(x[2]), np.zeros((3,)))

    @parameterized.parameters(dynamics.replicator, dynamics.boltzmannq, dynamics.qpg)
    def test_dynamics_rps_mixed_fixed_point(self, func):
        if False:
            print('Hello World!')
        game = pyspiel.load_matrix_game('matrix_rps')
        payoff_matrix = game_payoffs_array(game)
        dyn = dynamics.SinglePopulationDynamics(payoff_matrix, func)
        x = np.ones(shape=(3,)) / 3.0
        np.testing.assert_allclose(dyn(x), np.zeros((3,)), atol=1e-15)

    def test_multi_population_rps(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_matrix_game('matrix_rps')
        payoff_matrix = game_payoffs_array(game)
        rd = dynamics.replicator
        dyn = dynamics.MultiPopulationDynamics(payoff_matrix, [rd] * 2)
        x = np.concatenate([np.ones(k) / float(k) for k in payoff_matrix.shape[1:]])
        np.testing.assert_allclose(dyn(x), np.zeros((6,)), atol=1e-15)

    def test_multi_population_three_populations(self):
        if False:
            for i in range(10):
                print('nop')
        payoff_matrix = np.arange(3 * 2 * 3 * 4).reshape(3, 2, 3, 4)
        rd = dynamics.replicator
        dyn = dynamics.MultiPopulationDynamics(payoff_matrix, [rd] * 3)
        x = np.concatenate([np.ones(k) / float(k) for k in payoff_matrix.shape[1:]])
        self.assertEqual(dyn(x).shape, (9,))

    def test_multi_population_four_populations(self):
        if False:
            while True:
                i = 10
        payoff_matrix = np.zeros((4, 2, 2, 2, 2))
        payoff_matrix[:, 0, 0, 0, 0] = np.ones((4,))
        rd = dynamics.replicator
        dyn = dynamics.MultiPopulationDynamics(payoff_matrix, [rd] * 4)
        x = np.concatenate([np.ones(k) / float(k) for k in payoff_matrix.shape[1:]])
        avg_fitness = 1.0 / float(2 ** 4)
        dx = dyn(x)
        np.testing.assert_allclose(dx[::2], np.ones((4,)) * avg_fitness / 2.0)
        np.testing.assert_allclose(dx[1::2], np.ones((4,)) * -avg_fitness / 2.0)

    def test_time_average(self):
        if False:
            print('Hello World!')
        (n, k) = (10, 3)
        traj = np.ones(shape=(n, k))
        time_avg = dynamics.time_average(traj)
        np.testing.assert_allclose(time_avg, np.ones(shape=(n, k)))
        traj[1::2] = -1.0 * traj[1::2]
        time_avg = dynamics.time_average(traj)
        np.testing.assert_allclose(time_avg[-1], np.zeros(shape=(k,)))
        np.testing.assert_allclose(time_avg[-2], 1.0 / (n - 1.0) * np.ones(shape=(k,)))
if __name__ == '__main__':
    absltest.main()