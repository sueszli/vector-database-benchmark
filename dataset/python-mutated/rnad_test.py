"""Tests for RNaD algorithm under open_spiel."""
import pickle
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from open_spiel.python.algorithms.rnad import rnad

class RNADTest(parameterized.TestCase):

    def test_run_kuhn(self):
        if False:
            i = 10
            return i + 15
        solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name='kuhn_poker'))
        for _ in range(10):
            solver.step()

    def test_serialization(self):
        if False:
            return 10
        solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name='kuhn_poker'))
        solver.step()
        state_bytes = pickle.dumps(solver)
        solver2 = pickle.loads(state_bytes)
        self.assertEqual(solver.config, solver2.config)
        np.testing.assert_equal(jax.device_get(solver.params), jax.device_get(solver2.params))

    @parameterized.named_parameters(dict(testcase_name='3x2_5x1_6', sizes=[3, 5, 6], repeats=[2, 1, 1], cover_steps=24, expected=[(0, False), (2 / 3, False), (1, True), (0, False), (2 / 3, False), (1, True), (0, False), (0.4, False), (0.8, False), (1, False), (1, True), (0, False), (1 / 3, False), (2 / 3, False), (1, False), (1, False), (1, True), (0, False), (1 / 3, False), (2 / 3, False), (1, False), (1, False), (1, True), (0, False)]))
    def test_entropy_schedule(self, sizes, repeats, cover_steps, expected):
        if False:
            return 10
        schedule = rnad.EntropySchedule(sizes=sizes, repeats=repeats)
        computed = [schedule(i) for i in range(cover_steps)]
        np.testing.assert_almost_equal(computed, expected)
if __name__ == '__main__':
    absltest.main()