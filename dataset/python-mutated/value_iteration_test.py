"""Tests for open_spiel.python.algorithms.get_all_states."""
from absl.testing import absltest
from open_spiel.python.algorithms import value_iteration
import pyspiel

class ValueIterationTest(absltest.TestCase):

    def test_solve_tic_tac_toe(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('tic_tac_toe')
        values = value_iteration.value_iteration(game, depth_limit=-1, threshold=0.01)
        initial_state = '...\n...\n...'
        cross_win_state = '...\n...\n.ox'
        naught_win_state = 'x..\noo.\nxx.'
        self.assertEqual(values[initial_state], 0)
        self.assertEqual(values[cross_win_state], 1)
        self.assertEqual(values[naught_win_state], -1)

    def test_solve_small_goofspiel(self):
        if False:
            return 10
        game = pyspiel.load_game('goofspiel', {'num_cards': 3})
        values = value_iteration.value_iteration(game, depth_limit=-1, threshold=1e-06)
        initial_state = game.new_initial_state()
        assert initial_state.is_chance_node()
        root_value = 0
        for (action, action_prob) in initial_state.chance_outcomes():
            next_state = initial_state.child(action)
            root_value += action_prob * values[str(next_state)]
        self.assertAlmostEqual(root_value, 0)

    def test_solve_small_oshi_zumo(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('oshi_zumo', {'coins': 5, 'size': 2})
        values = value_iteration.value_iteration(game, depth_limit=-1, threshold=1e-06, cyclic_game=True)
        initial_state = game.new_initial_state()
        self.assertAlmostEqual(values[str(initial_state)], 0)
        game = pyspiel.load_game('oshi_zumo', {'coins': 5, 'size': 2, 'min_bid': 1})
        values = value_iteration.value_iteration(game, depth_limit=-1, threshold=1e-06, cyclic_game=False)
        initial_state = game.new_initial_state()
        self.assertAlmostEqual(values[str(initial_state)], 0)

    def test_solve_small_pig(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('pig', {'winscore': 20})
        values = value_iteration.value_iteration(game, depth_limit=-1, threshold=1e-06, cyclic_game=True)
        initial_state = game.new_initial_state()
        print('Value of Pig(20): ', values[str(initial_state)])
if __name__ == '__main__':
    absltest.main()