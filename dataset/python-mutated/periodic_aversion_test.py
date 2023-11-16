"""Tests for Python Periodic Aversion game."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.mfg.games import periodic_aversion
import pyspiel
MFG_STR_CONST = '_a'

class MFGPeriodicAversionTest(absltest.TestCase):

    def test_load(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('python_mfg_periodic_aversion')
        game.new_initial_state()

    def test_create(self):
        if False:
            print('Hello World!')
        'Checks we can create the game and clone states.'
        game = periodic_aversion.MFGPeriodicAversionGame()
        self.assertEqual(game.size, periodic_aversion._SIZE)
        self.assertEqual(game.horizon, periodic_aversion._HORIZON)
        self.assertEqual(game.get_type().dynamics, pyspiel.GameType.Dynamics.MEAN_FIELD)
        print('Num distinct actions:', game.num_distinct_actions())
        state = game.new_initial_state()
        clone = state.clone()
        print('Initial state:', state)
        print('Cloned initial state:', clone)

    def test_create_with_params(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('python_mfg_periodic_aversion(horizon=30,size=41)')
        self.assertEqual(game.size, 41)
        self.assertEqual(game.horizon, 30)

    def check_cloning(self, state):
        if False:
            for i in range(10):
                print('nop')
        cloned = state.clone()
        self.assertEqual(str(cloned), str(state))
        self.assertEqual(cloned._distribution, state._distribution)
        self.assertEqual(cloned._returns(), state._returns())
        self.assertEqual(cloned.current_player(), state.current_player())
        self.assertEqual(cloned.size, state.size)
        self.assertEqual(cloned.horizon, state.horizon)
        self.assertEqual(cloned._last_action, state._last_action)

    def test_random_game(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests basic API functions.'
        np.random.seed(7)
        horizon = 30
        size = 41
        game = periodic_aversion.MFGPeriodicAversionGame(params={'horizon': horizon, 'size': size})
        state = game.new_initial_state()
        t = 0
        while not state.is_terminal():
            if state.current_player() == pyspiel.PlayerId.CHANCE:
                (actions, probs) = zip(*state.chance_outcomes())
                action = np.random.choice(actions, p=probs)
                self.check_cloning(state)
                self.assertEqual(len(state.legal_actions()), len(state.chance_outcomes()))
                state.apply_action(action)
            elif state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
                self.assertEqual(state.legal_actions(), [])
                self.check_cloning(state)
                num_states = len(state.distribution_support())
                state.update_distribution([1 / num_states] * num_states)
            else:
                self.assertEqual(state.current_player(), 0)
                self.check_cloning(state)
                state.observation_string()
                state.information_state_string()
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
                t += 1
        self.assertEqual(t, horizon)
if __name__ == '__main__':
    absltest.main()