"""Tests for iterated_prisoners_dilemma.py."""
from absl.testing import absltest
from open_spiel.python.games import iterated_prisoners_dilemma
import pyspiel

class IteratedPrisonersDilemmaTest(absltest.TestCase):

    def test_default_param(self):
        if False:
            i = 10
            return i + 15
        'Check the game can be converted to a turn-based game.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma')
        self.assertEqual(game._termination_probability, 0.125)

    def test_non_default_param_from_string(self):
        if False:
            return 10
        'Check params can be given through the string loading.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma(termination_probability=0.5)')
        self.assertEqual(game._termination_probability, 0.5)

    def test_non_default_param_from_dict(self):
        if False:
            print('Hello World!')
        'Check params can be given through a dictionary.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma', {'termination_probability': 0.75})
        self.assertEqual(game._termination_probability, 0.75)

    def test_game_as_turn_based(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the game can be converted to a turn-based game.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma')
        turn_based = pyspiel.convert_to_turn_based(game)
        pyspiel.random_sim_test(turn_based, num_sims=10, serialize=False, verbose=True)

    def test_game_as_turn_based_via_string(self):
        if False:
            while True:
                i = 10
        'Check the game can be created as a turn-based game from a string.'
        game = pyspiel.load_game('turn_based_simultaneous_game(game=python_iterated_prisoners_dilemma())')
        pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

    def test_game_from_cc(self):
        if False:
            i = 10
            return i + 15
        'Runs our standard game tests, checking API consistency.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma')
        pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)
if __name__ == '__main__':
    absltest.main()