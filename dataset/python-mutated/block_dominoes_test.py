"""Tests for Python Block Dominoes."""
from absl.testing import absltest
from open_spiel.python.games import block_dominoes
import pyspiel

class DominoesBlockTest(absltest.TestCase):

    def test_game_from_cc(self):
        if False:
            while True:
                i = 10
        'Runs our standard game tests, checking API consistency.'
        game = pyspiel.load_game('python_block_dominoes')
        pyspiel.random_sim_test(game, num_sims=100, serialize=False, verbose=True)

    def test_single_deterministic_game_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs a single game where tiles and actions chose deterministically.'
        game = pyspiel.load_game('python_block_dominoes')
        state = game.new_initial_state()
        hand0 = [(6.0, 6.0), (0.0, 2.0), (4.0, 4.0), (3.0, 3.0), (2.0, 2.0), (1.0, 1.0), (0.0, 0.0)]
        hand1 = [(5.0, 6.0), (4.0, 5.0), (3.0, 4.0), (2.0, 3.0), (1.0, 2.0), (0.0, 1.0), (4.0, 6.0)]
        self.deal_hands(state, [hand0, hand1])
        self.apply_action(state, block_dominoes.Action(0, (6.0, 6.0), None))
        self.apply_action(state, block_dominoes.Action(1, (5.0, 6.0), 6.0))
        self.apply_action(state, block_dominoes.Action(1, (4.0, 5.0), 5.0))
        self.apply_action(state, block_dominoes.Action(0, (4.0, 4.0), 4.0))
        self.apply_action(state, block_dominoes.Action(1, (3.0, 4.0), 4.0))
        self.apply_action(state, block_dominoes.Action(0, (3.0, 3.0), 3.0))
        self.apply_action(state, block_dominoes.Action(1, (2.0, 3.0), 3.0))
        self.apply_action(state, block_dominoes.Action(0, (2.0, 2.0), 2.0))
        self.apply_action(state, block_dominoes.Action(1, (1.0, 2.0), 2.0))
        self.apply_action(state, block_dominoes.Action(0, (1.0, 1.0), 1.0))
        self.apply_action(state, block_dominoes.Action(1, (0.0, 1.0), 1.0))
        self.apply_action(state, block_dominoes.Action(0, (0.0, 0.0), 0.0))
        self.apply_action(state, block_dominoes.Action(1, (4.0, 6.0), 6.0))
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.returns()[0], -2)
        self.assertEqual(state.returns()[1], 2)

    def test_single_deterministic_game_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Runs a single game where tiles and actions chose deterministically.'
        game = pyspiel.load_game('python_block_dominoes')
        state = game.new_initial_state()
        hand0 = [(6.0, 6.0), (0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (4.0, 5.0), (5.0, 5.0)]
        hand1 = [(0.0, 4.0), (1.0, 4.0), (2.0, 4.0), (3.0, 4.0), (4.0, 4.0), (0.0, 3.0), (1.0, 3.0)]
        self.deal_hands(state, [hand0, hand1])
        self.apply_action(state, block_dominoes.Action(0, (6.0, 6.0), None))
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.returns()[0], -45)
        self.assertEqual(state.returns()[1], 45)

    @staticmethod
    def apply_action(state, action):
        if False:
            i = 10
            return i + 15
        actions_str = block_dominoes._ACTIONS_STR
        state.apply_action(actions_str.index(str(action)))

    @staticmethod
    def deal_hands(state, hands):
        if False:
            for i in range(10):
                print('nop')
        deck = block_dominoes._DECK
        for hand in hands:
            for t in hand:
                state.apply_action(deck.index(t))
if __name__ == '__main__':
    absltest.main()