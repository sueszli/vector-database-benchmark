"""Wrapper for loading pyspiel games as payoff tensors."""
from absl import logging
import numpy as np
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

class PyspielTensorGame(object):
    """Matrix Game."""

    def __init__(self, string_specifier='blotto(coins=10,fields=3,players=3)', tensor_game=False, seed=None):
        if False:
            for i in range(10):
                print('nop')
        'Ctor. Inits payoff tensor (players x actions x ... np.array).'
        self.pt = None
        self.string_specifier = string_specifier
        self.tensor_game = tensor_game
        if tensor_game:
            self.game = pyspiel.load_tensor_game(string_specifier)
        else:
            self.game = pyspiel.load_game(string_specifier)
        self.seed = seed

    def num_players(self):
        if False:
            i = 10
            return i + 15
        return self.game.num_players()

    def num_strategies(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.game.num_distinct_actions()] * self.num_players()

    def payoff_tensor(self):
        if False:
            while True:
                i = 10
        if self.pt is None:
            if not self.tensor_game:
                logging.info('reloading pyspiel game as tensor_game')
                self.game = pyspiel.load_tensor_game(self.string_specifier)
                self.tensor_game = True
            pt = np.asarray(game_payoffs_array(self.game))
            self.pt = pt - self.game.min_utility()
        return self.pt

    def get_payoffs_for_strategies(self, policies):
        if False:
            print('Hello World!')
        'Return vector of payoffs for all players given list of strategies.\n\n    Args:\n      policies: list of integers indexing strategies for each player\n    Returns:\n      np.array (length num players) of payoffs\n    '
        state = self.game.new_initial_state()
        state.apply_actions(policies)
        return np.asarray(state.returns()) - self.game.min_utility()