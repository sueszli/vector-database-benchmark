"""GAMUT games.

See https://github.com/deepmind/open_spiel/tree/master/open_spiel/games/gamut
for details on how to build OpenSpiel with support for GAMUT.
"""
from absl import logging
import numpy as np
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

class GAMUT(object):
    """GAMUT Games."""

    def __init__(self, config_list, java_path='', seed=None):
        if False:
            return 10
        "Ctor. Inits payoff tensor (players x actions x ... np.array).\n\n    Args:\n      config_list: a list or strings alternating between gamut flags and values\n        see http://gamut.stanford.edu/userdoc.pdf for more information\n        e.g., config_list = ['-g', 'CovariantGame', '-players', '6',\n                             '-normalize', '-min_payoff', '0',\n                             '-max_payoff', '1', '-actions', '5', '-r', '0']\n      java_path: string, java path\n      seed: random seed, some GAMUT games are randomly generated\n    "
        self.pt = None
        self.config_list = config_list
        self.seed = seed
        self.random = np.random.RandomState(seed)
        if '-r' in config_list:
            idx = next((i for (i, s) in enumerate(config_list) if s == '-r'))
            val = config_list[idx + 1]
            if not val.isnumeric() and val[0] in '([' and (val[-1] in ')]'):
                (a, b) = val.strip('[]()').split(',')
                a = float(a)
                b = float(b)
                rho = self.random.rand() * (b - a) + a
                config_list[idx + 1] = str(rho)
        if isinstance(seed, int):
            self.config_list += ['-random_seed', str(seed)]
        self.java_path = java_path
        if java_path:
            generator = pyspiel.GamutGenerator(java_path, 'gamut/gamut_main_deploy.jar')
        else:
            generator = pyspiel.GamutGenerator('gamut.jar')
        self.game = generator.generate_game(config_list)

    def num_players(self):
        if False:
            i = 10
            return i + 15
        return self.game.num_players()

    def num_strategies(self):
        if False:
            while True:
                i = 10
        return [self.game.num_distinct_actions()] * self.num_players()

    def payoff_tensor(self):
        if False:
            i = 10
            return i + 15
        if self.pt is None:
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