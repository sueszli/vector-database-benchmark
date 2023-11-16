"""Some helpers for normal-form games."""
import collections
import numpy as np

class StrategyAverager(object):
    """A helper class for averaging strategies for players."""

    def __init__(self, num_players, action_space_shapes, window_size=None):
        if False:
            return 10
        "Initialize the average strategy helper object.\n\n    Args:\n      num_players (int): the number of players in the game,\n      action_space_shapes:  an vector of n integers, where each element\n          represents the size of player i's actions space,\n      window_size (int or None): if None, computes the players' average\n          strategies over the entire sequence, otherwise computes the average\n          strategy over a finite-sized window of the k last entries.\n    "
        self._num_players = num_players
        self._action_space_shapes = action_space_shapes
        self._window_size = window_size
        self._num = 0
        if self._window_size is None:
            self._sum_meta_strategies = [np.zeros(action_space_shapes[p]) for p in range(num_players)]
        else:
            self._window = collections.deque(maxlen=self._window_size)

    def append(self, meta_strategies):
        if False:
            i = 10
            return i + 15
        'Append the meta-strategies to the averaged sequence.\n\n    Args:\n      meta_strategies: a list of strategies, one per player.\n    '
        if self._window_size is None:
            for p in range(self._num_players):
                self._sum_meta_strategies[p] += meta_strategies[p]
        else:
            self._window.append(meta_strategies)
        self._num += 1

    def average_strategies(self):
        if False:
            while True:
                i = 10
        "Return each player's average strategy.\n\n    Returns:\n      The averaged strategies, as a list containing one strategy per player.\n    "
        if self._window_size is None:
            avg_meta_strategies = [np.copy(x) for x in self._sum_meta_strategies]
            num_strategies = self._num
        else:
            avg_meta_strategies = [np.zeros(self._action_space_shapes[p]) for p in range(self._num_players)]
            for i in range(len(self._window)):
                for p in range(self._num_players):
                    avg_meta_strategies[p] += self._window[i][p]
            num_strategies = len(self._window)
        for p in range(self._num_players):
            avg_meta_strategies[p] /= num_strategies
        return avg_meta_strategies