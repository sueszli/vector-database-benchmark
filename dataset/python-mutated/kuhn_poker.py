"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""
import enum
import numpy as np
import pyspiel

class Action(enum.IntEnum):
    PASS = 0
    BET = 1
_NUM_PLAYERS = 2
_DECK = frozenset([0, 1, 2])
_GAME_TYPE = pyspiel.GameType(short_name='python_kuhn_poker', long_name='Python Kuhn Poker', dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL, chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC, information=pyspiel.GameType.Information.IMPERFECT_INFORMATION, utility=pyspiel.GameType.Utility.ZERO_SUM, reward_model=pyspiel.GameType.RewardModel.TERMINAL, max_num_players=_NUM_PLAYERS, min_num_players=_NUM_PLAYERS, provides_information_state_string=True, provides_information_state_tensor=True, provides_observation_string=True, provides_observation_tensor=True, provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(num_distinct_actions=len(Action), max_chance_outcomes=len(_DECK), num_players=_NUM_PLAYERS, min_utility=-2.0, max_utility=2.0, utility_sum=0.0, max_game_length=3)

class KuhnPokerGame(pyspiel.Game):
    """A Python version of Kuhn poker."""

    def __init__(self, params=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        if False:
            return 10
        'Returns a state corresponding to the start of a game.'
        return KuhnPokerState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        if False:
            while True:
                i = 10
        'Returns an object used for observing game state.'
        return KuhnPokerObserver(iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)

class KuhnPokerState(pyspiel.State):
    """A python version of the Kuhn poker state."""

    def __init__(self, game):
        if False:
            return 10
        'Constructor; should only be called by Game.new_initial_state.'
        super().__init__(game)
        self.cards = []
        self.bets = []
        self.pot = [1.0, 1.0]
        self._game_over = False
        self._next_player = 0

    def current_player(self):
        if False:
            i = 10
            return i + 15
        'Returns id of the next player to move, or TERMINAL if game is over.'
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif len(self.cards) < _NUM_PLAYERS:
            return pyspiel.PlayerId.CHANCE
        else:
            return self._next_player

    def _legal_actions(self, player):
        if False:
            return 10
        'Returns a list of legal actions, sorted in ascending order.'
        assert player >= 0
        return [Action.PASS, Action.BET]

    def chance_outcomes(self):
        if False:
            print('Hello World!')
        'Returns the possible chance outcomes and their probabilities.'
        assert self.is_chance_node()
        outcomes = sorted(_DECK - set(self.cards))
        p = 1.0 / len(outcomes)
        return [(o, p) for o in outcomes]

    def _apply_action(self, action):
        if False:
            while True:
                i = 10
        'Applies the specified action to the state.'
        if self.is_chance_node():
            self.cards.append(action)
        else:
            self.bets.append(action)
            if action == Action.BET:
                self.pot[self._next_player] += 1
            self._next_player = 1 - self._next_player
            if min(self.pot) == 2 or (len(self.bets) == 2 and action == Action.PASS) or len(self.bets) == 3:
                self._game_over = True

    def _action_to_string(self, player, action):
        if False:
            i = 10
            return i + 15
        'Action -> string.'
        if player == pyspiel.PlayerId.CHANCE:
            return f'Deal:{action}'
        elif action == Action.PASS:
            return 'Pass'
        else:
            return 'Bet'

    def is_terminal(self):
        if False:
            return 10
        'Returns True if the game is over.'
        return self._game_over

    def returns(self):
        if False:
            i = 10
            return i + 15
        'Total reward for each player over the course of the game so far.'
        pot = self.pot
        winnings = float(min(pot))
        if not self._game_over:
            return [0.0, 0.0]
        elif pot[0] > pot[1]:
            return [winnings, -winnings]
        elif pot[0] < pot[1]:
            return [-winnings, winnings]
        elif self.cards[0] > self.cards[1]:
            return [winnings, -winnings]
        else:
            return [-winnings, winnings]

    def __str__(self):
        if False:
            return 10
        'String for debug purposes. No particular semantics are required.'
        return ''.join([str(c) for c in self.cards] + ['pb'[b] for b in self.bets])

class KuhnPokerObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        if False:
            print('Hello World!')
        'Initializes an empty observation tensor.'
        if params:
            raise ValueError(f'Observation parameters not supported; passed {params}')
        pieces = [('player', 2, (2,))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(('private_card', 3, (3,)))
        if iig_obs_type.public_info:
            if iig_obs_type.perfect_recall:
                pieces.append(('betting', 6, (3, 2)))
            else:
                pieces.append(('pot_contribution', 2, (2,)))
        total_size = sum((size for (name, size, shape) in pieces))
        self.tensor = np.zeros(total_size, np.float32)
        self.dict = {}
        index = 0
        for (name, size, shape) in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        if False:
            return 10
        'Updates `tensor` and `dict` to reflect `state` from PoV of `player`.'
        self.tensor.fill(0)
        if 'player' in self.dict:
            self.dict['player'][player] = 1
        if 'private_card' in self.dict and len(state.cards) > player:
            self.dict['private_card'][state.cards[player]] = 1
        if 'pot_contribution' in self.dict:
            self.dict['pot_contribution'][:] = state.pot
        if 'betting' in self.dict:
            for (turn, action) in enumerate(state.bets):
                self.dict['betting'][turn, action] = 1

    def string_from(self, state, player):
        if False:
            i = 10
            return i + 15
        'Observation of `state` from the PoV of `player`, as a string.'
        pieces = []
        if 'player' in self.dict:
            pieces.append(f'p{player}')
        if 'private_card' in self.dict and len(state.cards) > player:
            pieces.append(f'card:{state.cards[player]}')
        if 'pot_contribution' in self.dict:
            pieces.append(f'pot[{int(state.pot[0])} {int(state.pot[1])}]')
        if 'betting' in self.dict and state.bets:
            pieces.append(''.join(('pb'[b] for b in state.bets)))
        return ' '.join((str(p) for p in pieces))
pyspiel.register_game(_GAME_TYPE, KuhnPokerGame)