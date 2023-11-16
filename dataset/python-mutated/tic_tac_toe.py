"""Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""
import numpy as np
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel
_NUM_PLAYERS = 2
_NUM_ROWS = 3
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(short_name='python_tic_tac_toe', long_name='Python Tic-Tac-Toe', dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL, chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC, information=pyspiel.GameType.Information.PERFECT_INFORMATION, utility=pyspiel.GameType.Utility.ZERO_SUM, reward_model=pyspiel.GameType.RewardModel.TERMINAL, max_num_players=_NUM_PLAYERS, min_num_players=_NUM_PLAYERS, provides_information_state_string=True, provides_information_state_tensor=False, provides_observation_string=True, provides_observation_tensor=True, parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(num_distinct_actions=_NUM_CELLS, max_chance_outcomes=0, num_players=2, min_utility=-1.0, max_utility=1.0, utility_sum=0.0, max_game_length=_NUM_CELLS)

class TicTacToeGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""

    def __init__(self, params=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        if False:
            while True:
                i = 10
        'Returns a state corresponding to the start of a game.'
        return TicTacToeState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        if False:
            i = 10
            return i + 15
        'Returns an object used for observing game state.'
        if iig_obs_type is None or (iig_obs_type.public_info and (not iig_obs_type.perfect_recall)):
            return BoardObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)

class TicTacToeState(pyspiel.State):
    """A python version of the Tic-Tac-Toe state."""

    def __init__(self, game):
        if False:
            while True:
                i = 10
        'Constructor; should only be called by Game.new_initial_state.'
        super().__init__(game)
        self._cur_player = 0
        self._player0_score = 0.0
        self._is_terminal = False
        self.board = np.full((_NUM_ROWS, _NUM_COLS), '.')

    def current_player(self):
        if False:
            while True:
                i = 10
        'Returns id of the next player to move, or TERMINAL if game is over.'
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        if False:
            while True:
                i = 10
        'Returns a list of legal actions, sorted in ascending order.'
        return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == '.']

    def _apply_action(self, action):
        if False:
            return 10
        'Applies the specified action to the state.'
        self.board[_coord(action)] = 'x' if self._cur_player == 0 else 'o'
        if _line_exists(self.board):
            self._is_terminal = True
            self._player0_score = 1.0 if self._cur_player == 0 else -1.0
        elif all(self.board.ravel() != '.'):
            self._is_terminal = True
        else:
            self._cur_player = 1 - self._cur_player

    def _action_to_string(self, player, action):
        if False:
            return 10
        'Action -> string.'
        (row, col) = _coord(action)
        return '{}({},{})'.format('x' if player == 0 else 'o', row, col)

    def is_terminal(self):
        if False:
            while True:
                i = 10
        'Returns True if the game is over.'
        return self._is_terminal

    def returns(self):
        if False:
            for i in range(10):
                print('nop')
        'Total reward for each player over the course of the game so far.'
        return [self._player0_score, -self._player0_score]

    def __str__(self):
        if False:
            print('Hello World!')
        'String for debug purposes. No particular semantics are required.'
        return _board_to_string(self.board)

class BoardObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        'Initializes an empty observation tensor.'
        if params:
            raise ValueError(f'Observation parameters not supported; passed {params}')
        shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {'observation': np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        if False:
            print('Hello World!')
        'Updates `tensor` and `dict` to reflect `state` from PoV of `player`.'
        del player
        obs = self.dict['observation']
        obs.fill(0)
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                cell_state = '.ox'.index(state.board[row, col])
                obs[cell_state, row, col] = 1

    def string_from(self, state, player):
        if False:
            return 10
        'Observation of `state` from the PoV of `player`, as a string.'
        del player
        return _board_to_string(state.board)

def _line_value(line):
    if False:
        return 10
    'Checks a possible line, returning the winning symbol if any.'
    if all(line == 'x') or all(line == 'o'):
        return line[0]

def _line_exists(board):
    if False:
        for i in range(10):
            print('nop')
    'Checks if a line exists, returns "x" or "o" if so, and None otherwise.'
    return _line_value(board[0]) or _line_value(board[1]) or _line_value(board[2]) or _line_value(board[:, 0]) or _line_value(board[:, 1]) or _line_value(board[:, 2]) or _line_value(board.diagonal()) or _line_value(np.fliplr(board).diagonal())

def _coord(move):
    if False:
        i = 10
        return i + 15
    'Returns (row, col) from an action id.'
    return (move // _NUM_COLS, move % _NUM_COLS)

def _board_to_string(board):
    if False:
        return 10
    'Returns a string representation of the board.'
    return '\n'.join((''.join(row) for row in board))
pyspiel.register_game(_GAME_TYPE, TicTacToeGame)