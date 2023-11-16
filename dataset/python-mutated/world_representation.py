"""API for world state representation."""
import abc
from typing import Any, List, Text, Tuple

class WorldState(abc.ABC):
    """Base class for world state representation.

    We can implement this class for world state representations in both
    sequential and matrix games.

    Attributes:
      chance_policy: Policy of the chance node in the game tree.
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.chance_policy = {0: 1.0}
        self._history = []

    @abc.abstractmethod
    def get_distinct_actions(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        'Returns all possible distinct actions in the game.'
        pass

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns if the current state of the game is a terminal or not.'
        pass

    @abc.abstractmethod
    def get_actions(self) -> List[Any]:
        if False:
            return 10
        'Returns the list of legal actions from the current state of the game.'
        pass

    @abc.abstractmethod
    def get_infostate_string(self, player: int) -> Text:
        if False:
            while True:
                i = 10
        'Returns the string form of infostate representation of a given player.\n\n    Args:\n      player: Index of player.\n\n    Returns:\n      The string representation of the infostate of player.\n    '
        pass

    @abc.abstractmethod
    def apply_actions(self, actions: Tuple[int, int, int]) -> None:
        if False:
            i = 10
            return i + 15
        "Applies the current player's action to change state of the world.\n\n    At each timestep of the game, the state of the world is changing by the\n    current player's action. At the same time, we should update self._history\n    with actions, by appending actions to self._history.\n\n    Args:\n      actions: List of actions for chance node, player 1 and player 2.\n\n    "
        pass

    @abc.abstractmethod
    def get_utility(self, player: int) -> float:
        if False:
            for i in range(10):
                print('nop')
        "Returns player's utility when the game reaches to a terminal state.\n\n    Args:\n      player: Index of player.\n\n    Returns:\n      Utility that player receives when we reach a terminal state in the game.\n    "
        pass