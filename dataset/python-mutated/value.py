"""Representation of a value for a game.

This is a standard representation for passing value functions into algorithms,
with currently the following implementations:

The main way of using a value is to call `value(state)`
or `value(state, action)`.

We will prevent calling a value on a state action on a MEAN_FIELD state.

The state can be a pyspiel.State object or its string representation. For a
particular ValueFunction instance, you should use only one or the other. The
behavior may be undefined for mixed usage depending on the implementation.
"""
import collections
from typing import Union
import pyspiel
ValueFunctionState = Union[pyspiel.State, str]

class ValueFunction(object):
    """Base class for values.

  A ValueFunction is something that returns a value given
  a state of the world or a state and an action.

  Attributes:
    game: the game for which this ValueFunction derives
  """

    def __init__(self, game):
        if False:
            while True:
                i = 10
        'Initializes a value.\n\n    Args:\n      game: the game for which this value derives\n    '
        self.game = game

    def value(self, state: ValueFunctionState, action=None) -> float:
        if False:
            return 10
        'Returns a float representing a value.\n\n    Args:\n      state: A `pyspiel.State` object or its string representation.\n      action: may be None or a legal action\n\n    Returns:\n      A value for the state (and eventuallu state action pair).\n    '
        raise NotImplementedError()

    def __call__(self, state: ValueFunctionState, action=None) -> float:
        if False:
            print('Hello World!')
        'Turns the value into a callable.\n\n    Args:\n      state: A `pyspiel.State` object or its string representation.\n      action: may be None or a legal action\n\n    Returns:\n      Float: the value of the state or the state action pair.\n    '
        return self.value(state, action=action)

    def set_value(self, state: ValueFunctionState, value: float, action=None):
        if False:
            return 10
        'Sets the value of the state.\n\n    Args:\n      state: A `pyspiel.State` object or its string representation.\n      value: Value of the state.\n      action: may be None or a legal action\n    '
        raise NotImplementedError()

    def has(self, state: ValueFunctionState, action=None) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns true if state(-action) has an explicit value.\n\n    Args:\n      state: A `pyspiel.State` object or its string representation.\n      action: may be None or a legal action\n\n    Returns:\n      True if there is an explicitly specified value.\n    '
        raise NotImplementedError()

    def add_value(self, state, value: float, action=None):
        if False:
            print('Hello World!')
        'Adds the value to the current value of the state.\n\n    Args:\n      state: A `pyspiel.State` object or its string representation.\n      value: Value to add.\n      action: may be None or a legal action\n    '
        self.set_value(state, self.value(state, action=action) + value, action=action)

class TabularValueFunction(ValueFunction):
    """Tabular value function backed by a dictionary."""

    def __init__(self, game):
        if False:
            i = 10
            return i + 15
        super().__init__(game)
        self._values = collections.defaultdict(float)

    def value(self, state: ValueFunctionState, action=None):
        if False:
            i = 10
            return i + 15
        return self._values[state, action]

    def set_value(self, state: ValueFunctionState, value: float, action=None):
        if False:
            print('Hello World!')
        self._values[state, action] = value

    def has(self, state: ValueFunctionState, action=None):
        if False:
            for i in range(10):
                print('nop')
        return (state, action) in self._values