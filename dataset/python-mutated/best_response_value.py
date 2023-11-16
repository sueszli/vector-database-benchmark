"""Does a backward pass to output a value of a best response policy."""
from typing import Optional
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
import pyspiel

class BestResponse(value.ValueFunction):
    """Computes a best response value."""

    def __init__(self, game, distribution: distribution_std.Distribution, state_value: Optional[value.ValueFunction]=None, root_state=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the best response calculation.\n\n    Args:\n      game: The game to analyze.\n      distribution: A `distribution_std.Distribution` object.\n      state_value: A state value function. Default to TabularValueFunction.\n      root_state: The state of the game at which to start. If `None`, the game\n        root state is used.\n    '
        super().__init__(game)
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
        self._distribution = distribution
        self._state_value = state_value if state_value else value.TabularValueFunction(game)
        self.evaluate()

    def eval_state(self, state):
        if False:
            while True:
                i = 10
        'Evaluate the value of a state.\n\n    Args:\n      state: a game state.\n\n    Returns:\n      the optimal value of the state\n\n    Recursively computes the value of the optimal policy given the fixed state\n    distribution. `self._state_value` is used as a cache for pre-computed\n    values.\n    '
        state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
        if self._state_value.has(state_str):
            return self._state_value(state_str)
        if state.is_terminal():
            self._state_value.set_value(state_str, state.rewards()[state.mean_field_population()])
            return self._state_value(state_str)
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            self._state_value.set_value(state_str, 0.0)
            for (action, prob) in state.chance_outcomes():
                new_state = state.child(action)
                self._state_value.add_value(state_str, prob * self.eval_state(new_state))
            return self._state_value(state_str)
        if state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
            dist = [self._distribution.value_str(str_state, 0.0) for str_state in state.distribution_support()]
            new_state = state.clone()
            new_state.update_distribution(dist)
            self._state_value.set_value(state_str, state.rewards()[state.mean_field_population()] + self.eval_state(new_state))
            return self._state_value(state_str)
        else:
            assert int(state.current_player()) >= 0, 'The player id should be >= 0'
            max_q = max((self.eval_state(state.child(action)) for action in state.legal_actions()))
            self._state_value.set_value(state_str, state.rewards()[state.mean_field_population()] + max_q)
            return self._state_value(state_str)

    def evaluate(self):
        if False:
            print('Hello World!')
        'Evaluate the best response value on all states.'
        for state in self._root_states:
            self.eval_state(state)

    def value(self, state, action=None):
        if False:
            return 10
        if action is None:
            return self._state_value(state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))
        new_state = state.child(action)
        return state.rewards()[state.mean_field_population()] + self._state_value(new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))