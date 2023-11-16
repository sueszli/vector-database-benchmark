"""Mirror Descent (https://arxiv.org/pdf/2103.00623.pdf)."""
from typing import Dict, List, Optional
import numpy as np
from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import distribution
import pyspiel

def softmax_projection(logits):
    if False:
        while True:
            i = 10
    max_l = max(logits)
    exp_l = [np.exp(l - max_l) for l in logits]
    norm_exp = sum(exp_l)
    return [l / norm_exp for l in exp_l]

class ProjectedPolicy(policy_lib.Policy):
    """Project values on the policy simplex."""

    def __init__(self, game: pyspiel.Game, player_ids: List[int], state_value: value.ValueFunction, coeff: float=1.0):
        if False:
            i = 10
            return i + 15
        'Initializes the projected policy.\n\n    Args:\n      game: The game to analyze.\n      player_ids: list of player ids for which this policy applies; each should\n        be in the range 0..game.num_players()-1.\n      state_value: The (cumulative) state value to project.\n      coeff: Coefficient for the values of the states.\n    '
        super(ProjectedPolicy, self).__init__(game, player_ids)
        self._state_value = state_value
        self._coeff = coeff

    def value(self, state: pyspiel.State, action: Optional[int]=None) -> float:
        if False:
            for i in range(10):
                print('nop')
        if action is None:
            return self._state_value(state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))
        else:
            new_state = state.child(action)
            return state.rewards()[0] + self._state_value(new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))

    def action_probabilities(self, state: pyspiel.State, player_id: Optional[int]=None) -> Dict[int, float]:
        if False:
            i = 10
            return i + 15
        del player_id
        action_logit = [(a, self._coeff * self.value(state, action=a)) for a in state.legal_actions()]
        (action, logit) = zip(*action_logit)
        return dict(zip(action, softmax_projection(logit)))

class MirrorDescent(object):
    """The mirror descent algorithm."""

    def __init__(self, game: pyspiel.Game, state_value: Optional[value.ValueFunction]=None, lr: float=0.01, root_state: Optional[pyspiel.State]=None):
        if False:
            while True:
                i = 10
        'Initializes mirror descent.\n\n    Args:\n      game: The game,\n      state_value: A state value function. Default to TabularValueFunction.\n      lr: The learning rate of mirror descent,\n      root_state: The state of the game at which to start. If `None`, the game\n        root state is used.\n    '
        self._game = game
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
        self._policy = policy_lib.UniformRandomPolicy(game)
        self._distribution = distribution.DistributionPolicy(game, self._policy)
        self._md_step = 0
        self._lr = lr
        self._state_value = state_value if state_value else value.TabularValueFunction(game)
        self._cumulative_state_value = value.TabularValueFunction(game)

    def get_state_value(self, state: pyspiel.State, learning_rate: float) -> float:
        if False:
            print('Hello World!')
        'Returns the value of the state.'
        if state.is_terminal():
            return state.rewards()[state.mean_field_population()]
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            v = 0.0
            for (action, prob) in state.chance_outcomes():
                new_state = state.child(action)
                v += prob * self.eval_state(new_state, learning_rate)
            return v
        if state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
            dist_to_register = state.distribution_support()
            dist = [self._distribution.value_str(str_state, 0.0) for str_state in dist_to_register]
            new_state = state.clone()
            new_state.update_distribution(dist)
            return state.rewards()[state.mean_field_population()] + self.eval_state(new_state, learning_rate)
        assert int(state.current_player()) >= 0, 'The player id should be >= 0'
        v = 0.0
        for (action, prob) in self._policy.action_probabilities(state).items():
            new_state = state.child(action)
            v += prob * self.eval_state(new_state, learning_rate)
        return state.rewards()[state.mean_field_population()] + v

    def eval_state(self, state: pyspiel.State, learning_rate: float) -> float:
        if False:
            i = 10
            return i + 15
        'Evaluate the value of a state and update the cumulative sum.'
        state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
        if self._state_value.has(state_str):
            return self._state_value(state_str)
        v = self.get_state_value(state, learning_rate)
        self._state_value.set_value(state_str, v)
        self._cumulative_state_value.add_value(state_str, learning_rate * v)
        return v

    def get_projected_policy(self) -> policy_lib.Policy:
        if False:
            print('Hello World!')
        'Returns the projected policy.'
        return ProjectedPolicy(self._game, list(range(self._game.num_players())), self._cumulative_state_value)

    def iteration(self, learning_rate: Optional[float]=None):
        if False:
            i = 10
            return i + 15
        'An iteration of Mirror Descent.'
        self._md_step += 1
        self._state_value = value.TabularValueFunction(self._game)
        for state in self._root_states:
            self.eval_state(state, learning_rate if learning_rate else self._lr)
        self._policy = self.get_projected_policy()
        self._distribution = distribution.DistributionPolicy(self._game, self._policy)

    def get_policy(self) -> policy_lib.Policy:
        if False:
            i = 10
            return i + 15
        return self._policy

    @property
    def distribution(self) -> distribution.DistributionPolicy:
        if False:
            while True:
                i = 10
        return self._distribution