"""Compute the value of action given a policy vs a best responder."""
import collections
from open_spiel.python import policy
from open_spiel.python.algorithms import action_value
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_utils
import pyspiel

def _transitions(state, policies):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of (action, prob) pairs from the specified state.'
    if state.is_chance_node():
        return state.chance_outcomes()
    else:
        pl = state.current_player()
        return list(policies[pl].action_probabilities(state).items())

def _tuples_from_policy(policy_vector):
    if False:
        while True:
            i = 10
    return [(action, probability) for (action, probability) in enumerate(policy_vector)]
_CalculatorReturn = collections.namedtuple('_CalculatorReturn', ['exploitability', 'values_vs_br', 'counterfactual_reach_probs_vs_br', 'player_reach_probs_vs_br'])

class Calculator(object):
    """Class to orchestrate the calculation."""

    def __init__(self, game):
        if False:
            for i in range(10):
                print('nop')
        if game.num_players() != 2:
            raise ValueError('Only supports 2-player games.')
        self.game = game
        self._num_players = game.num_players()
        self._num_actions = game.num_distinct_actions()
        self._action_value_calculator = action_value.TreeWalkCalculator(game)
        self._best_responder = {0: None, 1: None}
        self._all_states = None

    def __call__(self, player, player_policy, info_states):
        if False:
            for i in range(10):
                print('nop')
        'Computes action values per state for the player.\n\n    Args:\n      player: The id of the player (0 <= player < game.num_players()). This\n        player will play `player_policy`, while the opponent will play a best\n        response.\n      player_policy: A `policy.Policy` object.\n      info_states: A list of info state strings.\n\n    Returns:\n      A `_CalculatorReturn` nametuple. See its docstring for the documentation.\n    '
        self.player = player
        opponent = 1 - player

        def best_response_policy(state):
            if False:
                return 10
            infostate = state.information_state_string(opponent)
            action = best_response_actions[infostate]
            return [(action, 1.0)]
        if isinstance(player_policy, policy.TabularPolicy):
            tabular_policy = {key: _tuples_from_policy(player_policy.policy_for_key(key)) for key in player_policy.state_lookup}
        else:
            if self._all_states is None:
                self._all_states = get_all_states.get_all_states(self.game, depth_limit=-1, include_terminals=False, include_chance_states=False)
                self._state_to_information_state = {state: self._all_states[state].information_state_string() for state in self._all_states}
            tabular_policy = policy_utils.policy_to_dict(player_policy, self.game, self._all_states, self._state_to_information_state)
        if self._best_responder[player] is None:
            self._best_responder[player] = pyspiel.TabularBestResponse(self.game, opponent, tabular_policy)
        else:
            self._best_responder[player].set_policy(tabular_policy)
        best_response_value = self._best_responder[player].value_from_state(self.game.new_initial_state())
        best_response_actions = self._best_responder[player].get_best_response_actions()
        self._action_value_calculator.compute_all_states_action_values({player: player_policy, opponent: policy.tabular_policy_from_callable(self.game, best_response_policy, [opponent])})
        obj = self._action_value_calculator._get_tabular_statistics(((player, s) for s in info_states))
        return _CalculatorReturn(exploitability=best_response_value, values_vs_br=obj.action_values, counterfactual_reach_probs_vs_br=obj.counterfactual_reach_probs, player_reach_probs_vs_br=obj.player_reach_probs)