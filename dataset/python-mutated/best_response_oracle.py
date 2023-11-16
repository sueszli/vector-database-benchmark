"""An Oracle for Exact Best Responses.

This class computes the best responses against sets of policies.
"""
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import policy_utils
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import utils
import pyspiel

class BestResponseOracle(optimization_oracle.AbstractOracle):
    """Oracle using exact best responses to compute BR to policies."""

    def __init__(self, best_response_backend='cpp', game=None, all_states=None, state_to_information_state=None, prob_cut_threshold=-1.0, action_value_tolerance=-1.0, **kwargs):
        if False:
            return 10
        "Init function for the RLOracle.\n\n    Args:\n      best_response_backend: A string (either 'cpp' or 'py'), specifying the\n        best response backend to use (C++ or python, respectively). The cpp\n        backend should be preferred, generally, as it is significantly faster.\n      game: The game on which the optimization process takes place.\n      all_states: The result of calling get_all_states.get_all_states. Cached\n        for improved performance.\n      state_to_information_state: A dict mapping str(state) to\n        state.information_state for every state in the game. Cached for improved\n        performance.\n      prob_cut_threshold: For cpp backend, a partially computed best-response\n        can be computed when using a prob_cut_threshold >= 0.\n      action_value_tolerance: For cpp backend, the max-entropy best-response\n        policy is computed if a non-negative `action_value_tolerance` is used.\n      **kwargs: kwargs\n    "
        super(BestResponseOracle, self).__init__(**kwargs)
        self.best_response_backend = best_response_backend
        if self.best_response_backend == 'cpp':
            (self.all_states, self.state_to_information_state) = utils.compute_states_and_info_states_if_none(game, all_states, state_to_information_state)
            policy = openspiel_policy.UniformRandomPolicy(game)
            policy_to_dict = policy_utils.policy_to_dict(policy, game, self.all_states, self.state_to_information_state)
            self.best_response_processors = [pyspiel.TabularBestResponse(game, best_responder_id, policy_to_dict, prob_cut_threshold, action_value_tolerance) for best_responder_id in range(game.num_players())]
            self.best_responders = [best_response.CPPBestResponsePolicy(game, i_player, policy, self.all_states, self.state_to_information_state, self.best_response_processors[i_player]) for i_player in range(game.num_players())]

    def __call__(self, game, training_parameters, strategy_sampler=utils.sample_strategy, using_joint_strategies=False, **oracle_specific_execution_kwargs):
        if False:
            return 10
        'Call method for oracle, returns best responses for training_parameters.\n\n    Args:\n      game: The game on which the optimization process takes place.\n      training_parameters: List of list of dicts: one list per player, one dict\n        per selected agent in the pool for each player,\n        each dictionary containing the following fields:\n        - policy: the policy from which to start training.\n        - total_policies: A list of all policy.Policy strategies used for\n          training, including the one for the current player. Either\n          marginalized or joint strategies are accepted.\n        - current_player: Integer representing the current player.\n        - probabilities_of_playing_policies: A list of arrays representing, per\n          player, the probabilities of playing each policy in total_policies for\n          the same player.\n      strategy_sampler: Callable that samples strategies from `total_policies`\n        using `probabilities_of_playing_policies`. It only samples one joint\n        "action" for all players. Implemented to be able to take into account\n        joint probabilities of action.\n      using_joint_strategies: Whether the meta-strategies sent are joint (True)\n        or marginalized.\n      **oracle_specific_execution_kwargs: Other set of arguments, for\n        compatibility purposes. Can for example represent whether to Rectify\n        Training or not.\n\n    Returns:\n      A list of list of OpenSpiel Policy objects representing the expected\n      best response, following the same structure as training_parameters.\n    '
        new_policies = []
        for player_parameters in training_parameters:
            player_policies = []
            for params in player_parameters:
                current_player = params['current_player']
                total_policies = params['total_policies']
                probabilities_of_playing_policies = params['probabilities_of_playing_policies']
                if using_joint_strategies:
                    aggr_policy = utils.aggregate_joint_policies(game, utils.marginal_to_joint(total_policies), probabilities_of_playing_policies.reshape(-1))
                else:
                    aggr_policy = utils.aggregate_policies(game, total_policies, probabilities_of_playing_policies)
                if self.best_response_backend == 'py':
                    best_resp = best_response.BestResponsePolicy(game, current_player, aggr_policy)
                else:
                    self.best_response_processors[current_player].set_policy(policy_utils.policy_to_dict(aggr_policy, game, self.all_states, self.state_to_information_state))
                    self.best_responders[current_player] = best_response.CPPBestResponsePolicy(game, current_player, aggr_policy, self.all_states, self.state_to_information_state, self.best_response_processors[current_player])
                    best_resp = self.best_responders[current_player]
                player_policies.append(best_resp)
            new_policies.append(player_policies)
        return new_policies