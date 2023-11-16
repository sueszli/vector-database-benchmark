"""Class of Optimization Oracles generating best response against opponents.

Oracles are as defined in (Lanctot et Al., 2017,
https://arxiv.org/pdf/1711.00832.pdf ), functions generating a best response
against a probabilistic mixture of opponents. This class implements the abstract
class of oracles, and a simple oracle using Evolutionary Strategy as
optimization method.
"""
import numpy as np

def strategy_sampler_fun(total_policies, probabilities_of_playing_policies):
    if False:
        i = 10
        return i + 15
    'Samples strategies according to distribution over them.\n\n  Args:\n    total_policies: List of lists of policies for each player.\n    probabilities_of_playing_policies: List of numpy arrays representing the\n      probability of playing a strategy.\n\n  Returns:\n    One sampled joint strategy.\n  '
    policies_selected = []
    for k in range(len(total_policies)):
        selected_opponent = np.random.choice(total_policies[k], 1, p=probabilities_of_playing_policies[k]).reshape(-1)[0]
        policies_selected.append(selected_opponent)
    return policies_selected

class AbstractOracle(object):
    """The abstract class representing oracles, a hidden optimization process."""

    def __init__(self, number_policies_sampled=100, **oracle_specific_kwargs):
        if False:
            return 10
        'Initialization method for oracle.\n\n    Args:\n      number_policies_sampled: Number of different opponent policies sampled\n        during evaluation of policy.\n      **oracle_specific_kwargs: Oracle specific args, compatibility\n        purpose. Since oracles can vary so much in their implementation, no\n        specific argument constraint is put on this function.\n    '
        self._number_policies_sampled = number_policies_sampled
        self._kwargs = oracle_specific_kwargs

    def set_iteration_numbers(self, number_policies_sampled):
        if False:
            return 10
        'Changes the number of iterations used for computing episode returns.\n\n    Args:\n      number_policies_sampled: Number of different opponent policies sampled\n        during evaluation of policy.\n    '
        self._number_policies_sampled = number_policies_sampled

    def __call__(self, game, policy, total_policies, current_player, probabilities_of_playing_policies, **oracle_specific_execution_kwargs):
        if False:
            i = 10
            return i + 15
        'Call method for oracle, returns best response against a set of policies.\n\n    Args:\n      game: The game on which the optimization process takes place.\n      policy: The current policy, in policy.Policy, from which we wish to start\n        optimizing.\n      total_policies: A list of all policy.Policy strategies used for training,\n        including the one for the current player.\n      current_player: Integer representing the current player.\n      probabilities_of_playing_policies: A list of arrays representing, per\n        player, the probabilities of playing each policy in total_policies for\n        the same player.\n      **oracle_specific_execution_kwargs: Other set of arguments, for\n        compatibility purposes. Can for example represent whether to Rectify\n        Training or not.\n    '
        raise NotImplementedError('Calling Abstract class method.')

    def sample_episode(self, game, policies_selected):
        if False:
            print('Hello World!')
        raise NotImplementedError('Calling Abstract class method.')

    def evaluate_policy(self, game, pol, total_policies, current_player, probabilities_of_playing_policies, strategy_sampler=strategy_sampler_fun, **oracle_specific_execution_kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates a specific policy against a nash mixture of policies.\n\n    Args:\n      game: The game on which the optimization process takes place.\n      pol: The current policy, in policy.Policy, from which we wish to start\n        optimizing.\n      total_policies: A list of all policy.Policy strategies used for training,\n        including the one for the current player.\n      current_player: Integer representing the current player.\n      probabilities_of_playing_policies: A list of arrays representing, per\n        player, the probabilities of playing each policy in total_policies for\n        the same player.\n      strategy_sampler: callable sampling strategy.\n      **oracle_specific_execution_kwargs: Other set of arguments, for\n        compatibility purposes. Can for example represent whether to Rectify\n        Training or not.\n\n    Returns:\n      Average return for policy when played against policies_played_against.\n    '
        del oracle_specific_execution_kwargs
        totals = 0
        count = 0
        for _ in range(self._number_policies_sampled):
            policies_selected = strategy_sampler(total_policies, probabilities_of_playing_policies)
            policies_selected[current_player] = pol
            new_return = self.sample_episode(game, policies_selected)[current_player]
            totals += new_return
            count += 1
        return totals / max(1, count)