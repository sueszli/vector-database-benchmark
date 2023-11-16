"""Tests for open_spiel.python.algorithms.psro_v2.strategy_selectors."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms.psro_v2 import strategy_selectors

class FakeSolver(object):

    def __init__(self, strategies, policies):
        if False:
            return 10
        self.strategies = strategies
        self.policies = policies

    def get_policies(self):
        if False:
            i = 10
            return i + 15
        return self.policies

    def get_meta_strategies(self):
        if False:
            print('Hello World!')
        return self.strategies

def equal_to_transposition_lists(a, b):
    if False:
        print('Hello World!')
    return [set(x) for x in a] == [set(x) for x in b]
EPSILON_MIN_POSITIVE_PROBA = 1e-08

def rectified_alias(solver, number_policies_to_select):
    if False:
        for i in range(10):
            print('nop')
    'Returns every strategy with nonzero selection probability.\n\n  Args:\n    solver: A GenPSROSolver instance.\n    number_policies_to_select: Number policies to select\n\n  Returns:\n    used_policies: A list, each element a list of the policies used per player.\n  '
    del number_policies_to_select
    used_policies = []
    used_policy_indexes = []
    policies = solver.get_policies()
    num_players = len(policies)
    meta_strategy_probabilities = solver.get_meta_strategies()
    for k in range(num_players):
        current_policies = policies[k]
        current_probabilities = meta_strategy_probabilities[k]
        current_indexes = [i for i in range(len(current_policies)) if current_probabilities[i] > EPSILON_MIN_POSITIVE_PROBA]
        current_policies = [current_policies[i] for i in current_indexes]
        used_policy_indexes.append(current_indexes)
        used_policies.append(current_policies)
    return (used_policies, used_policy_indexes)

def probabilistic_alias(solver, number_policies_to_select):
    if False:
        return 10
    'Returns [kwargs] policies randomly, proportionally with selection probas.\n\n  Args:\n    solver: A GenPSROSolver instance.\n    number_policies_to_select: Number policies to select\n  '
    policies = solver.get_policies()
    num_players = len(policies)
    meta_strategy_probabilities = solver.get_meta_strategies()
    print(policies, meta_strategy_probabilities)
    used_policies = []
    used_policy_indexes = []
    for k in range(num_players):
        current_policies = policies[k]
        current_selection_probabilities = meta_strategy_probabilities[k]
        effective_number = min(number_policies_to_select, len(current_policies))
        selected_indexes = list(np.random.choice(list(range(len(current_policies))), effective_number, replace=False, p=current_selection_probabilities))
        selected_policies = [current_policies[i] for i in selected_indexes]
        used_policies.append(selected_policies)
        used_policy_indexes.append(selected_indexes)
    return (used_policies, used_policy_indexes)

def top_k_probabilities_alias(solver, number_policies_to_select):
    if False:
        i = 10
        return i + 15
    'Returns [kwargs] policies with highest selection probabilities.\n\n  Args:\n    solver: A GenPSROSolver instance.\n    number_policies_to_select: Number policies to select\n  '
    policies = solver.get_policies()
    num_players = len(policies)
    meta_strategy_probabilities = solver.get_meta_strategies()
    used_policies = []
    used_policy_indexes = []
    for k in range(num_players):
        current_policies = policies[k]
        current_selection_probabilities = meta_strategy_probabilities[k]
        effective_number = min(number_policies_to_select, len(current_policies))
        selected_indexes = [index for (_, index) in sorted(zip(current_selection_probabilities, list(range(len(current_policies)))), key=lambda pair: pair[0])][:effective_number]
        selected_policies = [current_policies[i] for i in selected_indexes]
        used_policies.append(selected_policies)
        used_policy_indexes.append(selected_indexes)
    return (used_policies, used_policy_indexes)

class StrategySelectorsTest(absltest.TestCase):

    def test_vital(self):
        if False:
            return 10
        n_tests = 1000
        number_strategies = 50
        number_players = 3
        for i in range(n_tests):
            probabilities = np.random.uniform(size=(number_players, number_strategies))
            probabilities /= np.sum(probabilities, axis=1).reshape(-1, 1)
            probabilities = list(probabilities)
            policies = [list(range(number_strategies)) for _ in range(number_players)]
            solver = FakeSolver(probabilities, policies)
            probabilities[0][0] = 0
            probabilities[-1][-1] = 0
            (a, b) = strategy_selectors.rectified(solver, 1)
            (c, d) = rectified_alias(solver, 1)
            self.assertEqual(a, c, 'Rectified failed.')
            self.assertEqual(b, d, 'Rectified failed.')
            (a, b) = strategy_selectors.top_k_probabilities(solver, 3)
            (c, d) = top_k_probabilities_alias(solver, 3)
            self.assertEqual(a, c, 'Top k failed.')
            self.assertEqual(b, d, 'Top k failed.')
            n_nonzero_policies = 2
            probabilities = [np.zeros(number_strategies) for _ in range(number_players)]
            for player in range(number_players):
                for _ in range(n_nonzero_policies):
                    i = np.random.randint(0, high=number_strategies)
                    while probabilities[player][i] > 1e-12:
                        i = np.random.randint(0, high=number_strategies)
                    probabilities[player][i] = 1.0 / n_nonzero_policies
                probabilities[player] /= np.sum(probabilities[player])
            solver = FakeSolver(probabilities, policies)
            (a, b) = strategy_selectors.probabilistic(solver, n_nonzero_policies)
            (c, d) = probabilistic_alias(solver, n_nonzero_policies)
            self.assertTrue(equal_to_transposition_lists(a, c), 'Probabilistic failed.')
            self.assertTrue(equal_to_transposition_lists(b, d), 'Probabilistic failed.')
if __name__ == '__main__':
    absltest.main()