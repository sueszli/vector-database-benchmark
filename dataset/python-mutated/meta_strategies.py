"""Meta-strategy solvers for PSRO."""
import numpy as np
from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.algorithms import regret_matching
import pyspiel
EPSILON_MIN_POSITIVE_PROBA = 1e-08

def uniform_strategy(solver, return_joint=False):
    if False:
        while True:
            i = 10
    'Returns a Random Uniform distribution on policies.\n\n  Args:\n    solver: GenPSROSolver instance.\n    return_joint: If true, only returns marginals. Otherwise marginals as well\n      as joint probabilities.\n\n  Returns:\n    uniform distribution on strategies.\n  '
    policies = solver.get_policies()
    policy_lengths = [len(pol) for pol in policies]
    result = [np.ones(pol_len) / pol_len for pol_len in policy_lengths]
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return (result, joint_strategies)

def softmax_on_range(number_policies):
    if False:
        i = 10
        return i + 15
    x = np.array(list(range(number_policies)))
    x = np.exp(x - x.max())
    x /= np.sum(x)
    return x

def uniform_biased_strategy(solver, return_joint=False):
    if False:
        print('Hello World!')
    'Returns a Biased Random Uniform distribution on policies.\n\n  The uniform distribution is biased to prioritize playing against more recent\n  policies (Policies that were appended to the policy list later in training)\n  instead of older ones.\n\n  Args:\n    solver: GenPSROSolver instance.\n    return_joint: If true, only returns marginals. Otherwise marginals as well\n      as joint probabilities.\n\n  Returns:\n    uniform distribution on strategies.\n  '
    policies = solver.get_policies()
    if not isinstance(policies[0], list):
        policies = [policies]
    policy_lengths = [len(pol) for pol in policies]
    result = [softmax_on_range(pol_len) for pol_len in policy_lengths]
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return (result, joint_strategies)

def renormalize(probabilities):
    if False:
        print('Hello World!')
    'Replaces all negative entries with zeroes and normalizes the result.\n\n  Args:\n    probabilities: probability vector to renormalize. Has to be one-dimensional.\n\n  Returns:\n    Renormalized probabilities.\n  '
    probabilities[probabilities < 0] = 0
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def get_joint_strategy_from_marginals(probabilities):
    if False:
        return 10
    'Returns a joint strategy matrix from a list of marginals.\n\n  Args:\n    probabilities: list of probabilities.\n\n  Returns:\n    A joint strategy from a list of marginals.\n  '
    probas = []
    for i in range(len(probabilities)):
        probas_shapes = [1] * len(probabilities)
        probas_shapes[i] = -1
        probas.append(probabilities[i].reshape(*probas_shapes))
    result = np.prod(probas)
    return result.reshape(-1)

def nash_strategy(solver, return_joint=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns nash distribution on meta game matrix.\n\n  This method only works for two player zero-sum games.\n\n  Args:\n    solver: GenPSROSolver instance.\n    return_joint: If true, only returns marginals. Otherwise marginals as well\n      as joint probabilities.\n\n  Returns:\n    Nash distribution on strategies.\n  '
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    meta_games = [x.tolist() for x in meta_games]
    if len(meta_games) != 2:
        raise NotImplementedError('nash_strategy solver works only for 2p zero-sumgames, but was invoked for a {} player game'.format(len(meta_games)))
    (nash_prob_1, nash_prob_2, _, _) = lp_solver.solve_zero_sum_matrix_game(pyspiel.create_matrix_game(*meta_games))
    result = [renormalize(np.array(nash_prob_1).reshape(-1)), renormalize(np.array(nash_prob_2).reshape(-1))]
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return (result, joint_strategies)

def prd_strategy(solver, return_joint=False):
    if False:
        while True:
            i = 10
    'Computes Projected Replicator Dynamics strategies.\n\n  Args:\n    solver: GenPSROSolver instance.\n    return_joint: If true, only returns marginals. Otherwise marginals as well\n      as joint probabilities.\n\n  Returns:\n    PRD-computed strategies.\n  '
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    kwargs = solver.get_kwargs()
    result = projected_replicator_dynamics.projected_replicator_dynamics(meta_games, **kwargs)
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return (result, joint_strategies)

def rm_strategy(solver, return_joint=False):
    if False:
        while True:
            i = 10
    'Computes regret-matching strategies.\n\n  Args:\n    solver: GenPSROSolver instance.\n    return_joint: If true, only returns marginals. Otherwise marginals as well\n      as joint probabilities.\n\n  Returns:\n    PRD-computed strategies.\n  '
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    kwargs = solver.get_kwargs()
    result = regret_matching.regret_matching(meta_games, **kwargs)
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return (result, joint_strategies)
META_STRATEGY_METHODS = {'uniform_biased': uniform_biased_strategy, 'uniform': uniform_strategy, 'nash': nash_strategy, 'prd': prd_strategy, 'rm': rm_strategy}