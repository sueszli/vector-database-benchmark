"""MIP-Nash.

Based on the first formulation of
 https://dl.acm.org/doi/10.5555/1619410.1619413.
Compute optimal Nash equilibrium of two-player general-sum games
by solving a mixed-integer programming problem.
"""
import cvxpy as cp
import numpy as np
from open_spiel.python.algorithms.projected_replicator_dynamics import _simplex_projection
from open_spiel.python.egt.utils import game_payoffs_array

def mip_nash(game, objective, solver='GLPK_MI'):
    if False:
        for i in range(10):
            print('nop')
    'Solves for the optimal Nash for two-player general-sum games.\n\n    Using mixed-integer programming:\n      min f(x_0, x_1, p_mat)\n      s.t.\n      (u_0, u_1 are Nash payoffs variables of player 0 and 1)\n      p_mat[0] * x_1 <= u_0\n      x_0^T*p_mat[1] <= u_1\n      (if a pure strategy is in the support then its payoff is Nash payoff)\n      u_0 - p_mat[0] * x_1 <= u_max_0 * b_0\n      u_1 - x_0^T*p_mat[1] <= u_max_1 * b_1\n      (if a pure strategy is not in the support its probability mass is 0)\n      x_0 <= 1 - b_0\n      x_1 <= 1 - b_1\n      (probability constraints)\n      x_0 >= 0\n      1^T * x_0 = 1\n      x_1 >= 0\n      1^T * x_1 = 1\n      for all n, b_0[n] in {0, 1},\n      for all m, b_1[m] in {0, 1},\n      u_max_0, u_max_1 are the maximum payoff differences of player 0 and 1.\n    Note: this formulation is a basic one that may only work well\n    for simple objective function or low-dimensional inputs.\n    GLPK_MI solver only handles linear objective.\n    To handle nonlinear and high-dimensional cases,\n    it is recommended to use advance solvers such as GUROBI,\n    or use a piecewise linear approximation of the objective.\n  Args:\n    game: a pyspiel matrix game object\n    objective: a string representing the objective (e.g., MAX_SOCIAL_WELFARE)\n    solver: the mixed-integer solver used by cvxpy\n\n  Returns:\n    optimal Nash (x_0, x_1)\n  '
    p_mat = game_payoffs_array(game)
    if len(p_mat) != 2:
        raise ValueError('MIP-Nash only works for two players.')
    assert len(p_mat) == 2
    assert p_mat[0].shape == p_mat[1].shape
    (m_0, m_1) = p_mat[0].shape
    u_max_0 = np.max(p_mat[0]) - np.min(p_mat[0])
    u_max_1 = np.max(p_mat[1]) - np.min(p_mat[1])
    x_0 = cp.Variable(m_0)
    x_1 = cp.Variable(m_1)
    u_0 = cp.Variable(1)
    u_1 = cp.Variable(1)
    b_0 = cp.Variable(m_0, boolean=True)
    b_1 = cp.Variable(m_1, boolean=True)
    u_m = p_mat[0] @ x_1
    u_n = x_0 @ p_mat[1]
    constraints = [x_0 >= 0, x_1 >= 0, cp.sum(x_0) == 1, cp.sum(x_1) == 1]
    constraints.extend([u_m <= u_0, u_0 - u_m <= u_max_0 * b_0, x_0 <= 1 - b_0])
    constraints.extend([u_n <= u_1, u_1 - u_n <= u_max_1 * b_1, x_1 <= 1 - b_1])
    variables = {'x_0': x_0, 'x_1': x_1, 'u_0': u_0, 'u_1': u_1, 'b_0': b_0, 'b_1': b_1, 'p_mat': p_mat}
    obj = TWO_PLAYER_OBJECTIVE[objective](variables)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver)
    return (_simplex_projection(x_0.value.reshape(-1)), _simplex_projection(x_1.value.reshape(-1)))

def max_social_welfare_two_player(variables):
    if False:
        i = 10
        return i + 15
    'Max social welfare objective.'
    return cp.Maximize(variables['u_0'] + variables['u_1'])

def min_social_welfare_two_player(variables):
    if False:
        return 10
    'Min social welfare objective.'
    return cp.Minimize(variables['u_0'] + variables['u_1'])

def max_support_two_player(variables):
    if False:
        print('Hello World!')
    'Max support objective.'
    return cp.Minimize(cp.sum(variables['b_0']) + cp.sum(variables['b_1']))

def min_support_two_player(variables):
    if False:
        while True:
            i = 10
    'Min support objective.'
    return cp.Maximize(cp.sum(variables['b_0']) + cp.sum(variables['b_1']))

def max_gini_two_player(variables):
    if False:
        while True:
            i = 10
    'Max gini objective.'
    return cp.Minimize(cp.sum(cp.square(variables['x_0'])) + cp.sum(cp.square(variables['x_1'])))
TWO_PLAYER_OBJECTIVE = {'MAX_SOCIAL_WELFARE': max_social_welfare_two_player, 'MIN_SOCIAL_WELFARE': min_social_welfare_two_player, 'MAX_SUPPORT': max_support_two_player, 'MIN_SUPPORT': min_support_two_player, 'MAX_GINI': max_gini_two_player}