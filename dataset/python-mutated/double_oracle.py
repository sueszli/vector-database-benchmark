"""Double Oracle algorithm.

Solves two-player zero-sum games, for more information see:
McMahan et al. (2003). Planning in the presence of cost functions controlled by
  an adversary. In Proceedings of the 20th International Conference on Machine
  Learning (ICML-03) (pp. 536-543).
"""
import numpy as np
from open_spiel.python.algorithms import lp_solver
from open_spiel.python.egt import utils
import pyspiel

def lens(lists):
    if False:
        while True:
            i = 10
    'Returns the sizes of lists in a list.'
    return list(map(len, lists))

def solve_subgame(subgame_payoffs):
    if False:
        return 10
    "Solves the subgame using OpenSpiel's LP solver."
    (p0_sol, p1_sol, _, _) = lp_solver.solve_zero_sum_matrix_game(pyspiel.create_matrix_game(*subgame_payoffs))
    (p0_sol, p1_sol) = (np.asarray(p0_sol), np.asarray(p1_sol))
    return [p0_sol / p0_sol.sum(), p1_sol / p1_sol.sum()]

class DoubleOracleSolver(object):
    """Double Oracle solver."""

    def __init__(self, game, enforce_symmetry=False):
        if False:
            while True:
                i = 10
        "Initializes the Double Oracle solver.\n\n    Args:\n      game: pyspiel.MatrixGame (zero-sum).\n      enforce_symmetry: If True, enforces symmetry in the strategies appended by\n        each player, by using the first player's best response for the second\n        player as well; also asserts the game is symmetric and that players are\n        seeded with identical initial_strategies, default: False.\n    "
        assert isinstance(game, pyspiel.MatrixGame)
        assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
        self.payoffs = utils.game_payoffs_array(game)
        self.subgame_strategies = [[], []]
        self.enforce_symmetry = enforce_symmetry
        if self.enforce_symmetry:
            assert utils.is_symmetric_matrix_game(self.payoffs), 'enforce_symmetry is True, but payoffs are asymmetric!'

    def subgame_payoffs(self):
        if False:
            print('Hello World!')
        assert all(lens(self.subgame_strategies)), 'Need > 0 strategies per player.'
        subgame_payoffs = np.copy(self.payoffs)
        for (player, indices) in enumerate(self.subgame_strategies):
            subgame_payoffs = np.take(subgame_payoffs, indices, axis=player + 1)
        return subgame_payoffs

    def oracle(self, subgame_solution):
        if False:
            while True:
                i = 10
        'Computes the best responses.\n\n    Args:\n      subgame_solution: List of subgame solution policies.\n\n    Returns:\n      best_response: For both players from the original set of pure strategies.\n      best_response_utility: Corresponding utility for both players.\n    '
        assert lens(subgame_solution) == lens(self.subgame_strategies), f'{lens(subgame_solution)} != {lens(self.subgame_strategies)}'
        best_response = [None, None]
        best_response_utility = [None, None]
        n_best_responders = 1 if self.enforce_symmetry else 2
        for player in range(n_best_responders):
            opponent = 1 - player
            payoffs = np.take(self.payoffs[player], self.subgame_strategies[opponent], axis=opponent)
            payoffs = np.transpose(payoffs, [player, opponent])
            avg_payoffs = (payoffs @ subgame_solution[opponent]).squeeze()
            best_response[player] = np.argmax(avg_payoffs)
            best_response_utility[player] = avg_payoffs[best_response[player]]
        if self.enforce_symmetry:
            best_response[1] = best_response[0]
            best_response_utility[1] = best_response_utility[0]
        return (best_response, best_response_utility)

    def step(self):
        if False:
            i = 10
            return i + 15
        'Performs one iteration.'
        subgame_payoffs = self.subgame_payoffs()
        subgame_solution = solve_subgame(subgame_payoffs)
        (best_response, best_response_utility) = self.oracle(subgame_solution)
        self.subgame_strategies = [sorted(set(strategies + [br])) for (strategies, br) in zip(self.subgame_strategies, best_response)]
        return (best_response, best_response_utility)

    def solve_yield(self, initial_strategies, max_steps, tolerance, verbose, yield_subgame=False):
        if False:
            print('Hello World!')
        'Solves game using Double Oracle, yielding intermediate results.\n\n    Args:\n      initial_strategies: List of pure strategies for both players, optional.\n      max_steps: Maximum number of iterations, default: 20.\n      tolerance: Stop if the estimated value of the game is below the tolerance.\n      verbose: If False, no warning is shown, default: True.\n      yield_subgame: If True, yields the subgame on each iteration. Otherwise,\n        yields the final results only, default: False.\n\n    Yields:\n      solution: Policies for both players.\n      iteration: The number of iterations performed.\n      value: Estimated value of the game.\n    '
        if self.enforce_symmetry and initial_strategies:
            assert np.array_equal(initial_strategies[0], initial_strategies[1]), f'Players must use same initial_strategies as symmetry is enforced.\ninitial_strategies[0]: {initial_strategies[0]}, \ninitial_strategies[1]: {initial_strategies[1]}'
        self.subgame_strategies = initial_strategies if initial_strategies else [[0], [0]]
        iteration = 0
        while iteration < max_steps:
            if yield_subgame:
                yield (None, iteration, None, self.subgame_payoffs())
            iteration += 1
            last_subgame_size = lens(self.subgame_strategies)
            (_, best_response_utility) = self.step()
            value = sum(best_response_utility)
            if abs(value) < tolerance:
                if verbose:
                    print('Last iteration={}; value below tolerance {} < {}.'.format(iteration, value, tolerance))
                break
            if lens(self.subgame_strategies) == last_subgame_size:
                if verbose:
                    print('Last iteration={}; no strategies added, increase tolerance={} or check subgame solver.'.format(iteration, tolerance))
                break
        subgame_solution = solve_subgame(self.subgame_payoffs())
        solution = [np.zeros(k) for k in self.payoffs.shape[1:]]
        for p in range(2):
            solution[p][self.subgame_strategies[p]] = subgame_solution[p].squeeze()
        yield (solution, iteration, value, self.subgame_payoffs())

    def solve(self, initial_strategies=None, max_steps=20, tolerance=5e-05, verbose=True):
        if False:
            i = 10
            return i + 15
        'Solves the game using Double Oracle, returning the final solution.'
        (solution, iteration, value) = (None, None, None)
        generator = self.solve_yield(initial_strategies, max_steps, tolerance, verbose, yield_subgame=False)
        for (solution, iteration, value, _) in generator:
            pass
        return (solution, iteration, value)