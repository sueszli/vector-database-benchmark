"""Solving matrix games with LP solver."""
from absl import app
from open_spiel.python.algorithms import lp_solver
import pyspiel

def main(_):
    if False:
        i = 10
        return i + 15
    (p0_sol, p1_sol, p0_sol_val, p1_sol_val) = lp_solver.solve_zero_sum_matrix_game(pyspiel.create_matrix_game([[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]], [[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]))
    print('p0 val = {}, policy = {}'.format(p0_sol_val, p0_sol))
    print('p1 val = {}, policy = {}'.format(p1_sol_val, p1_sol))
    payoff_matrix = [[1.0, 1.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 2.0]]
    mixture = lp_solver.is_dominated(0, payoff_matrix, 0, lp_solver.DOMINANCE_WEAK, return_mixture=True)
    print('mixture strategy : {}'.format(mixture))
    print('payoff vector    : {}'.format(mixture.dot(payoff_matrix)))
if __name__ == '__main__':
    app.run(main)