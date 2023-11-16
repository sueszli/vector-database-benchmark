"""Example use of the CFR algorithm on Kuhn Poker."""
from absl import app
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel

def main(_):
    if False:
        print('Hello World!')
    game = pyspiel.load_game('kuhn_poker')
    cfr_solver = cfr.CFRSolver(game)
    iterations = 1000
    for i in range(iterations):
        cfr_value = cfr_solver.evaluate_and_update_policy()
        print('Game util at iteration {}: {}'.format(i, cfr_value))
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(game.new_initial_state(), [average_policy] * 2)
    print('Computed player 0 value: {}'.format(average_policy_values[0]))
    print('Expected player 0 value: {}'.format(-1 / 18))
if __name__ == '__main__':
    app.run(main)