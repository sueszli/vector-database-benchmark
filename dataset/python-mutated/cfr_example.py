"""Example use of the CFR algorithm on Kuhn Poker."""
from absl import app
from absl import flags
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 100, 'Number of iterations')
flags.DEFINE_string('game', 'kuhn_poker', 'Name of the game')
flags.DEFINE_integer('players', 2, 'Number of players')
flags.DEFINE_integer('print_freq', 10, 'How often to print the exploitability')

def main(_):
    if False:
        for i in range(10):
            print('nop')
    game = pyspiel.load_game(FLAGS.game, {'players': FLAGS.players})
    cfr_solver = cfr.CFRSolver(game)
    for i in range(FLAGS.iterations):
        cfr_solver.evaluate_and_update_policy()
        if i % FLAGS.print_freq == 0:
            conv = exploitability.exploitability(game, cfr_solver.average_policy())
            print('Iteration {} exploitability {}'.format(i, conv))
if __name__ == '__main__':
    app.run(main)