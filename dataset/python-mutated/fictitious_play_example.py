"""Python XFP example."""
import sys
from absl import app
from absl import flags
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import fictitious_play
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 100, 'Number of iterations')
flags.DEFINE_string('game', 'leduc_poker', 'Name of the game')
flags.DEFINE_integer('players', 2, 'Number of players')
flags.DEFINE_integer('print_freq', 10, 'How often to print the exploitability')

def main(_):
    if False:
        print('Hello World!')
    game = pyspiel.load_game(FLAGS.game, {'players': FLAGS.players})
    xfp_solver = fictitious_play.XFPSolver(game)
    for i in range(FLAGS.iterations):
        xfp_solver.iteration()
        conv = exploitability.exploitability(game, xfp_solver.average_policy())
        if i % FLAGS.print_freq == 0:
            print('Iteration: {} Conv: {}'.format(i, conv))
            sys.stdout.flush()
if __name__ == '__main__':
    app.run(main)