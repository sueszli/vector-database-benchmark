"""Example: using MMD with dilated entropy to solve for QRE in a matrix Game."""
from absl import app
from absl import flags
from open_spiel.python.algorithms import mmd_dilated
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_integer('iterations', 1000, 'Number of iterations')
flags.DEFINE_float('alpha', 0.1, 'QRE parameter, larger value amounts to more regularization')
flags.DEFINE_integer('print_freq', 100, 'How often to print the gap')
game = pyspiel.create_matrix_game([[0, -1, 3], [1, 0, -3], [-3, 3, 0]], [[0, 1, -3], [-1, 0, 3], [3, -3, 0]])
game = pyspiel.convert_to_turn_based(game)

def main(_):
    if False:
        for i in range(10):
            print('nop')
    mmd = mmd_dilated.MMDDilatedEnt(game, FLAGS.alpha)
    for i in range(FLAGS.iterations):
        mmd.update_sequences()
        if i % FLAGS.print_freq == 0:
            conv = mmd.get_gap()
            print('Iteration {} gap {}'.format(i, conv))
    print(mmd.get_policies().action_probability_array)
    print(mmd.current_sequences())
if __name__ == '__main__':
    app.run(main)