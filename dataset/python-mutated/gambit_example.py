"""Export game in gambit .efg format."""
from absl import app
from absl import flags
from absl import logging
from open_spiel.python.algorithms.gambit import export_gambit
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'kuhn_poker', 'Name of the game')
flags.DEFINE_string('out', '/tmp/gametree.efg', 'Name of output file, e.g., [*.efg].')
flags.DEFINE_boolean('print', False, 'Print the tree to stdout instead of saving to file.')

def main(argv):
    if False:
        i = 10
        return i + 15
    del argv
    game = pyspiel.load_game(FLAGS.game)
    game_type = game.get_type()
    if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        logging.warn('%s is not turn-based. Trying to reload game as turn-based.', FLAGS.game)
        game = pyspiel.load_game_as_turn_based(FLAGS.game)
    gametree = export_gambit(game)
    if FLAGS.print:
        print(gametree)
    else:
        with open(FLAGS.out, 'w') as f:
            f.write(gametree)
        logging.info('Game tree for %s saved to file: %s', FLAGS.game, FLAGS.out)
if __name__ == '__main__':
    app.run(main)