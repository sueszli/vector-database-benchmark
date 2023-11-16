"""Python nfg_writer example."""
from absl import app
from absl import flags
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'matrix_rps', 'Name of the game')
flags.DEFINE_string('outfile', None, 'File to send the output to.')

def main(_):
    if False:
        i = 10
        return i + 15
    game = pyspiel.load_game(FLAGS.game)
    nfg_text = pyspiel.game_to_nfg_string(game)
    if FLAGS.outfile is None:
        print(nfg_text)
    else:
        print('Exporting to {}'.format(FLAGS.outfile))
        outfile = open(FLAGS.outfile, 'w')
        outfile.write(nfg_text)
        outfile.close()
if __name__ == '__main__':
    app.run(main)