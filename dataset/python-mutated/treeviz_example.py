"""Game tree visualization example."""
from absl import app
from absl import flags
from absl import logging
import pyspiel
from open_spiel.python.visualizations import treeviz
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'kuhn_poker', 'Name of the game')
flags.DEFINE_string('out', '/tmp/gametree.png', 'Name of output file, e.g., [*.png|*.pdf].')
flags.DEFINE_enum('prog', 'dot', ['dot', 'neato', 'circo'], 'Graphviz layout.')
flags.DEFINE_boolean('group_infosets', False, 'Whether to group infosets.')
flags.DEFINE_boolean('group_terminal', False, 'Whether to group terminal nodes.')
flags.DEFINE_boolean('group_pubsets', False, 'Whether to group public states.')
flags.DEFINE_string('target_pubset', '*', 'Limit grouping of public states only to specified state.')
flags.DEFINE_boolean('verbose', False, 'Whether to print verbose output.')

def _zero_sum_node_decorator(state):
    if False:
        for i in range(10):
            print('nop')
    'Custom node decorator that only shows the return of the first player.'
    attrs = treeviz.default_node_decorator(state)
    if state.is_terminal():
        attrs['label'] = str(int(state.returns()[0]))
    return attrs

def main(argv):
    if False:
        while True:
            i = 10
    del argv
    game = pyspiel.load_game(FLAGS.game)
    game_type = game.get_type()
    if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        logging.warn('%s is not turn-based. Trying to reload game as turn-based.', FLAGS.game)
        game = pyspiel.load_game_as_turn_based(FLAGS.game)
        game_type = game.get_type()
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError('Game must be sequential, not {}'.format(game_type.dynamics))
    if game_type.utility == pyspiel.GameType.Utility.ZERO_SUM and game.num_players() == 2:
        logging.info("Game is zero-sum: only showing first-player's returns.")
        gametree = treeviz.GameTree(game, node_decorator=_zero_sum_node_decorator, group_infosets=FLAGS.group_infosets, group_terminal=FLAGS.group_terminal, group_pubsets=FLAGS.group_pubsets, target_pubset=FLAGS.target_pubset)
    else:
        gametree = treeviz.GameTree(game, group_infosets=FLAGS.group_infosets, group_terminal=FLAGS.group_terminal, group_pubsets=FLAGS.group_pubsets, target_pubset=FLAGS.target_pubset)
    if FLAGS.verbose:
        logging.info('Game tree:\n%s', gametree.to_string())
    gametree.draw(FLAGS.out, prog=FLAGS.prog)
    logging.info('Game tree saved to file: %s', FLAGS.out)
if __name__ == '__main__':
    app.run(main)