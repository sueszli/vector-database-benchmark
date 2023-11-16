"""Game-specific query example."""
from absl import app
from absl import flags
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'negotiation', 'Name of the game')

def main(_):
    if False:
        print('Hello World!')
    print('Creating game: ' + FLAGS.game)
    game = pyspiel.load_game(FLAGS.game)
    state = game.new_initial_state()
    print(str(state))
    state.apply_action(0)
    print('Item pool: {}'.format(state.item_pool()))
    print('Player 0 utils: {}'.format(state.agent_utils(0)))
    print('Player 1 utils: {}'.format(state.agent_utils(1)))
    state = game.new_initial_state()
    print(str(state))
    state.apply_action(0)
    print('Item pool: {}'.format(state.item_pool()))
    print('Player 0 utils: {}'.format(state.agent_utils(0)))
    print('Player 1 utils: {}'.format(state.agent_utils(1)))
if __name__ == '__main__':
    app.run(main)