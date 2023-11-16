"""Plays a uniform random bot against the default scenarios for that game."""
import random
from absl import app
from absl import flags
from open_spiel.python.bots import scenarios
from open_spiel.python.bots import uniform_random
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('game_name', 'catch', 'Game to play scenarios for.')

def main(argv):
    if False:
        print('Hello World!')
    del argv
    game = pyspiel.load_game(FLAGS.game_name)
    bots = [uniform_random.UniformRandomBot(i, random) for i in range(game.num_players())]
    scenarios.play_bot_in_scenarios(game, bots)
if __name__ == '__main__':
    app.run(main)