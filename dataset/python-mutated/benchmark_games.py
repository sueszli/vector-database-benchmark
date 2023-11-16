"""Benchmark performance of games by counting the number of rollouts."""
import random
import time
from absl import app
from absl import flags
from absl import logging
import pandas as pd
from open_spiel.python import games
from open_spiel.python.mfg import games as mfg_games
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('games', '*', 'Benchmark only specific games (semicolon separated). Use * to benchmark all (loadable) games.')
flags.DEFINE_float('time_limit', 10.0, 'Time limit per game (in seconds).')
flags.DEFINE_integer('give_up_after', 100, 'Give up rollout when the history length is exceeded.')
flags.DEFINE_bool('if_simultaneous_convert_to_turn_based', False, 'If True, load any simultaneous game as turn based for the benchmark.')

def _rollout_until_timeout(game_name, time_limit, give_up_after, if_simultaneous_convert_to_turn_based=False):
    if False:
        print('Hello World!')
    'Run rollouts on the specified game until the time limit.\n\n  Args:\n    game_name:      str\n    time_limit:     In number of seconds\n    give_up_after:  Cuts off trajectories longer than specified\n    if_simultaneous_convert_to_turn_based: if the game is simultaneous and this\n      boolean is true, then the game is loaded as a turn based game.\n\n  Returns:\n    A dict of collected statistics.\n  '
    game = pyspiel.load_game(game_name)
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
        raise NotImplementedError('Benchmark on mean field games is not available yet.')
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS and if_simultaneous_convert_to_turn_based:
        game = pyspiel.convert_to_turn_based(game)
    is_time_out = lambda t: time.time() - t > time_limit
    num_rollouts = 0
    num_giveups = 0
    num_moves = 0
    start = time.time()
    while not is_time_out(start):
        state = game.new_initial_state()
        while not state.is_terminal():
            if len(state.history()) > give_up_after:
                num_giveups += 1
                break
            if state.is_simultaneous_node():

                def random_choice(actions):
                    if False:
                        print('Hello World!')
                    if actions:
                        return random.choice(actions)
                    return 0
                actions = [random_choice(state.legal_actions(i)) for i in range(state.num_players())]
                state.apply_actions(actions)
            else:
                action = random.choice(state.legal_actions(state.current_player()))
                state.apply_action(action)
            num_moves += 1
        num_rollouts += 1
    time_elapsed = time.time() - start
    return dict(game_name=game_name, ms_per_rollouts=time_elapsed / num_rollouts * 1000, ms_per_moves=time_elapsed / num_moves * 1000, giveups_per_rollout=num_giveups / num_rollouts, time_elapsed=time_elapsed)

def main(_):
    if False:
        while True:
            i = 10
    if FLAGS.games == '*':
        games_list = [game.short_name for game in pyspiel.registered_games() if game.default_loadable]
    else:
        games_list = FLAGS.games.split(';')
    logging.info('Running benchmark for %s games.', len(games_list))
    logging.info('This will take approximately %d seconds.', len(games_list) * FLAGS.time_limit)
    game_stats = []
    for game_name in games_list:
        logging.info('Running benchmark on %s', game_name)
        game_stats.append(_rollout_until_timeout(game_name, FLAGS.time_limit, FLAGS.give_up_after, FLAGS.if_simultaneous_convert_to_turn_based))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        df = pd.DataFrame(game_stats)
        df.rename(columns={'game_name': 'Game', 'ms_per_rollouts': 'msec/rollout', 'ms_per_moves': 'msec/move', 'giveups_per_rollout': 'Give ups/rollouts', 'time_elapsed': 'Time elapsed [sec]'}, inplace=True)
        print('---')
        print('Results for following benchmark configuration:')
        print('time_limit =', FLAGS.time_limit)
        print('give_up_after =', FLAGS.give_up_after)
        print('---')
        print(df)
if __name__ == '__main__':
    app.run(main)