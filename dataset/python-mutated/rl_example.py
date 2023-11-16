"""Python spiel example."""
import logging
from absl import app
from absl import flags
import numpy as np
from open_spiel.python import rl_environment
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'tic_tac_toe', 'Name of the game')
flags.DEFINE_integer('num_players', None, 'Number of players')

def select_actions(observations, cur_player):
    if False:
        return 10
    cur_legal_actions = observations['legal_actions'][cur_player]
    actions = [np.random.choice(cur_legal_actions)]
    return actions

def print_iteration(time_step, actions, player_id):
    if False:
        return 10
    'Print TimeStep information.'
    obs = time_step.observations
    logging.info('Player: %s', player_id)
    if time_step.step_type.first():
        logging.info('Info state: %s, - - %s', obs['info_state'][player_id], time_step.step_type)
    else:
        logging.info('Info state: %s, %s %s %s', obs['info_state'][player_id], time_step.rewards[player_id], time_step.discounts[player_id], time_step.step_type)
    logging.info('Action taken: %s', actions)
    logging.info('-' * 80)

def turn_based_example(unused_arg):
    if False:
        i = 10
        return i + 15
    'Example usage of the RL environment for turn-based games.'
    logging.info('Registered games: %s', rl_environment.registered_games())
    logging.info('Creating game %s', FLAGS.game)
    env_configs = {'players': FLAGS.num_players} if FLAGS.num_players else {}
    env = rl_environment.Environment(FLAGS.game, **env_configs)
    logging.info('Env specs: %s', env.observation_spec())
    logging.info('Action specs: %s', env.action_spec())
    time_step = env.reset()
    while not time_step.step_type.last():
        pid = time_step.observations['current_player']
        actions = select_actions(time_step.observations, pid)
        print_iteration(time_step, actions, pid)
        time_step = env.step(actions)
    for pid in range(env.num_players):
        print_iteration(time_step, actions, pid)
if __name__ == '__main__':
    app.run(turn_based_example)