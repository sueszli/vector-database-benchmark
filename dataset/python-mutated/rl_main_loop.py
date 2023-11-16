"""Python spiel example."""
import logging
from absl import app
from absl import flags
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'kuhn_poker', 'Name of the game.')
flags.DEFINE_integer('num_players', 2, 'Number of players.')
flags.DEFINE_integer('num_episodes', 2, 'Number of episodes.')

def print_iteration(time_step, player_id, action=None):
    if False:
        i = 10
        return i + 15
    'Print TimeStep information.'
    obs = time_step.observations
    logging.info('Player: %s', player_id)
    if time_step.first():
        logging.info('Info state: %s, - - %s', obs['info_state'][player_id], time_step.step_type)
    else:
        logging.info('Info state: %s, %s %s %s', obs['info_state'][player_id], time_step.rewards[player_id], time_step.discounts[player_id], time_step.step_type)
    if action is not None:
        logging.info('Action taken: %s', action)
    logging.info('-' * 80)

def main_loop(unused_arg):
    if False:
        for i in range(10):
            print('nop')
    'RL main loop example.'
    logging.info('Registered games: %s', rl_environment.registered_games())
    logging.info('Creating game %s', FLAGS.game)
    env_configs = {'players': FLAGS.num_players} if FLAGS.num_players else {}
    env = rl_environment.Environment(FLAGS.game, **env_configs)
    num_actions = env.action_spec()['num_actions']
    agents = [random_agent.RandomAgent(player_id=i, num_actions=num_actions) for i in range(FLAGS.num_players)]
    logging.info('Env specs: %s', env.observation_spec())
    logging.info('Action specs: %s', env.action_spec())
    for cur_episode in range(FLAGS.num_episodes):
        logging.info('Starting episode %s', cur_episode)
        time_step = env.reset()
        while not time_step.last():
            pid = time_step.observations['current_player']
            if env.is_turn_based:
                agent_output = agents[pid].step(time_step)
                action_list = [agent_output.action]
            else:
                agents_output = [agent.step(time_step) for agent in agents]
                action_list = [agent_output.action for agent_output in agents_output]
            print_iteration(time_step, pid, action_list)
            time_step = env.step(action_list)
        for agent in agents:
            agent.step(time_step)
        for pid in range(env.num_players):
            print_iteration(time_step, pid)
if __name__ == '__main__':
    app.run(main_loop)