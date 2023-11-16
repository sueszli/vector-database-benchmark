"""Tabular Q-Learner self-play example.

Two Q-Learning agents are trained by playing against each other.
"""
import sys
from absl import app
from absl import flags
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_train_episodes', int(1000000.0), 'Number of training episodes.')
flags.DEFINE_integer('num_eval_episodes', int(10000.0), 'Number of episodes to use during each evaluation.')
flags.DEFINE_integer('eval_freq', int(10000.0), 'The frequency (in episodes) to run evaluation.')
flags.DEFINE_string('epsilon_schedule', None, "Epsilon schedule: e.g. 'linear,init,final,num_steps' or 'constant,0.2'")
flags.DEFINE_string('game', 'tic_tac_toe', 'Game to load.')

def eval_agents(env, agents, num_episodes):
    if False:
        for i in range(10):
            print('nop')
    'Evaluate the agents, returning a numpy array of average returns.'
    rewards = np.array([0] * env.num_players, dtype=np.float64)
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations['current_player']
            agent_output = agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
        for i in range(env.num_players):
            rewards[i] += time_step.rewards[i]
    rewards /= num_episodes
    return rewards

def create_epsilon_schedule(sched_str):
    if False:
        return 10
    'Creates an epsilon schedule from the string as desribed in the flags.'
    values = FLAGS.epsilon_schedule.split(',')
    if values[0] == 'linear':
        assert len(values) == 4
        return rl_tools.LinearSchedule(float(values[1]), float(values[2]), int(values[3]))
    elif values[0] == 'constant':
        assert len(values) == 2
        return rl_tools.ConstantSchedule(float(values[1]))
    else:
        print('Unrecognized schedule string: {}'.format(sched_str))
        sys.exit()

def main(_):
    if False:
        for i in range(10):
            print('nop')
    env = rl_environment.Environment(FLAGS.game)
    num_players = env.num_players
    num_actions = env.action_spec()['num_actions']
    agents = []
    if FLAGS.epsilon_schedule is not None:
        for idx in range(num_players):
            agents.append(tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions, epsilon_schedule=create_epsilon_schedule(FLAGS.epsilon_schedule)))
    else:
        agents = [tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions) for idx in range(num_players)]
    training_episodes = FLAGS.num_train_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(FLAGS.eval_freq) == 0:
            avg_rewards = eval_agents(env, agents, FLAGS.num_eval_episodes)
            print('Training episodes: {}, Avg rewards: {}'.format(cur_episode, avg_rewards))
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations['current_player']
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        for agent in agents:
            agent.step(time_step)
if __name__ == '__main__':
    app.run(main)