"""Python spiel example."""
import logging
from absl import app
from absl import flags
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.environments import cliff_walking
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', int(500.0), 'Number of train episodes.')

def eval_agent(env, agent, num_episodes):
    if False:
        while True:
            i = 10
    'Evaluates `agent` for `num_episodes`.'
    rewards = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
            episode_reward += time_step.rewards[0]
        rewards += episode_reward
    return rewards / num_episodes

def main_loop(unused_arg):
    if False:
        i = 10
        return i + 15
    'Trains a tabular qlearner agent in the cliff walking environment.'
    env = cliff_walking.Environment(width=5, height=3)
    num_actions = env.action_spec()['num_actions']
    train_episodes = FLAGS.num_episodes
    eval_interval = 50
    agent = tabular_qlearner.QLearner(player_id=0, step_size=0.05, num_actions=num_actions)
    for ep in range(train_episodes):
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
        agent.step(time_step)
        if ep and ep % eval_interval == 0:
            logging.info('-' * 80)
            logging.info('Episode %s', ep)
            logging.info('Last loss: %s', agent.loss)
            avg_return = eval_agent(env, agent, 100)
            logging.info('Avg return: %s', avg_return)
if __name__ == '__main__':
    app.run(main_loop)