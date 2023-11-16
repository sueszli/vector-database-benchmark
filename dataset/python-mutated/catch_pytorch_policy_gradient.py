"""Python spiel example."""
import logging
from absl import app
from absl import flags
from open_spiel.python.environments import catch
from open_spiel.python.pytorch import policy_gradient
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', int(100000.0), 'Number of train episodes.')
flags.DEFINE_integer('eval_every', int(1000.0), "'How often to evaluate the policy.")
flags.DEFINE_enum('algorithm', 'a2c', ['rpg', 'qpg', 'rm', 'a2c'], 'Algorithms to run.')

def _eval_agent(env, agent, num_episodes):
    if False:
        return 10
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
        while True:
            i = 10
    'Trains a Policy Gradient agent in the catch environment.'
    env = catch.Environment()
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']
    train_episodes = FLAGS.num_episodes
    agent = policy_gradient.PolicyGradient(player_id=0, info_state_size=info_state_size, num_actions=num_actions, loss_str=FLAGS.algorithm, hidden_layers_sizes=[128, 128], batch_size=128, entropy_cost=0.01, critic_learning_rate=0.1, pi_learning_rate=0.1, num_critic_before_pi=3)
    for ep in range(train_episodes):
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
        agent.step(time_step)
        if ep and ep % FLAGS.eval_every == 0:
            logging.info('-' * 80)
            logging.info('Episode %s', ep)
            logging.info('Loss: %s', agent.loss)
            avg_return = _eval_agent(env, agent, 100)
            logging.info('Avg return: %s', avg_return)
if __name__ == '__main__':
    app.run(main_loop)