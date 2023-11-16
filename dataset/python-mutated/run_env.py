"""Random policy on an environment."""
import tensorflow as tf
import numpy as np
import random
from environments import create_maze_env
app = tf.app
flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'AntMaze', 'environment name: AntMaze, AntPush, or AntFall')
flags.DEFINE_integer('episode_length', 500, 'episode length')
flags.DEFINE_integer('num_episodes', 50, 'number of episodes')

def get_goal_sample_fn(env_name):
    if False:
        while True:
            i = 10
    if env_name == 'AntMaze':
        return lambda : np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda : np.array([0.0, 19.0])
    elif env_name == 'AntFall':
        return lambda : np.array([0.0, 27.0, 4.5])
    else:
        assert False, 'Unknown env'

def get_reward_fn(env_name):
    if False:
        return 10
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'

def success_fn(last_reward):
    if False:
        print('Hello World!')
    return last_reward > -5.0

class EnvWithGoal(object):

    def __init__(self, base_env, env_name):
        if False:
            return 10
        self.base_env = base_env
        self.goal_sample_fn = get_goal_sample_fn(env_name)
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None

    def reset(self):
        if False:
            return 10
        obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()
        return np.concatenate([obs, self.goal])

    def step(self, a):
        if False:
            return 10
        (obs, _, done, info) = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        return (np.concatenate([obs, self.goal]), reward, done, info)

    @property
    def action_space(self):
        if False:
            i = 10
            return i + 15
        return self.base_env.action_space

def run_environment(env_name, episode_length, num_episodes):
    if False:
        i = 10
        return i + 15
    env = EnvWithGoal(create_maze_env.create_maze_env(env_name).gym, env_name)

    def action_fn(obs):
        if False:
            while True:
                i = 10
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = action_space_mean + action_space_magn * np.random.uniform(low=-1.0, high=1.0, size=action_space.shape)
        return random_action
    rewards = []
    successes = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        for _ in range(episode_length):
            (obs, reward, done, _) = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            if done:
                break
        logging.info('Episode %d reward: %.2f, Success: %d', ep + 1, rewards[-1], successes[-1])
    logging.info('Average Reward over %d episodes: %.2f', num_episodes, np.mean(rewards))
    logging.info('Average Success over %d episodes: %.2f', num_episodes, np.mean(successes))

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    logging.set_verbosity(logging.INFO)
    run_environment(FLAGS.env, FLAGS.episode_length, FLAGS.num_episodes)
if __name__ == '__main__':
    app.run()