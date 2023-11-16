from __future__ import print_function, division
from builtins import range
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    if False:
        while True:
            i = 10
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
    if False:
        i = 10
        return i + 15
    observation = env.reset()
    done = False
    t = 0
    while not done and t < 10000:
        t += 1
        action = get_action(observation, params)
        (observation, reward, done, info) = env.step(action)
        if done:
            break
    return t

def play_multiple_episodes(env, T, params):
    if False:
        print('Hello World!')
    episode_lengths = np.empty(T)
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)
    avg_length = episode_lengths.mean()
    print('avg length:', avg_length)
    return avg_length

def random_search(env):
    if False:
        return 10
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)
        if avg_length > best:
            params = new_params
            best = avg_length
    return (episode_lengths, params)
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    (episode_lengths, params) = random_search(env)
    plt.plot(episode_lengths)
    plt.show()
    env = wrappers.RecordVideo(env, 'my_awesome_dir')
    print('***Final run with final weights***:', play_one_episode(env, params))