from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import q_learning
from q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg

class SGDRegressor:

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.w = None
        self.lr = 0.01

    def partial_fit(self, X, Y):
        if False:
            while True:
                i = 10
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        if False:
            return 10
        return X.dot(self.w)
q_learning.SGDRegressor = SGDRegressor

def play_one(model, eps, gamma, n=5):
    if False:
        return 10
    observation = env.reset()
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    iters = 0
    multiplier = np.array([gamma] * n) ** np.arange(n)
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        states.append(observation)
        actions.append(action)
        prev_observation = observation
        (observation, reward, done, info) = env.step(action)
        rewards.append(reward)
        if len(rewards) >= n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + gamma ** n * np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)
        totalreward += reward
        iters += 1
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n + 1:]
        states = states[-n + 1:]
        actions = actions[-n + 1:]
    if observation[0] >= 0.5:
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        while len(rewards) > 0:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    return totalreward
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.1 * 0.97 ** n
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print('episode:', n, 'total reward:', totalreward)
    print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
    print('total steps:', -totalrewards.sum())
    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()
    plot_running_avg(totalrewards)
    plot_cost_to_go(env, model)