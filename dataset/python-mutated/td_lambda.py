from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning import FeatureTransformer
from q_learning_bins import plot_running_avg

class SGDRegressor:

    def __init__(self, D):
        if False:
            while True:
                i = 10
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, x, y, e, lr=0.1):
        if False:
            return 10
        self.w += lr * (y - x.dot(self.w)) * e

    def predict(self, X):
        if False:
            print('Hello World!')
        X = np.array(X)
        return X.dot(self.w)

class Model:

    def __init__(self, env, feature_transformer):
        if False:
            return 10
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        sample_feature = feature_transformer.transform([env.reset()])
        D = sample_feature.shape[1]
        for i in range(env.action_space.n):
            model = SGDRegressor(D)
            self.models.append(model)
        self.eligibilities = np.zeros((env.action_space.n, D))

    def reset(self):
        if False:
            return 10
        self.eligibilities = np.zeros_like(self.eligibilities)

    def predict(self, s):
        if False:
            while True:
                i = 10
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, s, a, G, gamma, lambda_):
        if False:
            i = 10
            return i + 15
        X = self.feature_transformer.transform([s])
        self.eligibilities *= gamma * lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])

    def sample_action(self, s, eps):
        if False:
            print('Hello World!')
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, env, eps, gamma, lambda_):
    if False:
        while True:
            i = 10
    observation = env.reset()
    done = False
    totalreward = 0
    states_actions_rewards = []
    iters = 0
    model.reset()
    while not done and iters < 1000000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        (observation, reward, done, info) = env.step(action)
        if done:
            reward = -300
        next = model.predict(observation)
        assert next.shape == (1, env.action_space.n)
        G = reward + gamma * np.max(next[0])
        model.update(prev_observation, action, G, gamma, lambda_)
        states_actions_rewards.append((prev_observation, action, reward))
        if reward == 1:
            totalreward += reward
        iters += 1
    return (states_actions_rewards, totalreward)
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.999
    lambda_ = 0.7
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        (states_actions_rewards, totalreward) = play_one(model, env, eps, gamma, lambda_)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print('episode:', n, 'total reward:', totalreward, 'eps:', eps, 'avg reward (last 100):', totalrewards[max(0, n - 100):n + 1].mean())
    print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
    print('total steps:', totalrewards.sum())
    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()
    plot_running_avg(totalrewards)