import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ple import PLE
from ple.games.flappybird import FlappyBird
import sys
from threading import Thread
HISTORY_LENGTH = 1

class Env:

    def __init__(self):
        if False:
            print('Hello World!')
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=True)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.action_map = self.env.getActionSet()

    def step(self, action):
        if False:
            i = 10
            return i + 15
        action = self.action_map[action]
        reward = self.env.act(action)
        done = self.env.game_over()
        obs = self.get_observation()
        return (obs, reward, done)

    def reset(self):
        if False:
            print('Hello World!')
        self.env.reset_game()
        return self.get_observation()

    def get_observation(self):
        if False:
            for i in range(10):
                print('nop')
        obs = self.env.getGameState()
        return np.array(list(obs.values()))

    def set_display(self, boolean_value):
        if False:
            while True:
                i = 10
        self.env.display_screen = boolean_value
env = Env()
D = len(env.reset()) * HISTORY_LENGTH
M = 50
K = 2

def softmax(a):
    if False:
        for i in range(10):
            print('nop')
    c = np.max(a, axis=1, keepdims=True)
    e = np.exp(a - c)
    return e / e.sum(axis=-1, keepdims=True)

def relu(x):
    if False:
        for i in range(10):
            print('nop')
    return x * (x > 0)

class ANN:

    def __init__(self, D, M, K, f=relu):
        if False:
            return 10
        self.D = D
        self.M = M
        self.K = K
        self.f = f

    def init(self):
        if False:
            print('Hello World!')
        (D, M, K) = (self.D, self.M, self.K)
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M)
        self.b2 = np.zeros(K)

    def forward(self, X):
        if False:
            print('Hello World!')
        Z = self.f(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2)

    def sample_action(self, x):
        if False:
            for i in range(10):
                print('nop')
        X = np.atleast_2d(x)
        P = self.forward(X)
        p = P[0]
        return np.argmax(p)

    def score(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        P = np.argmax(self.forward(X), axis=1)
        return np.mean(Y == P)

    def get_params(self):
        if False:
            while True:
                i = 10
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        if False:
            return 10
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def set_params(self, params):
        if False:
            return 10
        (D, M, K) = (self.D, self.M, self.K)
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]
(env1, env2) = (Env(), Env())

def reward_function(params, env):
    if False:
        while True:
            i = 10
    model = ANN(D, M, K)
    model.set_params(params)
    episode_reward = 0
    episode_length = 0
    done = False
    obs = env.reset()
    obs_dim = len(obs)
    if HISTORY_LENGTH > 1:
        state = np.zeros(HISTORY_LENGTH * obs_dim)
        state[obs_dim:] = obs
    else:
        state = obs
    while not done:
        action = model.sample_action(state)
        (obs, reward, done) = env.step(action)
        episode_reward += reward
        episode_length += 1
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:] = obs
        else:
            state = obs
    print('Reward:', episode_reward)
if __name__ == '__main__':
    j = np.load('es_flappy_results.npz')
    best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
    (D, M) = j['W1'].shape
    K = len(j['b2'])
    t1 = Thread(target=reward_function, args=(best_params, env1))
    t2 = Thread(target=reward_function, args=(best_params, env2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()