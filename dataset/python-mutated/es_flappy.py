import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ple import PLE
from ple.games.flappybird import FlappyBird
import sys
HISTORY_LENGTH = 1

class Env:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=False)
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
            return 10
        obs = self.env.getGameState()
        return np.array(list(obs.values()))

    def set_display(self, boolean_value):
        if False:
            print('Hello World!')
        self.env.display_screen = boolean_value
env = Env()
D = len(env.reset()) * HISTORY_LENGTH
M = 50
K = 2

def softmax(a):
    if False:
        i = 10
        return i + 15
    c = np.max(a, axis=1, keepdims=True)
    e = np.exp(a - c)
    return e / e.sum(axis=-1, keepdims=True)

def relu(x):
    if False:
        return 10
    return x * (x > 0)

class ANN:

    def __init__(self, D, M, K, f=relu):
        if False:
            print('Hello World!')
        self.D = D
        self.M = M
        self.K = K
        self.f = f

    def init(self):
        if False:
            i = 10
            return i + 15
        (D, M, K) = (self.D, self.M, self.K)
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M)
        self.b2 = np.zeros(K)

    def forward(self, X):
        if False:
            return 10
        Z = self.f(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2)

    def sample_action(self, x):
        if False:
            i = 10
            return i + 15
        X = np.atleast_2d(x)
        P = self.forward(X)
        p = P[0]
        return np.argmax(p)

    def get_params(self):
        if False:
            for i in range(10):
                print('nop')
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        if False:
            while True:
                i = 10
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def set_params(self, params):
        if False:
            return 10
        (D, M, K) = (self.D, self.M, self.K)
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]

def evolution_strategy(f, population_size, sigma, lr, initial_params, num_iters):
    if False:
        print('Hello World!')
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)
        R = np.zeros(population_size)
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j] = f(params_try)
        m = R.mean()
        s = R.std()
        if s == 0:
            print('Skipping')
            continue
        A = (R - m) / s
        reward_per_iteration[t] = m
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)
        lr *= 0.992354
        print('Iter:', t, 'Avg Reward: %.3f' % m, 'Max:', R.max(), 'Duration:', datetime.now() - t0)
    return (params, reward_per_iteration)

def reward_function(params):
    if False:
        for i in range(10):
            print('nop')
    model = ANN(D, M, K)
    model.set_params(params)
    episode_reward = 0
    episode_length = 0
    done = False
    obs = env.reset()
    obs_dim = len(obs)
    if HISTORY_LENGTH > 1:
        state = np.zeros(HISTORY_LENGTH * obs_dim)
        state[-obs_dim:] = obs
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
    return episode_reward
if __name__ == '__main__':
    model = ANN(D, M, K)
    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        j = np.load('es_flappy_results.npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
        (D, M) = j['W1'].shape
        K = len(j['b2'])
        (model.D, model.M, model.K) = (D, M, K)
    else:
        model.init()
        params = model.get_params()
        (best_params, rewards) = evolution_strategy(f=reward_function, population_size=30, sigma=0.1, lr=0.03, initial_params=params, num_iters=300)
        model.set_params(best_params)
        np.savez('es_flappy_results.npz', train=rewards, **model.get_params_dict())
    env.set_display(True)
    for _ in range(5):
        print('Test:', reward_function(best_params))