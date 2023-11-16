from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ACTION2INT = {a: i for (i, a) in enumerate(ALL_POSSIBLE_ACTIONS)}
INT2ONEHOT = np.eye(len(ALL_POSSIBLE_ACTIONS))

def epsilon_greedy(model, s, eps=0.1):
    if False:
        while True:
            i = 10
    p = np.random.random()
    if p < 1 - eps:
        values = model.predict_all_actions(s)
        return ALL_POSSIBLE_ACTIONS[np.argmax(values)]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def one_hot(k):
    if False:
        return 10
    return INT2ONEHOT[k]

def merge_state_action(s, a):
    if False:
        return 10
    ai = one_hot(ACTION2INT[a])
    return np.concatenate((s, ai))

def gather_samples(grid, n_episodes=1000):
    if False:
        print('Hello World!')
    samples = []
    for _ in range(n_episodes):
        s = grid.reset()
        while not grid.game_over():
            a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            sa = merge_state_action(s, a)
            samples.append(sa)
            r = grid.move(a)
            s = grid.current_state()
    return samples

class Model:

    def __init__(self, grid):
        if False:
            for i in range(10):
                print('nop')
        samples = gather_samples(grid)
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)
        dims = self.featurizer.n_components
        self.w = np.zeros(dims)

    def predict(self, s, a):
        if False:
            for i in range(10):
                print('nop')
        sa = merge_state_action(s, a)
        x = self.featurizer.transform([sa])[0]
        return x @ self.w

    def predict_all_actions(self, s):
        if False:
            i = 10
            return i + 15
        return [self.predict(s, a) for a in ALL_POSSIBLE_ACTIONS]

    def grad(self, s, a):
        if False:
            for i in range(10):
                print('nop')
        sa = merge_state_action(s, a)
        x = self.featurizer.transform([sa])[0]
        return x
if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)
    print('rewards:')
    print_values(grid.rewards, grid)
    model = Model(grid)
    reward_per_episode = []
    state_visit_count = {}
    n_episodes = 20000
    for it in range(n_episodes):
        if (it + 1) % 100 == 0:
            print(it + 1)
        s = grid.reset()
        state_visit_count[s] = state_visit_count.get(s, 0) + 1
        episode_reward = 0
        while not grid.game_over():
            a = epsilon_greedy(model, s)
            r = grid.move(a)
            s2 = grid.current_state()
            state_visit_count[s2] = state_visit_count.get(s2, 0) + 1
            if grid.game_over():
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)
            g = model.grad(s, a)
            err = target - model.predict(s, a)
            model.w += ALPHA * err * g
            episode_reward += r
            s = s2
        reward_per_episode.append(episode_reward)
    plt.plot(reward_per_episode)
    plt.title('Reward per episode')
    plt.show()
    V = {}
    greedy_policy = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            values = model.predict_all_actions(s)
            V[s] = np.max(values)
            greedy_policy[s] = ALL_POSSIBLE_ACTIONS[np.argmax(values)]
        else:
            V[s] = 0
    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(greedy_policy, grid)
    print('state_visit_count:')
    state_sample_count_arr = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            if (i, j) in state_visit_count:
                state_sample_count_arr[i, j] = state_visit_count[i, j]
    df = pd.DataFrame(state_sample_count_arr)
    print(df)