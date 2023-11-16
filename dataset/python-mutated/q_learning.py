from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def epsilon_greedy(Q, s, eps=0.1):
    if False:
        print('Hello World!')
    if np.random.random() < eps:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        a_opt = max_dict(Q[s])[0]
        return a_opt
if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)
    print('rewards:')
    print_values(grid.rewards, grid)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    update_counts = {}
    reward_per_episode = []
    for it in range(10000):
        if it % 2000 == 0:
            print('it:', it)
        s = grid.reset()
        episode_reward = 0
        while not grid.game_over():
            a = epsilon_greedy(Q, s, eps=0.1)
            r = grid.move(a)
            s2 = grid.current_state()
            episode_reward += r
            maxQ = max_dict(Q[s2])[1]
            Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * maxQ - Q[s][a])
            update_counts[s] = update_counts.get(s, 0) + 1
            s = s2
        reward_per_episode.append(episode_reward)
    plt.plot(reward_per_episode)
    plt.title('reward_per_episode')
    plt.show()
    policy = {}
    V = {}
    for s in grid.actions.keys():
        (a, max_q) = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q
    print('update counts:')
    total = np.sum(list(update_counts.values()))
    for (k, v) in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)
    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)