from __future__ import print_function, division
from builtins import range
import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic import print_values, print_policy
SMALL_ENOUGH = 0.001
GAMMA = 0.9

def get_transition_probs_and_rewards(grid):
    if False:
        i = 10
        return i + 15
    transition_probs = {}
    rewards = {}
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[s, a, s2] = 1
                    if s2 in grid.rewards:
                        rewards[s, a, s2] = grid.rewards[s2]
    return (transition_probs, rewards)

def evaluate_deterministic_policy(grid, policy, initV=None):
    if False:
        print('Hello World!')
    if initV is None:
        V = {}
        for s in grid.all_states():
            V[s] = 0
    else:
        V = initV
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = 1 if policy.get(s) == a else 0
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        it += 1
        if biggest_change < SMALL_ENOUGH:
            break
    return V
if __name__ == '__main__':
    grid = standard_grid()
    (transition_probs, rewards) = get_transition_probs_and_rewards(grid)
    print('rewards:')
    print_values(grid.rewards, grid)
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)
    print('initial policy:')
    print_policy(policy, grid)
    V = None
    while True:
        V = evaluate_deterministic_policy(grid, policy, initV=V)
        is_policy_converged = True
        for s in grid.actions.keys():
            old_a = policy[s]
            new_a = None
            best_value = float('-inf')
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                if v > best_value:
                    best_value = v
                    new_a = a
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False
        if is_policy_converged:
            break
    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)