from __future__ import print_function, division
from builtins import range
import numpy as np
ACTION_SPACE = ('U', 'D', 'L', 'R')

class Grid:

    def __init__(self, rows, cols, start):
        if False:
            for i in range(10):
                print('nop')
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        if False:
            for i in range(10):
                print('nop')
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        if False:
            return 10
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        if False:
            print('Hello World!')
        return (self.i, self.j)

    def is_terminal(self, s):
        if False:
            return 10
        return s not in self.actions

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.i = 2
        self.j = 0
        return (self.i, self.j)

    def get_next_state(self, s, a):
        if False:
            return 10
        (i, j) = (s[0], s[1])
        if a in self.actions[i, j]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'R':
                j += 1
            elif a == 'L':
                j -= 1
        return (i, j)

    def move(self, action):
        if False:
            i = 10
            return i + 15
        if action in self.actions[self.i, self.j]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        if False:
            i = 10
            return i + 15
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        assert self.current_state() in self.all_states()

    def game_over(self):
        if False:
            i = 10
            return i + 15
        return (self.i, self.j) not in self.actions

    def all_states(self):
        if False:
            print('Hello World!')
        return set(self.actions.keys()) | set(self.rewards.keys())

def standard_grid():
    if False:
        for i in range(10):
            print('nop')
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {(0, 0): ('D', 'R'), (0, 1): ('L', 'R'), (0, 2): ('L', 'D', 'R'), (1, 0): ('U', 'D'), (1, 2): ('U', 'D', 'R'), (2, 0): ('U', 'R'), (2, 1): ('L', 'R'), (2, 2): ('L', 'R', 'U'), (2, 3): ('L', 'U')}
    g.set(rewards, actions)
    return g

def negative_grid(step_cost=-0.1):
    if False:
        print('Hello World!')
    g = standard_grid()
    g.rewards.update({(0, 0): step_cost, (0, 1): step_cost, (0, 2): step_cost, (1, 0): step_cost, (1, 2): step_cost, (2, 0): step_cost, (2, 1): step_cost, (2, 2): step_cost, (2, 3): step_cost})
    return g

class WindyGrid:

    def __init__(self, rows, cols, start):
        if False:
            print('Hello World!')
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions, probs):
        if False:
            print('Hello World!')
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def set_state(self, s):
        if False:
            print('Hello World!')
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        if False:
            print('Hello World!')
        return (self.i, self.j)

    def is_terminal(self, s):
        if False:
            while True:
                i = 10
        return s not in self.actions

    def move(self, action):
        if False:
            for i in range(10):
                print('nop')
        s = (self.i, self.j)
        a = action
        next_state_probs = self.probs[s, a]
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        next_state_idx = np.random.choice(len(next_states), p=next_probs)
        s2 = next_states[next_state_idx]
        (self.i, self.j) = s2
        return self.rewards.get(s2, 0)

    def game_over(self):
        if False:
            i = 10
            return i + 15
        return (self.i, self.j) not in self.actions

    def all_states(self):
        if False:
            return 10
        return set(self.actions.keys()) | set(self.rewards.keys())

def windy_grid():
    if False:
        print('Hello World!')
    g = WindyGrid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {(0, 0): ('D', 'R'), (0, 1): ('L', 'R'), (0, 2): ('L', 'D', 'R'), (1, 0): ('U', 'D'), (1, 2): ('U', 'D', 'R'), (2, 0): ('U', 'R'), (2, 1): ('L', 'R'), (2, 2): ('L', 'R', 'U'), (2, 3): ('L', 'U')}
    probs = {((2, 0), 'U'): {(1, 0): 1.0}, ((2, 0), 'D'): {(2, 0): 1.0}, ((2, 0), 'L'): {(2, 0): 1.0}, ((2, 0), 'R'): {(2, 1): 1.0}, ((1, 0), 'U'): {(0, 0): 1.0}, ((1, 0), 'D'): {(2, 0): 1.0}, ((1, 0), 'L'): {(1, 0): 1.0}, ((1, 0), 'R'): {(1, 0): 1.0}, ((0, 0), 'U'): {(0, 0): 1.0}, ((0, 0), 'D'): {(1, 0): 1.0}, ((0, 0), 'L'): {(0, 0): 1.0}, ((0, 0), 'R'): {(0, 1): 1.0}, ((0, 1), 'U'): {(0, 1): 1.0}, ((0, 1), 'D'): {(0, 1): 1.0}, ((0, 1), 'L'): {(0, 0): 1.0}, ((0, 1), 'R'): {(0, 2): 1.0}, ((0, 2), 'U'): {(0, 2): 1.0}, ((0, 2), 'D'): {(1, 2): 1.0}, ((0, 2), 'L'): {(0, 1): 1.0}, ((0, 2), 'R'): {(0, 3): 1.0}, ((2, 1), 'U'): {(2, 1): 1.0}, ((2, 1), 'D'): {(2, 1): 1.0}, ((2, 1), 'L'): {(2, 0): 1.0}, ((2, 1), 'R'): {(2, 2): 1.0}, ((2, 2), 'U'): {(1, 2): 1.0}, ((2, 2), 'D'): {(2, 2): 1.0}, ((2, 2), 'L'): {(2, 1): 1.0}, ((2, 2), 'R'): {(2, 3): 1.0}, ((2, 3), 'U'): {(1, 3): 1.0}, ((2, 3), 'D'): {(2, 3): 1.0}, ((2, 3), 'L'): {(2, 2): 1.0}, ((2, 3), 'R'): {(2, 3): 1.0}, ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5}, ((1, 2), 'D'): {(2, 2): 1.0}, ((1, 2), 'L'): {(1, 2): 1.0}, ((1, 2), 'R'): {(1, 3): 1.0}}
    g.set(rewards, actions, probs)
    return g

def windy_grid_no_wind():
    if False:
        while True:
            i = 10
    g = windy_grid()
    g.probs[(1, 2), 'U'] = {(0, 2): 1.0}
    return g

def windy_grid_penalized(step_cost=-0.1):
    if False:
        return 10
    g = WindyGrid(3, 4, (2, 0))
    rewards = {(0, 0): step_cost, (0, 1): step_cost, (0, 2): step_cost, (1, 0): step_cost, (1, 2): step_cost, (2, 0): step_cost, (2, 1): step_cost, (2, 2): step_cost, (2, 3): step_cost, (0, 3): 1, (1, 3): -1}
    actions = {(0, 0): ('D', 'R'), (0, 1): ('L', 'R'), (0, 2): ('L', 'D', 'R'), (1, 0): ('U', 'D'), (1, 2): ('U', 'D', 'R'), (2, 0): ('U', 'R'), (2, 1): ('L', 'R'), (2, 2): ('L', 'R', 'U'), (2, 3): ('L', 'U')}
    probs = {((2, 0), 'U'): {(1, 0): 1.0}, ((2, 0), 'D'): {(2, 0): 1.0}, ((2, 0), 'L'): {(2, 0): 1.0}, ((2, 0), 'R'): {(2, 1): 1.0}, ((1, 0), 'U'): {(0, 0): 1.0}, ((1, 0), 'D'): {(2, 0): 1.0}, ((1, 0), 'L'): {(1, 0): 1.0}, ((1, 0), 'R'): {(1, 0): 1.0}, ((0, 0), 'U'): {(0, 0): 1.0}, ((0, 0), 'D'): {(1, 0): 1.0}, ((0, 0), 'L'): {(0, 0): 1.0}, ((0, 0), 'R'): {(0, 1): 1.0}, ((0, 1), 'U'): {(0, 1): 1.0}, ((0, 1), 'D'): {(0, 1): 1.0}, ((0, 1), 'L'): {(0, 0): 1.0}, ((0, 1), 'R'): {(0, 2): 1.0}, ((0, 2), 'U'): {(0, 2): 1.0}, ((0, 2), 'D'): {(1, 2): 1.0}, ((0, 2), 'L'): {(0, 1): 1.0}, ((0, 2), 'R'): {(0, 3): 1.0}, ((2, 1), 'U'): {(2, 1): 1.0}, ((2, 1), 'D'): {(2, 1): 1.0}, ((2, 1), 'L'): {(2, 0): 1.0}, ((2, 1), 'R'): {(2, 2): 1.0}, ((2, 2), 'U'): {(1, 2): 1.0}, ((2, 2), 'D'): {(2, 2): 1.0}, ((2, 2), 'L'): {(2, 1): 1.0}, ((2, 2), 'R'): {(2, 3): 1.0}, ((2, 3), 'U'): {(1, 3): 1.0}, ((2, 3), 'D'): {(2, 3): 1.0}, ((2, 3), 'L'): {(2, 2): 1.0}, ((2, 3), 'R'): {(2, 3): 1.0}, ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5}, ((1, 2), 'D'): {(2, 2): 1.0}, ((1, 2), 'L'): {(1, 2): 1.0}, ((1, 2), 'R'): {(1, 3): 1.0}}
    g.set(rewards, actions, probs)
    return g

def grid_5x5(step_cost=-0.1):
    if False:
        for i in range(10):
            print('nop')
    g = Grid(5, 5, (4, 0))
    rewards = {(0, 4): 1, (1, 4): -1}
    actions = {(0, 0): ('D', 'R'), (0, 1): ('L', 'R'), (0, 2): ('L', 'R'), (0, 3): ('L', 'D', 'R'), (1, 0): ('U', 'D', 'R'), (1, 1): ('U', 'D', 'L'), (1, 3): ('U', 'D', 'R'), (2, 0): ('U', 'D', 'R'), (2, 1): ('U', 'L', 'R'), (2, 2): ('L', 'R', 'D'), (2, 3): ('L', 'R', 'U'), (2, 4): ('L', 'U', 'D'), (3, 0): ('U', 'D'), (3, 2): ('U', 'D'), (3, 4): ('U', 'D'), (4, 0): ('U', 'R'), (4, 1): ('L', 'R'), (4, 2): ('L', 'R', 'U'), (4, 3): ('L', 'R'), (4, 4): ('L', 'U')}
    g.set(rewards, actions)
    visitable_states = actions.keys()
    for s in visitable_states:
        g.rewards[s] = step_cost
    return g