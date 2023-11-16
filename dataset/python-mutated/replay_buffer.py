"""Replay buffer.

Implements replay buffer in Python.
"""
import random
import numpy as np
from six.moves import xrange

class ReplayBuffer(object):

    def __init__(self, max_size):
        if False:
            return 10
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}
        self.init_length = 0

    def __len__(self):
        if False:
            return 10
        return self.cur_size

    def seed_buffer(self, episodes):
        if False:
            return 10
        self.init_length = len(episodes)
        self.add(episodes, np.ones(self.init_length))

    def add(self, episodes, *args):
        if False:
            return 10
        'Add episodes to buffer.'
        idx = 0
        while self.cur_size < self.max_size and idx < len(episodes):
            self.buffer[self.cur_size] = episodes[idx]
            self.cur_size += 1
            idx += 1
        if idx < len(episodes):
            remove_idxs = self.remove_n(len(episodes) - idx)
            for remove_idx in remove_idxs:
                self.buffer[remove_idx] = episodes[idx]
                idx += 1
        assert len(self.buffer) == self.cur_size

    def remove_n(self, n):
        if False:
            i = 10
            return i + 15
        'Get n items for removal.'
        idxs = random.sample(xrange(self.init_length, self.cur_size), n)
        return idxs

    def get_batch(self, n):
        if False:
            return 10
        'Get batch of episodes to train on.'
        idxs = random.sample(xrange(self.cur_size), n)
        return ([self.buffer[idx] for idx in idxs], None)

    def update_last_batch(self, delta):
        if False:
            while True:
                i = 10
        pass

class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, max_size, alpha=0.2, eviction_strategy='rand'):
        if False:
            return 10
        self.max_size = max_size
        self.alpha = alpha
        self.eviction_strategy = eviction_strategy
        assert self.eviction_strategy in ['rand', 'fifo', 'rank']
        self.remove_idx = 0
        self.cur_size = 0
        self.buffer = {}
        self.priorities = np.zeros(self.max_size)
        self.init_length = 0

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.cur_size

    def add(self, episodes, priorities, new_idxs=None):
        if False:
            for i in range(10):
                print('nop')
        'Add episodes to buffer.'
        if new_idxs is None:
            idx = 0
            new_idxs = []
            while self.cur_size < self.max_size and idx < len(episodes):
                self.buffer[self.cur_size] = episodes[idx]
                new_idxs.append(self.cur_size)
                self.cur_size += 1
                idx += 1
            if idx < len(episodes):
                remove_idxs = self.remove_n(len(episodes) - idx)
                for remove_idx in remove_idxs:
                    self.buffer[remove_idx] = episodes[idx]
                    new_idxs.append(remove_idx)
                    idx += 1
        else:
            assert len(new_idxs) == len(episodes)
            for (new_idx, ep) in zip(new_idxs, episodes):
                self.buffer[new_idx] = ep
        self.priorities[new_idxs] = priorities
        self.priorities[0:self.init_length] = np.max(self.priorities[self.init_length:])
        assert len(self.buffer) == self.cur_size
        return new_idxs

    def remove_n(self, n):
        if False:
            i = 10
            return i + 15
        'Get n items for removal.'
        assert self.init_length + n <= self.cur_size
        if self.eviction_strategy == 'rand':
            idxs = random.sample(xrange(self.init_length, self.cur_size), n)
        elif self.eviction_strategy == 'fifo':
            idxs = [self.init_length + (self.remove_idx + i) % (self.max_size - self.init_length) for i in xrange(n)]
            self.remove_idx = idxs[-1] + 1 - self.init_length
        elif self.eviction_strategy == 'rank':
            idxs = np.argpartition(self.priorities, n)[:n]
        return idxs

    def sampling_distribution(self):
        if False:
            for i in range(10):
                print('nop')
        p = self.priorities[:self.cur_size]
        p = np.exp(self.alpha * (p - np.max(p)))
        norm = np.sum(p)
        if norm > 0:
            uniform = 0.0
            p = p / norm * (1 - uniform) + 1.0 / self.cur_size * uniform
        else:
            p = np.ones(self.cur_size) / self.cur_size
        return p

    def get_batch(self, n):
        if False:
            print('Hello World!')
        'Get batch of episodes to train on.'
        p = self.sampling_distribution()
        idxs = np.random.choice(self.cur_size, size=int(n), replace=False, p=p)
        self.last_batch = idxs
        return ([self.buffer[idx] for idx in idxs], p[idxs])

    def update_last_batch(self, delta):
        if False:
            while True:
                i = 10
        'Update last batch idxs with new priority.'
        self.priorities[self.last_batch] = np.abs(delta)
        self.priorities[0:self.init_length] = np.max(self.priorities[self.init_length:])