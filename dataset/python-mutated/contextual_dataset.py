"""Define a data buffer for contextual bandit algorithms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

class ContextualDataset(object):
    """The buffer is able to append new data, and sample random minibatches."""

    def __init__(self, context_dim, num_actions, buffer_s=-1, intercept=False):
        if False:
            for i in range(10):
                print('nop')
        'Creates a ContextualDataset object.\n\n    The data is stored in attributes: contexts and rewards.\n    The sequence of taken actions are stored in attribute actions.\n\n    Args:\n      context_dim: Dimension of the contexts.\n      num_actions: Number of arms for the multi-armed bandit.\n      buffer_s: Size of buffer for training. Only last buffer_s will be\n        returned as minibatch. If buffer_s = -1, all data will be used.\n      intercept: If True, it adds a constant (1.0) dimension to each context X,\n        at the end.\n    '
        self._context_dim = context_dim
        self._num_actions = num_actions
        self._contexts = None
        self._rewards = None
        self.actions = []
        self.buffer_s = buffer_s
        self.intercept = intercept

    def add(self, context, action, reward):
        if False:
            print('Hello World!')
        "Adds a new triplet (context, action, reward) to the dataset.\n\n    The reward for the actions that weren't played is assumed to be zero.\n\n    Args:\n      context: A d-dimensional vector with the context.\n      action: Integer between 0 and k-1 representing the chosen arm.\n      reward: Real number representing the reward for the (context, action).\n    "
        if self.intercept:
            c = np.array(context[:])
            c = np.append(c, 1.0).reshape((1, self.context_dim + 1))
        else:
            c = np.array(context[:]).reshape((1, self.context_dim))
        if self.contexts is None:
            self.contexts = c
        else:
            self.contexts = np.vstack((self.contexts, c))
        r = np.zeros((1, self.num_actions))
        r[0, action] = reward
        if self.rewards is None:
            self.rewards = r
        else:
            self.rewards = np.vstack((self.rewards, r))
        self.actions.append(action)

    def replace_data(self, contexts=None, actions=None, rewards=None):
        if False:
            return 10
        if contexts is not None:
            self.contexts = contexts
        if actions is not None:
            self.actions = actions
        if rewards is not None:
            self.rewards = rewards

    def get_batch(self, batch_size):
        if False:
            return 10
        'Returns a random minibatch of (contexts, rewards) with batch_size.'
        (n, _) = self.contexts.shape
        if self.buffer_s == -1:
            ind = np.random.choice(range(n), batch_size)
        else:
            ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        return (self.contexts[ind, :], self.rewards[ind, :])

    def get_data(self, action):
        if False:
            while True:
                i = 10
        'Returns all (context, reward) where the action was played.'
        (n, _) = self.contexts.shape
        ind = np.array([i for i in range(n) if self.actions[i] == action])
        return (self.contexts[ind, :], self.rewards[ind, action])

    def get_data_with_weights(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns all observations with one-hot weights for actions.'
        weights = np.zeros((self.contexts.shape[0], self.num_actions))
        a_ind = np.array([(i, val) for (i, val) in enumerate(self.actions)])
        weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        return (self.contexts, self.rewards, weights)

    def get_batch_with_weights(self, batch_size):
        if False:
            i = 10
            return i + 15
        'Returns a random mini-batch with one-hot weights for actions.'
        (n, _) = self.contexts.shape
        if self.buffer_s == -1:
            ind = np.random.choice(range(n), batch_size)
        else:
            ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        weights = np.zeros((batch_size, self.num_actions))
        sampled_actions = np.array(self.actions)[ind]
        a_ind = np.array([(i, val) for (i, val) in enumerate(sampled_actions)])
        weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        return (self.contexts[ind, :], self.rewards[ind, :], weights)

    def num_points(self, f=None):
        if False:
            return 10
        'Returns number of points in the buffer (after applying function f).'
        if f is not None:
            return f(self.contexts.shape[0])
        return self.contexts.shape[0]

    @property
    def context_dim(self):
        if False:
            return 10
        return self._context_dim

    @property
    def num_actions(self):
        if False:
            print('Hello World!')
        return self._num_actions

    @property
    def contexts(self):
        if False:
            return 10
        return self._contexts

    @contexts.setter
    def contexts(self, value):
        if False:
            while True:
                i = 10
        self._contexts = value

    @property
    def actions(self):
        if False:
            i = 10
            return i + 15
        return self._actions

    @actions.setter
    def actions(self, value):
        if False:
            return 10
        self._actions = value

    @property
    def rewards(self):
        if False:
            i = 10
            return i + 15
        return self._rewards

    @rewards.setter
    def rewards(self, value):
        if False:
            while True:
                i = 10
        self._rewards = value