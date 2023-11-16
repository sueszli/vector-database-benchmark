"""Contextual algorithm that keeps a full linear posterior for each arm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.stats import invgamma
from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset

class LinearFullPosteriorSampling(BanditAlgorithm):
    """Thompson Sampling with independent linear models and unknown noise var."""

    def __init__(self, name, hparams):
        if False:
            i = 10
            return i + 15
        'Initialize posterior distributions and hyperparameters.\n\n    Assume a linear model for each action i: reward = context^T beta_i + noise\n    Each beta_i has a Gaussian prior (lambda parameter), each sigma2_i (noise\n    level) has an inverse Gamma prior (a0, b0 parameters). Mean, covariance,\n    and precision matrices are initialized, and the ContextualDataset created.\n\n    Args:\n      name: Name of the algorithm.\n      hparams: Hyper-parameters of the algorithm.\n    '
        self.name = name
        self.hparams = hparams
        self._lambda_prior = self.hparams.lambda_prior
        self.mu = [np.zeros(self.hparams.context_dim + 1) for _ in range(self.hparams.num_actions)]
        self.cov = [1.0 / self.lambda_prior * np.eye(self.hparams.context_dim + 1) for _ in range(self.hparams.num_actions)]
        self.precision = [self.lambda_prior * np.eye(self.hparams.context_dim + 1) for _ in range(self.hparams.num_actions)]
        self._a0 = self.hparams.a0
        self._b0 = self.hparams.b0
        self.a = [self._a0 for _ in range(self.hparams.num_actions)]
        self.b = [self._b0 for _ in range(self.hparams.num_actions)]
        self.t = 0
        self.data_h = ContextualDataset(hparams.context_dim, hparams.num_actions, intercept=True)

    def action(self, context):
        if False:
            print('Hello World!')
        "Samples beta's from posterior, and chooses best action accordingly.\n\n    Args:\n      context: Context for which the action need to be chosen.\n\n    Returns:\n      action: Selected action for the context.\n    "
        if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            return self.t % self.hparams.num_actions
        sigma2_s = [self.b[i] * invgamma.rvs(self.a[i]) for i in range(self.hparams.num_actions)]
        try:
            beta_s = [np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i]) for i in range(self.hparams.num_actions)]
        except np.linalg.LinAlgError as e:
            print('Exception when sampling from {}.'.format(self.name))
            print('Details: {} | {}.'.format(e.message, e.args))
            d = self.hparams.context_dim + 1
            beta_s = [np.random.multivariate_normal(np.zeros(d), np.eye(d)) for i in range(self.hparams.num_actions)]
        vals = [np.dot(beta_s[i][:-1], context.T) + beta_s[i][-1] for i in range(self.hparams.num_actions)]
        return np.argmax(vals)

    def update(self, context, action, reward):
        if False:
            i = 10
            return i + 15
        'Updates action posterior using the linear Bayesian regression formula.\n\n    Args:\n      context: Last observed context.\n      action: Last observed action.\n      reward: Last observed reward.\n    '
        self.t += 1
        self.data_h.add(context, action, reward)
        (x, y) = self.data_h.get_data(action)
        s = np.dot(x.T, x)
        precision_a = s + self.lambda_prior * np.eye(self.hparams.context_dim + 1)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(x.T, y))
        a_post = self.a0 + x.shape[0] / 2.0
        b_upd = 0.5 * (np.dot(y.T, y) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
        b_post = self.b0 + b_upd
        self.mu[action] = mu_a
        self.cov[action] = cov_a
        self.precision[action] = precision_a
        self.a[action] = a_post
        self.b[action] = b_post

    @property
    def a0(self):
        if False:
            return 10
        return self._a0

    @property
    def b0(self):
        if False:
            print('Hello World!')
        return self._b0

    @property
    def lambda_prior(self):
        if False:
            return 10
        return self._lambda_prior