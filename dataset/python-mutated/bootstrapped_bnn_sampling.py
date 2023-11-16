"""Contextual algorithm based on boostrapping neural networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel

class BootstrappedBNNSampling(BanditAlgorithm):
    """Thompson Sampling algorithm based on training several neural networks."""

    def __init__(self, name, hparams, optimizer='RMS'):
        if False:
            for i in range(10):
                print('nop')
        'Creates a BootstrappedSGDSampling object based on a specific optimizer.\n\n      hparams.q: Number of models that are independently trained.\n      hparams.p: Prob of independently including each datapoint in each model.\n\n    Args:\n      name: Name given to the instance.\n      hparams: Hyperparameters for each individual model.\n      optimizer: Neural network optimization algorithm.\n    '
        self.name = name
        self.hparams = hparams
        self.optimizer_n = optimizer
        self.training_freq = hparams.training_freq
        self.training_epochs = hparams.training_epochs
        self.t = 0
        self.q = hparams.q
        self.p = hparams.p
        self.datasets = [ContextualDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s) for _ in range(self.q)]
        self.bnn_boot = [NeuralBanditModel(optimizer, hparams, '{}-{}-bnn'.format(name, i)) for i in range(self.q)]

    def action(self, context):
        if False:
            for i in range(10):
                print('nop')
        'Selects action for context based on Thompson Sampling using one BNN.'
        if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            return self.t % self.hparams.num_actions
        model_index = np.random.randint(self.q)
        with self.bnn_boot[model_index].graph.as_default():
            c = context.reshape((1, self.hparams.context_dim))
            output = self.bnn_boot[model_index].sess.run(self.bnn_boot[model_index].y_pred, feed_dict={self.bnn_boot[model_index].x: c})
            return np.argmax(output)

    def update(self, context, action, reward):
        if False:
            print('Hello World!')
        'Updates the data buffer, and re-trains the BNN every self.freq_update.'
        self.t += 1
        for i in range(self.q):
            if np.random.random() < self.p or self.t < 2:
                self.datasets[i].add(context, action, reward)
        if self.t % self.training_freq == 0:
            for i in range(self.q):
                if self.hparams.reset_lr:
                    self.bnn_boot[i].assign_lr()
                self.bnn_boot[i].train(self.datasets[i], self.training_epochs)