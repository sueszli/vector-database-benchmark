"""Contextual bandit algorithm based on Thompson Sampling and a Bayesian NN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.algorithms.bb_alpha_divergence_model import BBAlphaDivergence
from bandits.algorithms.bf_variational_neural_bandit_model import BfVariationalNeuralBanditModel
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.multitask_gp import MultitaskGP
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.variational_neural_bandit_model import VariationalNeuralBanditModel

class PosteriorBNNSampling(BanditAlgorithm):
    """Posterior Sampling algorithm based on a Bayesian neural network."""

    def __init__(self, name, hparams, bnn_model='RMSProp'):
        if False:
            i = 10
            return i + 15
        'Creates a PosteriorBNNSampling object based on a specific optimizer.\n\n    The algorithm has two basic tools: an Approx BNN and a Contextual Dataset.\n    The Bayesian Network keeps the posterior based on the optimizer iterations.\n\n    Args:\n      name: Name of the algorithm.\n      hparams: Hyper-parameters of the algorithm.\n      bnn_model: Type of BNN. By default RMSProp (point estimate).\n    '
        self.name = name
        self.hparams = hparams
        self.optimizer_n = hparams.optimizer
        self.training_freq = hparams.training_freq
        self.training_epochs = hparams.training_epochs
        self.t = 0
        self.data_h = ContextualDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s)
        bnn_name = '{}-bnn'.format(name)
        if bnn_model == 'Variational':
            self.bnn = VariationalNeuralBanditModel(hparams, bnn_name)
        elif bnn_model == 'AlphaDiv':
            self.bnn = BBAlphaDivergence(hparams, bnn_name)
        elif bnn_model == 'Variational_BF':
            self.bnn = BfVariationalNeuralBanditModel(hparams, bnn_name)
        elif bnn_model == 'GP':
            self.bnn = MultitaskGP(hparams)
        else:
            self.bnn = NeuralBanditModel(self.optimizer_n, hparams, bnn_name)

    def action(self, context):
        if False:
            print('Hello World!')
        'Selects action for context based on Thompson Sampling using the BNN.'
        if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            return self.t % self.hparams.num_actions
        with self.bnn.graph.as_default():
            c = context.reshape((1, self.hparams.context_dim))
            output = self.bnn.sess.run(self.bnn.y_pred, feed_dict={self.bnn.x: c})
            return np.argmax(output)

    def update(self, context, action, reward):
        if False:
            print('Hello World!')
        'Updates data buffer, and re-trains the BNN every training_freq steps.'
        self.t += 1
        self.data_h.add(context, action, reward)
        if self.t % self.training_freq == 0:
            if self.hparams.reset_lr:
                self.bnn.assign_lr()
            self.bnn.train(self.data_h, self.training_epochs)