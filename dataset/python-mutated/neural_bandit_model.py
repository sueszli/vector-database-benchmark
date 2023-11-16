"""Define a family of neural network architectures for bandits.

The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from absl import flags
from bandits.core.bayesian_nn import BayesianNN
FLAGS = flags.FLAGS

class NeuralBanditModel(BayesianNN):
    """Implements a neural network for bandit problems."""

    def __init__(self, optimizer, hparams, name):
        if False:
            while True:
                i = 10
        'Saves hyper-params and builds the Tensorflow graph.'
        self.opt_name = optimizer
        self.name = name
        self.hparams = hparams
        self.verbose = getattr(self.hparams, 'verbose', True)
        self.times_trained = 0
        self.build_model()

    def build_layer(self, x, num_units):
        if False:
            print('Hello World!')
        'Builds a layer with input x; dropout and layer norm if specified.'
        init_s = self.hparams.init_scale
        layer_n = getattr(self.hparams, 'layer_norm', False)
        dropout = getattr(self.hparams, 'use_dropout', False)
        nn = tf.contrib.layers.fully_connected(x, num_units, activation_fn=self.hparams.activation, normalizer_fn=None if not layer_n else tf.contrib.layers.layer_norm, normalizer_params={}, weights_initializer=tf.random_uniform_initializer(-init_s, init_s))
        if dropout:
            nn = tf.nn.dropout(nn, self.hparams.keep_prob)
        return nn

    def forward_pass(self):
        if False:
            while True:
                i = 10
        init_s = self.hparams.init_scale
        scope_name = 'prediction_{}'.format(self.name)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            nn = self.x
            for num_units in self.hparams.layer_sizes:
                if num_units > 0:
                    nn = self.build_layer(nn, num_units)
            y_pred = tf.layers.dense(nn, self.hparams.num_actions, kernel_initializer=tf.random_uniform_initializer(-init_s, init_s))
        return (nn, y_pred)

    def build_model(self):
        if False:
            i = 10
            return i + 15
        'Defines the actual NN model with fully connected layers.\n\n    The loss is computed for partial feedback settings (bandits), so only\n    the observed outcome is backpropagated (see weighted loss).\n    Selects the optimizer and, finally, it also initializes the graph.\n    '
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with tf.name_scope(self.name):
                self.global_step = tf.train.get_or_create_global_step()
                self.x = tf.placeholder(shape=[None, self.hparams.context_dim], dtype=tf.float32, name='{}_x'.format(self.name))
                self.y = tf.placeholder(shape=[None, self.hparams.num_actions], dtype=tf.float32, name='{}_y'.format(self.name))
                self.weights = tf.placeholder(shape=[None, self.hparams.num_actions], dtype=tf.float32, name='{}_w'.format(self.name))
                (self.nn, self.y_pred) = self.forward_pass()
                self.loss = tf.squared_difference(self.y_pred, self.y)
                self.weighted_loss = tf.multiply(self.weights, self.loss)
                self.cost = tf.reduce_sum(self.weighted_loss) / self.hparams.batch_size
                if self.hparams.activate_decay:
                    self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr, self.global_step, 1, self.hparams.lr_decay_rate)
                else:
                    self.lr = tf.Variable(self.hparams.initial_lr, trainable=False)
                self.create_summaries()
                self.summary_writer = tf.summary.FileWriter('{}/graph_{}'.format(FLAGS.logdir, self.name), self.sess.graph)
                tvars = tf.trainable_variables()
                (grads, _) = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.hparams.max_grad_norm)
                self.optimizer = self.select_optimizer()
                self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
                self.init = tf.global_variables_initializer()
                self.initialize_graph()

    def initialize_graph(self):
        if False:
            for i in range(10):
                print('nop')
        'Initializes all variables.'
        with self.graph.as_default():
            if self.verbose:
                print('Initializing model {}.'.format(self.name))
            self.sess.run(self.init)

    def assign_lr(self):
        if False:
            i = 10
            return i + 15
        'Resets the learning rate in dynamic schedules for subsequent trainings.\n\n    In bandits settings, we do expand our dataset over time. Then, we need to\n    re-train the network with the new data. The algorithms that do not keep\n    the step constant, can reset it at the start of each *training* process.\n    '
        decay_steps = 1
        if self.hparams.activate_decay:
            current_gs = self.sess.run(self.global_step)
            with self.graph.as_default():
                self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr, self.global_step - current_gs, decay_steps, self.hparams.lr_decay_rate)

    def select_optimizer(self):
        if False:
            while True:
                i = 10
        'Selects optimizer. To be extended (SGLD, KFAC, etc).'
        return tf.train.RMSPropOptimizer(self.lr)

    def create_summaries(self):
        if False:
            i = 10
            return i + 15
        'Defines summaries including mean loss, learning rate, and global step.'
        with self.graph.as_default():
            with tf.name_scope(self.name + '_summaries'):
                tf.summary.scalar('cost', self.cost)
                tf.summary.scalar('lr', self.lr)
                tf.summary.scalar('global_step', self.global_step)
                self.summary_op = tf.summary.merge_all()

    def train(self, data, num_steps):
        if False:
            for i in range(10):
                print('nop')
        'Trains the network for num_steps, using the provided data.\n\n    Args:\n      data: ContextualDataset object that provides the data.\n      num_steps: Number of minibatches to train the network for.\n    '
        if self.verbose:
            print('Training {} for {} steps...'.format(self.name, num_steps))
        with self.graph.as_default():
            for step in range(num_steps):
                (x, y, w) = data.get_batch_with_weights(self.hparams.batch_size)
                (_, cost, summary, lr) = self.sess.run([self.train_op, self.cost, self.summary_op, self.lr], feed_dict={self.x: x, self.y: y, self.weights: w})
                if step % self.hparams.freq_summary == 0:
                    if self.hparams.show_training:
                        print('{} | step: {}, lr: {}, loss: {}'.format(self.name, step, lr, cost))
                    self.summary_writer.add_summary(summary, step)
            self.times_trained += 1