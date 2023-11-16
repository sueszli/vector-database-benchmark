"""Bayesian NN using factorized VI (Bayes By Backprop. Blundell et al. 2014).

See https://arxiv.org/abs/1505.05424 for details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from absl import flags
from bandits.core.bayesian_nn import BayesianNN
FLAGS = flags.FLAGS
tfd = tf.contrib.distributions
tfl = tf.contrib.layers

def log_gaussian(x, mu, sigma, reduce_sum=True):
    if False:
        while True:
            i = 10
    'Returns log Gaussian pdf.'
    res = tfd.Normal(mu, sigma).log_prob(x)
    if reduce_sum:
        return tf.reduce_sum(res)
    else:
        return res

def analytic_kl(mu_1, sigma_1, mu_2, sigma_2):
    if False:
        print('Hello World!')
    'KL for two Gaussian distributions with diagonal covariance matrix.'
    kl = tfd.kl_divergence(tfd.MVNDiag(mu_1, sigma_1), tfd.MVNDiag(mu_2, sigma_2))
    return kl

class BfVariationalNeuralBanditModel(BayesianNN):
    """Implements an approximate Bayesian NN using Variational Inference."""

    def __init__(self, hparams, name='BBBNN'):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.hparams = hparams
        self.n_in = self.hparams.context_dim
        self.n_out = self.hparams.num_actions
        self.layers = self.hparams.layer_sizes
        self.init_scale = self.hparams.init_scale
        self.f_num_points = None
        if 'f_num_points' in hparams:
            self.f_num_points = self.hparams.f_num_points
        self.cleared_times_trained = self.hparams.cleared_times_trained
        self.initial_training_steps = self.hparams.initial_training_steps
        self.training_schedule = np.linspace(self.initial_training_steps, self.hparams.training_epochs, self.cleared_times_trained)
        self.verbose = getattr(self.hparams, 'verbose', True)
        self.weights_m = {}
        self.weights_std = {}
        self.biases_m = {}
        self.biases_std = {}
        self.times_trained = 0
        if self.hparams.use_sigma_exp_transform:
            self.sigma_transform = tf.exp
            self.inverse_sigma_transform = np.log
        else:
            self.sigma_transform = tf.nn.softplus
            self.inverse_sigma_transform = lambda y: y + np.log(1.0 - np.exp(-y))
        self.use_local_reparameterization = True
        self.build_graph()

    def build_mu_variable(self, shape):
        if False:
            i = 10
            return i + 15
        'Returns a mean variable initialized as N(0, 0.05).'
        return tf.Variable(tf.random_normal(shape, 0.0, 0.05))

    def build_sigma_variable(self, shape, init=-5.0):
        if False:
            return 10
        'Returns a sigma variable initialized as N(init, 0.05).'
        return tf.Variable(tf.random_normal(shape, init, 0.05))

    def build_layer(self, input_x, input_x_local, shape, layer_id, activation_fn=tf.nn.relu):
        if False:
            for i in range(10):
                print('nop')
        'Builds a variational layer, and computes KL term.\n\n    Args:\n      input_x: Input to the variational layer.\n      input_x_local: Input when the local reparameterization trick was applied.\n      shape: [number_inputs, number_outputs] for the layer.\n      layer_id: Number of layer in the architecture.\n      activation_fn: Activation function to apply.\n\n    Returns:\n      output_h: Output of the variational layer.\n      output_h_local: Output when local reparameterization trick was applied.\n      neg_kl: Negative KL term for the layer.\n    '
        w_mu = self.build_mu_variable(shape)
        w_sigma = self.sigma_transform(self.build_sigma_variable(shape))
        w_noise = tf.random_normal(shape)
        w = w_mu + w_sigma * w_noise
        b_mu = self.build_mu_variable([1, shape[1]])
        b_sigma = self.sigma_transform(self.build_sigma_variable([1, shape[1]]))
        b = b_mu
        self.weights_m[layer_id] = w_mu
        self.weights_std[layer_id] = w_sigma
        self.biases_m[layer_id] = b_mu
        self.biases_std[layer_id] = b_sigma
        output_h = activation_fn(tf.matmul(input_x, w) + b)
        if self.use_local_reparameterization:
            neg_kl = -analytic_kl(w_mu, w_sigma, 0.0, tf.to_float(np.sqrt(2.0 / shape[0])))
        else:
            log_p = log_gaussian(w, 0.0, tf.to_float(np.sqrt(2.0 / shape[0])))
            log_q = log_gaussian(w, tf.stop_gradient(w_mu), tf.stop_gradient(w_sigma))
            neg_kl = log_p - log_q
        m_h = tf.matmul(input_x_local, w_mu) + b
        v_h = tf.matmul(tf.square(input_x_local), tf.square(w_sigma))
        output_h_local = m_h + tf.sqrt(v_h + 1e-06) * tf.random_normal(tf.shape(v_h))
        output_h_local = activation_fn(output_h_local)
        return (output_h, output_h_local, neg_kl)

    def build_action_noise(self):
        if False:
            return 10
        'Defines a model for additive noise per action, and its KL term.'
        noise_sigma_mu = self.build_mu_variable([1, self.n_out]) + self.inverse_sigma_transform(self.hparams.noise_sigma)
        noise_sigma_sigma = self.sigma_transform(self.build_sigma_variable([1, self.n_out]))
        pre_noise_sigma = noise_sigma_mu + tf.random_normal([1, self.n_out]) * noise_sigma_sigma
        self.noise_sigma = self.sigma_transform(pre_noise_sigma)
        if getattr(self.hparams, 'infer_noise_sigma', False):
            neg_kl_term = log_gaussian(pre_noise_sigma, self.inverse_sigma_transform(self.hparams.noise_sigma), self.hparams.prior_sigma)
            neg_kl_term -= log_gaussian(pre_noise_sigma, noise_sigma_mu, noise_sigma_sigma)
        else:
            neg_kl_term = 0.0
        return neg_kl_term

    def build_model(self, activation_fn=tf.nn.relu):
        if False:
            return 10
        'Defines the actual NN model with fully connected layers.\n\n    The loss is computed for partial feedback settings (bandits), so only\n    the observed outcome is backpropagated (see weighted loss).\n    Selects the optimizer and, finally, it also initializes the graph.\n\n    Args:\n      activation_fn: the activation function used in the nn layers.\n    '

        def weight_prior(dtype, shape, c, d, e):
            if False:
                i = 10
                return i + 15
            del c, d, e
            return tfd.Independent(tfd.Normal(loc=tf.zeros(shape, dtype), scale=tf.to_float(np.sqrt(2) / shape[0])), reinterpreted_batch_ndims=tf.size(shape))
        if self.verbose:
            print('Initializing model {}.'.format(self.name))
        neg_kl_term = self.build_action_noise()
        input_x = self.x
        model_layers = [tfl.DenseLocalReparameterization(n_nodes, activation=tf.nn.relu, kernel_prior_fn=weight_prior) for n_nodes in self.layers if n_nodes > 0]
        output_layer = tfl.DenseLocalReparameterization(self.n_out, activation=lambda x: x, kernel_prior_fn=weight_prior)
        model_layers.append(output_layer)
        model = tf.keras.Sequential(model_layers)
        self.y_pred = model(input_x)
        neg_kl_term -= tf.add_n(model.losses)
        if getattr(self.hparams, 'infer_noise_sigma', False):
            log_likelihood = log_gaussian(self.y, self.y_pred, self.noise_sigma, reduce_sum=False)
        else:
            log_likelihood = log_gaussian(self.y, self.y_pred, self.hparams.noise_sigma, reduce_sum=False)
        batch_size = tf.to_float(tf.shape(self.x)[0])
        weighted_log_likelihood = tf.reduce_sum(log_likelihood * self.weights) / batch_size
        elbo = weighted_log_likelihood + neg_kl_term / self.n
        self.loss = -elbo
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = tf.train.AdamOptimizer(self.hparams.initial_lr).minimize(self.loss, global_step=self.global_step)
        self.create_summaries()
        self.summary_writer = tf.summary.FileWriter('{}/graph_{}'.format(FLAGS.logdir, self.name), self.sess.graph)

    def build_graph(self):
        if False:
            i = 10
            return i + 15
        'Defines graph, session, placeholders, and model.\n\n    Placeholders are: n (size of the dataset), x and y (context and observed\n    reward for each action), and weights (one-hot encoding of selected action\n    for each context, i.e., only possibly non-zero element in each y).\n    '
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.n = tf.placeholder(shape=[], dtype=tf.float32)
            self.x = tf.placeholder(shape=[None, self.n_in], dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.n_out], dtype=tf.float32)
            self.weights = tf.placeholder(shape=[None, self.n_out], dtype=tf.float32)
            self.build_model()
            self.sess.run(tf.global_variables_initializer())

    def create_summaries(self):
        if False:
            while True:
                i = 10
        'Defines summaries including mean loss, and global step.'
        with self.graph.as_default():
            with tf.name_scope(self.name + '_summaries'):
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('global_step', self.global_step)
                self.summary_op = tf.summary.merge_all()

    def assign_lr(self):
        if False:
            return 10
        'Resets the learning rate in dynamic schedules for subsequent trainings.\n\n    In bandits settings, we do expand our dataset over time. Then, we need to\n    re-train the network with the new data. The algorithms that do not keep\n    the step constant, can reset it at the start of each *training* process.\n    '
        decay_steps = 1
        if self.hparams.activate_decay:
            current_gs = self.sess.run(self.global_step)
            with self.graph.as_default():
                self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr, self.global_step - current_gs, decay_steps, self.hparams.lr_decay_rate)

    def train(self, data, num_steps):
        if False:
            while True:
                i = 10
        "Trains the BNN for num_steps, using the data in 'data'.\n\n    Args:\n      data: ContextualDataset object that provides the data.\n      num_steps: Number of minibatches to train the network for.\n\n    Returns:\n      losses: Loss history during training.\n    "
        if self.times_trained < self.cleared_times_trained:
            num_steps = int(self.training_schedule[self.times_trained])
        self.times_trained += 1
        losses = []
        with self.graph.as_default():
            if self.verbose:
                print('Training {} for {} steps...'.format(self.name, num_steps))
            for step in range(num_steps):
                (x, y, weights) = data.get_batch_with_weights(self.hparams.batch_size)
                (_, summary, global_step, loss) = self.sess.run([self.train_op, self.summary_op, self.global_step, self.loss], feed_dict={self.x: x, self.y: y, self.weights: weights, self.n: data.num_points(self.f_num_points)})
                losses.append(loss)
                if step % self.hparams.freq_summary == 0:
                    if self.hparams.show_training:
                        print('{} | step: {}, loss: {}'.format(self.name, global_step, loss))
                    self.summary_writer.add_summary(summary, global_step)
        return losses