from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
from scipy.misc import logsumexp
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
import utils as U
try:
    xrange
except NameError:
    xrange = range
FLAGS = tf.flags.FLAGS
Q_COLLECTION = 'q_collection'
P_COLLECTION = 'p_collection'

class SBN(object):

    def __init__(self, hparams, activation_func=tf.nn.sigmoid, mean_xs=None, eval_mode=False):
        if False:
            print('Hello World!')
        self.eval_mode = eval_mode
        self.hparams = hparams
        self.mean_xs = mean_xs
        self.train_bias = -np.log(1.0 / np.clip(mean_xs, 0.001, 0.999) - 1.0).astype(np.float32)
        self.activation_func = activation_func
        self.n_samples = tf.placeholder('int32')
        self.x = tf.placeholder('float', [None, self.hparams.n_input])
        self._x = tf.tile(self.x, [self.n_samples, 1])
        self.batch_size = tf.shape(self._x)[0]
        self.uniform_samples = dict()
        self.uniform_samples_v = dict()
        self.prior = tf.Variable(tf.zeros([self.hparams.n_hidden], dtype=tf.float32), name='p_prior', collections=[tf.GraphKeys.GLOBAL_VARIABLES, P_COLLECTION])
        self.run_recognition_network = False
        self.run_generator_network = False
        self.pre_temperature_variable = tf.Variable(np.log(self.hparams.temperature), trainable=False, dtype=tf.float32)
        self.temperature_variable = tf.exp(self.pre_temperature_variable)
        self.global_step = tf.Variable(0, trainable=False)
        self.baseline_loss = []
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.maintain_ema_ops = []
        self.optimizer_class = tf.train.AdamOptimizer(learning_rate=1 * self.hparams.learning_rate, beta2=self.hparams.beta2)
        self._generate_randomness()
        self._create_network()

    def initialize(self, sess):
        if False:
            for i in range(10):
                print('nop')
        self.sess = sess

    def _create_eta(self, shape=[], collection='CV'):
        if False:
            print('Hello World!')
        return 2 * tf.sigmoid(tf.Variable(tf.zeros(shape), trainable=False, collections=[collection, tf.GraphKeys.GLOBAL_VARIABLES, Q_COLLECTION]))

    def _create_baseline(self, n_output=1, n_hidden=100, is_zero_init=False, collection='BASELINE'):
        if False:
            for i in range(10):
                print('nop')
        h = self._x
        if self.mean_xs is not None:
            h -= self.mean_xs
        if is_zero_init:
            initializer = init_ops.zeros_initializer()
        else:
            initializer = slim.variance_scaling_initializer()
        with slim.arg_scope([slim.fully_connected], variables_collections=[collection, Q_COLLECTION], trainable=False, weights_initializer=initializer):
            h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.tanh)
            baseline = slim.fully_connected(h, n_output, activation_fn=None)
            if n_output == 1:
                baseline = tf.reshape(baseline, [-1])
        return baseline

    def _create_transformation(self, input, n_output, reuse, scope_prefix):
        if False:
            return 10
        'Create the deterministic transformation between stochastic layers.\n\n    If self.hparam.nonlinear:\n        2 x tanh layers\n    Else:\n        1 x linear layer\n    '
        if self.hparams.nonlinear:
            h = slim.fully_connected(input, self.hparams.n_hidden, reuse=reuse, activation_fn=tf.nn.tanh, scope='%s_nonlinear_1' % scope_prefix)
            h = slim.fully_connected(h, self.hparams.n_hidden, reuse=reuse, activation_fn=tf.nn.tanh, scope='%s_nonlinear_2' % scope_prefix)
            h = slim.fully_connected(h, n_output, reuse=reuse, activation_fn=None, scope='%s' % scope_prefix)
        else:
            h = slim.fully_connected(input, n_output, reuse=reuse, activation_fn=None, scope='%s' % scope_prefix)
        return h

    def _recognition_network(self, sampler=None, log_likelihood_func=None):
        if False:
            return 10
        'x values -> samples from Q and return log Q(h|x).'
        samples = {}
        reuse = None if not self.run_recognition_network else True
        if sampler is None:
            sampler = self._random_sample
        if log_likelihood_func is None:
            log_likelihood_func = lambda sample, log_params: U.binary_log_likelihood(sample['activation'], log_params)
        logQ = []
        if self.hparams.task in ['sbn', 'omni']:
            samples[-1] = {'activation': self._x}
            if self.mean_xs is not None:
                samples[-1]['activation'] -= self.mean_xs
            samples[-1]['activation'] = (samples[-1]['activation'] + 1) / 2.0
            with slim.arg_scope([slim.fully_connected], weights_initializer=slim.variance_scaling_initializer(), variables_collections=[Q_COLLECTION]):
                for i in xrange(self.hparams.n_layer):
                    input = 2.0 * samples[i - 1]['activation'] - 1.0
                    h = self._create_transformation(input, n_output=self.hparams.n_hidden, reuse=reuse, scope_prefix='q_%d' % i)
                    samples[i] = sampler(h, self.uniform_samples[i], i)
                    logQ.append(log_likelihood_func(samples[i], h))
            self.run_recognition_network = True
            return (logQ, samples)
        elif self.hparams.task == 'sp':
            samples[-1] = {'activation': tf.split(self._x, num_or_size_splits=2, axis=1)[0]}
            if self.mean_xs is not None:
                samples[-1]['activation'] -= np.split(self.mean_xs, 2, 0)[0]
            samples[-1]['activation'] = (samples[-1]['activation'] + 1) / 2.0
            with slim.arg_scope([slim.fully_connected], weights_initializer=slim.variance_scaling_initializer(), variables_collections=[Q_COLLECTION]):
                for i in xrange(self.hparams.n_layer):
                    input = 2.0 * samples[i - 1]['activation'] - 1.0
                    h = self._create_transformation(input, n_output=self.hparams.n_hidden, reuse=reuse, scope_prefix='q_%d' % i)
                    samples[i] = sampler(h, self.uniform_samples[i], i)
                    logQ.append(log_likelihood_func(samples[i], h))
            self.run_recognition_network = True
            return (logQ, samples)

    def _generator_network(self, samples, logQ, log_likelihood_func=None):
        if False:
            return 10
        'Returns learning signal and function.\n\n    This is the implementation for SBNs for the ELBO.\n\n    Args:\n      samples: dictionary of sampled latent variables\n      logQ: list of log q(h_i) terms\n      log_likelihood_func: function used to compute log probs for the latent\n        variables\n\n    Returns:\n      learning_signal: the "reward" function\n      function_term: part of the function that depends on the parameters\n        and needs to have the gradient taken through\n    '
        reuse = None if not self.run_generator_network else True
        if self.hparams.task in ['sbn', 'omni']:
            if log_likelihood_func is None:
                log_likelihood_func = lambda sample, log_params: U.binary_log_likelihood(sample['activation'], log_params)
            logPPrior = log_likelihood_func(samples[self.hparams.n_layer - 1], tf.expand_dims(self.prior, 0))
            with slim.arg_scope([slim.fully_connected], weights_initializer=slim.variance_scaling_initializer(), variables_collections=[P_COLLECTION]):
                for i in reversed(xrange(self.hparams.n_layer)):
                    if i == 0:
                        n_output = self.hparams.n_input
                    else:
                        n_output = self.hparams.n_hidden
                    input = 2.0 * samples[i]['activation'] - 1.0
                    h = self._create_transformation(input, n_output, reuse=reuse, scope_prefix='p_%d' % i)
                    if i == 0:
                        logP = U.binary_log_likelihood(self._x, h + self.train_bias)
                    else:
                        logPPrior += log_likelihood_func(samples[i - 1], h)
            self.run_generator_network = True
            return (logP + logPPrior - tf.add_n(logQ), logP + logPPrior)
        elif self.hparams.task == 'sp':
            with slim.arg_scope([slim.fully_connected], weights_initializer=slim.variance_scaling_initializer(), variables_collections=[P_COLLECTION]):
                n_output = int(self.hparams.n_input / 2)
                i = self.hparams.n_layer - 1
                input = 2.0 * samples[i]['activation'] - 1.0
                h = self._create_transformation(input, n_output, reuse=reuse, scope_prefix='p_%d' % i)
                logP = U.binary_log_likelihood(tf.split(self._x, num_or_size_splits=2, axis=1)[1], h + np.split(self.train_bias, 2, 0)[1])
            self.run_generator_network = True
            return (logP, logP)

    def _create_loss(self):
        if False:
            i = 10
            return i + 15
        (logQHard, samples) = self._recognition_network()
        (reinforce_learning_signal, reinforce_model_grad) = self._generator_network(samples, logQHard)
        logQHard = tf.add_n(logQHard)
        learning_signal = tf.stop_gradient(U.center(reinforce_learning_signal))
        self.optimizerLoss = -(learning_signal * logQHard + reinforce_model_grad)
        self.lHat = map(tf.reduce_mean, [reinforce_learning_signal, U.rms(learning_signal)])
        return reinforce_learning_signal

    def _reshape(self, t):
        if False:
            while True:
                i = 10
        return tf.transpose(tf.reshape(t, [self.n_samples, -1]))

    def compute_tensor_variance(self, t):
        if False:
            print('Hello World!')
        'Compute the mean per component variance.\n\n    Use a moving average to estimate the required moments.\n    '
        t_sq = tf.reduce_mean(tf.square(t))
        self.maintain_ema_ops.append(self.ema.apply([t, t_sq]))
        variance_estimator = self.ema.average(t_sq) - tf.reduce_mean(tf.square(self.ema.average(t)))
        return variance_estimator

    def _create_train_op(self, grads_and_vars, extra_grads_and_vars=[]):
        if False:
            print('Hello World!')
        '\n    Args:\n      grads_and_vars: gradients to apply and compute running average variance\n      extra_grads_and_vars: gradients to apply (not used to compute average variance)\n    '
        first_moment = U.vectorize(grads_and_vars, skip_none=True)
        second_moment = tf.square(first_moment)
        self.maintain_ema_ops.append(self.ema.apply([first_moment, second_moment]))
        if len(self.baseline_loss) > 0:
            mean_baseline_loss = tf.reduce_mean(tf.add_n(self.baseline_loss))
            extra_grads_and_vars += self.optimizer_class.compute_gradients(mean_baseline_loss, var_list=tf.get_collection('BASELINE'))
        extra_optimizer = tf.train.AdamOptimizer(learning_rate=10 * self.hparams.learning_rate, beta2=self.hparams.beta2)
        with tf.control_dependencies([tf.group(*[g for (g, _) in grads_and_vars + extra_grads_and_vars if g is not None])]):
            if self.eval_mode:
                grads_and_vars = [(g, v) for (g, v) in grads_and_vars if v not in tf.get_collection(P_COLLECTION)]
            train_op = self.optimizer_class.apply_gradients(grads_and_vars, global_step=self.global_step)
            if len(extra_grads_and_vars) > 0:
                extra_train_op = extra_optimizer.apply_gradients(extra_grads_and_vars)
            else:
                extra_train_op = tf.no_op()
            self.optimizer = tf.group(train_op, extra_train_op, *self.maintain_ema_ops)
        variance_estimator = self.ema.average(second_moment) - tf.square(self.ema.average(first_moment))
        self.grad_variance = tf.reduce_mean(variance_estimator)

    def _create_network(self):
        if False:
            print('Hello World!')
        logF = self._create_loss()
        self.optimizerLoss = tf.reduce_mean(self.optimizerLoss)
        grads_and_vars = self.optimizer_class.compute_gradients(self.optimizerLoss)
        self._create_train_op(grads_and_vars)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

    def partial_fit(self, X, n_samples=1):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'grad_variances'):
            grad_variance_field_to_return = self.grad_variances
        else:
            grad_variance_field_to_return = self.grad_variance
        (_, res, grad_variance, step, temperature) = self.sess.run((self.optimizer, self.lHat, grad_variance_field_to_return, self.global_step, self.temperature_variable), feed_dict={self.x: X, self.n_samples: n_samples})
        return (res, grad_variance, step, temperature)

    def partial_grad(self, X, n_samples=1):
        if False:
            print('Hello World!')
        (control_variate_grads, step) = self.sess.run((self.control_variate_grads, self.global_step), feed_dict={self.x: X, self.n_samples: n_samples})
        return (control_variate_grads, step)

    def partial_eval(self, X, n_samples=5):
        if False:
            for i in range(10):
                print('nop')
        if n_samples < 1000:
            (res, iwae) = self.sess.run((self.lHat, self.iwae), feed_dict={self.x: X, self.n_samples: n_samples})
            res = [iwae] + res
        else:
            assert n_samples % 100 == 0, 'When using large # of samples, it must be divisble by 100'
            res = []
            for i in xrange(int(n_samples / 100)):
                (logF,) = self.sess.run((self.logF,), feed_dict={self.x: X, self.n_samples: 100})
                res.append(logsumexp(logF, axis=1))
            res = [np.mean(logsumexp(res, axis=0) - np.log(n_samples))]
        return res

    def _mean_sample(self, log_alpha, _, layer):
        if False:
            return 10
        'Returns mean of random variables parameterized by log_alpha.'
        mu = tf.nn.sigmoid(log_alpha)
        return {'preactivation': mu, 'activation': mu, 'log_param': log_alpha}

    def _generate_randomness(self):
        if False:
            print('Hello World!')
        for i in xrange(self.hparams.n_layer):
            self.uniform_samples[i] = tf.stop_gradient(tf.random_uniform([self.batch_size, self.hparams.n_hidden]))

    def _u_to_v(self, log_alpha, u, eps=1e-08):
        if False:
            i = 10
            return i + 15
        'Convert u to tied randomness in v.'
        u_prime = tf.nn.sigmoid(-log_alpha)
        v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1)
        v_1 = tf.clip_by_value(v_1, 0, 1)
        v_1 = tf.stop_gradient(v_1)
        v_1 = v_1 * (1 - u_prime) + u_prime
        v_0 = u / tf.clip_by_value(u_prime, eps, 1)
        v_0 = tf.clip_by_value(v_0, 0, 1)
        v_0 = tf.stop_gradient(v_0)
        v_0 = v_0 * u_prime
        v = tf.where(u > u_prime, v_1, v_0)
        v = tf.check_numerics(v, 'v sampling is not numerically stable.')
        v = v + tf.stop_gradient(-v + u)
        return v

    def _random_sample(self, log_alpha, u, layer):
        if False:
            return 10
        'Returns sampled random variables parameterized by log_alpha.'
        if layer not in self.uniform_samples_v:
            self.uniform_samples_v[layer] = self._u_to_v(log_alpha, u)
        x = log_alpha + U.safe_log_prob(u) - U.safe_log_prob(1 - u)
        samples = tf.stop_gradient(tf.to_float(x > 0))
        return {'preactivation': x, 'activation': samples, 'log_param': log_alpha}

    def _random_sample_soft(self, log_alpha, u, layer, temperature=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns sampled random variables parameterized by log_alpha.'
        if temperature is None:
            temperature = self.hparams.temperature
        x = log_alpha + U.safe_log_prob(u) - U.safe_log_prob(1 - u)
        x /= tf.expand_dims(temperature, -1)
        if self.hparams.muprop_relaxation:
            y = tf.nn.sigmoid(x + log_alpha * tf.expand_dims(temperature / (temperature + 1), -1))
        else:
            y = tf.nn.sigmoid(x)
        return {'preactivation': x, 'activation': y, 'log_param': log_alpha}

    def _random_sample_soft_v(self, log_alpha, _, layer, temperature=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns sampled random variables parameterized by log_alpha.'
        v = self.uniform_samples_v[layer]
        return self._random_sample_soft(log_alpha, v, layer, temperature)

    def get_gumbel_gradient(self):
        if False:
            return 10
        (logQ, softSamples) = self._recognition_network(sampler=self._random_sample_soft)
        logQ = tf.add_n(logQ)
        (logPPrior, logP) = self._generator_network(softSamples)
        softELBO = logPPrior + logP - logQ
        gumbel_gradient = self.optimizer_class.compute_gradients(softELBO)
        debug = {'softELBO': softELBO}
        return (gumbel_gradient, debug)

    def _random_sample_switch(self, log_alpha, u, layer, switch_layer, temperature=None):
        if False:
            while True:
                i = 10
        'Run partial discrete, then continuous path.\n\n       Args:\n        switch_layer: this layer and beyond will be continuous\n    '
        if layer < switch_layer:
            return self._random_sample(log_alpha, u, layer)
        else:
            return self._random_sample_soft(log_alpha, u, layer, temperature)

    def _random_sample_switch_v(self, log_alpha, u, layer, switch_layer, temperature=None):
        if False:
            return 10
        'Run partial discrete, then continuous path.\n\n       Args:\n        switch_layer: this layer and beyond will be continuous\n    '
        if layer < switch_layer:
            return self._random_sample(log_alpha, u, layer)
        else:
            return self._random_sample_soft_v(log_alpha, u, layer, temperature)

    def get_nvil_gradient(self):
        if False:
            i = 10
            return i + 15
        'Compute the NVIL gradient.'
        (logQHard, samples) = self._recognition_network()
        (ELBO, reinforce_model_grad) = self._generator_network(samples, logQHard)
        logQHard = tf.add_n(logQHard)
        learning_signal = tf.stop_gradient(ELBO) - self._create_baseline()
        self.baseline_loss.append(tf.square(learning_signal))
        optimizerLoss = -(tf.stop_gradient(learning_signal) * logQHard + reinforce_model_grad)
        optimizerLoss = tf.reduce_mean(optimizerLoss)
        nvil_gradient = self.optimizer_class.compute_gradients(optimizerLoss)
        debug = {'ELBO': ELBO, 'RMS of centered learning signal': U.rms(learning_signal)}
        return (nvil_gradient, debug)

    def get_simple_muprop_gradient(self):
        if False:
            while True:
                i = 10
        ' Computes the simple muprop gradient.\n\n    This muprop control variate does not include the linear term.\n    '
        (logQHard, hardSamples) = self._recognition_network()
        (hardELBO, reinforce_model_grad) = self._generator_network(hardSamples, logQHard)
        (logQ, muSamples) = self._recognition_network(sampler=self._mean_sample)
        (muELBO, _) = self._generator_network(muSamples, logQ)
        scaling_baseline = self._create_eta(collection='BASELINE')
        learning_signal = hardELBO - scaling_baseline * muELBO - self._create_baseline()
        self.baseline_loss.append(tf.square(learning_signal))
        optimizerLoss = -(tf.stop_gradient(learning_signal) * tf.add_n(logQHard) + reinforce_model_grad)
        optimizerLoss = tf.reduce_mean(optimizerLoss)
        simple_muprop_gradient = self.optimizer_class.compute_gradients(optimizerLoss)
        debug = {'ELBO': hardELBO, 'muELBO': muELBO, 'RMS': U.rms(learning_signal)}
        return (simple_muprop_gradient, debug)

    def get_muprop_gradient(self):
        if False:
            while True:
                i = 10
        '\n    random sample function that actually returns mean\n    new forward pass that returns logQ as a list\n\n    can get x_i from samples\n    '
        (logQHard, hardSamples) = self._recognition_network()
        (hardELBO, reinforce_model_grad) = self._generator_network(hardSamples, logQHard)
        (logQ, muSamples) = self._recognition_network(sampler=self._mean_sample)
        (muELBO, _) = self._generator_network(muSamples, logQ)
        muELBOGrads = tf.gradients(tf.reduce_sum(muELBO), [muSamples[i]['activation'] for i in xrange(self.hparams.n_layer)])
        learning_signal = hardELBO
        optimizerLoss = 0.0
        learning_signals = []
        for i in xrange(self.hparams.n_layer):
            dfDiff = tf.reduce_sum(muELBOGrads[i] * (hardSamples[i]['activation'] - muSamples[i]['activation']), axis=1)
            dfMu = tf.reduce_sum(tf.stop_gradient(muELBOGrads[i]) * tf.nn.sigmoid(hardSamples[i]['log_param']), axis=1)
            scaling_baseline_0 = self._create_eta(collection='BASELINE')
            scaling_baseline_1 = self._create_eta(collection='BASELINE')
            learning_signals.append(learning_signal - scaling_baseline_0 * muELBO - scaling_baseline_1 * dfDiff - self._create_baseline())
            self.baseline_loss.append(tf.square(learning_signals[i]))
            optimizerLoss += logQHard[i] * tf.stop_gradient(learning_signals[i]) + tf.stop_gradient(scaling_baseline_1) * dfMu
        optimizerLoss += reinforce_model_grad
        optimizerLoss *= -1
        optimizerLoss = tf.reduce_mean(optimizerLoss)
        muprop_gradient = self.optimizer_class.compute_gradients(optimizerLoss)
        debug = {'ELBO': hardELBO, 'muELBO': muELBO}
        debug.update(dict([('RMS learning signal layer %d' % i, U.rms(learning_signal)) for (i, learning_signal) in enumerate(learning_signals)]))
        return (muprop_gradient, debug)

    def _create_gumbel_control_variate(self, logQHard, temperature=None):
        if False:
            i = 10
            return i + 15
        'Calculate gumbel control variate.\n    '
        if temperature is None:
            temperature = self.hparams.temperature
        (logQ, softSamples) = self._recognition_network(sampler=functools.partial(self._random_sample_soft, temperature=temperature))
        (softELBO, _) = self._generator_network(softSamples, logQ)
        logQ = tf.add_n(logQ)
        (logQ_v, softSamples_v) = self._recognition_network(sampler=functools.partial(self._random_sample_soft_v, temperature=temperature))
        (softELBO_v, _) = self._generator_network(softSamples_v, logQ_v)
        logQ_v = tf.add_n(logQ_v)
        learning_signal = tf.stop_gradient(softELBO_v)
        h = tf.stop_gradient(learning_signal) * tf.add_n(logQHard) - softELBO + softELBO_v
        extra = (softELBO_v, -softELBO + softELBO_v)
        return (h, extra)

    def _create_gumbel_control_variate_quadratic(self, logQHard, temperature=None):
        if False:
            return 10
        'Calculate gumbel control variate.\n    '
        if temperature is None:
            temperature = self.hparams.temperature
        h = 0
        extra = []
        for layer in xrange(self.hparams.n_layer):
            (logQ, softSamples) = self._recognition_network(sampler=functools.partial(self._random_sample_switch, switch_layer=layer, temperature=temperature))
            (softELBO, _) = self._generator_network(softSamples, logQ)
            (logQ_v, softSamples_v) = self._recognition_network(sampler=functools.partial(self._random_sample_switch_v, switch_layer=layer, temperature=temperature))
            (softELBO_v, _) = self._generator_network(softSamples_v, logQ_v)
            learning_signal = tf.stop_gradient(softELBO_v)
            h += tf.stop_gradient(learning_signal) * logQHard[layer] - softELBO + softELBO_v
            extra.append((softELBO_v, -softELBO + softELBO_v))
        return (h, extra)

    def _create_hard_elbo(self):
        if False:
            while True:
                i = 10
        (logQHard, hardSamples) = self._recognition_network()
        (hardELBO, reinforce_model_grad) = self._generator_network(hardSamples, logQHard)
        reinforce_learning_signal = tf.stop_gradient(hardELBO)
        baseline = self._create_baseline(collection='CV')
        reinforce_learning_signal = tf.stop_gradient(reinforce_learning_signal) - baseline
        nvil_gradient = (tf.stop_gradient(hardELBO) - baseline) * tf.add_n(logQHard) + reinforce_model_grad
        return (hardELBO, nvil_gradient, logQHard)

    def multiply_by_eta(self, h_grads, eta):
        if False:
            print('Hello World!')
        res = []
        eta_statistics = []
        for (g, v) in h_grads:
            if g is None:
                res.append((g, v))
            else:
                if 'network' not in eta:
                    eta['network'] = self._create_eta()
                res.append((g * eta['network'], v))
        eta_statistics.append(eta['network'])
        return (res, eta_statistics)

    def multiply_by_eta_per_layer(self, h_grads, eta):
        if False:
            for i in range(10):
                print('nop')
        res = []
        eta_statistics = []
        for (g, v) in h_grads:
            if g is None:
                res.append((g, v))
            else:
                if v not in eta:
                    eta[v] = self._create_eta()
                res.append((g * eta[v], v))
                eta_statistics.append(eta[v])
        return (res, eta_statistics)

    def multiply_by_eta_per_unit(self, h_grads, eta):
        if False:
            for i in range(10):
                print('nop')
        res = []
        eta_statistics = []
        for (g, v) in h_grads:
            if g is None:
                res.append((g, v))
            else:
                if v not in eta:
                    g_shape = g.shape_as_list()
                    assert len(g_shape) <= 2, 'Gradient has too many dimensions'
                    if len(g_shape) == 1:
                        eta[v] = self._create_eta(g_shape)
                    else:
                        eta[v] = self._create_eta([1, g_shape[1]])
                h_grads.append((g * eta[v], v))
                eta_statistics.extend(tf.nn.moments(tf.squeeze(eta[v]), axes=[0]))
        return (res, eta_statistics)

    def get_dynamic_rebar_gradient(self):
        if False:
            return 10
        'Get the dynamic rebar gradient (t, eta optimized).'
        tiled_pre_temperature = tf.tile([self.pre_temperature_variable], [self.batch_size])
        temperature = tf.exp(tiled_pre_temperature)
        (hardELBO, nvil_gradient, logQHard) = self._create_hard_elbo()
        if self.hparams.quadratic:
            (gumbel_cv, extra) = self._create_gumbel_control_variate_quadratic(logQHard, temperature=temperature)
        else:
            (gumbel_cv, extra) = self._create_gumbel_control_variate(logQHard, temperature=temperature)
        f_grads = self.optimizer_class.compute_gradients(tf.reduce_mean(-nvil_gradient))
        eta = {}
        (h_grads, eta_statistics) = self.multiply_by_eta_per_layer(self.optimizer_class.compute_gradients(tf.reduce_mean(gumbel_cv)), eta)
        model_grads = U.add_grads_and_vars(f_grads, h_grads)
        total_grads = model_grads
        g = U.vectorize(model_grads, set_none_to_zero=True)
        self.maintain_ema_ops.append(self.ema.apply([g]))
        gbar = 0
        variance_objective = tf.reduce_mean(tf.square(g - gbar))
        reinf_g_t = 0
        if self.hparams.quadratic:
            for layer in xrange(self.hparams.n_layer):
                (gumbel_learning_signal, _) = extra[layer]
                df_dt = tf.gradients(gumbel_learning_signal, tiled_pre_temperature)[0]
                (reinf_g_t_i, _) = self.multiply_by_eta_per_layer(self.optimizer_class.compute_gradients(tf.reduce_mean(tf.stop_gradient(df_dt) * logQHard[layer])), eta)
                reinf_g_t += U.vectorize(reinf_g_t_i, set_none_to_zero=True)
            reparam = tf.add_n([reparam_i for (_, reparam_i) in extra])
        else:
            (gumbel_learning_signal, reparam) = extra
            df_dt = tf.gradients(gumbel_learning_signal, tiled_pre_temperature)[0]
            (reinf_g_t, _) = self.multiply_by_eta_per_layer(self.optimizer_class.compute_gradients(tf.reduce_mean(tf.stop_gradient(df_dt) * tf.add_n(logQHard))), eta)
            reinf_g_t = U.vectorize(reinf_g_t, set_none_to_zero=True)
        (reparam_g, _) = self.multiply_by_eta_per_layer(self.optimizer_class.compute_gradients(tf.reduce_mean(reparam)), eta)
        reparam_g = U.vectorize(reparam_g, set_none_to_zero=True)
        reparam_g_t = tf.gradients(tf.reduce_mean(2 * tf.stop_gradient(g - gbar) * reparam_g), self.pre_temperature_variable)[0]
        variance_objective_grad = tf.reduce_mean(2 * (g - gbar) * reinf_g_t) + reparam_g_t
        debug = {'ELBO': hardELBO, 'etas': eta_statistics, 'variance_objective': variance_objective}
        return (total_grads, debug, variance_objective, variance_objective_grad)

    def get_rebar_gradient(self):
        if False:
            return 10
        'Get the rebar gradient.'
        (hardELBO, nvil_gradient, logQHard) = self._create_hard_elbo()
        if self.hparams.quadratic:
            (gumbel_cv, _) = self._create_gumbel_control_variate_quadratic(logQHard)
        else:
            (gumbel_cv, _) = self._create_gumbel_control_variate(logQHard)
        f_grads = self.optimizer_class.compute_gradients(tf.reduce_mean(-nvil_gradient))
        eta = {}
        (h_grads, eta_statistics) = self.multiply_by_eta_per_layer(self.optimizer_class.compute_gradients(tf.reduce_mean(gumbel_cv)), eta)
        model_grads = U.add_grads_and_vars(f_grads, h_grads)
        total_grads = model_grads
        variance_objective = tf.reduce_mean(tf.square(U.vectorize(model_grads, set_none_to_zero=True)))
        debug = {'ELBO': hardELBO, 'etas': eta_statistics, 'variance_objective': variance_objective}
        return (total_grads, debug, variance_objective)

class SBNSimpleMuProp(SBN):

    def _create_loss(self):
        if False:
            return 10
        (simple_muprop_gradient, debug) = self.get_simple_muprop_gradient()
        self.lHat = map(tf.reduce_mean, [debug['ELBO'], debug['muELBO']])
        return (debug['ELBO'], simple_muprop_gradient)

    def _create_network(self):
        if False:
            i = 10
            return i + 15
        (logF, loss_grads) = self._create_loss()
        self._create_train_op(loss_grads)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

class SBNMuProp(SBN):

    def _create_loss(self):
        if False:
            for i in range(10):
                print('nop')
        (muprop_gradient, debug) = self.get_muprop_gradient()
        self.lHat = map(tf.reduce_mean, [debug['ELBO'], debug['muELBO']])
        return (debug['ELBO'], muprop_gradient)

    def _create_network(self):
        if False:
            i = 10
            return i + 15
        (logF, loss_grads) = self._create_loss()
        self._create_train_op(loss_grads)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

class SBNNVIL(SBN):

    def _create_loss(self):
        if False:
            while True:
                i = 10
        (nvil_gradient, debug) = self.get_nvil_gradient()
        self.lHat = map(tf.reduce_mean, [debug['ELBO']])
        return (debug['ELBO'], nvil_gradient)

    def _create_network(self):
        if False:
            return 10
        (logF, loss_grads) = self._create_loss()
        self._create_train_op(loss_grads)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

class SBNRebar(SBN):

    def _create_loss(self):
        if False:
            print('Hello World!')
        (rebar_gradient, debug, variance_objective) = self.get_rebar_gradient()
        self.lHat = map(tf.reduce_mean, [debug['ELBO']])
        self.lHat.extend(map(tf.reduce_mean, debug['etas']))
        return (debug['ELBO'], rebar_gradient, variance_objective)

    def _create_network(self):
        if False:
            print('Hello World!')
        (logF, loss_grads, variance_objective) = self._create_loss()
        eta_grads = self.optimizer_class.compute_gradients(variance_objective, var_list=tf.get_collection('CV'))
        self._create_train_op(loss_grads, eta_grads)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

class SBNDynamicRebar(SBN):

    def _create_loss(self):
        if False:
            return 10
        (rebar_gradient, debug, variance_objective, variance_objective_grad) = self.get_dynamic_rebar_gradient()
        self.lHat = map(tf.reduce_mean, [debug['ELBO'], self.temperature_variable])
        self.lHat.extend(debug['etas'])
        return (debug['ELBO'], rebar_gradient, variance_objective, variance_objective_grad)

    def _create_network(self):
        if False:
            for i in range(10):
                print('nop')
        (logF, loss_grads, variance_objective, variance_objective_grad) = self._create_loss()
        eta_grads = self.optimizer_class.compute_gradients(variance_objective, var_list=tf.get_collection('CV')) + [(variance_objective_grad, self.pre_temperature_variable)]
        self._create_train_op(loss_grads, eta_grads)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

class SBNTrackGradVariances(SBN):
    """Follow NVIL, compute gradient variances for NVIL, MuProp and REBAR."""

    def compute_gradient_moments(self, grads_and_vars):
        if False:
            return 10
        first_moment = U.vectorize(grads_and_vars, set_none_to_zero=True)
        second_moment = tf.square(first_moment)
        self.maintain_ema_ops.append(self.ema.apply([first_moment, second_moment]))
        return (self.ema.average(first_moment), self.ema.average(second_moment))

    def _create_loss(self):
        if False:
            for i in range(10):
                print('nop')
        self.losses = [('NVIL', self.get_nvil_gradient), ('SimpleMuProp', self.get_simple_muprop_gradient), ('MuProp', self.get_muprop_gradient)]
        moments = []
        for (k, v) in self.losses:
            print(k)
            (gradient, debug) = v()
            if k == 'SimpleMuProp':
                ELBO = debug['ELBO']
                gradient_to_follow = gradient
            moments.append(self.compute_gradient_moments(gradient))
        self.losses.append(('DynamicREBAR', self.get_dynamic_rebar_gradient))
        (dynamic_rebar_gradient, _, variance_objective, variance_objective_grad) = self.get_dynamic_rebar_gradient()
        moments.append(self.compute_gradient_moments(dynamic_rebar_gradient))
        self.losses.append(('REBAR', self.get_rebar_gradient))
        (rebar_gradient, _, variance_objective2) = self.get_rebar_gradient()
        moments.append(self.compute_gradient_moments(rebar_gradient))
        mu = tf.reduce_mean(tf.stack([f for (f, _) in moments]), axis=0)
        self.grad_variances = []
        deviations = []
        for (f, s) in moments:
            self.grad_variances.append(tf.reduce_mean(s - tf.square(mu)))
            deviations.append(tf.reduce_mean(tf.square(f - mu)))
        self.lHat = map(tf.reduce_mean, [ELBO, self.temperature_variable, variance_objective_grad, variance_objective_grad * variance_objective_grad])
        self.lHat.extend(deviations)
        self.lHat.append(tf.log(tf.reduce_mean(mu * mu)))
        return (ELBO, gradient_to_follow, variance_objective + variance_objective2, variance_objective_grad)

    def _create_network(self):
        if False:
            while True:
                i = 10
        (logF, loss_grads, variance_objective, variance_objective_grad) = self._create_loss()
        eta_grads = self.optimizer_class.compute_gradients(variance_objective, var_list=tf.get_collection('CV')) + [(variance_objective_grad, self.pre_temperature_variable)]
        self._create_train_op(loss_grads, eta_grads)
        self.logF = self._reshape(logF)
        self.iwae = tf.reduce_mean(U.logSumExp(self.logF, axis=1) - tf.log(tf.to_float(self.n_samples)))

class SBNGumbel(SBN):

    def _random_sample_soft(self, log_alpha, u, layer, temperature=None):
        if False:
            print('Hello World!')
        'Returns sampled random variables parameterized by log_alpha.'
        if temperature is None:
            temperature = self.hparams.temperature
        x = log_alpha + U.safe_log_prob(u) - U.safe_log_prob(1 - u)
        x /= temperature
        if self.hparams.muprop_relaxation:
            x += temperature / (temperature + 1) * log_alpha
        y = tf.nn.sigmoid(x)
        return {'preactivation': x, 'activation': y, 'log_param': log_alpha}

    def _create_loss(self):
        if False:
            for i in range(10):
                print('nop')
        (logQHard, hardSamples) = self._recognition_network()
        (hardELBO, _) = self._generator_network(hardSamples, logQHard)
        (logQ, softSamples) = self._recognition_network(sampler=self._random_sample_soft)
        (softELBO, _) = self._generator_network(softSamples, logQ)
        self.optimizerLoss = -softELBO
        self.lHat = map(tf.reduce_mean, [hardELBO, softELBO])
        return hardELBO
default_hparams = tf.contrib.training.HParams(model='SBNGumbel', n_hidden=200, n_input=784, n_layer=1, nonlinear=False, learning_rate=0.001, temperature=0.5, n_samples=1, batch_size=24, trial=1, muprop_relaxation=True, dynamic_b=False, quadratic=True, beta2=0.99999, task='sbn')